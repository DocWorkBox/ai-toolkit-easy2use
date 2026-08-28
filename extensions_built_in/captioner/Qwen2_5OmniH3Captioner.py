from collections import OrderedDict
import os

import torch
import tqdm

from toolkit.basic import flush

from .BaseCaptioner import BaseCaptioner
from .caption_output import (
    extract_first_h3_caption,
    extract_transcribed_dialogue,
    inject_h3_dialogue,
)


VIDEO_FPS = 2
USER_INSTRUCTION = (
    "Inspect the complete video and audio track before answering. Return the "
    "three required sections now. If dialogue or singing is audible, "
    "transcribe every confirmed utterance verbatim in "
    "integrated_multimodal_description "
    "using stable speaker IDs and <d>[Language]...</d>. If no speech is audible, "
    "do not invent any. Do not answer in multiple turns. Stop immediately after "
    "the non_diegetic_music section."
)
TRANSCRIPTION_SYSTEM_PROMPT = (
    "You are a precise speech transcription engine. Listen to the complete "
    "audio, focusing especially on sung vocals mixed with instruments. "
    "Transcribe every intelligible lyric as well as spoken dialogue. Use "
    "[unclear] only for words masked by music or noise; do not omit an entire "
    "vocal line because part of it is unclear. If there is no intelligible "
    "dialogue or singing, return exactly "
    "N/A. Otherwise return only one line per speaker in the form "
    "(S1) <d>[Language] exact original words</d>. Preserve the original "
    "language. Do not translate, complete, or guess unclear speech."
)
FOLLOWUP_STOP_STRINGS = [
    "\nAssistant\n",
    "\nHuman:",
    "\nUser:",
    "\nassistant\n",
    "\nhuman:",
    "\nuser:",
]


class Qwen2_5OmniH3Captioner(BaseCaptioner):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)

    def load_model(self):
        from optimum.quanto import freeze
        from toolkit.util.quantize import get_qtype, quantize
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        model_path = self.caption_config.model_name_or_path
        self.print_and_status_update("Loading Qwen2.5-Omni H3 prompt rewriter")
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            dtype=self.torch_dtype,
            device_map="cpu",
            attn_implementation="sdpa",
        ).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path, use_fast=False
        )

        self.model.to(self.device_torch)
        if self.caption_config.quantize:
            self.print_and_status_update("Quantizing Qwen2.5-Omni H3 prompt rewriter")
            quantize(self.model, weights=get_qtype(self.caption_config.qtype))
            freeze(self.model)
            flush()
        if self.caption_config.layer_offloading:
            from toolkit.memory_management import MemoryManager

            self.print_and_status_update(
                " - layer offloading enabled: linears stream from system RAM"
            )
            MemoryManager.attach(
                self.model,
                self.device_torch,
                offload_percent=self.caption_config.layer_offloading_percent,
                ignore_modules=[self.model.lm_head],
            )
            self.model.to(self.device_torch)
        if self.caption_config.low_vram and not self.caption_config.layer_offloading:
            self.model.to("cpu")
        flush()

    def _build_messages(self, file_path: str, dialogue: str = ""):
        instruction = USER_INSTRUCTION
        if dialogue:
            instruction += (
                "\n\nConfirmed audio transcription follows. Include every line "
                "verbatim in integrated_multimodal_description:\n" + dialogue
            )
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.caption_config.caption_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": file_path},
                    {"type": "text", "text": instruction},
                ],
            },
        ]

    @staticmethod
    def _build_transcription_messages():
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": TRANSCRIPTION_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "audio"},
                    {
                        "type": "text",
                        "text": (
                            "Separate the vocal from the accompaniment and transcribe "
                            "all intelligible dialogue and sung lyrics now."
                        ),
                    },
                ],
            },
        ]

    def _size_kwargs(self):
        max_pixels = self.caption_config.max_res * self.caption_config.max_res
        return {
            "shortest_edge": min(131072, max_pixels),
            "longest_edge": max_pixels,
        }

    def _load_media(self, file_path: str):
        from transformers.audio_utils import load_audio
        from transformers.video_utils import load_video

        frames = load_video(file_path, fps=VIDEO_FPS)
        if isinstance(frames, tuple):
            frames = frames[0]

        audio = None
        try:
            candidate = load_audio(file_path, sampling_rate=16000)
            if candidate is not None and candidate.size > 0:
                audio = candidate
        except Exception as error:
            print(
                f"Could not read audio from {file_path}; "
                f"captioning video only: {error}"
            )
        return frames, audio

    def _generate_texts(self, inputs, use_audio_in_video: bool, max_new_tokens: int):
        inputs = inputs.to(self.device_torch).to(self.torch_dtype)
        generated_ids = self.model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            max_new_tokens=max_new_tokens,
            stop_strings=FOLLOWUP_STOP_STRINGS,
            tokenizer=self.processor.tokenizer,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        return [
            text.strip()
            for text in self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]

    def _generate_text(self, inputs, use_audio_in_video: bool, max_new_tokens: int):
        return self._generate_texts(inputs, use_audio_in_video, max_new_tokens)[0]

    def _transcribe_audio(self, audio) -> str:
        return self._transcribe_audio_batch([audio])[0]

    def _transcribe_audio_batch(self, audio_items) -> list[str]:
        text = self.processor.tokenizer.apply_chat_template(
            self._build_transcription_messages(),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text] * len(audio_items),
            audio=audio_items,
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
            use_audio_in_video=False,
        )
        raw_transcripts = self._generate_texts(
            inputs,
            use_audio_in_video=False,
            max_new_tokens=max(128, min(1024, self.caption_config.max_new_tokens)),
        )
        dialogues = []
        for index, raw_transcript in enumerate(raw_transcripts, start=1):
            print(
                f"Qwen2.5-Omni raw audio transcription {index}: "
                f"{raw_transcript[:1000]}"
            )
            dialogue = extract_transcribed_dialogue(raw_transcript)
            if not dialogue and raw_transcript.strip().upper() != "N/A":
                print(
                    "Qwen2.5-Omni did not return usable dialogue or lyrics: "
                    f"{raw_transcript[:300]}"
                )
            dialogues.append(dialogue)
        return dialogues

    def _prepare_item(self, file_path: str) -> dict:
        frames, audio = self._load_media(file_path)
        return {"file": file_path, "frames": frames, "audio": audio}

    def _caption_batch(self, items: list[dict]) -> list[str]:
        use_audio = items[0]["audio"] is not None
        if any((item["audio"] is not None) != use_audio for item in items):
            raise ValueError("Qwen2.5-Omni batches must have matching audio modes")

        dialogues = (
            self._transcribe_audio_batch([item["audio"] for item in items])
            if use_audio
            else [""] * len(items)
        )
        texts = [
            self.processor.tokenizer.apply_chat_template(
                self._build_messages(item["file"], dialogue),
                tokenize=False,
                add_generation_prompt=True,
            )
            for item, dialogue in zip(items, dialogues)
        ]
        inputs = self.processor(
            text=texts,
            audio=[item["audio"] for item in items] if use_audio else None,
            videos=[item["frames"] for item in items],
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio,
            fps=VIDEO_FPS,
            do_sample_frames=False,
            size=self._size_kwargs(),
        )
        captions = self._generate_texts(
            inputs,
            use_audio_in_video=use_audio,
            max_new_tokens=self.caption_config.max_new_tokens,
        )
        return [
            inject_h3_dialogue(extract_first_h3_caption(caption), dialogue)
            for caption, dialogue in zip(captions, dialogues)
        ]

    def run_caption_loop(self):
        batch_size = max(1, int(self.caption_config.batch_size))
        buckets = {True: [], False: []}
        progress = tqdm.tqdm(
            total=len(self.file_paths), desc="Captioning files", unit="file"
        )

        def finish(item, caption=None, error=None):
            if error is None:
                if caption is not None:
                    self.save_caption_for_file(item["file"], caption)
                    self.caption_success_count += 1
            else:
                print(f"Error captioning file {item['file']}: {error}")
                self.caption_failure_count += 1
                self.caption_failures.append((item["file"], str(error)))
            self.step_num += 1
            self.update_step()
            progress.update(1)

        def process_items(items):
            if not items:
                return
            use_low_vram_moves = (
                self.caption_config.low_vram
                and not self.caption_config.layer_offloading
            )
            try:
                if use_low_vram_moves:
                    self.model.to(self.device_torch)
                captions = self._caption_batch(items)
                for item, caption in zip(items, captions):
                    finish(item, caption=caption)
            except Exception as batch_error:
                if len(items) == 1:
                    finish(items[0], error=batch_error)
                    return
                print(f"Batch failed ({batch_error}); retrying files individually")
                for item in items:
                    try:
                        finish(item, caption=self._caption_batch([item])[0])
                    except Exception as item_error:
                        finish(item, error=item_error)
            finally:
                if use_low_vram_moves:
                    self.model.to("cpu")
                    flush()

        try:
            for index, file_path in enumerate(self.file_paths, start=1):
                if self.is_ui_captioner:
                    self.maybe_stop()
                    if self.is_stopping:
                        break
                self.update_status(
                    "running",
                    f"正在准备 {index}/{len(self.file_paths)}：{os.path.basename(file_path)}",
                )
                try:
                    item = self._prepare_item(file_path)
                except Exception as error:
                    finish({"file": file_path}, error=error)
                    continue
                bucket = buckets[item["audio"] is not None]
                bucket.append(item)
                if len(bucket) >= batch_size:
                    process_items(bucket[:])
                    bucket.clear()
            for bucket in buckets.values():
                process_items(bucket)
        finally:
            progress.close()

    def get_caption_for_file(self, file_path: str) -> str:
        if (
            not self.caption_config.layer_offloading
            and self.model.device == torch.device("cpu")
        ):
            self.model.to(self.device_torch)
        try:
            return self._caption_batch([self._prepare_item(file_path)])[0]
        finally:
            if self.caption_config.low_vram and not self.caption_config.layer_offloading:
                self.model.to("cpu")
                flush()
