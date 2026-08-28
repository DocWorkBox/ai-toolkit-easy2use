from collections import OrderedDict

import torch

from toolkit.basic import flush

from .BaseCaptioner import BaseCaptioner
from .caption_output import (
    extract_first_h3_caption,
    extract_tagged_dialogue,
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
    "audio. If there is no intelligible dialogue or singing, return exactly "
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
        if self.caption_config.low_vram:
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
                        "text": "Transcribe all intelligible dialogue and lyrics now.",
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

    def _generate_text(self, inputs, use_audio_in_video: bool, max_new_tokens: int):
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
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _transcribe_audio(self, audio) -> str:
        text = self.processor.tokenizer.apply_chat_template(
            self._build_transcription_messages(),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            audio=[audio],
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
            use_audio_in_video=False,
        )
        raw_transcript = self._generate_text(
            inputs,
            use_audio_in_video=False,
            max_new_tokens=max(128, min(1024, self.caption_config.max_new_tokens)),
        )
        dialogue = extract_tagged_dialogue(raw_transcript)
        if not dialogue and raw_transcript.strip().upper() != "N/A":
            print(
                "Qwen2.5-Omni detected possible speech but did not return tagged "
                f"dialogue: {raw_transcript[:300]}"
            )
        return dialogue

    def get_caption_for_file(self, file_path: str) -> str:
        frames, audio = self._load_media(file_path)
        use_audio = audio is not None
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        try:
            dialogue = self._transcribe_audio(audio) if use_audio else ""
            # Render through the tokenizer because this captioner intentionally
            # uses a task-specific system prompt and only generates text.
            text = self.processor.tokenizer.apply_chat_template(
                self._build_messages(file_path, dialogue),
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text],
                audio=[audio] if use_audio else None,
                videos=[frames],
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio,
                fps=VIDEO_FPS,
                do_sample_frames=False,
                size=self._size_kwargs(),
            )
            caption = self._generate_text(
                inputs,
                use_audio_in_video=use_audio,
                max_new_tokens=self.caption_config.max_new_tokens,
            )
            caption = extract_first_h3_caption(caption)
            return inject_h3_dialogue(caption, dialogue)
        finally:
            if self.caption_config.low_vram:
                self.model.to("cpu")
                flush()
