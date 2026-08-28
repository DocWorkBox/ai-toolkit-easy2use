from collections import OrderedDict

import torch

from toolkit.basic import flush

from .BaseCaptioner import BaseCaptioner


VIDEO_FPS = 2
USER_INSTRUCTION = "Analyze the supplied video and its audio track."


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

    def _build_messages(self, file_path: str):
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
                    {"type": "text", "text": USER_INSTRUCTION},
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

    def get_caption_for_file(self, file_path: str) -> str:
        frames, audio = self._load_media(file_path)
        use_audio = audio is not None
        # Qwen2.5-Omni warns when its processor wrapper sees a custom system
        # prompt because speech generation expects the stock prompt. This
        # captioner only generates text, so render the same bundled template
        # through the tokenizer and keep the task-specific system prompt.
        text = self.processor.tokenizer.apply_chat_template(
            self._build_messages(file_path),
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

        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        inputs = inputs.to(self.device_torch).to(self.torch_dtype)
        try:
            generated_ids = self.model.generate(
                **inputs,
                use_audio_in_video=use_audio,
                max_new_tokens=self.caption_config.max_new_tokens,
            )
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return caption.strip()
        finally:
            if self.caption_config.low_vram:
                self.model.to("cpu")
                flush()
