import json

from PIL import Image

from .Ideogram4Captioner import Ideogram4Captioner, MIN_NEW_TOKENS
from .RemoteAPICaptioner import RemoteAPICaptioner
from .prompts.ideogram4_caption_prompt import ideogram4_caption_prompt


class Ideogram4APICaptioner(RemoteAPICaptioner):
    compute_aspect_ratio = Ideogram4Captioner.compute_aspect_ratio
    _extract_json = Ideogram4Captioner._extract_json
    _normalize_caption = Ideogram4Captioner._normalize_caption

    def __init__(self, process_id: int, job, config, **kwargs):
        super(Ideogram4APICaptioner, self).__init__(process_id, job, config, **kwargs)
        if self.caption_config.max_new_tokens < MIN_NEW_TOKENS:
            print(
                f"[Ideogram4APICaptioner] Raising max_new_tokens "
                f"{self.caption_config.max_new_tokens} -> {MIN_NEW_TOKENS} "
                f"(the deconstruction JSON is long)."
            )
            self.caption_config.max_new_tokens = MIN_NEW_TOKENS

    def _aspect_ratio_for_file(self, file_path: str) -> str:
        with Image.open(file_path) as probe:
            return self.compute_aspect_ratio(probe.width, probe.height)

    def build_prompt(self, aspect_ratio: str) -> str:
        user_instructions = (self.caption_config.caption_prompt or "").strip()
        if not user_instructions:
            user_instructions = "None."
        prompt = ideogram4_caption_prompt.replace("{{aspect_ratio}}", aspect_ratio)
        prompt = prompt.replace("{{user_instructions}}", user_instructions)
        return prompt

    def build_prompt_for_file(self, file_path: str) -> str:
        return self.build_prompt(self._aspect_ratio_for_file(file_path))

    def get_caption_for_file(self, file_path: str) -> str | None:
        output_text = super().get_caption_for_file(file_path)
        data = self._extract_json(output_text)
        if data is None:
            print(
                f"[Ideogram4APICaptioner] Could not parse JSON for {file_path}; "
                f"saving raw output."
            )
            return output_text

        data = self._normalize_caption(data, self._aspect_ratio_for_file(file_path))
        return json.dumps(data, ensure_ascii=False, indent=2)
