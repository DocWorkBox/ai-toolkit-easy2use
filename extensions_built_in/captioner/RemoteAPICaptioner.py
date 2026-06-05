import json
import os
import concurrent.futures
from collections import OrderedDict
from urllib import error, request

from .BaseCaptioner import BaseCaptioner
from .remote_api_utils import (
    build_anthropic_payload,
    build_openai_payload,
    encode_image_to_jpeg_base64,
    parse_anthropic_caption,
    parse_openai_caption,
    resolve_api_endpoint,
)


class RemoteAPICaptioner(BaseCaptioner):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(RemoteAPICaptioner, self).__init__(process_id, job, config, **kwargs)
        self.endpoint = None

    def load_model(self):
        self.print_and_status_update("Initializing remote API captioner")
        if not self.caption_config.model_name_or_path or not self.caption_config.model_name_or_path.strip():
            raise ValueError("model_name_or_path is required for RemoteAPICaptioner")
        if not self.caption_config.api_base_url or not self.caption_config.api_base_url.strip():
            raise ValueError("api_base_url is required for RemoteAPICaptioner")
        if not self.caption_config.api_key or not self.caption_config.api_key.strip():
            raise ValueError("api_key is required for RemoteAPICaptioner")
        self.endpoint = resolve_api_endpoint(
            self.caption_config.api_base_url, self.caption_config.api_protocol
        )

    def _build_headers(self):
        protocol = self.caption_config.api_protocol
        if protocol == "anthropic":
            return {
                "Content-Type": "application/json",
                "x-api-key": self.caption_config.api_key,
                "anthropic-version": "2023-06-01",
            }
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.caption_config.api_key}",
        }

    def build_prompt_for_file(self, file_path: str) -> str:
        return self.caption_config.caption_prompt

    def get_caption_for_file(self, file_path: str) -> str:
        img = self.load_pil_image(file_path, max_res=self.caption_config.max_res)
        image_base64 = encode_image_to_jpeg_base64(img)
        prompt = self.build_prompt_for_file(file_path)
        protocol = self.caption_config.api_protocol
        if protocol == "anthropic":
            payload = build_anthropic_payload(
                model_name=self.caption_config.model_name_or_path,
                prompt=prompt,
                image_base64=image_base64,
                media_type="image/jpeg",
                max_new_tokens=self.caption_config.max_new_tokens,
            )
        else:
            payload = build_openai_payload(
                model_name=self.caption_config.model_name_or_path,
                prompt=prompt,
                image_data_url=f"data:image/jpeg;base64,{image_base64}",
                max_new_tokens=self.caption_config.max_new_tokens,
            )

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=body,
            headers=self._build_headers(),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                response_json = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as e:
            details = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Remote API request failed with status {e.code}: {details}"
            ) from e
        except error.URLError as e:
            raise RuntimeError(f"Remote API request failed: {e}") from e

        if protocol == "anthropic":
            return parse_anthropic_caption(response_json)
        return parse_openai_caption(response_json)

    def _caption_and_save_file(self, file_path: str) -> str | None:
        file_caption = self.get_caption_for_file(file_path)
        if file_caption is not None:
            self.save_caption_for_file(file_path, file_caption)
        return file_caption

    def run_caption_loop(self):
        total_files = len(self.file_paths)
        max_workers = max(1, self.caption_config.api_concurrency)
        if max_workers == 1 or total_files <= 1:
            return super().run_caption_loop()

        self.print_and_status_update(f"Remote API 并发数：{max_workers}")
        processed_count = 0
        next_index = 0
        pending: dict[concurrent.futures.Future, str] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while next_index < total_files and len(pending) < max_workers:
                file_path = self.file_paths[next_index]
                pending[executor.submit(self._caption_and_save_file, file_path)] = file_path
                next_index += 1

            while pending:
                if self.is_ui_captioner:
                    self.maybe_stop()
                    if self.is_stopping:
                        break

                done, _ = concurrent.futures.wait(
                    list(pending.keys()),
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    continue

                for future in done:
                    file_path = pending.pop(future)
                    processed_count += 1
                    try:
                        file_caption = future.result()
                        if file_caption is not None:
                            self.caption_success_count += 1
                        self.update_status(
                            "running",
                            f"正在打标 {processed_count}/{total_files}：{os.path.basename(file_path)}",
                        )
                    except Exception as e:
                        print(f"Error captioning file {file_path}: {e}")
                        self.caption_failure_count += 1
                        self.caption_failures.append((file_path, str(e)))
                        self.update_status(
                            "running",
                            f"打标失败 {self.caption_failure_count} 个，最近失败：{os.path.basename(file_path)} - {e}",
                        )

                    while next_index < total_files and len(pending) < max_workers:
                        next_file_path = self.file_paths[next_index]
                        pending[executor.submit(self._caption_and_save_file, next_file_path)] = next_file_path
                        next_index += 1

            if self.is_stopping:
                for future in pending:
                    future.cancel()
