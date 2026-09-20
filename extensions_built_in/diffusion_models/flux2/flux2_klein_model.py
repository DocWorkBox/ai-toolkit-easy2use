import os

from .flux2_model import Flux2Model
from transformers import Qwen3ForCausalLM, Qwen2Tokenizer
from toolkit.models.v2.text_encoders.qwen3 import Qwen3TextEncoder
from toolkit.config_modules import ModelConfig
from toolkit.basic import flush
from .src.model import Klein9BParams, Klein4BParams


class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "/datasets/studio/huggingface/models/flux2_vae/ae.safetensors"
    flux2_is_guidance_distilled: bool = False

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs,
        )
        # use the new format on this new model by default
        self.use_old_lokr_format = False

    def load_te(self):
        if self.flux2_klein_te_path is None:
            raise ValueError("flux2_klein_te_path must be set for Flux2KleinModel")
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen3")

        te_path = self.flux2_klein_te_path
        tokenizer_path = self.flux2_klein_te_path
        model_path = self.model_config.name_or_path
        possible_paths = [
            os.path.join(model_path, "text_encoder"),
            os.path.join(model_path, "qwen"),
            os.path.join(model_path, "Qwen"),
            os.path.join(model_path, "Qwen3"),
            model_path,
        ]
        if os.path.isfile(model_path):
            model_dir = os.path.dirname(model_path)
            possible_paths.extend(
                [
                    os.path.join(model_dir, "text_encoder"),
                    os.path.join(model_dir, "qwen"),
                    os.path.join(model_dir, "Qwen"),
                    os.path.join(model_dir, "Qwen3"),
                    model_dir,
                ]
            )
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "config.json")):
                te_path = path
                tokenizer_path = path
                for candidate in (
                    path,
                    os.path.join(os.path.dirname(path), "tokenizer"),
                    os.path.join(model_path, "tokenizer"),
                ):
                    if os.path.exists(os.path.join(candidate, "tokenizer_config.json")):
                        tokenizer_path = candidate
                        break
                self.print_and_status_update(f"Found local Qwen at {te_path}")
                break

        # load + quantize + offload + placement, all driven by model_config
        text_encoder = Qwen3TextEncoder.load(
            te_path, subfolder="", **self.component_load_kwargs("te")
        )
        flush()

        tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
        return text_encoder, tokenizer


class Flux2Klein4BModel(Flux2KleinModel):
    arch = "flux2_klein_4b"
    flux2_klein_te_path: str = "Qwen/Qwen3-4B"
    flux2_te_filename: str = "flux-2-klein-base-4b.safetensors"

    def get_flux2_params(self):
        return Klein4BParams()

    def get_base_model_version(self):
        return "flux2_klein_4b"


class Flux2Klein9BModel(Flux2KleinModel):
    arch = "flux2_klein_9b"
    flux2_klein_te_path: str = "Qwen/Qwen3-8B"
    flux2_te_filename: str = "flux-2-klein-base-9b.safetensors"

    def get_flux2_params(self):
        return Klein9BParams()

    def get_base_model_version(self):
        return "flux2_klein_9b"
