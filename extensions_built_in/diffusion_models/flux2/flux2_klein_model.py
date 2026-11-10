import os
from .flux2_model import Flux2Model
from transformers import Qwen3ForCausalLM, Qwen2Tokenizer
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush
from .src.model import Klein9BParams, Klein4BParams


class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "ai-toolkit/flux2_vae"
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
        
        # Check for local Qwen model
        te_path = self.flux2_klein_te_path
        tokenizer_path = self.flux2_klein_te_path
        model_path = self.model_config.name_or_path
        
        # Check priority: 1. model_path/text_encoder, 2. model_path/qwen, 3. model_path itself
        possible_paths = [
            os.path.join(model_path, "text_encoder"),
            os.path.join(model_path, "qwen"),
            os.path.join(model_path, "Qwen"),
            os.path.join(model_path, "Qwen3"),
            model_path
        ]

        if os.path.isfile(model_path):
            model_dir = os.path.dirname(model_path)
            possible_paths.extend([
                os.path.join(model_dir, "text_encoder"),
                os.path.join(model_dir, "qwen"),
                os.path.join(model_dir, "Qwen"),
                os.path.join(model_dir, "Qwen3"),
                model_dir
            ])
        
        found_local = False
        for p in possible_paths:
             # Check for config.json as a marker for a transformer model
            if os.path.exists(os.path.join(p, "config.json")):
                te_path = p
                self.print_and_status_update(f"Found local Qwen at {te_path}")
                
                # Check for tokenizer in the same folder or in 'tokenizer' subfolder
                if os.path.exists(os.path.join(p, "tokenizer_config.json")):
                    tokenizer_path = p
                elif os.path.exists(os.path.join(os.path.dirname(p), "tokenizer", "tokenizer_config.json")):
                     tokenizer_path = os.path.join(os.path.dirname(p), "tokenizer")
                elif os.path.exists(os.path.join(model_path, "tokenizer", "tokenizer_config.json")):
                    tokenizer_path = os.path.join(model_path, "tokenizer")
                else:
                    # Fallback to te_path if we can't find it elsewhere, might download
                    tokenizer_path = p
                
                found_local = True
                break

        if not found_local:
            self.print_and_status_update(f"Could not find local Qwen in possible paths. Will attempt to download {te_path}")


        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen3")

        text_encoder: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained(
            te_path,
            torch_dtype=dtype,
        )
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Qwen3")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()
        elif not self.model_config.low_vram:
            text_encoder.to(self.device_torch, dtype=dtype)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

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
