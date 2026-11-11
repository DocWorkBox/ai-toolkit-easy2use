from diffusers import AutoencoderKLWan

from .._mixin import OstrisModelMixin


class WanVAE(AutoencoderKLWan, OstrisModelMixin):
    """The wan 2.1/2.2 causal video VAE."""

    aitk_subfolder = "vae"

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        prefixes = ("encoder.downsamples.", "decoder.upsamples.")
        original = any(k.startswith(prefixes + ("encoder.conv1.", "decoder.conv1.", "conv1.")) for k in state_dict)
        if not original:
            return state_dict
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

        nested = any(k.startswith(prefixes) and k.split(".")[3] in ("downsamples", "upsamples") for k in state_dict)
        if not nested:
            return convert_wan_vae_to_diffusers(dict(state_dict))

        # Wan2.2 groups residual blocks and resamplers inside each stage.
        # Match diffusers/scripts/convert_wan_to_diffusers.py:convert_vae_22.
        converted = convert_wan_vae_to_diffusers({k: v for k, v in state_dict.items() if not k.startswith(prefixes)})
        residual_names = {
            ".residual.0.": ".norm1.", ".residual.2.": ".conv1.",
            ".residual.3.": ".norm2.", ".residual.6.": ".conv2.",
            ".shortcut.": ".conv_shortcut.",
        }
        for key, value in state_dict.items():
            if not key.startswith(prefixes):
                continue
            parts = key.split(".")
            down = parts[0] == "encoder"
            parts[1] = "down_blocks" if down else "up_blocks"
            if "residual" in parts or "shortcut" in parts:
                parts[3] = "resnets"
            elif "resample" in parts or "time_conv" in parts:
                parts = parts[:3] + ["downsampler" if down else "upsampler"] + parts[5:]
            new_key = ".".join(parts)
            for old, new in residual_names.items():
                new_key = new_key.replace(old, new)
            converted[new_key] = value
        return converted
