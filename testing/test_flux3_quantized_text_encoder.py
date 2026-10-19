"""Load tied Qwen embeddings through the real Comfy quantization importer."""

import json
import unittest
from unittest.mock import patch

import torch

from testing.test_flux3_staged_sampling import load_flux3
from toolkit.util.comfy_quant_import import Int8Embedding
from toolkit.util.ostris_quant import OstrisLinear


EMBED = "model.language_model.embed_tokens"


def tiny_encoder():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.embed_tokens = torch.nn.Embedding(16, 8, device="meta")
    model.lm_head = torch.nn.Linear(8, 16, bias=False, device="meta")
    return model


def quantized_embedding(prefix=EMBED, convrot=True):
    config = {"format": "int8_tensorwise"}
    if convrot:
        config.update(convrot=True, convrot_groupsize=4)
    return {
        f"{prefix}.weight": (torch.arange(128).reshape(16, 8) % 15 - 7).to(torch.int8),
        f"{prefix}.weight_scale": torch.linspace(0.01, 0.16, 16),
        f"{prefix}.comfy_quant": torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8),
    }


class Flux3QuantizedTextEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = load_flux3().Flux3TextEncoder

    def test_tied_head_inherits_embedding_quantization_metadata(self):
        state = quantized_embedding()
        converted = self.encoder.convert_state_dict_on_load(state)
        for suffix in ("weight", "weight_scale", "comfy_quant"):
            self.assertIn(f"lm_head.{suffix}", converted)
            self.assertIs(converted[f"lm_head.{suffix}"], converted[f"{EMBED}.{suffix}"])

    def test_legacy_comfy_key_conversion_preserves_tied_quantization(self):
        state = quantized_embedding("model.embed_tokens")
        state["model.layers.0.input_layernorm.weight"] = torch.ones(8)
        converted = self.encoder.convert_state_dict_on_load(state)
        self.assertIn(f"{EMBED}.comfy_quant", converted)
        self.assertIn("model.language_model.layers.0.input_layernorm.weight", converted)
        self.assertIn("lm_head.comfy_quant", converted)
        self.assertNotIn("model.embed_tokens.comfy_quant", converted)

    def test_explicit_output_head_is_not_overwritten_or_requantized(self):
        state = quantized_embedding()
        explicit = torch.randn(16, 8)
        state["lm_head.weight"] = explicit
        converted = self.encoder.convert_state_dict_on_load(state)
        self.assertIs(converted["lm_head.weight"], explicit)
        self.assertNotIn("lm_head.comfy_quant", converted)
        self.assertNotIn("lm_head.weight_scale", converted)

    def test_float_embedding_still_supplies_the_tied_head(self):
        weight = torch.randn(16, 8)
        converted = self.encoder.convert_state_dict_on_load({f"{EMBED}.weight": weight})
        self.assertIs(converted["lm_head.weight"], weight)
        self.assertEqual(set(converted), {f"{EMBED}.weight", "lm_head.weight"})

    def test_quantized_embedding_and_tied_head_load_without_integer_parameters(self):
        for convrot in (True, False):
            with self.subTest(convrot=convrot):
                with patch.object(self.encoder, "aitk_from_config", side_effect=lambda _: tiny_encoder()):
                    try:
                        model = self.encoder.load_from_state_dict(
                            quantized_embedding(convrot=convrot), torch.float32, config=object()
                        )
                    except RuntimeError as error:
                        self.fail(f"Quantized tied embedding/head failed to load: {error}")
                self.assertTrue(model.aitk_is_quantized)
                self.assertIsInstance(model.lm_head, OstrisLinear)
                self.assertFalse(any(parameter.is_meta for parameter in model.parameters()))
                self.assertTrue(all(parameter.is_floating_point() for parameter in model.parameters()))
                embedding = model.model.language_model.embed_tokens
                if not convrot:
                    self.assertIsInstance(embedding, Int8Embedding)
                head_weight = model.lm_head.ostris_quantizer.dequantize_folded(model.lm_head)
                torch.testing.assert_close(head_weight, embedding.weight)

    def test_float_checkpoint_load_path_still_succeeds(self):
        weight = torch.randn(16, 8)
        with patch.object(self.encoder, "aitk_from_config", side_effect=lambda _: tiny_encoder()):
            model = self.encoder.load_from_state_dict(
                {f"{EMBED}.weight": weight}, torch.float32, config=object()
            )
        torch.testing.assert_close(model.model.language_model.embed_tokens.weight, weight)
        torch.testing.assert_close(model.lm_head.weight, weight)


if __name__ == "__main__":
    unittest.main()
