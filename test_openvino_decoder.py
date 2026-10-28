"""Validate cached GPU decoding and CPU recovery without GPU hardware."""

import copy
import importlib.util
import unittest
from unittest.mock import Mock, patch

import torch
from whisper.model import ModelDimensions, Whisper

from inference_backend import backend_name
from openvino_decoder import (
    CrossAttentionGraph,
    DecoderGraph,
    OpenVINODecoder,
    compile_decoder,
)


def small_model():
    torch.manual_seed(13)
    model = Whisper(
        ModelDimensions(
            n_mels=4,
            n_audio_ctx=4,
            n_audio_state=8,
            n_audio_head=2,
            n_audio_layer=1,
            n_vocab=32,
            n_text_ctx=16,
            n_text_state=8,
            n_text_head=2,
            n_text_layer=2,
        )
    )
    torch.nn.init.normal_(model.decoder.positional_embedding, std=0.02)
    return model.eval()


class GraphRequest:
    """Emulate compiled graphs using independent weights and no Python hooks."""

    def __init__(self, graph):
        self.graph = graph
        self.fail = False

    def infer(self, inputs):
        if self.fail:
            raise RuntimeError("GPU disappeared")
        tensors = [torch.from_numpy(inputs[index]) for index in range(len(inputs))]
        with torch.no_grad():
            outputs = self.graph(*tensors)
        return {index: value.numpy() for index, value in enumerate(outputs)}


class CompiledGraph:
    def __init__(self, graph):
        self.request = GraphRequest(graph)

    def create_infer_request(self):
        return self.request

    def output(self, index):
        return index


def accelerated_model():
    model = small_model()
    exported = copy.deepcopy(model.decoder)
    cross = CompiledGraph(CrossAttentionGraph(exported))
    compiled = CompiledGraph(DecoderGraph(exported))
    model.decoder = OpenVINODecoder(model.decoder, cross, compiled, "GPU")
    return model


class DecoderTests(unittest.TestCase):
    def test_prefill_cached_steps_and_cpu_recovery_preserve_logits(self):
        model = accelerated_model()
        reference = copy.deepcopy(model.decoder.original)
        audio = torch.randn(1, 4, 8)
        tokens = torch.tensor([[1, 2, 3]])
        cache, hooks = model.install_kv_cache_hooks()
        try:
            with torch.no_grad():
                actual = model.decoder(tokens, audio, cache)
                expected = reference(tokens, audio)
                torch.testing.assert_close(actual, expected)
                for token in (4, 5):
                    step = torch.tensor([[token]])
                    tokens = torch.cat((tokens, step), dim=1)
                    actual = model.decoder(step, audio, cache)
                    torch.testing.assert_close(actual, reference(tokens, audio)[:, -1:])
                model.decoder.request.fail = True
                step = torch.tensor([[6]])
                tokens = torch.cat((tokens, step), dim=1)
                with self.assertLogs("openvino_decoder", level="ERROR"):
                    actual = model.decoder(step, audio, cache)
                torch.testing.assert_close(actual, reference(tokens, audio)[:, -1:])
                self.assertFalse(model.decoder.active)
                self.assertEqual(backend_name(model), "CPU encoder / CPU decoder")
                self.assertEqual(cache[model.decoder.blocks[0].attn.key].shape[1], 6)
        finally:
            for hook in hooks:
                hook.remove()

    def test_beam_reordering_uses_whisper_cache_keys(self):
        from whisper.decoding import PyTorchInference

        model = accelerated_model()
        reference = copy.deepcopy(model.decoder.original)
        audio = torch.randn(1, 4, 8).repeat(2, 1, 1)
        tokens = torch.tensor([[1, 2], [3, 4]])
        inference = PyTorchInference(model, initial_token_length=2)
        try:
            with torch.no_grad():
                inference.logits(tokens, audio)
                inference.rearrange_kv_cache([1, 0])
                tokens = torch.cat((tokens[[1, 0]], torch.tensor([[5], [6]])), dim=1)
                actual = inference.logits(tokens, audio)
                expected = reference(tokens, audio)[:, -1:]
                torch.testing.assert_close(actual, expected)
        finally:
            inference.cleanup_caching()

    @unittest.skipUnless(importlib.util.find_spec("openvino"), "OpenVINO not installed")
    def test_slower_gpu_or_invalid_scores_keeps_cpu_decoder(self):
        for gpu_time, cpu_time, valid, selected in (
            (0.001, 0.002, True, True),
            (0.003, 0.002, True, False),
            (0.001, 0.002, False, False),
        ):
            with self.subTest(gpu_time=gpu_time, valid=valid):
                model = small_model()
                decoder = Mock(active=True)
                expected = torch.ones(1, 1, 32)
                actual = expected if valid else expected + 1
                core = Mock()
                with (
                    patch("openvino.convert_model"),
                    patch("openvino_decoder.OpenVINODecoder", return_value=decoder),
                    patch(
                        "openvino_decoder.decoder_latency",
                        side_effect=[(gpu_time, actual), (cpu_time, expected)],
                    ),
                ):
                    if not valid:
                        with self.assertLogs("openvino_decoder", level="WARNING"):
                            result = compile_decoder(model, ["GPU"], core, {})
                    else:
                        result = compile_decoder(model, ["GPU"], core, {})
                self.assertEqual(result is decoder, selected)


if __name__ == "__main__":
    unittest.main()
