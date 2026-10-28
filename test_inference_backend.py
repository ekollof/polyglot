"""Accelerator selection and failure recovery, without GPU hardware/downloads."""

import importlib.util
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from inference_backend import (
    CachedAudioEncoder,
    OpenVINOEncoder,
    _load_torch,
    _openvino_encoder,
    _warmup_torch,
    load_whisper,
    openvino_devices,
    torch_devices,
)


class Encoder(torch.nn.Module):
    def forward(self, mel):
        return mel.transpose(1, 2)[:, ::2, :]


class BackendTests(unittest.TestCase):
    def test_encoder_reuses_identical_inputs_but_not_changed_samples_or_dtype(self):
        original = Encoder()
        with patch.object(original, "forward", wraps=original.forward) as forward:
            cached = CachedAudioEncoder(original)
            mel = torch.ones(1, 2, 4)
            with torch.no_grad():
                first = cached(mel)
                torch.testing.assert_close(cached(mel.clone()), first)
                self.assertEqual(forward.call_count, 1)
                mel.add_(1)
                torch.testing.assert_close(cached(mel), original(mel))
                calls = forward.call_count
                cached(mel.double())
                self.assertEqual(forward.call_count, calls + 1)
            self.assertEqual(cached.cache_hits, 1)

    def test_encoder_cache_does_not_disable_gradients(self):
        original = torch.nn.Linear(2, 2)
        cached = CachedAudioEncoder(original)
        value = torch.ones(1, 2)
        with torch.no_grad():
            cached(value)
        result = cached(value)
        result.sum().backward()
        self.assertIsNotNone(original.weight.grad)
        self.assertTrue(result.requires_grad)

    def test_encoder_retains_both_padding_variants_and_bounds_memory(self):
        original = Encoder()
        with patch.object(original, "forward", wraps=original.forward) as forward:
            cached = CachedAudioEncoder(original)
            first = torch.zeros(1, 2, 4)
            second = torch.ones(1, 2, 4)
            with torch.no_grad():
                cached(first)
                cached(second)
                cached(first)
                self.assertEqual(forward.call_count, 2)
                cached(second * 2)
                self.assertEqual(len(cached._entries), 2)
                cached(second)
                self.assertEqual(forward.call_count, 4)

    def test_device_priority_includes_numbered_gpus(self):
        self.assertEqual(
            openvino_devices(["CPU", "GPU.1", "NPU", "GPU.0", "AUTO"]),
            ["NPU", "GPU.1", "GPU.0", "CPU"],
        )

    def test_all_native_gpu_runtimes_are_probed(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.xpu.is_available", return_value=True),
            patch("torch.xpu.device_count", return_value=1),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(torch_devices(), ["cuda:0", "cuda:1", "xpu:0", "mps"])

    def test_broken_runtime_probe_does_not_hide_other_gpus(self):
        with (
            patch("torch.cuda.is_available", side_effect=RuntimeError("driver broken")),
            patch("torch.xpu.is_available", return_value=True),
            patch("torch.xpu.device_count", return_value=1),
            patch("torch.backends.mps.is_available", return_value=False),
            self.assertLogs("inference_backend", level="WARNING"),
        ):
            self.assertEqual(torch_devices(), ["xpu:0"])

    def test_registered_third_party_accelerators_are_detected(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.xpu.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch(
                "torch.accelerator.current_accelerator",
                return_value=SimpleNamespace(type="npu"),
            ),
            patch("torch.accelerator.device_count", return_value=2),
        ):
            self.assertEqual(torch_devices(), ["npu:0", "npu:1"])

    def test_native_gpu_selection_and_precision(self):
        from whisper.model import MultiHeadAttention

        for selected, hip, label in (
            ("cuda:0", None, "CUDA"),
            ("cuda:0", "7.2", "ROCm"),
            ("xpu:0", None, "XPU"),
            ("mps", None, "MPS"),
        ):
            with (
                self.subTest(label=label),
                patch("inference_backend.torch_devices", return_value=[selected]),
                patch("torch.version.hip", hip),
                patch("inference_backend._load_torch") as load,
                patch("inference_backend._warmup_torch", return_value=False),
                patch("inference_backend._openvino_encoder") as ov,
            ):
                model, backend = load_whisper("tiny")
                self.assertIs(model, load.return_value)
                self.assertEqual(backend, f"PyTorch {label}")
                self.assertFalse(model.inference_fp16)
                self.assertEqual(MultiHeadAttention.use_sdpa, hip is None)
                load.assert_called_once_with("tiny", selected)
                ov.assert_not_called()
        MultiHeadAttention.use_sdpa = True

    def test_failed_gpu_warmup_tries_another_gpu(self):
        with (
            patch("inference_backend.torch_devices", return_value=["cuda:0", "xpu:0"]),
            patch("inference_backend._load_torch") as load,
            patch(
                "inference_backend._warmup_torch",
                side_effect=[RuntimeError("unsupported kernel"), True],
            ),
            self.assertLogs("inference_backend", level="WARNING"),
        ):
            model, backend = load_whisper("tiny")
        self.assertEqual(backend, "PyTorch XPU")
        self.assertTrue(model.inference_fp16)
        self.assertEqual(
            [call.args[1] for call in load.call_args_list], ["cuda:0", "xpu:0"]
        )

    def test_broken_gpu_falls_back_to_cpu(self):
        model = SimpleNamespace(encoder=Encoder(), decoder=None)
        with (
            patch("inference_backend.torch_devices", return_value=["cuda:0"]),
            patch(
                "inference_backend._load_torch",
                side_effect=[RuntimeError("GPU failed"), model],
            ) as load,
            patch("inference_backend._openvino_encoder", return_value=None),
            patch("inference_backend._openvino_decoder", return_value=None),
            self.assertLogs("inference_backend", level="WARNING"),
        ):
            _, backend = load_whisper("tiny")
        self.assertEqual(backend, "PyTorch CPU")
        self.assertEqual(load.call_args.args[1], "cpu")

    def test_explicit_cpu_bypasses_accelerators(self):
        with (
            patch("inference_backend._load_torch") as load,
            patch("inference_backend._warmup_torch", return_value=False),
            patch("inference_backend._openvino_encoder") as ov,
        ):
            _, backend = load_whisper("tiny", device="cpu")
        load.assert_called_once_with("tiny", "cpu")
        ov.assert_not_called()
        self.assertEqual(backend, "PyTorch CPU")

    def test_openvino_missing_or_conversion_failure_preserves_cpu_model(self):
        for error in (
            ImportError("not installed"),
            RuntimeError("unsupported conversion"),
        ):
            model = SimpleNamespace(encoder=Encoder(), decoder=None)
            with (
                self.subTest(error=error),
                patch("inference_backend.torch_devices", return_value=[]),
                patch("inference_backend._load_torch", return_value=model),
                patch("inference_backend._openvino_encoder", side_effect=error),
                patch("inference_backend._openvino_decoder", side_effect=error),
                self.assertLogs("inference_backend", level="WARNING"),
            ):
                loaded, backend = load_whisper("tiny")
                self.assertIs(loaded, model)
                self.assertEqual(backend, "PyTorch CPU")

    def test_sparse_alignment_metadata_stays_on_cpu(self):
        model = torch.nn.Linear(2, 2)
        alignment = torch.eye(2).to_sparse()
        model.register_buffer("alignment_heads", alignment, persistent=False)

        def move(device):
            self.assertNotIn("alignment_heads", model._buffers)
            self.assertEqual(device, "mps")

        with (
            patch("whisper.load_model", return_value=model),
            patch.object(model, "to", side_effect=move),
        ):
            self.assertIs(_load_torch("tiny", "mps"), model)
        self.assertIs(model.alignment_heads, alignment)

    def test_half_precision_failure_retries_full_precision(self):
        model = SimpleNamespace(
            dims=SimpleNamespace(n_mels=2, n_audio_ctx=2),
            embed_audio=Mock(
                side_effect=[RuntimeError("FP16 unsupported"), torch.ones(1, 2, 2)]
            ),
            logits=Mock(return_value=torch.ones(1, 1, 3)),
            is_multilingual=False,
            install_kv_cache_hooks=lambda: ({}, []),
            decoder=Mock(),
        )
        real_zeros = torch.zeros

        def cpu_zeros(*args, **kwargs):
            kwargs.pop("device", None)
            return real_zeros(*args, **kwargs)

        with (
            patch("torch.zeros", side_effect=cpu_zeros),
            self.assertLogs("inference_backend", level="WARNING"),
        ):
            self.assertFalse(_warmup_torch(model, "xpu:0"))
        self.assertEqual(model.embed_audio.call_args.args[0].dtype, torch.float32)

    def test_encoder_output_does_not_alias_infer_request_memory(self):
        compiled = Mock()
        compiled.output.return_value = "output"
        buffer = np.ones((1, 2, 2), dtype=np.float32)
        compiled.create_infer_request.return_value.infer.return_value = {
            "output": buffer
        }
        encoder = OpenVINOEncoder(Encoder(), compiled, "NPU")
        result = encoder(torch.zeros(1, 2, 4))
        buffer.fill(99)
        self.assertTrue(torch.equal(result, torch.ones(1, 2, 2)))

    def test_inference_failure_disables_openvino_and_preserves_cpu_output(self):
        for failure in (
            RuntimeError("device lost"),
            {"output": np.full((1, 2, 2), np.nan)},
        ):
            with self.subTest(failure=failure):
                compiled = Mock()
                compiled.output.return_value = "output"
                request = compiled.create_infer_request.return_value
                if isinstance(failure, Exception):
                    request.infer.side_effect = failure
                else:
                    request.infer.return_value = failure
                encoder = OpenVINOEncoder(Encoder(), compiled, "NPU")
                mel = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
                with self.assertLogs("inference_backend", level="ERROR"):
                    result = encoder(mel)
                self.assertTrue(torch.equal(result, Encoder()(mel)))
                self.assertTrue(torch.equal(encoder(mel), result))
                self.assertFalse(encoder.active)
                self.assertIn("CPU", encoder.backend_name)
                request.infer.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec("openvino"), "OpenVINO not installed")
    def test_failed_npu_compilation_tries_next_device(self):
        model = SimpleNamespace(
            encoder=Encoder(),
            dims=SimpleNamespace(n_mels=2, n_audio_ctx=2, n_audio_state=2),
        )
        compiled = Mock()
        compiled.output.return_value = "output"
        compiled.create_infer_request.return_value.infer.return_value = {
            "output": np.zeros((1, 2, 2), dtype=np.float32)
        }
        with (
            tempfile.TemporaryDirectory() as cache,
            patch.dict("os.environ", {"XDG_CACHE_HOME": cache}),
            patch("openvino.Core") as core,
            patch("openvino.convert_model"),
            self.assertLogs("inference_backend", level="WARNING"),
        ):
            core.return_value.available_devices = ["CPU", "NPU"]
            core.return_value.compile_model.side_effect = [
                RuntimeError("cannot compile"),
                compiled,
            ]
            encoder = _openvino_encoder(model, "tiny")
            self.assertEqual(encoder.ov_device, "CPU")
            calls = core.return_value.compile_model.call_args_list
            self.assertEqual([call.args[1] for call in calls], ["NPU", "CPU"])


if __name__ == "__main__":
    unittest.main()
