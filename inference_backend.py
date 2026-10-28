"""Whisper accelerator selection and OpenVINO encoder/decoder bridges.

Only the inference thread calls this code. Whisper retains its decoding loop,
language probabilities, and segment confidence on every backend.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def torch_devices() -> list[str]:
    """Probe installed runtimes, rather than guessing from the OS/CPU vendor."""
    devices = []
    for kind in ("cuda", "xpu"):
        runtime = getattr(torch, kind, None)
        try:
            if runtime is not None and runtime.is_available():
                devices.extend(
                    f"{kind}:{index}" for index in range(runtime.device_count())
                )
        except Exception:
            logger.warning("Unable to probe PyTorch %s", kind, exc_info=True)
    try:
        if torch.backends.mps.is_available():
            devices.append("mps")
    except AttributeError, RuntimeError:
        logger.debug("MPS is unavailable", exc_info=True)
    try:
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        if accelerator is not None and accelerator.type not in (
            "cpu",
            "cuda",
            "xpu",
            "mps",
        ):
            devices.extend(
                f"{accelerator.type}:{index}"
                for index in range(torch.accelerator.device_count())
            )
    except Exception:
        logger.debug("No additional registered PyTorch accelerator", exc_info=True)
    return devices


def backend_name(model, default: str = "PyTorch CPU") -> str:
    """Report actual component placement, including a runtime CPU fallback."""
    from openvino_decoder import OpenVINODecoder

    encoder = getattr(model, "encoder", None)
    if isinstance(encoder, CachedAudioEncoder):
        encoder = encoder.original
    decoder = getattr(model, "decoder", None)
    if isinstance(encoder, OpenVINOEncoder) or isinstance(decoder, OpenVINODecoder):
        enc = encoder.device_name if isinstance(encoder, OpenVINOEncoder) else "CPU"
        dec = decoder.device_name if isinstance(decoder, OpenVINODecoder) else "CPU"
        return f"{enc} encoder / {dec} decoder"
    return default


class CachedAudioEncoder(torch.nn.Module):
    """Reuse identical mel inputs for language detection and token decoding.

    The last features also provide speaker embeddings without another encoder
    pass. All access is on the existing inference thread.
    """

    def __init__(self, original):
        super().__init__()
        self.original = original
        self.last_input = None
        self.last_features = None
        self.cache_hits = 0
        self._entries = []

    def forward(self, mel):
        if torch.is_grad_enabled():
            return self.original(mel)
        # Whisper uses distinct silence padding for language detection and
        # decoding. Keep both exact inputs rather than treating them as equal.
        for index, (previous, features) in enumerate(self._entries):
            if (
                previous.shape == mel.shape
                and previous.dtype == mel.dtype
                and previous.device == mel.device
                and torch.equal(previous, mel)
            ):
                self.cache_hits += 1
                self._entries.append(self._entries.pop(index))
                self.last_input, self.last_features = previous, features
                return features
        features = self.original(mel)
        self.last_input = mel.detach().clone()
        self.last_features = features.detach()
        self._entries.append((self.last_input, self.last_features))
        self._entries = self._entries[-2:]
        return features


def _openvino_config():
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = cache / "polyglot" / "openvino"
    config = {"PERFORMANCE_HINT": "LATENCY"}
    try:
        cache.mkdir(parents=True, exist_ok=True)
        config["CACHE_DIR"] = str(cache)
    except OSError:
        logger.warning("OpenVINO cache unavailable: %s", cache, exc_info=True)
    return config


def openvino_devices(devices: list[str]) -> list[str]:
    """Order actual device IDs, including numbered GPUs, by preference."""
    return [
        device
        for kind in ("NPU", "GPU", "CPU")
        for device in devices
        if device.split(".")[0] == kind
    ]


class OpenVINOEncoder(torch.nn.Module):
    """Keep a PyTorch encoder available if an accelerator fails at runtime."""

    def __init__(self, original, compiled, device: str):
        super().__init__()
        self.original = original
        self.request = compiled.create_infer_request()
        self.output = compiled.output(0)
        self.ov_device = device
        self.active = True

    @property
    def device_name(self) -> str:
        return f"OpenVINO {self.ov_device}" if self.active else "CPU"

    @property
    def backend_name(self) -> str:
        if self.active:
            return f"OpenVINO {self.ov_device} encoder / CPU decoder"
        return "PyTorch CPU (OpenVINO failed)"

    def forward(self, mel):
        if self.active:
            try:
                inputs = mel.detach().cpu().float().contiguous().numpy()
                # Copy: the infer request reuses its output memory next time.
                result = self.request.infer({0: inputs})[self.output].copy()
                if not np.isfinite(result).all():
                    raise RuntimeError("OpenVINO encoder returned non-finite values")
                return torch.from_numpy(result).to(device=mel.device, dtype=mel.dtype)
            except Exception:
                logger.exception("OpenVINO encoder failed; reverting to PyTorch CPU")
                self.active = False
        return self.original(mel)


def _openvino_encoder(model, model_name: str):
    import openvino as ov

    core = ov.Core()
    devices = openvino_devices(core.available_devices)
    logger.info("OpenVINO devices: %s", devices)
    if not devices:
        return None

    # Whisper always pads to 30 s.  A static shape allows NPU compilation.
    shape = (1, model.dims.n_mels, model.dims.n_audio_ctx * 2)
    example = torch.zeros(shape)
    with torch.no_grad():
        converted = ov.convert_model(
            model.encoder.eval(), example_input=example, input=[shape]
        )
    config = _openvino_config()

    for device in devices:
        try:
            logger.info(
                "Compiling Whisper %s encoder for OpenVINO %s…", model_name, device
            )
            compiled = core.compile_model(converted, device, config)
            encoder = OpenVINOEncoder(model.encoder, compiled, device)
            # Test execution as well as compilation before advertising readiness.
            with torch.no_grad():
                result = encoder(example)
            expected = (1, model.dims.n_audio_ctx, model.dims.n_audio_state)
            if not encoder.active or tuple(result.shape) != expected:
                raise RuntimeError("OpenVINO encoder warmup failed")
            return encoder
        except Exception:
            logger.warning(
                "OpenVINO %s unavailable for this model", device, exc_info=True
            )
    return None


def _openvino_decoder(model):
    import openvino as ov

    from openvino_decoder import compile_decoder

    core = ov.Core()
    config = _openvino_config()
    # Confidence scores and language probabilities must retain FP32 accuracy.
    config["INFERENCE_PRECISION_HINT"] = "f32"
    return compile_decoder(
        model, openvino_devices(core.available_devices), core, config
    )


def _load_torch(model_name, device):
    import whisper

    model = whisper.load_model(model_name, device="cpu")
    if device != "cpu":
        # MPS/XPU need not implement sparse tensors. Alignment metadata is only
        # used for optional word timestamps, so keep that buffer on CPU.
        alignment = model._buffers.pop("alignment_heads")
        try:
            model.to(device)
        finally:
            model.register_buffer("alignment_heads", alignment, persistent=False)
    return model


def _warmup_torch(model, device):
    """Reject unsupported kernels before the pipeline advertises readiness."""
    precisions = (True, False) if device.split(":")[0] in ("cuda", "xpu") else (False,)
    for fp16 in precisions:
        try:
            dtype = torch.float16 if fp16 else torch.float32
            shape = (1, model.dims.n_mels, model.dims.n_audio_ctx * 2)
            with torch.no_grad():
                encoded = model.embed_audio(
                    torch.zeros(shape, dtype=dtype, device=device)
                )
                tokens = torch.zeros((1, 1), dtype=torch.int64, device=device)
                logits = model.logits(tokens, encoded)
                if (
                    not torch.isfinite(logits).all()
                    or not torch.isfinite(encoded).all()
                ):
                    raise RuntimeError("Whisper warmup returned non-finite values")
                if model.is_multilingual:
                    model.detect_language(encoded)
                cache, hooks = model.install_kv_cache_hooks()
                try:
                    prefill = tokens.repeat(1, 3)
                    model.decoder(prefill, encoded, kv_cache=cache)
                    model.decoder(tokens, encoded, kv_cache=cache)
                finally:
                    for hook in hooks:
                        hook.remove()
            return fp16
        except Exception:
            if not fp16:
                raise
            logger.warning(
                "FP16 warmup failed on %s; retrying FP32", device, exc_info=True
            )


def load_whisper(model_name: str, device: str | None = None):
    """Prefer validated native GPUs, OpenVINO components, then PyTorch CPU.

    An explicit PyTorch device bypasses automatic selection. GPU packages
    must match the installed driver; torch exposes ROCm through its cuda API.
    """
    from whisper.model import MultiHeadAttention

    candidates = [device] if device is not None else torch_devices()
    logger.info("PyTorch accelerator candidates: %s", candidates)
    for selected in candidates:
        rocm = selected.startswith("cuda") and bool(torch.version.hip)
        MultiHeadAttention.use_sdpa = not rocm
        try:
            model = _load_torch(model_name, selected)
            model.inference_fp16 = _warmup_torch(model, selected)
            name = (
                "ROCm"
                if rocm
                else (
                    "CUDA"
                    if selected.startswith("cuda")
                    else selected.split(":")[0].upper()
                )
            )
            model.encoder = CachedAudioEncoder(model.encoder)
            return model, f"PyTorch {name}"
        except Exception:
            if device is not None:
                raise
            logger.warning(
                "PyTorch %s validation failed; trying another backend",
                selected,
                exc_info=True,
            )
            if "model" in locals():
                del model

    MultiHeadAttention.use_sdpa = True
    model = _load_torch(model_name, "cpu")
    model.inference_fp16 = False
    try:
        encoder = _openvino_encoder(model, model_name)
        if encoder is not None:
            model.encoder = encoder
    except Exception:
        logger.warning(
            "OpenVINO encoder setup failed; using PyTorch CPU", exc_info=True
        )
    try:
        decoder = _openvino_decoder(model)
        if decoder is not None:
            model.decoder = decoder
    except Exception:
        logger.warning(
            "OpenVINO decoder setup failed; using PyTorch CPU", exc_info=True
        )
    model.encoder = CachedAudioEncoder(model.encoder)
    return model, backend_name(model)
