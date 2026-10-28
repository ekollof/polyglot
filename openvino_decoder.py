"""OpenVINO GPU decoding with Whisper-compatible attention caches.

The Python Whisper decoding loop still owns language detection, sampling,
timestamps, and confidence scores. Only its tensor computations move to GPU.
"""

from __future__ import annotations

import logging
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class CrossAttentionGraph(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, audio):
        keys = [block.cross_attn.key(audio) for block in self.decoder.blocks]
        values = [block.cross_attn.value(audio) for block in self.decoder.blocks]
        return torch.stack(keys), torch.stack(values)


def _attention(attention, query, key, value, mask=None):
    batch, length, state = query.shape
    heads = attention.n_head
    query = query.reshape(batch, length, heads, state // heads).transpose(1, 2)
    key = key.reshape(batch, -1, heads, state // heads).transpose(1, 2)
    value = value.reshape(batch, -1, heads, state // heads).transpose(1, 2)
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    return attention.out(output.transpose(1, 2).reshape(batch, length, state))


class DecoderGraph(torch.nn.Module):
    """Explicit cache inputs/outputs allow dynamic token lengths on GPU."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens, past_keys, past_values, cross_keys, cross_values, offset):
        positions = offset[0] + torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.decoder.token_embedding(tokens)
        hidden = hidden + self.decoder.positional_embedding[positions]
        length = past_keys.shape[2] + tokens.shape[1]
        allowed = (
            torch.arange(length, device=tokens.device)[None, :] <= positions[:, None]
        )
        mask = torch.where(allowed, 0.0, float("-inf"))
        keys, values = [], []
        for index, block in enumerate(self.decoder.blocks):
            normalized = block.attn_ln(hidden)
            key = torch.cat((past_keys[index], block.attn.key(normalized)), dim=1)
            value = torch.cat((past_values[index], block.attn.value(normalized)), dim=1)
            hidden = hidden + _attention(
                block.attn, block.attn.query(normalized), key, value, mask
            )
            normalized = block.cross_attn_ln(hidden)
            hidden = hidden + _attention(
                block.cross_attn,
                block.cross_attn.query(normalized),
                cross_keys[index],
                cross_values[index],
            )
            hidden = hidden + block.mlp(block.mlp_ln(hidden))
            keys.append(key)
            values.append(value)
        hidden = self.decoder.ln(hidden)
        logits = (hidden @ self.decoder.token_embedding.weight.T).float()
        return logits, torch.stack(keys), torch.stack(values)


class OpenVINODecoder(torch.nn.Module):
    def __init__(self, original, cross_compiled, compiled, device):
        super().__init__()
        self.original = original
        self.cross_request = cross_compiled.create_infer_request()
        self.cross_outputs = [cross_compiled.output(i) for i in range(2)]
        self.request = compiled.create_infer_request()
        self.outputs = [compiled.output(i) for i in range(3)]
        self.ov_device = device
        self.active = True

    @property
    def blocks(self):
        # Whisper uses these module objects as attention-cache keys.
        return self.original.blocks

    @property
    def device_name(self):
        return f"OpenVINO {self.ov_device}" if self.active else "CPU"

    def forward(self, tokens, audio, kv_cache=None):
        if self.active:
            try:
                return self._infer(tokens, audio, kv_cache)
            except Exception:
                logger.exception("OpenVINO decoder failed; reverting to PyTorch CPU")
                self.active = False
        # Existing Whisper hooks append to the same cache after a fallback.
        return self.original(tokens, audio, kv_cache=kv_cache)

    def _infer(self, tokens, audio, kv_cache):
        blocks = self.original.blocks
        cache = kv_cache if kv_cache is not None else {}
        if cache:
            past_keys = torch.stack([cache[block.attn.key] for block in blocks]).numpy()
            past_values = torch.stack(
                [cache[block.attn.value] for block in blocks]
            ).numpy()
            cross_keys = torch.stack(
                [cache[block.cross_attn.key] for block in blocks]
            ).numpy()
            cross_values = torch.stack(
                [cache[block.cross_attn.value] for block in blocks]
            ).numpy()
        else:
            cross = self.cross_request.infer({0: audio.detach().cpu().float().numpy()})
            cross_keys, cross_values = [
                cross[port].copy() for port in self.cross_outputs
            ]
            shape = (len(blocks), tokens.shape[0], 0, audio.shape[-1])
            past_keys = np.zeros(shape, dtype=np.float32)
            past_values = np.zeros(shape, dtype=np.float32)
        outputs = self.request.infer(
            {
                0: tokens.detach().cpu().numpy(),
                1: past_keys,
                2: past_values,
                3: cross_keys,
                4: cross_values,
                5: np.array([past_keys.shape[2]], dtype=np.int64),
            }
        )
        logits, keys, values = [outputs[port].copy() for port in self.outputs]
        if not all(np.isfinite(value).all() for value in (logits, keys, values)):
            raise RuntimeError("OpenVINO decoder returned non-finite values")
        # Publish cache updates only after all outputs have passed validation.
        if kv_cache is not None:
            for index, block in enumerate(blocks):
                cache[block.attn.key] = torch.from_numpy(keys[index])
                cache[block.attn.value] = torch.from_numpy(values[index])
                cache[block.cross_attn.key] = torch.from_numpy(cross_keys[index])
                cache[block.cross_attn.value] = torch.from_numpy(cross_values[index])
        return torch.from_numpy(logits)


def decoder_latency(model, decoder):
    """Compare equivalent prefill/cached work with independent request caches."""
    audio = torch.zeros(1, model.dims.n_audio_ctx, model.dims.n_audio_state)
    tokens = torch.zeros(1, 4, dtype=torch.int64)
    timings = []
    last = None
    for _ in range(3):
        hooks = []
        if decoder is model.decoder:
            cache, hooks = model.install_kv_cache_hooks()
        else:
            cache = {}
        try:
            with torch.no_grad():
                start = time.perf_counter()
                decoder(tokens, audio, kv_cache=cache)
                for _ in range(4):
                    last = decoder(tokens[:, :1], audio, kv_cache=cache)
                timings.append(time.perf_counter() - start)
        finally:
            for hook in hooks:
                hook.remove()
    return statistics.median(timings), last


def compile_decoder(model, devices, core, config):
    """Try GPUs for autoregressive decoding; NPU keeps the static encoder."""
    import openvino as ov

    gpus = [device for device in devices if device.split(".")[0] == "GPU"]
    if not gpus:
        return None
    dims = model.dims
    layers, state = dims.n_text_layer, dims.n_text_state
    audio = torch.zeros(1, dims.n_audio_ctx, dims.n_audio_state)
    tokens = torch.zeros(1, 3, dtype=torch.int64)
    past = torch.zeros(layers, 1, 1, state)
    cross = torch.zeros(layers, 1, dims.n_audio_ctx, state)
    with torch.no_grad():
        cross_graph = ov.convert_model(
            CrossAttentionGraph(model.decoder).eval(),
            example_input=audio,
            input=[(-1, dims.n_audio_ctx, dims.n_audio_state)],
        )
        decoder_graph = ov.convert_model(
            DecoderGraph(model.decoder).eval(),
            example_input=(tokens, past, past, cross, cross, torch.tensor([1])),
            input=[
                (-1, -1),
                (layers, -1, -1, state),
                (layers, -1, -1, state),
                (layers, -1, dims.n_audio_ctx, state),
                (layers, -1, dims.n_audio_ctx, state),
                (1,),
            ],
        )
    for device in gpus:
        try:
            logger.info("Compiling Whisper decoder for OpenVINO %s…", device)
            decoder = OpenVINODecoder(
                model.decoder,
                core.compile_model(cross_graph, device, config),
                core.compile_model(decoder_graph, device, config),
                device,
            )
            # Exercise prefill and a cached step; both must work on the device.
            cache = {}
            with torch.no_grad():
                decoder(tokens, audio, cache)
                decoder(tokens[:, :1], audio, cache)
            if not decoder.active:
                raise RuntimeError("OpenVINO decoder warmup failed")
            gpu_latency, actual = decoder_latency(model, decoder)
            cpu_latency, expected = decoder_latency(model, model.decoder)
            if not torch.allclose(actual, expected, rtol=0.01, atol=0.02):
                raise RuntimeError("OpenVINO decoder validation differs from PyTorch")
            logger.info(
                "Decoder probe: %s %.1f ms, CPU %.1f ms",
                device,
                gpu_latency * 1000,
                cpu_latency * 1000,
            )
            if not decoder.active or gpu_latency >= cpu_latency * 0.9:
                logger.info(
                    "Keeping CPU decoder; %s offers no measured improvement", device
                )
                continue
            return decoder
        except Exception:
            logger.warning("OpenVINO %s decoder unavailable", device, exc_info=True)
    return None
