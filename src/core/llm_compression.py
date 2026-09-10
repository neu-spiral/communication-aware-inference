from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from . import llmint8 as _llmint8


def detect_llm_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Best-effort decoder-layer detection across common HF architectures.
    We hook decoder blocks at indices to approximate "split points".
    """
    # LLaMA / Gemma-like
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if isinstance(layers, nn.ModuleList):
            return layers

    # Some models expose layers directly
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model.layers

    # GPT-NeoX-like
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = model.gpt_neox.layers
        if isinstance(layers, nn.ModuleList):
            return layers

    # GPT-2 like
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        if isinstance(layers, nn.ModuleList):
            return layers

    # T5-style encoder-decoder. The compressed links all sit at encoder block
    # boundaries (the last one carries the encoder output into the decoder), so
    # the encoder stack is the hookable sequence here.
    if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        blocks = model.encoder.block
        if isinstance(blocks, nn.ModuleList):
            return blocks

    raise ValueError("Could not detect decoder layers for this HF architecture.")


def _topk_mask_by_abs(flat_abs: torch.Tensor, k: int) -> torch.Tensor:
    """
    Returns a boolean mask keeping the top-k magnitudes.
    This is used for magnitude sparsification.
    """
    if k <= 0:
        return torch.zeros_like(flat_abs, dtype=torch.bool)
    if k >= flat_abs.numel():
        return torch.ones_like(flat_abs, dtype=torch.bool)
    # Use threshold from top-k so we can build a mask in O(numel).
    vals = flat_abs.topk(k).values
    thresh = vals.min()
    return flat_abs >= thresh


_FP16_ABS_MAX = 65504.0  # largest finite magnitude representable in IEEE FP16


def _paper_quantize_per_row(hidden: torch.Tensor, eta: float) -> torch.Tensor:
    """Symmetric uniform activation quantization (paper spec).

    Per-token (per-row) abs-max scale along the LAST (hidden) dimension; the
    native precision is FP32, so eta is the compression ratio to the 32-bit
    width and maps to a target bit-width in {2, 4, 8, 16, 32}:

        eta >= 1.0   -> FP32  (native, no-op)
        eta >= 0.5   -> FP16  (direct datatype conversion + value clipping)
        eta >= 0.25  -> INT8
        eta >= 0.125 -> INT4  (stochastic rounding: error unbiased in expectation)
        else         -> INT2

    INT4 rounds up with probability equal to the fractional part (stochastic
    rounding), which removes the systematic bias of round-to-nearest at very
    low bit-widths. Every other rung rounds to nearest. A row of all-zero
    activations maps to zero (the scale is floored to avoid division by zero).
    """
    if eta >= 1.0:
        return hidden  # FP32 native reference — no compression.
    orig_dtype = hidden.dtype
    x = hidden.to(torch.float32)
    if eta >= 0.5:  # FP16: datatype conversion with clipping to avoid overflow.
        return x.clamp(-_FP16_ABS_MAX, _FP16_ABS_MAX).half().to(orig_dtype)
    if eta >= 0.25:
        levels, stochastic = 127.0, False   # INT8
    elif eta >= 0.125:
        levels, stochastic = 7.0, True       # INT4 (stochastic rounding)
    else:
        levels, stochastic = 1.0, False      # INT2
    # Per-token (per-row) symmetric scale along the hidden dimension.
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = (amax / levels).clamp(min=1e-8)
    xs = x / scale
    if stochastic:
        floor_x = torch.floor(xs)
        prob = (xs - floor_x).clamp(0.0, 1.0)
        q = floor_x + torch.bernoulli(prob)
    else:
        q = torch.round(xs)
    q = q.clamp(-levels, levels)
    return (q * scale).to(orig_dtype)


@dataclass(frozen=True)
class ActivationCompressionConfig:
    # magnitude | topk_per_token | random | quantization | llmint8
    strategy: str
    # Cut-points in "layer index" space where hooks are attached.
    layer_indices: Sequence[int]
    # Seed used to make random compression deterministic across eta evaluations.
    random_seed: int = 1234


class LLMActivationCompressor:
    """
    Registers forward hooks at given decoder-layer indices and applies activation
    sparsification controlled by `eta` (a vector, one eta per hooked layer).

    The compressor keeps hooks installed; only `eta` is updated in-place.

    Strategies
    ----------
    magnitude        top-k by absolute value over the whole tensor
    topk_per_token   top-k per token position along the hidden dimension
    random           Bernoulli mask with a deterministic seed
    quantization     symmetric uniform per-token quantization; eta is the
                     compression ratio to FP32 and selects a width from
                     {2, 4, 8, 16, 32}
    llmint8          two-band mixed precision; eta is the compression ratio
                     (see :mod:`src.core.llmint8`)

    For every strategy eta is a compression ratio in [0, 1], so they are
    comparable at equal eta.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ActivationCompressionConfig,
        layers: Optional[nn.ModuleList] = None,
    ) -> None:
        """
        Parameters
        ----------
        layers
            Modules whose outputs `config.layer_indices` refers to. Defaults to
            the auto-detected decoder stack. Pass an explicit list when the
            compressed boundary is not a plain decoder block -- for T5 the
            encoder-to-decoder link carries the tensor *after*
            ``encoder.final_layer_norm``, which is not a member of
            ``encoder.block``.
        """
        self.model = model
        self.config = config
        self.layers = layers if layers is not None else detect_llm_decoder_layers(model)

        self.d = len(config.layer_indices)
        if self.d <= 0:
            raise ValueError("ActivationCompressionConfig.layer_indices must be non-empty.")

        # Persist eta as floats for cheap .item()-free access in hooks.
        self._eta: List[float] = [1.0 for _ in range(self.d)]

        handles: List[torch.utils.hooks.RemovableHandle] = []
        for local_i, layer_idx in enumerate(config.layer_indices):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                raise ValueError(
                    f"layer index {layer_idx} out of range (n_layers={len(self.layers)})"
                )
            handles.append(self.layers[layer_idx].register_forward_hook(self._make_hook(local_i)))
        self._handles = handles

    def remove_hooks(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def set_eta(self, eta: torch.Tensor) -> None:
        if eta.numel() != self.d:
            raise ValueError(f"Expected eta of length {self.d}, got {eta.numel()}")
        # Clamp to [0,1] to keep sparsifiers well-behaved.
        eta_cpu = eta.detach().to(torch.float32).cpu()
        self._eta = [float(torch.clamp(v, 0.0, 1.0).item()) for v in eta_cpu]

    def _compress(self, hidden: torch.Tensor, eta: float) -> torch.Tensor:
        if eta >= 1.0:
            return hidden
        if eta <= 0.0:
            return torch.zeros_like(hidden)

        if self.config.strategy == "llmint8":
            return _llmint8.quantize_activations(hidden, eta)

        if self.config.strategy == "quantization":
            # Symmetric uniform activation quantization, per-token (per-row)
            # scale along the hidden dim, native FP32. eta is the compression
            # ratio to the 32-bit width and selects a target bit-width from
            # {2, 4, 8, 16, 32}; INT4 uses stochastic rounding. See
            # _paper_quantize_per_row for the full ladder.
            return _paper_quantize_per_row(hidden, eta)

        if self.config.strategy == "magnitude":
            flat = hidden.flatten()
            k = max(1, int(eta * flat.numel()))
            mask = _topk_mask_by_abs(flat.abs(), k=k)
            return (flat * mask.to(flat.dtype)).reshape_as(hidden)

        if self.config.strategy == "topk_per_token":
            # Keep top-k hidden dimensions per token position.
            # Expected hidden: (batch, seq_len, hidden_dim) -> mask over last dim.
            D = hidden.size(-1)
            k = max(1, int(eta * D))
            abs_h = hidden.abs()
            _, idx = abs_h.topk(k, dim=-1)
            mask = torch.zeros_like(hidden)
            mask.scatter_(-1, idx, 1.0)
            return hidden * mask

        if self.config.strategy == "random":
            # Bernoulli mask with deterministic generator seed.
            g = torch.Generator(device=hidden.device)
            g.manual_seed(int(self.config.random_seed))
            mask = torch.rand(hidden.shape, generator=g, device=hidden.device) < float(eta)
            return hidden * mask.to(hidden.dtype)

        raise ValueError(f"Unknown activation compression strategy: {self.config.strategy}")

    def _make_hook(self, local_i: int):
        def hook(_module: nn.Module, _inputs, output):
            # HF decoder blocks typically return:
            #   - hidden_states tensor, or
            #   - tuple(hidden_states, past_key_value, ...)
            if isinstance(output, tuple):
                hidden = output[0]
                compressed = self._compress(hidden, self._eta[local_i])
                return (compressed, *output[1:])
            return self._compress(output, self._eta[local_i])

        return hook

