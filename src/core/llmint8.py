"""
LLM.int8()-style mixed-precision compression.

This module is the only definition of what ``llmint8`` means in this
repository. Both call sites use it:

* activation compression during inference, via
  :func:`quantize_activations` (see :mod:`src.core.llm_compression`);
* codec parameters for :class:`src.core.compressors.LLMInt8Compressor`, via
  :func:`codec_params`, or :func:`resolve_codec_execution_plan` for a whole
  eta vector.

Semantics
---------
``eta`` is the compression ratio against the FP16 activation baseline, so the
target average width is ``16 * eta`` bits. Elements are split by magnitude into
two bands:

* the **high band** is always FP16 and holds the largest-magnitude elements,
  the outliers that dominate inference quality;
* the **low band** holds everything else and steps down the ladder
  ``int8 -> int4 -> int2 -> drop`` as ``eta`` falls.

Given a target width ``b = 16 * eta`` and a low band of ``b_low`` bits, the
high-band fraction that hits the target exactly is::

    f_high = (b - b_low) / (16 - b_low)

Because ``eta`` is a ratio on the same [0, 1] axis as ``topk_per_token`` and
``quantization``, the three compressors are directly comparable at equal
``eta``.

The high-band fraction is floored at :data:`FP16_OUTLIER_FLOOR`. Without it
``f_high`` is exactly 0 at the phase boundaries (eta 0.5, 0.25, 0.125), leaving
no outlier in FP16; the unprotected low-precision activations then collapse the
model at precisely those points. The floor costs under 0.3% deviation from the
requested ratio.

Quantization is deterministic symmetric abs-max, per band. Deterministic
rounding keeps the eta-to-utility surface smooth, which the concavity analysis
in ``experiments/concavity/`` depends on.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Sequence

import numpy as np
import torch

__all__ = [
    "FP16_OUTLIER_FLOOR",
    "BitPlan",
    "bit_plan",
    "bits_per_element",
    "codec_params",
    "quantize_activations",
    "resolve_codec_execution_plan",
    "validate_llmint8_mapping_entries",
]

#: Precision rungs, low to high, with their widths in bits.
_LADDER = (("drop", 0), ("int2", 2), ("int4", 4), ("int8", 8), ("fp16", 16))

#: Symmetric quantization levels per integer precision.
_LEVELS = {"int8": 127.0, "int4": 7.0, "int2": 1.0}

#: Smallest high-band fraction ever used. See the module docstring.
FP16_OUTLIER_FLOOR = 0.005

_BASELINE_BITS = 16.0


class BitPlan(NamedTuple):
    """How one ``eta`` splits into two precision bands."""

    low_precision: str
    high_precision: str
    #: Fraction of elements, by descending magnitude, placed in the high band.
    f_high: float


def bit_plan(eta: float) -> BitPlan:
    """Return the two-band plan for *eta*.

    >>> bit_plan(1.0)
    BitPlan(low_precision='fp16', high_precision='fp16', f_high=1.0)
    >>> bit_plan(0.75).low_precision, round(bit_plan(0.75).f_high, 3)
    ('int8', 0.5)
    """
    b = _BASELINE_BITS * float(eta)
    if b >= _BASELINE_BITS:
        return BitPlan("fp16", "fp16", 1.0)

    if b >= 8.0:
        low, low_bits = "int8", 8
    elif b >= 4.0:
        low, low_bits = "int4", 4
    elif b >= 2.0:
        low, low_bits = "int2", 2
    else:
        low, low_bits = "drop", 0

    f_high = (b - low_bits) / (_BASELINE_BITS - low_bits)
    f_high = min(1.0, max(f_high, FP16_OUTLIER_FLOOR))
    return BitPlan(low, "fp16", f_high)


def bits_per_element(eta: float) -> float:
    """Average width the plan for *eta* actually delivers, in bits.

    Equals ``16 * eta`` except where :data:`FP16_OUTLIER_FLOOR` binds.
    """
    plan = bit_plan(eta)
    low_bits = dict(_LADDER)[plan.low_precision]
    return plan.f_high * _BASELINE_BITS + (1.0 - plan.f_high) * low_bits


def codec_params(eta: float, *, regular_precision: str | None = None) -> List:
    """Return ``[outlier_ratio, outlier_precision, regular_precision]``.

    The parameter triple :class:`src.core.compressors.LLMInt8Compressor` and its
    GPU counterpart expect. *regular_precision* overrides the ladder rung the
    plan chose, for callers pinned to a fixed low band.
    """
    plan = bit_plan(eta)
    low = regular_precision or plan.low_precision
    # The codec's low band is one of int8/int4/int2. "drop" clamps to its floor;
    # "fp16" only arises at eta >= 1, where the low band holds nothing.
    if low in ("drop", "fp16"):
        low = "int2" if low == "drop" else "int8"
    return [float(plan.f_high), plan.high_precision, str(low)]


def _quantize_band(x: torch.Tensor, precision: str) -> torch.Tensor:
    """Dequantized float32 values, same shape as *x*."""
    if precision == "fp16":
        return x.half().float()
    if precision == "drop":
        return torch.zeros_like(x)
    levels = _LEVELS[precision]
    scale = (x.abs().max() / levels).clamp(min=1e-8)
    return (x / scale).round().clamp(-levels, levels) * scale


def quantize_activations(hidden: torch.Tensor, eta: float) -> torch.Tensor:
    """Two-band mixed-precision round trip of *hidden* at ratio *eta*.

    Returns a tensor of the input shape and dtype. Runs wherever the input
    lives; the magnitude threshold uses ``kthvalue``, which stays on device for
    any tensor size.
    """
    original_dtype = hidden.dtype
    flat = hidden.reshape(-1).float()
    n = flat.numel()

    plan = bit_plan(eta)
    n_high = int(round(plan.f_high * n))

    if n_high <= 0:
        out = _quantize_band(flat, plan.low_precision)
    elif n_high >= n:
        out = _quantize_band(flat, plan.high_precision)
    else:
        magnitude = flat.abs()
        threshold = torch.kthvalue(magnitude, n - n_high).values
        high = magnitude > threshold
        out = torch.empty_like(flat)
        out[high] = _quantize_band(flat[high], plan.high_precision)
        out[~high] = _quantize_band(flat[~high], plan.low_precision)

    return out.reshape_as(hidden).to(original_dtype)


def plan_table(etas: Sequence[float]) -> str:
    """Human-readable plan for each of *etas*, for logs and docs."""
    rows = ["   eta   bits  high band          low band",
            "  ----- ------ ------------------ ---------"]
    for eta in etas:
        plan = bit_plan(eta)
        rows.append(f"  {eta:5.3f} {bits_per_element(eta):6.2f} "
                    f"{plan.high_precision} x {plan.f_high:<12.4f} "
                    f"{plan.low_precision}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Whole-vector plans, for the task instances
# ---------------------------------------------------------------------------

def _as_float_list(value: Any) -> List[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def validate_llmint8_mapping_entries(entries: Any, *, num_links: int) -> List[Dict[str, Any]]:
    """
    Normalize llmint8 mapping entries and keep only valid rows.

    Expected per-entry schema:
      - eta or eta_vec: list[float] length == num_links
      - compression_params_list (or aliases): list[float] length == num_links
    """
    if not isinstance(entries, list):
        return []

    normalized: List[Dict[str, Any]] = []
    expected = int(num_links)
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        eta = _as_float_list(raw.get("eta", raw.get("eta_vec")))
        params = _as_float_list(
            raw.get("compression_params_list", raw.get("compression_params", raw.get("params")))
        )
        if eta is None or params is None:
            continue
        if len(eta) != expected or len(params) != expected:
            continue
        normalized.append({**raw, "eta": eta, "compression_params_list": params})
    return normalized


def resolve_codec_execution_plan(
    *,
    codec_name: str,
    eta: List[float],
    outlier_precision: str = "fp16",
    regular_precision: str | None = None,
    llmint8_mapping_entries: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Map optimizer eta to llmint8 runtime compression parameters.

    Non-llmint8 codecs take eta unchanged. For llmint8 the parameters come from
    :func:`src.core.llmint8.codec_params`, the repository's single definition of
    the scheme. Supplying *llmint8_mapping_entries* overrides that with a
    nearest-neighbour lookup in eta-space, which is how a measured table of
    on-device overheads can replace the analytic plan.
    """
    codec = str(codec_name).strip().lower()
    eta_vec = np.asarray([float(x) for x in eta], dtype=float).reshape(-1)

    if codec != "llmint8":
        return {"compression_params_list": eta_vec.tolist()}

    entries = list(llmint8_mapping_entries or [])
    if not entries:
        return {
            "codec_name": codec,
            "outlier_precision": str(outlier_precision),
            "regular_precision": None,
            "compression_params_list": [
                codec_params(v, regular_precision=regular_precision)
                for v in eta_vec.tolist()
            ],
        }

    best_params: List[float] | None = None
    best_dist = float("inf")
    for entry in entries:
        e = np.asarray(entry.get("eta", []), dtype=float).reshape(-1)
        p = entry.get("compression_params_list", [])
        if e.shape != eta_vec.shape or not isinstance(p, list) or len(p) != int(eta_vec.size):
            continue
        dist = float(np.linalg.norm(e - eta_vec))
        if dist < best_dist:
            best_dist = dist
            best_params = [float(x) for x in p]

    if best_params is None:
        best_params = [
            codec_params(v, regular_precision=regular_precision)
            for v in eta_vec.tolist()
        ]

    return {
        "codec_name": codec,
        "outlier_precision": str(outlier_precision),
        "regular_precision": regular_precision,
        "compression_params_list": best_params,
    }
