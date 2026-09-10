"""
Central compressor dispatch.

Single source of truth for the ``COMPRESSOR_BACKEND`` environment variable, so
modules that need a compressor do not each repeat the backend-selection logic.

Environment
-----------
``COMPRESSOR_BACKEND=cpu`` (default)
    Use :mod:`src.core.compressors` — compress on CPU, transfer back to device.
``COMPRESSOR_BACKEND=gpu``
    Use :mod:`src.core.gpu_compressors` — payload stays on the input device.

The variable is read once at import time; changing it afterwards has no effect.

Usage
-----
    from src.core.compressor_registry import get_compressor, is_gpu_compressor

    comp = get_compressor("llmint8")
    if is_gpu_compressor(comp):
        ...   # GPU-native path, no CPU round-trip
"""

import os

from .compressors import BaseCompressor, get_compressor as _cpu_get_compressor
from .gpu_compressors import GPUBaseCompressor, get_gpu_compressor as _gpu_get_compressor

__all__ = [
    "COMPRESSOR_BACKEND",
    "BaseCompressor",
    "GPUBaseCompressor",
    "describe_compressor",
    "get_compressor",
    "is_gpu_backend",
    "is_gpu_compressor",
]

COMPRESSOR_BACKEND: str = os.environ.get("COMPRESSOR_BACKEND", "cpu").strip().lower()

if COMPRESSOR_BACKEND not in ("cpu", "gpu"):
    raise ValueError(
        f"COMPRESSOR_BACKEND={COMPRESSOR_BACKEND!r} is invalid. Choose from: cpu, gpu."
    )


def is_gpu_backend() -> bool:
    """True if ``COMPRESSOR_BACKEND=gpu`` is active."""
    return COMPRESSOR_BACKEND == "gpu"


def is_gpu_compressor(obj) -> bool:
    """True if *obj* is a GPU-native compressor instance."""
    return isinstance(obj, GPUBaseCompressor)


def get_compressor(name: str):
    """
    Instantiate a compressor by name, respecting ``COMPRESSOR_BACKEND``.

    Parameters
    ----------
    name : str
        One of ``topk``, ``quantization``/``quant``, ``llmint8``/``llm_int8``.

    Returns
    -------
    BaseCompressor or GPUBaseCompressor
        Depending on the active backend.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    if COMPRESSOR_BACKEND == "gpu":
        return _gpu_get_compressor(name)
    return _cpu_get_compressor(name)


def describe_compressor(instance, module_tag: str = "") -> str:
    """
    One-line human-readable description of a compressor instance.

    Example
    -------
    ``[ResNet backend] compressor=LLMInt8    backend=cpu  path=CPU (compress->cpu->decompress->device)``
    """
    backend_actual = "gpu" if is_gpu_compressor(instance) else "cpu"
    execution_path = (
        "GPU-native (no CPU round-trip)"
        if backend_actual == "gpu"
        else "CPU (compress->cpu->decompress->device)"
    )
    name = getattr(instance, "get_name", lambda: type(instance).__name__)()
    tag = f"[{module_tag}] " if module_tag else ""
    return f"{tag}compressor={name:<20} backend={backend_actual}  path={execution_path}"
