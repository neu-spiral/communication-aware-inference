"""
Resolve task family by name and build :class:`~src.core.task.InferenceTask` instances.

Setup (model training, callables, etc.) lives in ``src/core/task_instances/*.py``.
This module caches that result per task name so scripts only pass the name to
:func:`get_task`.

Task instance contract
----------------------
Each registered module must implement:

* ``setup_model_and_callables(**kwargs) -> (api, extra)`` where ``api`` exposes
  ``accuracy_callable``, ``gradient_callable``, and ``accuracy_callable_true``
  (each maps ``np.ndarray`` → float / ndarray).
* ``make_task(task_id, api, w_k=..., R_k=..., **kwargs) -> InferenceTask``

Optional kwargs such as ``compressor_name`` are forwarded when the registry name
implies a codec (e.g. ``resnet56_cifar10_quantization`` → ``quantization``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .task import InferenceTask
from .task_instances import flant5_sst2
from .task_instances import gemma2b_sharegpt_ppl
from .task_instances import gemma7b_sharegpt_ppl
from .task_instances import llama31_8b_mmlu
from .task_instances import llama31_8b_wikitext_ppl
from .task_instances import resnet56_cifar10_topk
from .task_instances import toy_mlp_mnist

# name -> (module, optional compressor_name forwarded to setup/make_task)
# The five LLM scenarios from the paper, each in the three codecs. Names are
# "<task>_<codec>"; the codec is forwarded to setup/make_task, exactly as for
# the ResNet rows. Paper scenario names, for cross-referencing the results
# table: gemma2b_sharegpt_ppl = G1-2B-SGPT, gemma7b_sharegpt_ppl = G1-7B-SGPT,
# llama31_8b_mmlu = Ll3-8B-MMLU, llama31_8b_wikitext_ppl = Ll3-8B-WT,
# flant5_sst2 = FT5-SST2.
_LLM_MODULES = {
    "gemma2b_sharegpt_ppl": gemma2b_sharegpt_ppl,
    "gemma7b_sharegpt_ppl": gemma7b_sharegpt_ppl,
    "llama31_8b_mmlu": llama31_8b_mmlu,
    "llama31_8b_wikitext_ppl": llama31_8b_wikitext_ppl,
    "flant5_sst2": flant5_sst2,
}

_REGISTRY: Dict[str, Tuple[Any, Optional[str]]] = {
    "resnet56_cifar10_topk": (resnet56_cifar10_topk, "topk"),
    "resnet56_cifar10_quantization": (resnet56_cifar10_topk, "quantization"),
    "resnet56_cifar10_llmint8": (resnet56_cifar10_topk, "llmint8"),
    "toy_mlp_mnist": (toy_mlp_mnist, None),
    **{
        f"{task}_{codec}": (module, codec)
        for task, module in _LLM_MODULES.items()
        for codec in ("topk", "quantization", "llmint8")
    },
}

_API_CACHE: Dict[str, Tuple[Any, Any]] = {}


def register_task(
    name: str,
    module: Any,
    *,
    compressor_name: Optional[str] = None,
) -> None:
    """Register a task-instance module under ``name``."""
    _REGISTRY[name] = (module, compressor_name)


def registered_task_names() -> list[str]:
    return sorted(_REGISTRY.keys())


def clear_task_api_cache(name: Optional[str] = None) -> None:
    """Drop cached setup for one task name, or all names (e.g. for tests)."""
    if name is None:
        _API_CACHE.clear()
    else:
        _API_CACHE.pop(name, None)


def _entry(name: str) -> Tuple[Any, Optional[str]]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown task {name!r}. Registered: {', '.join(registered_task_names())}"
        )
    return _REGISTRY[name]


def _module(name: str) -> Any:
    return _entry(name)[0]


def _setup_kwargs(name: str) -> Dict[str, Any]:
    compressor_name = _entry(name)[1]
    return {"compressor_name": compressor_name} if compressor_name is not None else {}


def _ensure_setup(name: str) -> Tuple[Any, Any]:
    if name not in _API_CACHE:
        mod = _module(name)
        _API_CACHE[name] = mod.setup_model_and_callables(**_setup_kwargs(name))
    return _API_CACHE[name]


def setup_task_api(name: str) -> Tuple[Any, Any]:
    """Return cached ``(api, extra)`` from the task instance module, creating it if needed."""
    return _ensure_setup(name)


def get_task(
    name: str,
    *,
    task_id: int = 1,
    w_k: float = 1.0,
    R_k: float = 10.0,
) -> InferenceTask:
    """
    Return a configured :class:`~src.core.task.InferenceTask` for ``name``.

    The task family's setup runs at most once per process per ``name``; further calls reuse
    the same callables so multiple tasks can share one trained model without passing ``api``.
    """
    api, _ = _ensure_setup(name)
    kwargs = _setup_kwargs(name)
    return _module(name).make_task(task_id=task_id, api=api, w_k=w_k, R_k=R_k, **kwargs)
