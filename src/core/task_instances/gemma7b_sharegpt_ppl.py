"""
Gemma-7B on ShareGPT, scored by bounded perplexity ratio.

Paper scenario ``G1-7B-SGPT``. See :mod:`src.core.task_instances._llm_base` for
the shared pipeline, what ``eta`` means, and the env overrides.
"""

from __future__ import annotations

from typing import Optional

from ._llm_base import LLMTaskSpec, make_llm_task, setup_llm_task
from ..task import InferenceTask

SPEC = LLMTaskSpec(
    name="gemma7b_sharegpt_ppl",
    model_name="google/gemma-7b",
    metric="perplexity",
    corpus="sharegpt",
    n_stages=5,
    eta_min=0.1,
    fast_samples=32,
    true_samples=128,
    max_length=512,
    batch_size=1,
)


def setup_model_and_callables(*, compressor_name: Optional[str] = None):
    return setup_llm_task(SPEC, compressor_name=compressor_name)


def make_task(
    task_id: int,
    api,
    w_k: float = 1.0,
    R_k: float = 10.0,
    compressor_name: Optional[str] = None,
) -> InferenceTask:
    return make_llm_task(task_id=task_id, api=api, w_k=w_k, R_k=R_k,
                         compressor_name=compressor_name)
