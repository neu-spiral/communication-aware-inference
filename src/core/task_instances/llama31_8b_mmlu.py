"""
Llama-3.1-8B on 5-shot MMLU, scored by accuracy.

Paper scenario ``Ll3-8B-MMLU``. See :mod:`src.core.task_instances._llm_base` for
the shared pipeline, what ``eta`` means, and the env overrides.
"""

from __future__ import annotations

from typing import Optional

from ._llm_base import LLMTaskSpec, make_llm_task, setup_llm_task
from ..task import InferenceTask

SPEC = LLMTaskSpec(
    name="llama31_8b_mmlu",
    model_name="meta-llama/Llama-3.1-8B",
    metric="mmlu_accuracy",
    n_stages=5,
    eta_min=0.1,
    # Questions PER SUBJECT, over 6 subjects. 5/subject (30 prompts) was
    # measured to be too coarse to use: granularity is 1/30, and the utility
    # came out non-monotone in eta (0.667 at eta=1, 0.600 at 0.5, 0.700 at 0.2
    # -- all within two questions of each other). 20/subject is the setting the
    # concavity campaign established as adequate at tol=0.02. It costs roughly
    # 50 s per accuracy_callable on a V100, so about 5 min per Stein gradient
    # at grad_N=3; drop LLM_TASK_FAST_SAMPLES for a quick smoke test, knowing
    # the utility will be noisy.
    fast_samples=20,
    true_samples=40,
    # 5-shot MMLU prompts run ~1500-2000 tokens, and MMLUEvaluator raises its
    # own tokenization ceiling to 2048 whenever n_shot > 0. Declaring 2048 here
    # keeps the profiled link payload honest: at 512 the profile would describe
    # a quarter of the activation this task actually ships.
    max_length=2048,
    batch_size=4,
    # Wider Stein probe than the perplexity tasks, as a precaution rather than
    # a measured need here: on flan-t5/SST-2 at eta=0.6, sigma=0.05 returned an
    # all-zero gradient because no perturbation flipped a prediction, and only
    # sigma=0.10 was nonzero. MMLU at 5 questions/subject did produce a nonzero
    # gradient at 0.05, but that was with granularity 1/30; at the 20/subject
    # default below it is 1/120, i.e. closer to the regime where T5 went flat.
    grad_sigma=0.1,
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
