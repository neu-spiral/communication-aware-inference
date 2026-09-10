"""
Shared machinery for the LLM task instances.

The five LLM scenarios differ only in model, corpus, metric and stage count, so
they all reduce to one pipeline:

    load model -> install activation-compression hooks at the cut layers
               -> evaluate the task metric under a given eta
               -> wrap that as (accuracy, gradient, accuracy_true) callables

Each concrete task module is a thin wrapper: it declares an :class:`LLMTaskSpec`
and delegates ``setup_model_and_callables`` / ``make_task`` to this module. See
``src/core/task_instances/__init__.py`` for the contract those two functions
have to satisfy.

What eta means here
-------------------
``eta`` has one entry per compressed link, so ``dim(eta) = L_k - 1``. Each entry
is the compression ratio requested on that link, and the compressor name selects
how it is realised:

    topk         -> topk_per_token   keep the top-eta fraction per token
    quantization -> quantization     eta picks a bit-width rung
    llmint8      -> llmint8          two-band mixed precision, FP16 outliers

See :mod:`src.core.llmint8` for the two-band scheme behind the third row.

Cost of an evaluation
---------------------
Every ``accuracy_callable`` call is a real forward pass over the eval subset,
and the Stein gradient oracle needs ``2 * grad_N`` of them. These tasks are
therefore orders of magnitude more expensive per optimizer step than the toy
MLP, and they need the model weights locally (the Llama and Gemma checkpoints
are gated on the Hub). Size the subsets with the ``*_SAMPLES`` env knobs below
before running a long sweep.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from ..llm_compression import (
    ActivationCompressionConfig,
    LLMActivationCompressor,
    detect_llm_decoder_layers,
)
from ..llm_text_eval import (
    load_sharegpt_texts,
    load_wikitext_texts,
    make_cut_points,
    per_sample_mean_nll,
    sample_scores,
)
from ..task import InferenceTask
from ..toy_A import grad_oracle

_ROOT = Path(__file__).resolve().parents[3]
_STAGE_PROFILE = _ROOT / "assets" / "llm_stage_profiles.json"

# compressor name (registry-facing) -> activation strategy (llm_compression)
STRATEGY_FOR_CODEC = {
    "topk": "topk_per_token",
    "quantization": "quantization",
    "llmint8": "llmint8",
}

_DEFAULT_MMLU_SUBJECTS = (
    "college_computer_science",
    "high_school_mathematics",
    "professional_law",
    "global_facts",
    "miscellaneous",
    "business_ethics",
)


@dataclass(frozen=True)
class LLMTaskSpec:
    """Everything that distinguishes one LLM scenario from another."""

    #: Registry-facing task name, e.g. ``"llama31_8b_mmlu"``.
    name: str
    #: Hub id of the model, e.g. ``"meta-llama/Llama-3.1-8B"``.
    model_name: str
    #: ``"perplexity"`` (bounded PPL-ratio score) or ``"mmlu_accuracy"``.
    metric: str
    #: Corpus for the perplexity metric: ``"wikitext"`` or ``"sharegpt"``.
    corpus: Optional[str] = None
    #: L_k, the number of pipeline stages. dim(eta) is ``n_stages - 1``.
    n_stages: int = 5
    #: Per-link lower clamp on eta, applied inside every evaluation.
    eta_min: float = 0.1
    #: Texts / MMLU questions per subject for the cheap evaluation.
    fast_samples: int = 32
    #: Same for the reference evaluation behind ``accuracy_callable_true``.
    true_samples: int = 128
    max_length: int = 512
    batch_size: int = 1
    eval_seed: int = 0
    grad_sigma: float = 0.05
    grad_N: int = 3
    mmlu_subjects: Sequence[str] = field(default_factory=lambda: _DEFAULT_MMLU_SUBJECTS)
    mmlu_n_shot: int = 5

    @property
    def n_links(self) -> int:
        return self.n_stages - 1

    def with_env_overrides(self) -> "LLMTaskSpec":
        """Apply ``LLM_TASK_*`` env overrides for the knobs worth tuning per run."""
        prefix = "LLM_TASK_"

        def _get(key, cast, current):
            raw = os.environ.get(prefix + key)
            return cast(raw) if raw not in (None, "") else current

        return LLMTaskSpec(
            name=self.name,
            model_name=_get("MODEL", str, self.model_name),
            metric=self.metric,
            corpus=self.corpus,
            n_stages=_get("N_STAGES", int, self.n_stages),
            eta_min=_get("ETA_MIN", float, self.eta_min),
            fast_samples=_get("FAST_SAMPLES", int, self.fast_samples),
            true_samples=_get("TRUE_SAMPLES", int, self.true_samples),
            max_length=_get("MAX_LENGTH", int, self.max_length),
            batch_size=_get("BATCH_SIZE", int, self.batch_size),
            eval_seed=_get("SEED", int, self.eval_seed),
            grad_sigma=_get("GRAD_SIGMA", float, self.grad_sigma),
            grad_N=_get("GRAD_N", int, self.grad_N),
            mmlu_subjects=self.mmlu_subjects,
            mmlu_n_shot=_get("MMLU_N_SHOT", int, self.mmlu_n_shot),
        )


# ---------------------------------------------------------------------------
# Topology: measured tau and a
# ---------------------------------------------------------------------------

def load_stage_profile(
    spec: LLMTaskSpec,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Return ``(tau, a)`` for *spec* from the measured stage profile.

    ``tau[i]`` is stage ``i``'s compute time in seconds and ``a[i]`` the bytes
    on link ``i``, as written by ``profile_llm_stages.py``.

    Raises
    ------
    FileNotFoundError, KeyError
        If the profile has no entry for this (model, stages, batch, seq_len).
        Deliberately fatal: a silently invented latency would flow straight
        into the optimizer's objective and look like a measurement.
    """
    if not _STAGE_PROFILE.exists():
        raise FileNotFoundError(
            f"No stage profile at {_STAGE_PROFILE}. Measure it first:\n"
            f"  python profile_llm_stages.py --model {spec.model_name} "
            f"--n_stages {spec.n_stages} --batch_size {spec.batch_size} "
            f"--seq_len {spec.max_length}"
        )
    profiles = json.loads(_STAGE_PROFILE.read_text())
    try:
        rec = (profiles[spec.model_name][str(spec.n_stages)]
                       [str(spec.batch_size)][str(spec.max_length)])
    except KeyError as exc:
        have = json.dumps(
            {m: sorted(v.keys()) for m, v in profiles.items()}, indent=2)
        raise KeyError(
            f"Stage profile has no entry for model={spec.model_name} "
            f"n_stages={spec.n_stages} batch_size={spec.batch_size} "
            f"seq_len={spec.max_length}. Measured stage counts:\n{have}\n"
            f"Add it with:\n  python profile_llm_stages.py "
            f"--model {spec.model_name} --n_stages {spec.n_stages} "
            f"--batch_size {spec.batch_size} --seq_len {spec.max_length}"
        ) from exc

    tau_s, a_bytes = rec["tau_s"], rec["a_bytes"]
    if len(tau_s) != spec.n_stages or len(a_bytes) != spec.n_links:
        raise ValueError(
            f"Profile shape mismatch for {spec.name}: got {len(tau_s)} stages "
            f"and {len(a_bytes)} links, expected {spec.n_stages} and "
            f"{spec.n_links}."
        )
    tau = {i: float(v) for i, v in enumerate(tau_s)}
    a = {i: float(v) for i, v in enumerate(a_bytes)}
    return tau, a


# ---------------------------------------------------------------------------
# Metric evaluators
# ---------------------------------------------------------------------------

class _PerplexityMetric:
    """Bounded perplexity score against an uncompressed reference.

    The reference per-sample NLL is measured once with the hooks disabled
    (eta = 1) and cached, so each later evaluation costs one forward pass.
    """

    def __init__(self, model, tokenizer, texts, *, device, max_length, batch_size):
        self.model, self.tokenizer, self.texts = model, tokenizer, texts
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self._hbar_ref: Optional[np.ndarray] = None

    def set_reference(self, hbar_ref: np.ndarray) -> None:
        self._hbar_ref = hbar_ref

    def measure_nll(self) -> np.ndarray:
        return per_sample_mean_nll(
            self.model, self.tokenizer, self.texts,
            device=self.device, max_length=self.max_length,
            batch_size=self.batch_size,
        )

    def score(self) -> float:
        if self._hbar_ref is None:
            raise RuntimeError("reference NLL not measured yet")
        return float(np.mean(sample_scores(self._hbar_ref, self.measure_nll())))


class _MMLUMetric:
    """Few-shot MMLU accuracy under the current eta."""

    def __init__(self, model, evaluator, *, device):
        self.model, self.evaluator, self.device = model, evaluator, device

    def score(self) -> float:
        return float(self.evaluator.accuracy(self.model, device=self.device))


# ---------------------------------------------------------------------------
# Callables
# ---------------------------------------------------------------------------

class LLMTaskCallables:
    """``accuracy_callable`` / ``gradient_callable`` / ``accuracy_callable_true``.

    ``accuracy_callable`` uses the cheap subset and is what the optimizer and
    the Stein oracle call; ``accuracy_callable_true`` uses the larger subset and
    is for reporting.
    """

    def __init__(self, *, spec: LLMTaskSpec, model, compressor, fast, true, device):
        self.spec = spec
        self.model = model
        self.compressor = compressor
        self._fast = fast
        self._true = true
        self.device = device

    # -- internals ---------------------------------------------------------
    def _clamped(self, eta_vec: np.ndarray) -> torch.Tensor:
        eta = torch.as_tensor(np.asarray(eta_vec, dtype=np.float32).reshape(-1))
        if eta.numel() != self.spec.n_links:
            raise ValueError(
                f"{self.spec.name}: expected eta of length {self.spec.n_links} "
                f"(L_k={self.spec.n_stages}), got {eta.numel()}"
            )
        return torch.clamp(eta, self.spec.eta_min, 1.0)

    def _score(self, eta_vec: np.ndarray, *, true: bool) -> float:
        self.compressor.set_eta(self._clamped(eta_vec))
        return (self._true if true else self._fast).score()

    # -- public API --------------------------------------------------------
    def accuracy_callable(self, eta_vec: np.ndarray) -> float:
        return self._score(eta_vec, true=False)

    def accuracy_callable_true(self, eta_vec: np.ndarray) -> float:
        return self._score(eta_vec, true=True)

    def gradient_callable(self, eta_vec: np.ndarray) -> np.ndarray:
        eta = torch.as_tensor(np.asarray(eta_vec, dtype=np.float32).reshape(-1))

        def f(e: torch.Tensor) -> torch.Tensor:
            return torch.tensor(self._score(e.detach().cpu().numpy(), true=False),
                                dtype=torch.float32, device=e.device)

        g = grad_oracle(f, eta, sigma=float(self.spec.grad_sigma),
                        N=int(self.spec.grad_N))
        return g.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def resolve_compressor_name(override: Optional[str]) -> str:
    raw = override if override is not None else os.environ.get("LLM_COMPRESSOR_NAME", "topk")
    name = str(raw).strip().lower()
    aliases = {"topk": "topk", "top_k": "topk",
               "quant": "quantization", "quantization": "quantization",
               "llmint8": "llmint8", "llm_int8": "llmint8", "llm.int8": "llmint8"}
    if name not in aliases:
        raise ValueError(
            f"Unsupported compressor {raw!r}. Expected one of: "
            f"topk, quantization, llmint8."
        )
    return aliases[name]


def setup_llm_task(spec: LLMTaskSpec, *, compressor_name: Optional[str] = None):
    """Load the model, install the hooks, and build the callables.

    Returns ``(api, extra)`` per the task-instance contract, where ``extra`` is
    a dict of the resolved setup (cut layers, strategy, device) that is handy
    for logging and for tests.
    """
    spec = spec.with_env_overrides()
    codec = resolve_compressor_name(compressor_name)
    strategy = STRATEGY_FOR_CODEC[codec]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()

    n_layers = len(detect_llm_decoder_layers(model))
    cuts = make_cut_points(n_layers, spec.n_links)
    compressor = LLMActivationCompressor(
        model,
        ActivationCompressionConfig(
            strategy=strategy, layer_indices=cuts, random_seed=spec.eval_seed
        ),
    )
    print(f"[{spec.name}] {spec.model_name}: {n_layers} layers, L_k="
          f"{spec.n_stages}, cuts at {cuts}, codec={codec} -> {strategy}",
          flush=True)

    if spec.metric == "perplexity":
        loader = {"wikitext": load_wikitext_texts,
                  "sharegpt": load_sharegpt_texts}[spec.corpus]
        # The cheap subset is a prefix of the reference one, so both metrics see
        # the same texts and their scores stay comparable.
        texts_true = loader(spec.true_samples)
        texts_fast = texts_true[:min(spec.fast_samples, len(texts_true))]
        fast = _PerplexityMetric(model, tokenizer, texts_fast, device=device,
                                 max_length=spec.max_length, batch_size=spec.batch_size)
        true = _PerplexityMetric(model, tokenizer, texts_true, device=device,
                                 max_length=spec.max_length, batch_size=spec.batch_size)
        # Reference NLL with compression off, measured once.
        compressor.set_eta(torch.ones(spec.n_links))
        fast.set_reference(fast.measure_nll())
        true.set_reference(true.measure_nll())
    elif spec.metric == "mmlu_accuracy":
        from ..mmlu_eval import MMLUEvalConfig, MMLUEvaluator

        def _ev(samples, seed):
            return MMLUEvaluator(tokenizer=tokenizer, config=MMLUEvalConfig(
                subjects=tuple(spec.mmlu_subjects), samples_per_subject=samples,
                seed=seed, max_length=spec.max_length,
                batch_size=spec.batch_size, n_shot=spec.mmlu_n_shot))

        fast = _MMLUMetric(model, _ev(spec.fast_samples, spec.eval_seed), device=device)
        true = _MMLUMetric(model, _ev(spec.true_samples, spec.eval_seed + 1), device=device)
    else:
        raise ValueError(f"Unknown metric {spec.metric!r} for task {spec.name}")

    api = LLMTaskCallables(spec=spec, model=model, compressor=compressor,
                           fast=fast, true=true, device=device)
    extra = {"spec": spec, "codec": codec, "strategy": strategy,
             "cuts": cuts, "n_layers": n_layers, "device": str(device)}
    return api, extra


def make_llm_task(
    *,
    task_id: int,
    api: LLMTaskCallables,
    w_k: float = 1.0,
    R_k: float = 10.0,
    compressor_name: Optional[str] = None,
) -> InferenceTask:
    """Build the :class:`InferenceTask` behind *api*, with measured tau and a.

    The spec comes from ``api`` rather than being passed in, so the topology
    always matches the spec the callables were actually built on -- including
    any ``LLM_TASK_*`` override applied during setup.
    """
    spec = api.spec
    tau, a = load_stage_profile(spec)
    task = InferenceTask(
        task_id=task_id,
        b_k=0,
        L_k=spec.n_stages,
        tau=tau,
        a=a,
        eta_min={i: float(spec.eta_min) for i in range(spec.n_links)},
        R_k_callable=lambda t, _R=R_k: float(_R),
        w_k=w_k,
        accuracy_callable=api.accuracy_callable,
        accuracy_callable_true=api.accuracy_callable_true,
        gradient_callable=api.gradient_callable,
    )
    task.compressor_name = resolve_compressor_name(compressor_name)
    return task
