"""
Flan-T5-base on SST-2, scored by classification accuracy.

Paper scenario ``FT5-SST2``. This is the one encoder-decoder task, so it does
not go through :mod:`src.core.task_instances._llm_base`: the metric is a
two-way verbalizer comparison rather than perplexity or multiple-choice MMLU,
and the compressed links sit at *encoder* block boundaries.

Topology (L_k = 4, dim(eta) = 3), matching the concavity harness in
``experiments/concavity/flant5_sst2/``::

    Enc0 (blocks 0-3) -> Enc1 (blocks 4-7) -> Enc2 (blocks 8-11) -> Decoder
                      tp0                 tp1                    tp2

tp0 and tp1 hook encoder blocks 3 and 7. tp2 hooks ``encoder.final_layer_norm``
rather than block 11, because that is the tensor actually handed to the decoder
and it is where the harness compresses.

Scoring: the sentence goes into a fixed sentiment prompt, the decoder is run for
one step from its start token, and the label is whichever of the single-token
verbalizers ("positive" / "negative") gets the higher logit. Accuracy is over a
fixed subset of the SST-2 validation split, so repeated evaluations at the same
eta are deterministic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ._llm_base import STRATEGY_FOR_CODEC, load_stage_profile, resolve_compressor_name
from ._llm_base import LLMTaskSpec
from ..llm_compression import ActivationCompressionConfig, LLMActivationCompressor
from ..task import InferenceTask
from ..toy_A import grad_oracle

_ROOT = Path(__file__).resolve().parents[3]

# Single canonical copy of the validation split, committed with the concavity
# harness so both run without Hub access. Override with FLANT5_SST2_DATA.
_DEFAULT_SST2 = (_ROOT / "experiments" / "concavity" / "flant5_sst2" / "data"
                 / "sst2_validation.jsonl")

_PROMPT = (
    "Classify the sentiment of the following sentence as positive or negative.\n"
    "Sentence: {sentence}\n"
    "Sentiment:"
)
# Tried in order; the first pair that tokenizes to one distinct token each wins.
_VERBALIZERS = [("positive", "negative"), ("yes", "no"), ("true", "false")]

#: Encoder blocks after which tp0 and tp1 sit. tp2 is handled separately: the
#: encoder-to-decoder link carries the tensor AFTER encoder.final_layer_norm,
#: so compressing block 11's output would compress the wrong tensor (the
#: concavity harness applies the final layer norm inside its last encoder
#: partition, before transport). encoder.dropout follows the norm but is
#: identity in eval mode.
_CUT_BLOCKS = [3, 7]

SPEC = LLMTaskSpec(
    name="flant5_sst2",
    model_name="google/flan-t5-base",
    metric="sst2_accuracy",
    n_stages=4,
    eta_min=0.1,
    fast_samples=128,
    true_samples=512,
    max_length=128,
    batch_size=16,
    # Accuracy is a discrete metric, so the Stein oracle needs a wider probe
    # than the continuous perplexity tasks: measured on flan-t5/SST-2 at
    # eta=0.6, sigma=0.05 returns an all-zero gradient (no perturbation flips a
    # prediction), while sigma=0.10 is nonzero on every link. The perplexity
    # tasks keep 0.05, where they already produce usable gradients.
    grad_sigma=0.1,
)


def _load_samples(limit: int) -> List[Tuple[str, int]]:
    """First *limit* (sentence, label) pairs of the SST-2 validation split."""
    path = Path(os.environ.get("FLANT5_SST2_DATA", _DEFAULT_SST2))
    if path.exists():
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset

        rows = [dict(r) for r in load_dataset("glue", "sst2", split="validation")]

    out: List[Tuple[str, int]] = []
    for r in rows:
        sentence = r.get("sentence", r.get("text"))
        label = r.get("label", r.get("gold"))
        if sentence is None or label is None:
            continue
        out.append((str(sentence), int(label)))
        if len(out) >= limit:
            break
    if not out:
        raise RuntimeError(f"No SST-2 samples loaded (looked at {path}).")
    return out


def _resolve_verbalizers(tokenizer) -> Tuple[int, int]:
    """(negative_id, positive_id) for the first usable single-token pair."""
    for pos, neg in _VERBALIZERS:
        pos_ids = tokenizer.encode(pos, add_special_tokens=False)
        neg_ids = tokenizer.encode(neg, add_special_tokens=False)
        if len(pos_ids) == 1 and len(neg_ids) == 1 and pos_ids[0] != neg_ids[0]:
            return int(neg_ids[0]), int(pos_ids[0])
    raise RuntimeError("No single-token verbalizer pair works for this tokenizer.")


class _SST2Metric:
    def __init__(self, model, tokenizer, samples, *, device, max_length,
                 batch_size, verbalizers, decoder_start_id):
        self.model, self.tokenizer, self.samples = model, tokenizer, samples
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.neg_id, self.pos_id = verbalizers
        self.decoder_start_id = decoder_start_id
        self.labels = np.asarray([lab for _, lab in samples], dtype=np.int64)
        self.prompts = [_PROMPT.format(sentence=s.strip()) for s, _ in samples]

    @torch.no_grad()
    def score(self) -> float:
        preds: List[int] = []
        for start in range(0, len(self.prompts), self.batch_size):
            batch = self.prompts[start:start + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            # One decoder step from the start token; only that position is scored.
            dec_in = torch.full((input_ids.size(0), 1), self.decoder_start_id,
                                dtype=torch.long, device=self.device)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                decoder_input_ids=dec_in, use_cache=False).logits
            step = logits[:, 0, :].float()
            preds.extend(
                (step[:, self.pos_id] >= step[:, self.neg_id]).long().cpu().tolist()
            )
        return float((np.asarray(preds) == self.labels).mean())


class FlanT5SST2Callables:
    """The three callables the task contract requires."""

    def __init__(self, *, spec, model, compressor, fast, true):
        self.spec, self.model, self.compressor = spec, model, compressor
        self._fast, self._true = fast, true

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


def setup_model_and_callables(*, compressor_name: Optional[str] = None):
    spec = SPEC.with_env_overrides()
    codec = resolve_compressor_name(compressor_name)
    strategy = STRATEGY_FOR_CODEC[codec]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(spec.model_name)
    # FP32 throughout: this model is small, and the SST-2 margin between the two
    # verbalizer logits is narrow enough that FP16 flips labels.
    model = T5ForConditionalGeneration.from_pretrained(spec.model_name).to(device)
    model.eval()

    n_enc = len(model.encoder.block)
    if n_enc != 12:
        raise ValueError(
            f"{spec.name} assumes the 12-block flan-t5-base encoder for the "
            f"fixed 3-transfer-point split; got {n_enc} blocks for "
            f"{spec.model_name}."
        )

    # tp0, tp1 hang off encoder blocks; tp2 hangs off the encoder's final layer
    # norm, so the three hookable modules are collected explicitly and indexed
    # 0, 1, 2 (see the LLMActivationCompressor `layers` argument).
    hookable = nn.ModuleList([
        model.encoder.block[_CUT_BLOCKS[0]],
        model.encoder.block[_CUT_BLOCKS[1]],
        model.encoder.final_layer_norm,
    ])
    compressor = LLMActivationCompressor(
        model,
        ActivationCompressionConfig(strategy=strategy, layer_indices=[0, 1, 2],
                                    random_seed=spec.eval_seed),
        layers=hookable,
    )

    decoder_start_id = model.config.decoder_start_token_id
    if decoder_start_id is None:
        decoder_start_id = model.config.pad_token_id
    verbalizers = _resolve_verbalizers(tokenizer)

    samples_true = _load_samples(spec.true_samples)
    samples_fast = samples_true[:min(spec.fast_samples, len(samples_true))]

    def _metric(samples):
        return _SST2Metric(model, tokenizer, samples, device=device,
                           max_length=spec.max_length, batch_size=spec.batch_size,
                           verbalizers=verbalizers, decoder_start_id=int(decoder_start_id))

    print(f"[{spec.name}] {spec.model_name}: encoder blocks={n_enc}, L_k="
          f"{spec.n_stages}, cuts after blocks {_CUT_BLOCKS} + encoder "
          f"final_layer_norm, codec={codec} -> {strategy}", flush=True)

    api = FlanT5SST2Callables(spec=spec, model=model, compressor=compressor,
                              fast=_metric(samples_fast), true=_metric(samples_true))
    extra = {"spec": spec, "codec": codec, "strategy": strategy,
             "cuts": list(_CUT_BLOCKS) + ["encoder.final_layer_norm"],
             "device": str(device)}
    return api, extra


def make_task(
    task_id: int,
    api: FlanT5SST2Callables,
    w_k: float = 1.0,
    R_k: float = 10.0,
    compressor_name: Optional[str] = None,
) -> InferenceTask:
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
