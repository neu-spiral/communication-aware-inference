#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Monte-Carlo concavity test for Flan-T5-base on SST-2.

The model, dataset, prompt, verbalizers, and fixed 3-transfer-point split match
the original Flan-T5/SST-2 local estimator experiment. The concavity workflow
matches mc_concavity.py: random rays in eta-space, discrete second-difference
testing, eta_min search, and phase-2 validation in the estimated sub-cube.

Direct run:
  python flan_t5_sst2_concavity.py

Override defaults:
  python flan_t5_sst2_concavity.py --device cuda --max_samples 100

Recompute eta_min from saved rays:
  python flan_t5_sst2_concavity.py etamin --rays_json path/to/*_rays.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from flan_t5_sst2_runtime import (  # noqa: E402
    DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SPLIT,
    NUM_TRANSFER_POINTS,
    TRANSFER_POINT_LABELS,
    FlanT5SST2PartitionRunner,
    build_prompt,
    decide_label,
    load_sst2_samples,
    resolve_single_token_verbalizers,
    _extract_two_way_logits,
)


METRICS = ("accuracy", "neg_ce", "ppl_score")
DEFAULT_TOL = {
    "accuracy": 0.01,
    "neg_ce": 0.001,
    "ppl_score": 0.001,
}

# Default experiment configuration. Edit this block for the common direct-run
# case; every value can still be overridden from the command line.
RUN_DEFAULT_MODEL_NAME = DEFAULT_MODEL_NAME
RUN_DEFAULT_DATASET_PATH = None
RUN_DEFAULT_SPLIT = DEFAULT_SPLIT
RUN_DEFAULT_DEVICE = "cuda"
RUN_DEFAULT_MAX_SAMPLES = 0  # 0 or negative uses the full split.
RUN_DEFAULT_MAX_INPUT_LENGTH = DEFAULT_MAX_INPUT_LENGTH
RUN_DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE
RUN_DEFAULT_POSITIVE_TOKEN = None
RUN_DEFAULT_NEGATIVE_TOKEN = None
RUN_DEFAULT_COMPRESSOR_NAME = "topk"
RUN_DEFAULT_LLMINT8_OUTLIER_PRECISION = "fp16"
RUN_DEFAULT_LLMINT8_REGULAR_PRECISION = "int8"
RUN_DEFAULT_METRICS = ["all"]
RUN_DEFAULT_N_RAYS = 160
RUN_DEFAULT_PHASE2_RAYS = 80
RUN_DEFAULT_N_POINTS = 7
RUN_DEFAULT_EPS = 1e-3
RUN_DEFAULT_MIN_SEG = 0.05
RUN_DEFAULT_TARGET = 0.95
RUN_DEFAULT_MIN_USABLE = 20
RUN_DEFAULT_SEED = 0
RUN_DEFAULT_TOL_ACCURACY = DEFAULT_TOL["accuracy"]
RUN_DEFAULT_TOL_NEG_CE = DEFAULT_TOL["neg_ce"]
RUN_DEFAULT_TOL_PPL_SCORE = DEFAULT_TOL["ppl_score"]
RUN_DEFAULT_OUT_DIR = str(SCRIPT_DIR / "flant5_sst2_concavity_result")

ETAMIN_DEFAULT_TOL = None
ETAMIN_DEFAULT_TARGET = 0.95
ETAMIN_DEFAULT_MIN_USABLE = 20
ETAMIN_DEFAULT_OUT = None


def _json_dumps(obj) -> str:
    return json.dumps(obj, indent=2)


def _model_slug(model_name: str) -> str:
    return str(model_name).replace("/", "__")


def _sanitize_tag(text: str) -> str:
    chars = []
    for ch in str(text).strip().lower():
        chars.append(ch if ch.isalnum() else "_")
    return "".join(chars).strip("_")


def _parse_metrics(values: Sequence[str]) -> List[str]:
    if not values:
        return list(METRICS)
    out: List[str] = []
    for value in values:
        for item in str(value).split(","):
            name = item.strip().lower()
            if not name:
                continue
            if name == "all":
                out.extend(METRICS)
            elif name in METRICS:
                out.append(name)
            else:
                raise ValueError(
                    f"Unsupported metric '{name}'. Choose from {METRICS} or all."
                )
    deduped = []
    seen = set()
    for name in out:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _metric_tolerances(args, metrics: Iterable[str]) -> Dict[str, float]:
    explicit = {
        "accuracy": args.tol_accuracy,
        "neg_ce": args.tol_neg_ce,
        "ppl_score": args.tol_ppl_score,
    }
    return {m: float(explicit[m]) for m in metrics}


def _stem(args, metric: str) -> str:
    strategy = _sanitize_tag(args.compressor_name)
    return (
        f"{_model_slug(args.model_name)}__sst2__{metric}__"
        f"{strategy}__cuts{NUM_TRANSFER_POINTS}"
    )


def _eta_to_compression_params(
    eta_vec: np.ndarray,
    *,
    compressor_name: str,
    llmint8_outlier_precision: str,
    llmint8_regular_precision: str,
) -> List[object]:
    name = str(compressor_name).strip().lower()
    eta = [float(np.clip(x, 0.0, 1.0)) for x in eta_vec]
    if name in ("", "identity", "none"):
        return [None for _ in eta]
    if name == "llmint8":
        # eta is a compression ratio, not an outlier fraction: the parameter
        # triple comes from src.core.llmint8, the single definition of the
        # scheme. llmint8_regular_precision pins the low band when set.
        from src.core import llmint8

        return [
            llmint8.codec_params(x, regular_precision=llmint8_regular_precision or None)
            for x in eta
        ]
    return eta


def second_difference_test_on(ys: Sequence[float], tol: float = 1e-3) -> dict:
    ys = list(ys)
    violations = 0
    worst = 0.0
    for j in range(1, len(ys) - 1):
        v = ys[j - 1] + ys[j + 1] - 2.0 * ys[j]
        if v > worst:
            worst = v
        if v > tol:
            violations += 1
    return {
        "n_interior": max(0, len(ys) - 2),
        "violations": violations,
        "worst_d2": float(worst),
        "tol": float(tol),
    }


def sample_random_ray(
    rng: np.random.Generator,
    n: int,
    *,
    lo: np.ndarray,
    hi: float = 1.0,
    min_seg: float = 0.05,
    max_tries: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo, dtype=np.float64)
    start = lo.copy()
    end = lo.copy()
    for _ in range(max_tries):
        start = lo + (hi - lo) * rng.random(n)
        end = lo + (hi - lo) * rng.random(n)
        if np.linalg.norm(end - start) >= min_seg:
            return start, end
    return start, end


def ray_eta_points(start: np.ndarray, end: np.ndarray, n_points: int):
    ts = np.linspace(0.0, 1.0, n_points)
    etas = [(1.0 - t) * start + t * end for t in ts]
    return ts, etas


def concavity_of(rays: List[dict]) -> Tuple[float, int]:
    if not rays:
        return float("nan"), 0
    n_concave = sum(1 for r in rays if r["concave"])
    return n_concave / len(rays), len(rays)


def clip_indices(start, end, ts, v, atol: float = 1e-9) -> List[int]:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)

    t_lo, t_hi = 0.0, 1.0
    for k in range(len(v)):
        delta = end[k] - start[k]
        rhs = v[k] - start[k]
        if abs(delta) <= atol:
            if start[k] < v[k] - atol:
                return []
            continue
        bound = rhs / delta
        if delta > 0:
            t_lo = max(t_lo, bound)
        else:
            t_hi = min(t_hi, bound)
    if t_lo > t_hi + atol:
        return []
    return [j for j, t in enumerate(ts) if (t_lo - atol) <= t <= (t_hi + atol)]


def concavity_fraction(
    rays: List[dict],
    v: np.ndarray,
    tol: float,
    min_pts: int = 3,
) -> Tuple[Optional[float], int]:
    usable = 0
    concave = 0
    for ray in rays:
        idx = clip_indices(ray["start"], ray["end"], ray["t"], v)
        if len(idx) < min_pts:
            continue
        usable += 1
        ys = [ray["y"][j] for j in idx]
        if second_difference_test_on(ys, tol)["violations"] == 0:
            concave += 1
    frac = (concave / usable) if usable > 0 else None
    return frac, usable


def _meets(frac: Optional[float], usable: int, target: float, min_usable: int) -> bool:
    return frac is not None and usable >= min_usable and frac >= target


def find_eta_min(
    rays: List[dict],
    n: int,
    *,
    tol: float,
    target: float = 0.95,
    min_usable: int = 30,
    coarse_step: float = 0.1,
    fine_step: float = 0.02,
    v_max: float = 0.95,
) -> dict:
    history: List[dict] = []

    def frac_usable(v):
        return concavity_fraction(rays, v, tol)

    v = np.zeros(n, dtype=np.float64)
    for _ in range(n * int(round(v_max / coarse_step)) + 2):
        frac, usable = frac_usable(v)
        if _meets(frac, usable, target, min_usable):
            break
        best = None
        for k in range(n):
            if v[k] >= v_max:
                continue
            cand = v.copy()
            cand[k] = min(v_max, v[k] + coarse_step)
            f, u = frac_usable(cand)
            if u < min_usable:
                continue
            key = (f if f is not None else -1.0, u)
            if best is None or key > best[0]:
                best = (key, cand)
        if best is None:
            break
        v = best[1]
        f, u = frac_usable(v)
        history.append({"pass": "ascent", "v": v.tolist(), "frac": f, "usable": u})

    improved = True
    while improved:
        improved = False
        best = None
        for k in range(n):
            if v[k] <= 0.0:
                continue
            cand = v.copy()
            cand[k] = max(0.0, v[k] - fine_step)
            f, u = frac_usable(cand)
            if _meets(f, u, target, min_usable):
                drop = v[k] - cand[k]
                if best is None or drop > best[0]:
                    best = (drop, cand)
        if best is not None:
            v = best[1]
            f, u = frac_usable(v)
            history.append({"pass": "descent", "v": v.tolist(), "frac": f, "usable": u})
            improved = True

    frac, usable = frac_usable(v)
    scalar_sweep = []
    for c in np.round(np.arange(0.0, v_max + 1e-9, 0.05), 3):
        f, u = frac_usable(np.full(n, float(c)))
        scalar_sweep.append({"c": float(c), "frac": f, "usable": u})

    return {
        "eta_min": v.tolist(),
        "frac_at_eta_min": frac,
        "usable_at_eta_min": usable,
        "met_target": _meets(frac, usable, target, min_usable),
        "target": target,
        "min_usable": min_usable,
        "tol": tol,
        "history": history,
        "scalar_floor_sweep": scalar_sweep,
    }


class FlanT5SST2ConcavityEvaluator:
    """Evaluate accuracy, neg_ce, and baseline-normalized PPL-score."""

    def __init__(
        self,
        *,
        model_name: str,
        dataset_path: Optional[str],
        split: str,
        device: str,
        max_samples: int,
        max_input_length: int,
        prompt_template: str,
        positive_token: Optional[str],
        negative_token: Optional[str],
        compressor_name: str,
        llmint8_outlier_precision: str,
        llmint8_regular_precision: str,
    ) -> None:
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.split = split
        self.device = device
        self.max_samples = int(max_samples)
        self.max_input_length = int(max_input_length)
        self.prompt_template = prompt_template
        self.compressor_name = compressor_name
        self.llmint8_outlier_precision = llmint8_outlier_precision
        self.llmint8_regular_precision = llmint8_regular_precision
        self.eval_counter = 0

        self.runner = FlanT5SST2PartitionRunner(
            model_name=model_name,
            device=device,
            max_input_length=max_input_length,
        )
        samples = load_sst2_samples(dataset_path=dataset_path, split=split)
        if self.max_samples > 0:
            samples = samples[: self.max_samples]
        if not samples:
            raise ValueError("No SST-2 samples loaded.")
        self.samples = samples
        self.verbalizers = resolve_single_token_verbalizers(
            self.runner.tokenizer,
            positive_text=positive_token,
            negative_text=negative_token,
        )
        self.encoded_samples = self._encode_samples(samples)

        print(
            "[FlanT5/SST2] model={} split={} samples={} device={} compressor={}".format(
                model_name,
                split,
                len(self.encoded_samples),
                device,
                compressor_name,
            ),
            flush=True,
        )
        print(
            "[FlanT5/SST2] verbalizers: positive='{}'({}) negative='{}'({})".format(
                self.verbalizers["positive_text"],
                self.verbalizers["positive_id"],
                self.verbalizers["negative_text"],
                self.verbalizers["negative_id"],
            ),
            flush=True,
        )

        baseline_rows = self._evaluate_per_sample(
            np.ones(NUM_TRANSFER_POINTS),
            label="baseline",
        )
        self.baseline_ce = np.array(
            [row["ce"] for row in baseline_rows],
            dtype=np.float64,
        )
        self.baseline = self._reduce_metrics(baseline_rows)

    def _encode_samples(self, samples) -> List[dict]:
        encoded = []
        for sample in samples:
            prompt_text = build_prompt(sample.sentence, self.prompt_template)
            payload = self.runner.encode_prompt(prompt_text)
            encoded.append(
                {
                    "sample": sample,
                    "prompt_text": prompt_text,
                    "input_ids": payload["input_ids"],
                    "attention_mask": payload["attention_mask"],
                }
            )
        return encoded

    def _evaluate_per_sample(self, eta_vec: np.ndarray, *, label: str) -> List[dict]:
        params = _eta_to_compression_params(
            eta_vec,
            compressor_name=self.compressor_name,
            llmint8_outlier_precision=self.llmint8_outlier_precision,
            llmint8_regular_precision=self.llmint8_regular_precision,
        )
        rows = []
        self.eval_counter += 1
        for sample_idx, sample_item in enumerate(self.encoded_samples):
            sample = sample_item["sample"]
            result = self.runner.run_partitioned_first_step(
                input_ids=sample_item["input_ids"],
                attention_mask=sample_item["attention_mask"],
                task_id=(
                    f"{label}_eval{self.eval_counter}_"
                    f"sample{sample.sample_id}_{sample_idx}"
                ),
                compressor_name=self.compressor_name,
                tp_compression_params=params,
            )
            logits = result["logits"]
            pred_label = decide_label(logits[0], self.verbalizers)
            true_label = int(sample.label)

            label_logits = _extract_two_way_logits(logits, self.verbalizers)
            labels = torch.tensor([true_label], dtype=torch.long, device=label_logits.device)
            ce = F.cross_entropy(label_logits, labels, reduction="none")
            positive_logit = float(logits[0, self.verbalizers["positive_id"]].item())
            negative_logit = float(logits[0, self.verbalizers["negative_id"]].item())
            rows.append(
                {
                    "correct": int(pred_label == true_label),
                    "ce": float(ce[0].detach().cpu().item()),
                    "margin_abs": abs(positive_logit - negative_logit),
                    "positive_logit": positive_logit,
                    "negative_logit": negative_logit,
                    "transfer_stats": result["transfer_stats"],
                }
            )
        return rows

    def _reduce_metrics(self, rows: List[dict]) -> Dict[str, float]:
        correct = np.array([row["correct"] for row in rows], dtype=np.float64)
        ce = np.array([row["ce"] for row in rows], dtype=np.float64)
        out = {
            "accuracy": float(correct.mean()),
            "neg_ce": float(-ce.mean()),
        }
        if hasattr(self, "baseline_ce"):
            delta = np.clip(self.baseline_ce - ce, -80.0, 80.0)
            ratios = np.minimum(1.0, np.exp(delta))
            out["ppl_score"] = float(ratios.mean())
        else:
            out["ppl_score"] = 1.0
        return out

    def evaluate_all(self, eta_vec: np.ndarray) -> Dict[str, float]:
        rows = self._evaluate_per_sample(np.asarray(eta_vec, dtype=np.float64), label="eta")
        return self._reduce_metrics(rows)


def _build_ray(
    *,
    ray_id: int,
    start: np.ndarray,
    end: np.ndarray,
    ts: np.ndarray,
    etas: np.ndarray,
    ys: List[float],
    tol: float,
) -> dict:
    sd = second_difference_test_on(ys, tol)
    return {
        "id": int(ray_id),
        "start": [float(x) for x in start],
        "end": [float(x) for x in end],
        "t": [float(x) for x in ts],
        "eta": [[float(x) for x in e] for e in etas],
        "y": [float(x) for x in ys],
        "concave": bool(sd["violations"] == 0),
        "worst_d2": sd["worst_d2"],
        "violations": sd["violations"],
    }


def adaptive_fill_phase(
    evaluate: Callable[[np.ndarray], float],
    rays: List[dict],
    n: int,
    *,
    rng: np.random.Generator,
    min_box_samples: int,
    floor_step: float,
    n_points: int,
    eps: float,
    min_seg: float,
    tol: float,
    lo_full: np.ndarray,
    out_dir: Path,
    stem: str,
    checkpoint_every: int = 5,
    floor_max: float = 0.9,
    prior_fill: Optional[List[dict]] = None,
    on_checkpoint: Optional[Callable[[List[dict]], None]] = None,
) -> List[dict]:
    """Sample extra rays so each floor level has >= min_box_samples usable rays.

    Ported from mc_concavity.adaptive_fill_phase. Scans floor levels low to high:
    the lowest under-sampled floor c is filled first with rays from [c, 1]^n whose
    grid points spread across the shells above c, so the intermediate 0.1 bands get
    populated before each higher floor is topped up with only its residual deficit
    (the old high->low order left the 0.6-0.9 shells empty behind a corner spike).
    `evaluate` returns the single target metric.

    Checkpointing is crash/timeout safe: `_fill.partial.json` is written as the
    CUMULATIVE set (prior_fill + this run's fill_rays), never just this run's, so a
    timed-out link can never regress the resumed set. `on_checkpoint` (if given) is
    called with this run's fills at every flush so the caller can also fold them
    into the real `_rays.json` mid-run rather than only on clean completion.
    """
    floors = np.round(np.arange(0.0, floor_max + 1e-9, floor_step), 3)
    fill_rays: List[dict] = []
    prior = list(prior_fill or [])
    all_rays = list(rays)
    cp_path = out_dir / f"{stem}_fill.partial.json"

    def _flush() -> None:
        # Persist prior + new together so the next link resumes from the full set.
        cp_path.write_text(json.dumps(prior + fill_rays), encoding="utf-8")
        if on_checkpoint is not None:
            on_checkpoint(list(fill_rays))

    # Bank the resumed state immediately, before evaluating any new ray, so a link
    # that dies before its first checkpoint still leaves prior_fill on disk / in
    # _rays.json instead of dropping it.
    _flush()

    for c in floors:
        v = np.full(n, float(c))
        _, usable = concavity_fraction(all_rays, v, tol)
        if usable >= min_box_samples:
            continue
        lo_fill = np.maximum(lo_full, v)
        batch_idx = 0
        while True:
            _, usable = concavity_fraction(all_rays, v, tol)
            if usable >= min_box_samples:
                break
            start, end = sample_random_ray(rng, n, lo=lo_fill, min_seg=min_seg)
            start = np.clip(start, eps, 1.0)
            end = np.clip(end, eps, 1.0)
            ts, etas = ray_eta_points(start, end, n_points)
            ys: List[float] = []
            t0 = time.time()
            for eta_vec in etas:
                ys.append(float(evaluate(eta_vec)))
            sd = second_difference_test_on(ys, tol)
            r = {
                "id": len(fill_rays),
                "start": [float(x) for x in start],
                "end": [float(x) for x in end],
                "t": [float(x) for x in ts],
                "eta": [[float(x) for x in e] for e in etas],
                "y": [float(x) for x in ys],
                "concave": bool(sd["violations"] == 0),
                "worst_d2": sd["worst_d2"],
                "violations": sd["violations"],
                "fill_floor": float(c),
            }
            fill_rays.append(r)
            all_rays.append(r)
            batch_idx += 1
            print(
                f"  [fill c={c:.2f}] ray {batch_idx}  "
                f"concave={r['concave']}  worst_d2={sd['worst_d2']:+.4f}  "
                f"({time.time()-t0:.1f}s)  usable={usable}/{min_box_samples}",
                flush=True,
            )
            if batch_idx % checkpoint_every == 0:
                _flush()

    if fill_rays:
        _flush()
    return fill_rays


def run_mc_phase_multi_metric(
    evaluator: FlanT5SST2ConcavityEvaluator,
    *,
    metrics: List[str],
    tolerances: Dict[str, float],
    n_rays: int,
    n_points: int,
    rng: np.random.Generator,
    lo: np.ndarray,
    eps: float,
    min_seg: float,
    label: str,
    out_dir: Path,
    stems: Dict[str, str],
    checkpoint_every: int = 5,
) -> Dict[str, List[dict]]:
    rays_by_metric: Dict[str, List[dict]] = {m: [] for m in metrics}
    for i in range(n_rays):
        start, end = sample_random_ray(rng, NUM_TRANSFER_POINTS, lo=lo, min_seg=min_seg)
        start = np.clip(start, eps, 1.0)
        end = np.clip(end, eps, 1.0)
        ts, etas = ray_eta_points(start, end, n_points)
        ys_by_metric: Dict[str, List[float]] = {m: [] for m in metrics}
        t0 = time.time()
        for eta_vec in etas:
            values = evaluator.evaluate_all(eta_vec)
            for metric in metrics:
                ys_by_metric[metric].append(float(values[metric]))

        status_bits = []
        for metric in metrics:
            ray = _build_ray(
                ray_id=i,
                start=start,
                end=end,
                ts=ts,
                etas=etas,
                ys=ys_by_metric[metric],
                tol=tolerances[metric],
            )
            rays_by_metric[metric].append(ray)
            status_bits.append(
                "{}:{} worst_d2={:+.4f}".format(
                    metric,
                    "C" if ray["concave"] else "NC",
                    float(ray["worst_d2"]),
                )
            )

        print(
            "  [{}] ray {}/{}  {}  ({:.1f}s)".format(
                label,
                i + 1,
                n_rays,
                " | ".join(status_bits),
                time.time() - t0,
            ),
            flush=True,
        )
        if (i + 1) % checkpoint_every == 0 or (i + 1) == n_rays:
            for metric in metrics:
                cp = out_dir / f"{stems[metric]}_{label}.partial.json"
                cp.write_text(json.dumps(rays_by_metric[metric]), encoding="utf-8")
    return rays_by_metric


def _base_summary(args, metric: str, tol: float, baseline: float) -> dict:
    return {
        "model": args.model_name,
        "dataset": "sst2",
        "split": args.split,
        "metric": metric,
        "strategy": args.compressor_name,
        "compressor": args.compressor_name,
        "n_cuts": NUM_TRANSFER_POINTS,
        "cuts": ["tp0", "tp1", "tp2"],
        "transfer_points": dict(TRANSFER_POINT_LABELS),
        "n_points": args.n_points,
        "seed": args.seed,
        "tol": tol,
        "eta_min_mode": "estimated_from_rays",
        "baseline": baseline,
        "max_samples": args.max_samples,
        "max_input_length": args.max_input_length,
        "prompt_template": args.prompt_template,
        "metric_note": (
            "ppl_score = mean_i min(1, exp(CE_ref_i - CE_eta_i)); "
            "CE is two-way SST-2 verbalizer cross entropy."
        ),
    }


def _save_rays_plot_with_score(
    rays: List[dict],
    path: Path,
    *,
    title: str,
    phase_label: str,
    max_lines: int = 40,
) -> None:
    if not rays:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(skip plot: {exc})", flush=True)
        return

    frac, n_used = concavity_of(rays)
    n_concave = sum(1 for ray in rays if ray.get("concave"))
    score_text = f"concavity score = {n_concave}/{n_used} = {frac:.3f}"

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ray in rays[:max_lines]:
        color = "tab:green" if ray["concave"] else "tab:red"
        ax.plot(ray["t"], ray["y"], marker=".", alpha=0.5, color=color, linewidth=0.8)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.4)
    ax.set_xlabel("ray parameter t")
    ax.set_ylabel("task metric along ray")
    ax.set_title(f"{title}\n{phase_label}")
    ax.text(
        0.02,
        0.98,
        score_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "0.7",
        },
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_metric_outputs(
    *,
    out_dir: Path,
    stem: str,
    summary: dict,
    rays: List[dict],
    phase2_rays: List[dict],
    eta_min_phase1: Optional[dict],
    eta_min_phase2: Optional[dict],
    final: bool,
) -> None:
    payload = {
        **summary,
        "rays": rays,
        "phase2_rays": phase2_rays,
        "fill_rays": [],
    }
    (out_dir / f"{stem}_rays.json").write_text(_json_dumps(payload), encoding="utf-8")

    summary_payload = dict(summary)
    if eta_min_phase1 is not None:
        summary_payload["eta_min_phase1"] = eta_min_phase1
    if eta_min_phase2 is not None:
        summary_payload["eta_min_phase2"] = eta_min_phase2
    (out_dir / f"{stem}_summary.json").write_text(
        _json_dumps(summary_payload),
        encoding="utf-8",
    )

    if final:
        title = (
            f"{summary['model']} SST-2/{summary['metric']} "
            f"{summary['strategy']} ({summary['n_cuts']} cuts)"
        )
        _save_rays_plot_with_score(
            rays,
            out_dir / f"{stem}_rays.png",
            title=title,
            phase_label="Phase 1",
        )
        _save_rays_plot_with_score(
            phase2_rays,
            out_dir / f"{stem}_phase2_rays.png",
            title=title,
            phase_label="Phase 2",
        )


def cmd_run(args) -> None:
    metrics = _parse_metrics(args.metrics)
    tolerances = _metric_tolerances(args, metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = {metric: _stem(args, metric) for metric in metrics}

    evaluator = FlanT5SST2ConcavityEvaluator(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        split=args.split,
        device=args.device,
        max_samples=args.max_samples,
        max_input_length=args.max_input_length,
        prompt_template=args.prompt_template,
        positive_token=args.positive_token,
        negative_token=args.negative_token,
        compressor_name=args.compressor_name,
        llmint8_outlier_precision=args.llmint8_outlier_precision,
        llmint8_regular_precision=args.llmint8_regular_precision,
    )
    for metric in metrics:
        print(
            "  baseline {} (eta=1) = {:.6f}".format(metric, evaluator.baseline[metric]),
            flush=True,
        )

    rng = np.random.default_rng(args.seed)
    eps = float(args.eps)
    lo_full = np.full(NUM_TRANSFER_POINTS, eps, dtype=np.float64)

    print(
        f"  -- Phase 1: {args.n_rays} rays over [{eps}, 1]^{NUM_TRANSFER_POINTS} --",
        flush=True,
    )
    rays_by_metric = run_mc_phase_multi_metric(
        evaluator,
        metrics=metrics,
        tolerances=tolerances,
        n_rays=args.n_rays,
        n_points=args.n_points,
        rng=rng,
        lo=lo_full,
        eps=eps,
        min_seg=args.min_seg,
        label="phase1",
        out_dir=out_dir,
        stems=stems,
    )

    summaries: Dict[str, dict] = {}
    eta_min_phase1: Dict[str, dict] = {}
    phase2_rays_by_metric: Dict[str, List[dict]] = {metric: [] for metric in metrics}
    eta_min_phase2: Dict[str, Optional[dict]] = {metric: None for metric in metrics}

    for metric in metrics:
        rays = rays_by_metric[metric]
        frac, n_used = concavity_of(rays)
        summaries[metric] = _base_summary(
            args,
            metric,
            tolerances[metric],
            evaluator.baseline[metric],
        )
        summaries[metric]["phase1_concavity"] = frac
        summaries[metric]["phase1_n_rays"] = n_used
        eta_min_phase1[metric] = find_eta_min(
            rays,
            NUM_TRANSFER_POINTS,
            tol=tolerances[metric],
            target=args.target,
            min_usable=args.min_usable,
        )
        print(
            "  Phase-1 {} concavity = {:.3f} (M={}) eta_min={}".format(
                metric,
                frac,
                n_used,
                np.round(eta_min_phase1[metric]["eta_min"], 3).tolist(),
            ),
            flush=True,
        )
        _write_metric_outputs(
            out_dir=out_dir,
            stem=stems[metric],
            summary=summaries[metric],
            rays=rays,
            phase2_rays=[],
            eta_min_phase1=eta_min_phase1[metric],
            eta_min_phase2=None,
            final=False,
        )

    if args.phase2_rays > 0:
        for metric in metrics:
            lo_cand = np.asarray(eta_min_phase1[metric]["eta_min"], dtype=np.float64)
            print(
                f"  -- Phase 2 [{metric}]: {args.phase2_rays} rays over "
                f"[eta_min_cand, 1]^{NUM_TRANSFER_POINTS} --",
                flush=True,
            )
            phase2 = run_mc_phase_multi_metric(
                evaluator,
                metrics=[metric],
                tolerances={metric: tolerances[metric]},
                n_rays=args.phase2_rays,
                n_points=args.n_points,
                rng=rng,
                lo=lo_cand,
                eps=eps,
                min_seg=args.min_seg,
                label="phase2",
                out_dir=out_dir,
                stems=stems,
            )[metric]
            phase2_rays_by_metric[metric] = phase2
            p2_frac, p2_n = concavity_of(phase2)
            summaries[metric]["phase2_concavity"] = p2_frac
            summaries[metric]["phase2_n_rays"] = p2_n
            eta_min_phase2[metric] = find_eta_min(
                rays_by_metric[metric] + phase2,
                NUM_TRANSFER_POINTS,
                tol=tolerances[metric],
                target=args.target,
                min_usable=args.min_usable,
            )
            print(
                "  Phase-2 {} concavity = {:.3f} (M={}) refined_eta_min={}".format(
                    metric,
                    p2_frac,
                    p2_n,
                    np.round(eta_min_phase2[metric]["eta_min"], 3).tolist(),
                ),
                flush=True,
            )

    for metric in metrics:
        _write_metric_outputs(
            out_dir=out_dir,
            stem=stems[metric],
            summary=summaries[metric],
            rays=rays_by_metric[metric],
            phase2_rays=phase2_rays_by_metric[metric],
            eta_min_phase1=eta_min_phase1[metric],
            eta_min_phase2=eta_min_phase2[metric],
            final=True,
        )
        print(
            f"Done. Wrote {stems[metric]}_rays.json / _summary.json / "
            f"_rays.png / _phase2_rays.png to {out_dir}",
            flush=True,
        )


def cmd_etamin(args) -> None:
    data = json.loads(Path(args.rays_json).read_text(encoding="utf-8"))
    rays = data["rays"] + data.get("phase2_rays", []) + data.get("fill_rays", [])
    metric = str(data.get("metric", "")).lower()
    tol = (
        float(args.tol)
        if args.tol is not None
        else float(data.get("tol", DEFAULT_TOL.get(metric, 0.001)))
    )
    n_cuts = int(data.get("n_cuts", NUM_TRANSFER_POINTS))
    em = find_eta_min(
        rays,
        n_cuts,
        tol=tol,
        target=args.target,
        min_usable=args.min_usable,
    )
    print(_json_dumps(em))
    if args.out:
        Path(args.out).write_text(_json_dumps(em), encoding="utf-8")
        print(f"Wrote {args.out}", flush=True)


def cmd_fill(args) -> None:
    """Standalone adaptive-fill pass on an existing Flan-T5 *_rays.json.

    Mirrors mc_concavity.cmd_fill: loads a rays file, checks per-floor usable-ray
    coverage up to floor_max, and (only if under-filled) loads the model and
    samples extra rays inside the high-floor sub-cubes. Resumes from a
    <stem>_fill.partial.json checkpoint and never clobbers the base rays.
    """
    rays_path = Path(args.rays_json)
    data = json.loads(rays_path.read_text(encoding="utf-8"))
    n_cuts = int(data.get("n_cuts", NUM_TRANSFER_POINTS))
    metric = str(data.get("metric", "")).lower()
    if metric not in METRICS:
        raise ValueError(
            f"rays_json metric '{metric}' not in supported {METRICS}"
        )
    tol = (
        float(args.tol)
        if args.tol is not None
        else float(data.get("tol", DEFAULT_TOL[metric]))
    )
    compressor_name = args.compressor_name or str(
        data.get("strategy") or data.get("compressor") or "topk"
    )

    out_dir = Path(args.out_dir) if args.out_dir else rays_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive the stem from the rays filename so the fill checkpoint and the
    # rewritten output line up exactly with the file we were handed.
    if rays_path.name.endswith("_rays.json"):
        stem = rays_path.name[: -len("_rays.json")]
    else:
        stem = rays_path.stem

    base_rays = data["rays"] + data.get("phase2_rays", []) + data.get("fill_rays", [])
    fill_cp = out_dir / f"{stem}_fill.partial.json"
    prior_fill: List[dict] = []
    if fill_cp.exists():
        prior_fill = json.loads(fill_cp.read_text(encoding="utf-8"))
        base_rays = base_rays + prior_fill
        print(f"  resuming: {len(prior_fill)} fill rays already on disk", flush=True)

    # De-duplicate before deciding coverage (see mc_concavity.cmd_fill): the
    # persisted fill_rays and the _fill.partial.json overlap, and counting dupes
    # inflates the usable-per-floor tally so a thin task looks satisfied. Key on
    # (start, end, y) — matches jensen_concavity._ray_hash.
    def _rkey(r: dict) -> str:
        return json.dumps([
            [round(float(x), 6) for x in r.get("start", [])],
            [round(float(x), 6) for x in r.get("end", [])],
            [round(float(x), 6) for x in r.get("y", [])],
        ])
    _seen: set = set()
    _deduped = []
    for r in base_rays:
        k = _rkey(r)
        if k in _seen:
            continue
        _seen.add(k)
        _deduped.append(r)
    if len(_deduped) != len(base_rays):
        print(
            f"  de-duplicated base rays: {len(base_rays)} -> {len(_deduped)} "
            f"unique (dropped {len(base_rays) - len(_deduped)})",
            flush=True,
        )
    base_rays = _deduped

    eps = float(args.eps)
    lo_full = np.full(n_cuts, eps, dtype=np.float64)
    floors = np.round(np.arange(0.0, args.floor_max + 1e-9, args.adaptive_floor_step), 3)
    under_filled = []
    for c in floors:
        _, usable = concavity_fraction(base_rays, np.full(n_cuts, float(c)), tol)
        if usable < args.min_box_samples:
            under_filled.append((float(c), usable))
    if not under_filled:
        print(
            f"  All floors already have >= {args.min_box_samples} usable rays "
            f"({len(base_rays)} base rays on disk). Skipping model load.",
            flush=True,
        )
        merged = dict(data)
        merged["fill_rays"] = data.get("fill_rays", []) + prior_fill
        (out_dir / f"{stem}_rays.json").write_text(_json_dumps(merged), encoding="utf-8")
        return
    print(f"  Under-filled floors: {under_filled}", flush=True)

    evaluator = FlanT5SST2ConcavityEvaluator(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        split=args.split,
        device=args.device,
        max_samples=args.max_samples,
        max_input_length=args.max_input_length,
        prompt_template=args.prompt_template,
        positive_token=args.positive_token,
        negative_token=args.negative_token,
        compressor_name=compressor_name,
        llmint8_outlier_precision=args.llmint8_outlier_precision,
        llmint8_regular_precision=args.llmint8_regular_precision,
    )
    print(
        f"  base rays={len(base_rays)}  metric={metric}  tol={tol}  "
        f"strategy={compressor_name}  target min_box_samples={args.min_box_samples}",
        flush=True,
    )

    def evaluate(eta_vec: np.ndarray) -> float:
        return float(evaluator.evaluate_all(np.asarray(eta_vec, dtype=np.float64))[metric])

    out_path = out_dir / f"{stem}_rays.json"

    def _persist(new_fills: List[dict]) -> None:
        # Fold this run's fills into the real _rays.json at every checkpoint, so a
        # timed-out link leaves fill_rays populated instead of 0. Mirrors the clean-
        # completion merge below exactly (data fill_rays + prior_fill + new_fill).
        all_fill = prior_fill + new_fills
        for i, r in enumerate(all_fill):
            r["id"] = i
        merged = dict(data)
        merged["fill_rays"] = data.get("fill_rays", []) + all_fill
        merged["n_cuts"] = n_cuts
        out_path.write_text(_json_dumps(merged), encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    new_fill = adaptive_fill_phase(
        evaluate,
        base_rays,
        n_cuts,
        rng=rng,
        min_box_samples=args.min_box_samples,
        floor_step=args.adaptive_floor_step,
        n_points=args.n_points,
        eps=eps,
        min_seg=args.min_seg,
        tol=tol,
        lo_full=lo_full,
        out_dir=out_dir,
        stem=stem,
        floor_max=args.floor_max,
        prior_fill=prior_fill,
        on_checkpoint=_persist,
    )

    all_fill = prior_fill + new_fill
    for i, r in enumerate(all_fill):
        r["id"] = i
    merged = dict(data)
    merged["fill_rays"] = data.get("fill_rays", []) + all_fill
    merged["n_cuts"] = n_cuts
    out_path = out_dir / f"{stem}_rays.json"
    out_path.write_text(_json_dumps(merged), encoding="utf-8")
    print(
        f"  Added {len(new_fill)} new fill rays this run "
        f"({len(all_fill)} total fill rays). Wrote {out_path}",
        flush=True,
    )


def _enum_key(vec) -> str:
    return json.dumps([round(float(x), 6) for x in vec])


def _enum_chord_concave(ts, ys, tol):
    """Eta-aware concavity of a straight-line ray: every triple i<j<k must sit
    on/above its chord. On a straight segment t is affine in eta, so the t-based
    chord test equals the correct non-uniform eta-space test (matches
    jensen_pass). Returns (all_pass, worst_violation)."""
    ts = [float(x) for x in ts]
    ys = [float(x) for x in ys]
    n = len(ts)
    ok, worst = True, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                span = ts[k] - ts[i]
                if span <= 0:
                    continue
                lam = (ts[k] - ts[j]) / span
                viol = (lam * ys[i] + (1.0 - lam) * ys[k]) - ys[j]
                worst = max(worst, viol)
                if viol > tol:
                    ok = False
    return ok, worst


def _enum_axis_rays(levels, grid_metric, n, tol):
    """Full set of axis-aligned straight-line rays through the level lattice
    (one per varying-axis x fixed-others assignment; len(levels) points each),
    in the mc_concavity *_rays.json schema."""
    import itertools

    lo, hi = min(levels), max(levels)
    span = (hi - lo) if hi > lo else 1.0
    ordered = sorted(levels)
    rays: List[dict] = []
    rid = 0
    for k in range(n):
        rest = [a for a in range(n) if a != k]
        for combo in itertools.product(ordered, repeat=len(rest)):
            fixed = dict(zip(rest, combo))
            etas, ts, ys = [], [], []
            for lvl in ordered:
                vec = [float(lvl) if a == k else float(fixed[a]) for a in range(n)]
                etas.append(vec)
                ts.append((float(lvl) - lo) / span)
                ys.append(float(grid_metric[_enum_key(vec)]))
            ok, worst = _enum_chord_concave(ts, ys, tol)
            rays.append({
                "id": rid, "axis": k,
                "start": list(etas[0]), "end": list(etas[-1]),
                "t": ts, "eta": etas, "y": ys,
                "concave": bool(ok), "worst_d2": float(worst),
                "violations": int(0 if ok else 1),
            })
            rid += 1
    return rays


def cmd_enumerate(args) -> None:
    """Exhaustive quantization enumeration for Flan-T5/SST-2 (cuts3).

    Evaluates every level^n lattice point exactly (both metrics share the model
    load) and writes axis-aligned rays that jensen folds in as the quantization
    curve. See experiments/enumerate_quant_concavity.py for the rationale."""
    import itertools
    import torch

    torch.manual_seed(args.seed)  # reproducible INT4 stochastic rounding
    metrics = _parse_metrics(args.metrics)
    tolerances = _metric_tolerances(args, metrics)
    levels = sorted(float(x) for x in args.levels.split(","))
    n = NUM_TRANSFER_POINTS
    args.n_points = len(levels)
    base = Path(args.out_dir)

    evaluator = FlanT5SST2ConcavityEvaluator(
        model_name=args.model_name, dataset_path=args.dataset_path,
        split=args.split, device=args.device, max_samples=args.max_samples,
        max_input_length=args.max_input_length,
        prompt_template=args.prompt_template,
        positive_token=args.positive_token, negative_token=args.negative_token,
        compressor_name=args.compressor_name,
        llmint8_outlier_precision=args.llmint8_outlier_precision,
        llmint8_regular_precision=args.llmint8_regular_precision,
    )
    for metric in metrics:
        print(f"  baseline {metric} (eta=1) = {evaluator.baseline[metric]:.6f}",
              flush=True)

    grids: Dict[str, Dict[str, float]] = {m: {} for m in metrics}
    all_points = list(itertools.product(levels, repeat=n))
    print(f"  enumerating {len(all_points)} lattice points "
          f"({len(levels)} levels ^ {n} cuts) ...", flush=True)
    for i, pt in enumerate(all_points):
        res = evaluator.evaluate_all(np.asarray(pt, dtype=np.float64))
        for m in metrics:
            grids[m][_enum_key(pt)] = float(res[m])
        print(f"  [{i+1}/{len(all_points)}] eta={list(pt)} -> "
              + " ".join(f"{m}={res[m]:.4f}" for m in metrics), flush=True)

    for metric in metrics:
        rays = _enum_axis_rays(levels, grids[metric], n, tolerances[metric])
        n_conc = sum(1 for r in rays if r["concave"])
        summary = _base_summary(args, metric, tolerances[metric],
                                evaluator.baseline[metric])
        summary.update({
            "levels": levels, "enumeration": True,
            "n_grid_points": len(all_points),
            "exact_concavity": (n_conc / len(rays)) if rays else None,
            "n_axis_rays": len(rays), "n_concave_axis_rays": n_conc,
        })
        out_dir = base / f"sst2_{metric}" / "quantization" / f"cuts{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _stem(args, metric)
        _write_metric_outputs(
            out_dir=out_dir, stem=stem, summary=summary, rays=rays,
            phase2_rays=[], eta_min_phase1=None, eta_min_phase2=None, final=False)
        ec = summary["exact_concavity"]
        print(f"Done [{metric}]. exact_concavity="
              f"{ec:.3f} ({n_conc}/{len(rays)}). Wrote {stem}_rays.json to {out_dir}",
              flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["run"]
    elif argv[0] not in ("run", "etamin", "fill", "enumerate", "-h", "--help") and argv[0].startswith("-"):
        argv = ["run"] + argv

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run Flan-T5/SST-2 MC concavity experiment")
    r.add_argument("--model_name", default=RUN_DEFAULT_MODEL_NAME)
    r.add_argument("--dataset_path", default=RUN_DEFAULT_DATASET_PATH)
    r.add_argument("--split", default=RUN_DEFAULT_SPLIT)
    r.add_argument("--device", default=RUN_DEFAULT_DEVICE)
    r.add_argument(
        "--max_samples",
        type=int,
        default=RUN_DEFAULT_MAX_SAMPLES,
        help="0 or negative uses the full split.",
    )
    r.add_argument("--max_input_length", type=int, default=RUN_DEFAULT_MAX_INPUT_LENGTH)
    r.add_argument("--prompt_template", default=RUN_DEFAULT_PROMPT_TEMPLATE)
    r.add_argument("--positive_token", default=RUN_DEFAULT_POSITIVE_TOKEN)
    r.add_argument("--negative_token", default=RUN_DEFAULT_NEGATIVE_TOKEN)
    r.add_argument(
        "--compressor_name",
        default=RUN_DEFAULT_COMPRESSOR_NAME,
        choices=["topk", "quantization", "llmint8", "identity", "none"],
    )
    r.add_argument(
        "--llmint8_outlier_precision",
        default=RUN_DEFAULT_LLMINT8_OUTLIER_PRECISION,
    )
    r.add_argument(
        "--llmint8_regular_precision",
        default=RUN_DEFAULT_LLMINT8_REGULAR_PRECISION,
    )
    r.add_argument(
        "--metrics",
        nargs="+",
        default=RUN_DEFAULT_METRICS,
        help="accuracy neg_ce ppl_score or all",
    )
    r.add_argument("--n_rays", type=int, default=RUN_DEFAULT_N_RAYS)
    r.add_argument("--phase2_rays", type=int, default=RUN_DEFAULT_PHASE2_RAYS)
    r.add_argument("--n_points", type=int, default=RUN_DEFAULT_N_POINTS)
    r.add_argument("--eps", type=float, default=RUN_DEFAULT_EPS)
    r.add_argument("--min_seg", type=float, default=RUN_DEFAULT_MIN_SEG)
    r.add_argument("--target", type=float, default=RUN_DEFAULT_TARGET)
    r.add_argument("--min_usable", type=int, default=RUN_DEFAULT_MIN_USABLE)
    r.add_argument("--seed", type=int, default=RUN_DEFAULT_SEED)
    r.add_argument("--tol_accuracy", type=float, default=RUN_DEFAULT_TOL_ACCURACY)
    r.add_argument("--tol_neg_ce", type=float, default=RUN_DEFAULT_TOL_NEG_CE)
    r.add_argument("--tol_ppl_score", type=float, default=RUN_DEFAULT_TOL_PPL_SCORE)
    r.add_argument("--out_dir", default=RUN_DEFAULT_OUT_DIR)
    r.set_defaults(func=cmd_run)

    en = sub.add_parser(
        "enumerate",
        help="exhaustive quantization concavity enumeration (all lattice points)")
    en.add_argument("--model_name", default=RUN_DEFAULT_MODEL_NAME)
    en.add_argument("--dataset_path", default=RUN_DEFAULT_DATASET_PATH)
    en.add_argument("--split", default=RUN_DEFAULT_SPLIT)
    en.add_argument("--device", default=RUN_DEFAULT_DEVICE)
    en.add_argument("--max_samples", type=int, default=RUN_DEFAULT_MAX_SAMPLES,
                    help="0 or negative uses the full split.")
    en.add_argument("--max_input_length", type=int, default=RUN_DEFAULT_MAX_INPUT_LENGTH)
    en.add_argument("--prompt_template", default=RUN_DEFAULT_PROMPT_TEMPLATE)
    en.add_argument("--positive_token", default=RUN_DEFAULT_POSITIVE_TOKEN)
    en.add_argument("--negative_token", default=RUN_DEFAULT_NEGATIVE_TOKEN)
    en.add_argument("--compressor_name", default="quantization",
                    choices=["quantization"])
    en.add_argument("--llmint8_outlier_precision",
                    default=RUN_DEFAULT_LLMINT8_OUTLIER_PRECISION)
    en.add_argument("--llmint8_regular_precision",
                    default=RUN_DEFAULT_LLMINT8_REGULAR_PRECISION)
    en.add_argument("--metrics", nargs="+", default=RUN_DEFAULT_METRICS,
                    help="accuracy neg_ce ppl_score or all")
    en.add_argument("--levels", default="0.0625,0.125,0.25,0.5,1.0",
                    help="Paper ladder relative to FP32: {2,4,8,16,32}-bit.")
    en.add_argument("--seed", type=int, default=RUN_DEFAULT_SEED)
    en.add_argument("--tol_accuracy", type=float, default=RUN_DEFAULT_TOL_ACCURACY)
    en.add_argument("--tol_neg_ce", type=float, default=RUN_DEFAULT_TOL_NEG_CE)
    en.add_argument("--tol_ppl_score", type=float, default=RUN_DEFAULT_TOL_PPL_SCORE)
    en.add_argument("--out_dir", default=RUN_DEFAULT_OUT_DIR)
    en.set_defaults(func=cmd_enumerate)

    e = sub.add_parser("etamin", help="model-free eta_min search on a *_rays.json")
    e.add_argument("--rays_json", required=True)
    e.add_argument("--tol", type=float, default=ETAMIN_DEFAULT_TOL)
    e.add_argument("--target", type=float, default=ETAMIN_DEFAULT_TARGET)
    e.add_argument("--min_usable", type=int, default=ETAMIN_DEFAULT_MIN_USABLE)
    e.add_argument("--out", default=ETAMIN_DEFAULT_OUT)
    e.set_defaults(func=cmd_etamin)

    f = sub.add_parser(
        "fill",
        help="adaptive-fill pass on an existing *_rays.json (per-floor coverage)",
    )
    f.add_argument("--rays_json", required=True)
    f.add_argument(
        "--out_dir",
        default=None,
        help="where to write fill checkpoint + rewritten rays "
        "(default: the rays_json's own directory)",
    )
    f.add_argument(
        "--tol",
        type=float,
        default=None,
        help="override the concavity tol (default: read from the rays_json)",
    )
    f.add_argument("--min_box_samples", type=int, default=100)
    f.add_argument("--adaptive_floor_step", type=float, default=0.1)
    f.add_argument("--floor_max", type=float, default=0.7)
    # Evaluator construction (metric + n_cuts come from the rays_json).
    f.add_argument("--model_name", default=RUN_DEFAULT_MODEL_NAME)
    f.add_argument("--dataset_path", default=RUN_DEFAULT_DATASET_PATH)
    f.add_argument("--split", default=RUN_DEFAULT_SPLIT)
    f.add_argument("--device", default=RUN_DEFAULT_DEVICE)
    f.add_argument("--max_samples", type=int, default=RUN_DEFAULT_MAX_SAMPLES)
    f.add_argument("--max_input_length", type=int, default=RUN_DEFAULT_MAX_INPUT_LENGTH)
    f.add_argument("--prompt_template", default=RUN_DEFAULT_PROMPT_TEMPLATE)
    f.add_argument("--positive_token", default=RUN_DEFAULT_POSITIVE_TOKEN)
    f.add_argument("--negative_token", default=RUN_DEFAULT_NEGATIVE_TOKEN)
    f.add_argument(
        "--compressor_name",
        default=None,
        help="override the strategy (default: read from the rays_json)",
    )
    f.add_argument(
        "--llmint8_outlier_precision", default=RUN_DEFAULT_LLMINT8_OUTLIER_PRECISION
    )
    f.add_argument(
        "--llmint8_regular_precision", default=RUN_DEFAULT_LLMINT8_REGULAR_PRECISION
    )
    f.add_argument("--n_points", type=int, default=RUN_DEFAULT_N_POINTS)
    f.add_argument("--eps", type=float, default=RUN_DEFAULT_EPS)
    f.add_argument("--min_seg", type=float, default=RUN_DEFAULT_MIN_SEG)
    f.add_argument("--seed", type=int, default=RUN_DEFAULT_SEED)
    f.set_defaults(func=cmd_fill)

    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
