"""
Monte-Carlo concavity estimation in eta-space + per-cut-point eta_min search.

This REFRAMES the fixed-ray test in ``experiments/concavity/ray_concavity_score.py``.
Instead of a handful of hand-designed rays we treat concavity as a Monte-Carlo
property of the whole eta-hypercube [0,1]^n (n = number of activation
cut-points):

  1. Sample M random rays (segments) in the hypercube.
  2. Evaluate the task metric at K equally spaced points along each ray.
  3. Classify each ray concave / non-concave (discrete second-difference test).
  4. Report  concavity = (# concave rays) / M.

We also SAVE every ray's full profile (sampled eta-vectors + metric values) so
that a model-free post-hoc step can search for the per-cut-point lower-bound
vector ``eta_min`` below which concavity breaks: the smallest sub-cube
[eta_min, 1]^n in which the concavity fraction stays >= target (default 0.95).
The theory: very low eta enters a "weird-behaviour" regime that destroys
concavity, so excluding it should restore concavity.

Metric per task
---------------
* perplexity tasks (ShareGPT, WikiText):  mean per-sample score
      min{1, PPL_ref / PPL_comp}            (reuses ray_concavity_score helpers)
* accuracy tasks (MMLU):                   ABSOLUTE accuracy(eta) in [0, 1]
      (raw, not normalised by baseline)

Subcommands
-----------
  run     : load a model, run the Monte-Carlo phase(s), save profiles + eta_min.
  etamin  : model-free; re-run the eta_min search on an existing *_rays.json.

Example (smoke):
  python experiments/concavity/mc_concavity.py run \
      --model meta-llama/Llama-3.1-8B --dataset wikitext --metric perplexity \
      --strategy topk_per_token --n_cuts 4 --n_rays 20 --n_points 7 \
      --max_texts 16 --out_dir outputs/mc_concavity/smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Allow running as a top-level script and importing the repo modules / the
# sibling ray_concavity_score helpers (which we reuse verbatim).
HERE = Path(__file__).resolve().parent      # experiments/concavity
REPO_ROOT = HERE.parents[1]                 # repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "core"))
sys.path.insert(0, str(HERE))

from src.core.llm_compression import (  # noqa: E402
    ActivationCompressionConfig,
    LLMActivationCompressor,
    detect_llm_decoder_layers,
)
from ray_concavity_score import (  # noqa: E402
    load_wikitext_texts,
    make_cut_points,
    per_sample_mean_nll,
    sample_scores,
)


# ──────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────

# ShareGPT mirrors on the Hub. We try parquet first (robust in the pinned
# `easy` env where datasets.load_dataset is broken — see the WikiText loader),
# then fall back to JSON, then to datasets.load_dataset.
# ShareGPT loading lives in src/core alongside the WikiText loader, so the
# task instances and these runners share one implementation.
from src.core.llm_text_eval import load_sharegpt_texts  # noqa: E402,F401


# ──────────────────────────────────────────────────────────────────────────
# Concavity test (list variant of ray_concavity_score.second_difference_test)
# ──────────────────────────────────────────────────────────────────────────

def second_difference_test_on(ys: Sequence[float], tol: float = 1e-3) -> dict:
    """Concavity on a uniform grid: D2[j] = y[j-1] + y[j+1] - 2 y[j] <= tol.

    Returns dict with violations / worst_d2 / n_interior. A positive D2 at an
    interior point means the function is locally convex (non-concave) there.
    """
    ys = list(ys)

    # A non-finite utility means the evaluation failed, not that the curve is
    # concave. Guard explicitly: `v > tol` is False for NaN, so without this a
    # broken ray silently scores concave (this is what inflated the T5
    # ppl_score quantization result before the FP16 clipping fix).
    if not all(math.isfinite(float(v)) for v in ys):
        return {
            "n_interior": max(0, len(ys) - 2),
            "violations": max(0, len(ys) - 2),
            "worst_d2": float("inf"),
            "tol": tol,
            "non_finite": True,
        }

    d2: List[float] = []
    violations = 0
    worst = 0.0
    for j in range(1, len(ys) - 1):
        v = ys[j - 1] + ys[j + 1] - 2.0 * ys[j]
        d2.append(v)
        if v > worst:
            worst = v
        if v > tol:
            violations += 1
    return {
        "n_interior": max(0, len(ys) - 2),
        "violations": violations,
        "worst_d2": float(worst),
        "tol": tol,
    }


# ──────────────────────────────────────────────────────────────────────────
# Metric-agnostic evaluators  (evaluate(eta_vec) -> float)
# ──────────────────────────────────────────────────────────────────────────

class PerplexityEvaluator:
    """ShareGPT / WikiText: mean per-sample min{1, PPL_ref/PPL_comp}."""

    def __init__(self, *, model, tokenizer, compressor, texts, device,
                 max_length: int, batch_size: int, n_cuts: int):
        self.model = model
        self.tokenizer = tokenizer
        self.compressor = compressor
        self.texts = texts
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_cuts = n_cuts
        # Reference (eta = 1, uncompressed) captured once.
        self.compressor.set_eta(torch.ones(n_cuts, dtype=torch.float32))
        self.hbar_ref = per_sample_mean_nll(
            model, tokenizer, texts, device=device,
            max_length=max_length, batch_size=batch_size,
        )
        self.baseline = float(np.mean(sample_scores(self.hbar_ref, self.hbar_ref)))

    def evaluate(self, eta_vec: np.ndarray) -> float:
        self.compressor.set_eta(torch.tensor(eta_vec, dtype=torch.float32))
        hbar_comp = per_sample_mean_nll(
            self.model, self.tokenizer, self.texts, device=self.device,
            max_length=self.max_length, batch_size=self.batch_size,
        )
        return float(np.mean(sample_scores(self.hbar_ref, hbar_comp)))


class AccuracyEvaluator:
    """MMLU: ABSOLUTE accuracy(eta) in [0, 1] (no baseline normalisation)."""

    def __init__(self, *, model, compressor, mmlu_evaluator, device, n_cuts: int):
        self.model = model
        self.compressor = compressor
        self.ev = mmlu_evaluator
        self.device = device
        self.n_cuts = n_cuts
        self.compressor.set_eta(torch.ones(n_cuts, dtype=torch.float32))
        self.baseline = float(self.ev.accuracy(model, device=device))  # acc_ref

    def evaluate(self, eta_vec: np.ndarray) -> float:
        self.compressor.set_eta(torch.tensor(eta_vec, dtype=torch.float32))
        return float(self.ev.accuracy(self.model, device=self.device))


# ──────────────────────────────────────────────────────────────────────────
# Random ray sampling + Monte-Carlo phase
# ──────────────────────────────────────────────────────────────────────────

def sample_random_ray(
    rng: np.random.Generator, n: int, *,
    lo: np.ndarray, hi: float = 1.0, min_seg: float = 0.05, max_tries: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two independent uniform endpoints in the box [lo, hi]^n.

    `lo` is a per-coordinate lower bound (array of length n). The monotonicity
    constraint of the old fixed-ray code (end >= start) is intentionally
    dropped so the segments are isotropic. Degenerate (near-constant) segments
    are rejected and resampled.
    """
    lo = np.asarray(lo, dtype=np.float64)
    for _ in range(max_tries):
        start = lo + (hi - lo) * rng.random(n)
        end = lo + (hi - lo) * rng.random(n)
        if np.linalg.norm(end - start) >= min_seg:
            return start, end
    # Give up on the rejection and return whatever we have.
    return start, end


def ray_eta_points(start: np.ndarray, end: np.ndarray, n_points: int):
    """Equally spaced eta-vectors along the segment, t in [0, 1]."""
    ts = np.linspace(0.0, 1.0, n_points)
    etas = [(1.0 - t) * start + t * end for t in ts]
    return ts, etas


def run_mc_phase(
    evaluate: Callable[[np.ndarray], float], *,
    n: int, n_rays: int, n_points: int, rng: np.random.Generator,
    lo: np.ndarray, eps: float, min_seg: float, tol: float,
    label: str = "phase1",
    checkpoint_path: Optional[Path] = None, checkpoint_every: int = 5,
) -> List[dict]:
    """Sample `n_rays` random rays, evaluate, classify, return ray dicts.

    If `checkpoint_path` is given, the partial ray list is dumped there every
    `checkpoint_every` rays (and at the end) so a walltime kill doesn't
    discard completed work.
    """
    rays: List[dict] = []
    for i in range(n_rays):
        start, end = sample_random_ray(rng, n, lo=lo, min_seg=min_seg)
        start = np.clip(start, eps, 1.0)
        end = np.clip(end, eps, 1.0)
        ts, etas = ray_eta_points(start, end, n_points)
        ys: List[float] = []
        t0 = time.time()
        for eta_vec in etas:
            ys.append(evaluate(eta_vec))
        sd = second_difference_test_on(ys, tol)
        rays.append({
            "id": i,
            "start": [float(x) for x in start],
            "end": [float(x) for x in end],
            "t": [float(x) for x in ts],
            "eta": [[float(x) for x in e] for e in etas],
            "y": [float(x) for x in ys],
            "concave": bool(sd["violations"] == 0),
            "worst_d2": sd["worst_d2"],
            "violations": sd["violations"],
        })
        print(f"  [{label}] ray {i+1}/{n_rays}  "
              f"concave={rays[-1]['concave']}  worst_d2={sd['worst_d2']:+.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if checkpoint_path is not None and (
                (i + 1) % checkpoint_every == 0 or (i + 1) == n_rays):
            Path(checkpoint_path).write_text(json.dumps(rays))
    return rays


def concavity_of(rays: List[dict]) -> Tuple[float, int]:
    """Global concavity fraction over a list of rays (each fully evaluated)."""
    if not rays:
        return float("nan"), 0
    n_conc = sum(1 for r in rays if r["concave"])
    return n_conc / len(rays), len(rays)


def adaptive_fill_phase(
    evaluate: Callable[[np.ndarray], float], rays: List[dict], n: int, *,
    rng: np.random.Generator, min_box_samples: int, floor_step: float,
    n_points: int, eps: float, min_seg: float, tol: float,
    lo_full: np.ndarray, out_dir: Path, stem: str, checkpoint_every: int = 5,
    floor_max: float = 0.9,
) -> List[dict]:
    """Sample extra rays to ensure each floor level has >= min_box_samples usable rays.

    Scans floor levels from low to high. Filling the lowest under-sampled floor c
    first draws rays from [c, 1]^n whose grid points spread across the shells above
    c (not just the top corner), so the intermediate 0.1 bands get populated before
    each higher floor is topped up with only its residual deficit. (The old high->low
    order filled the top corner once and let every lower floor ride on it, which left
    the 0.6-0.9 shells empty behind a corner spike.)
    """
    floors = np.round(np.arange(0.0, floor_max + 1e-9, floor_step), 3)
    fill_rays: List[dict] = []
    all_rays = list(rays)

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
                ys.append(evaluate(eta_vec))
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
            print(f"  [fill c={c:.2f}] ray {batch_idx}  "
                  f"concave={r['concave']}  worst_d2={sd['worst_d2']:+.4f}  "
                  f"({time.time()-t0:.1f}s)  usable={usable}/{min_box_samples}",
                  flush=True)
            if batch_idx % checkpoint_every == 0:
                cp = out_dir / f"{stem}_fill.partial.json"
                cp.write_text(json.dumps(fill_rays))

    if fill_rays:
        cp = out_dir / f"{stem}_fill.partial.json"
        cp.write_text(json.dumps(fill_rays))
    return fill_rays


# ──────────────────────────────────────────────────────────────────────────
# Per-cut-point eta_min search  (model-free; operates on saved ray profiles)
# ──────────────────────────────────────────────────────────────────────────

def clip_indices(start, end, ts, v, atol: float = 1e-9) -> List[int]:
    """Indices of saved grid points whose eta-vector lies in the sub-cube [v,1]^n.

    Since each ray is a single line segment, {t : eta(t) >= v (all coords)} is a
    convex sub-interval of [0,1]; the in-range saved grid points are therefore a
    contiguous block, preserving uniform spacing for the D2 test.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)

    t_lo, t_hi = 0.0, 1.0
    for k in range(len(v)):
        delta = end[k] - start[k]
        rhs = v[k] - start[k]            # need t*delta >= rhs
        if abs(delta) <= atol:
            if start[k] < v[k] - atol:   # constant coord below v -> empty
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
    rays: List[dict], v: np.ndarray, tol: float, min_pts: int = 3,
) -> Tuple[Optional[float], int]:
    """Fraction of clipped ray-segments in [v,1]^n that are concave.

    Returns (frac, usable). `usable` counts rays with >= min_pts contiguous
    saved points inside the sub-cube; frac is None when usable == 0.
    """
    usable = 0
    concave = 0
    for r in rays:
        idx = clip_indices(r["start"], r["end"], r["t"], v)
        if len(idx) < min_pts:
            continue
        usable += 1
        ys = [r["y"][j] for j in idx]
        if second_difference_test_on(ys, tol)["violations"] == 0:
            concave += 1
    frac = (concave / usable) if usable > 0 else None
    return frac, usable


def _meets(frac: Optional[float], usable: int, target: float, min_usable: int) -> bool:
    return frac is not None and usable >= min_usable and frac >= target


def find_eta_min(
    rays: List[dict], n: int, *, tol: float, target: float = 0.95,
    min_usable: int = 30, coarse_step: float = 0.1, fine_step: float = 0.02,
    v_max: float = 0.95,
) -> dict:
    """Per-cut-point eta_min: the smallest sub-cube corner [v,1]^n with
    concavity-fraction >= target.

    Searched in two passes. Concavity rises as we raise v (we exclude the
    low-eta "weird-behaviour" regime) but the usable ray count falls, so we
    must ASCEND from v=0 (data-rich) rather than descend from v=1 (data-starved):

      (A) Greedy ASCENT (coarse step): from v=0, repeatedly raise the coordinate
          whose increase most improves the concavity fraction, until the target
          is met (or no coordinate helps / data runs out).
      (B) Fine DESCENT (fine step): peel back any coarse overshoot — repeatedly
          lower a coordinate while the target still holds — to land on a
          Pareto-minimal v at the fine resolution.
    """
    history: List[dict] = []

    def frac_usable(v):
        return concavity_fraction(rays, v, tol)

    # ── (A) coarse greedy ascent from 0 ───────────────────────────────────
    v = np.zeros(n, dtype=np.float64)
    for _ in range(n * int(round(v_max / coarse_step)) + 2):
        frac, usable = frac_usable(v)
        if _meets(frac, usable, target, min_usable):
            break
        best = None  # ((frac_or_-1, usable), cand)
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
            break  # nowhere left with enough data
        v = best[1]
        f, u = frac_usable(v)
        history.append({"pass": "ascent", "v": v.tolist(), "frac": f, "usable": u})

    # ── (B) fine descent to minimise v while target holds ─────────────────
    improved = True
    while improved:
        improved = False
        best = None  # prefer the move that lowers a coordinate the most
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

    # Interpretable companion: concavity-fraction vs a UNIFORM scalar floor
    # v = c * 1 (the diagonal of the eta_min search), c in [0, v_max].
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


# ──────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────

def save_rays_plot(rays: List[dict], path: Path, title: str, max_lines: int = 40):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skip plot: {e})")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for r in rays[:max_lines]:
        color = "tab:green" if r["concave"] else "tab:red"
        ax.plot(r["t"], r["y"], marker=".", alpha=0.5, color=color, linewidth=0.8)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.4)
    ax.set_xlabel("ray parameter t")
    ax.set_ylabel("task metric along ray")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# `run` subcommand: load model, run MC phase(s), save profiles + eta_min
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_MMLU_SUBJECTS = (
    "college_computer_science", "high_school_mathematics", "professional_law",
    "global_facts", "miscellaneous", "business_ethics",
)


def build_evaluator(args, *, model, tokenizer, compressor, device, n_cuts):
    if args.metric == "perplexity":
        if args.dataset == "wikitext":
            texts = load_wikitext_texts(args.max_texts)
        elif args.dataset == "sharegpt":
            texts = load_sharegpt_texts(args.max_texts)
        else:
            raise ValueError(f"Unknown perplexity dataset: {args.dataset}")
        print(f"  loaded {len(texts)} {args.dataset} texts", flush=True)
        return PerplexityEvaluator(
            model=model, tokenizer=tokenizer, compressor=compressor, texts=texts,
            device=device, max_length=args.max_length, batch_size=args.batch_size,
            n_cuts=n_cuts,
        )
    if args.metric == "accuracy":
        from mmlu_eval import MMLUEvaluator, MMLUEvalConfig
        subjects = (tuple(args.mmlu_subjects.split(","))
                    if args.mmlu_subjects else DEFAULT_MMLU_SUBJECTS)
        mmlu = MMLUEvaluator(tokenizer=tokenizer, config=MMLUEvalConfig(
            subjects=subjects, samples_per_subject=args.samples_per_subject,
            seed=args.seed, max_length=args.max_length, batch_size=args.batch_size,
            n_shot=args.n_shot,
        ))
        return AccuracyEvaluator(model=model, compressor=compressor,
                                 mmlu_evaluator=mmlu, device=device, n_cuts=n_cuts)
    raise ValueError(f"Unknown metric: {args.metric}")


def load_model_and_compressor(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model dtype. Default (`auto`) keeps the historical behaviour (FP16 on GPU)
    # so the topk / llmint8 arms are unchanged; the quantization arm passes
    # `fp32` because its paper spec uses an FP32 native reference (eta=1 is a
    # true uncompressed operating point).
    dtype_arg = getattr(args, "dtype", "auto")
    if dtype_arg == "fp32":
        torch_dtype = torch.float32
    elif dtype_arg == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"=== {args.model} | {args.dataset}/{args.metric} | "
          f"{args.strategy} | n_cuts={args.n_cuts} ===", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    layers = detect_llm_decoder_layers(model)
    cuts = make_cut_points(len(layers), args.n_cuts)
    print(f"  layers={len(layers)}  cuts={cuts}", flush=True)

    compressor = LLMActivationCompressor(
        model, ActivationCompressionConfig(
            strategy=args.strategy, layer_indices=cuts, random_seed=args.seed),
    )
    return device, tokenizer, model, compressor, cuts


def cmd_run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device, tokenizer, model, compressor, cuts = load_model_and_compressor(args)
    rng = np.random.default_rng(args.seed)
    summary: dict = {
        "model": args.model, "dataset": args.dataset, "metric": args.metric,
        "strategy": args.strategy, "n_cuts": args.n_cuts, "cuts": cuts,
        "n_points": args.n_points, "seed": args.seed, "tol": args.tol,
    }

    slug = args.model.replace("/", "__")
    stem = f"{slug}__{args.dataset}__{args.metric}__{args.strategy}__cuts{args.n_cuts}"

    def _write(rays, phase2_rays, em, phase2_em, final: bool, fill_rays=None):
        """Persist current results. Called after Phase 1 and again at the end so
        a walltime kill during Phase 2 still leaves Phase-1 results + eta_min."""
        (out_dir / f"{stem}_rays.json").write_text(json.dumps({
            **summary, "rays": rays, "phase2_rays": phase2_rays,
            "fill_rays": fill_rays or []}, indent=2))
        s = dict(summary)
        if em is not None:
            s["eta_min_phase1"] = em
        if phase2_em is not None:
            s["eta_min_phase2"] = phase2_em
        (out_dir / f"{stem}_summary.json").write_text(json.dumps(s, indent=2))
        if final:
            save_rays_plot(rays, out_dir / f"{stem}_rays.png",
                           title=f"{args.model} {args.dataset}/{args.strategy} "
                                 f"({args.n_cuts} cuts)")

    rays: List[dict] = []
    phase2_rays: List[dict] = []
    fill_rays: List[dict] = []
    em = None
    phase2_em = None
    try:
        evaluator = build_evaluator(
            args, model=model, tokenizer=tokenizer, compressor=compressor,
            device=device, n_cuts=args.n_cuts)
        summary["baseline"] = evaluator.baseline
        print(f"  baseline (eta=1) = {evaluator.baseline:.4f}", flush=True)

        # ── Phase 1: isotropic MC over the whole hypercube ──────────────────
        eps = args.eps
        lo_full = np.full(args.n_cuts, eps)
        print(f"  -- Phase 1: {args.n_rays} rays over [{eps}, 1]^{args.n_cuts} --",
              flush=True)
        rays = run_mc_phase(
            evaluator.evaluate, n=args.n_cuts, n_rays=args.n_rays,
            n_points=args.n_points, rng=rng, lo=lo_full, eps=eps,
            min_seg=args.min_seg, tol=args.tol, label="phase1",
            checkpoint_path=out_dir / f"{stem}_phase1.partial.json")
        frac, M = concavity_of(rays)
        summary["phase1_concavity"] = frac
        summary["phase1_n_rays"] = M
        print(f"  Phase-1 global concavity = {frac:.3f}  (M={M})", flush=True)

        # ── eta_min search on Phase-1 rays ──────────────────────────────────
        em = find_eta_min(rays, args.n_cuts, tol=args.tol, target=args.target,
                          min_usable=args.min_usable)
        print(f"  Phase-1 eta_min = {np.round(em['eta_min'], 3).tolist()}  "
              f"(frac={em['frac_at_eta_min']}, usable={em['usable_at_eta_min']})",
              flush=True)
        # Persist Phase-1 results immediately (survives a Phase-2 timeout).
        _write(rays, phase2_rays, em, None, final=False, fill_rays=fill_rays)

        # ── Phase 2: targeted resampling inside [eta_min_candidate, 1]^n ─────
        if args.phase2_rays > 0:
            lo_cand = np.asarray(em["eta_min"], dtype=np.float64)
            print(f"  -- Phase 2: {args.phase2_rays} rays over "
                  f"[eta_min_cand, 1]^{args.n_cuts} --", flush=True)
            phase2_rays = run_mc_phase(
                evaluator.evaluate, n=args.n_cuts, n_rays=args.phase2_rays,
                n_points=args.n_points, rng=rng, lo=lo_cand, eps=eps,
                min_seg=args.min_seg, tol=args.tol, label="phase2",
                checkpoint_path=out_dir / f"{stem}_phase2.partial.json")
            p2_frac, p2_M = concavity_of(phase2_rays)
            summary["phase2_concavity"] = p2_frac
            summary["phase2_n_rays"] = p2_M
            print(f"  Phase-2 concavity in candidate sub-cube = {p2_frac:.3f} "
                  f"(M={p2_M})", flush=True)
            # Refine eta_min using the combined ray set (clean, no clipping loss).
            phase2_em = find_eta_min(rays + phase2_rays, args.n_cuts, tol=args.tol,
                                     target=args.target, min_usable=args.min_usable)
            print(f"  Phase-2 refined eta_min = "
                  f"{np.round(phase2_em['eta_min'], 3).tolist()}  "
                  f"(frac={phase2_em['frac_at_eta_min']}, "
                  f"usable={phase2_em['usable_at_eta_min']})", flush=True)

        # ── Adaptive fill: ensure min_box_samples usable rays per floor level ─
        if args.min_box_samples > 0:
            print(f"  -- Adaptive fill: min_box_samples={args.min_box_samples} "
                  f"floor_step={args.adaptive_floor_step} --", flush=True)
            fill_rays = adaptive_fill_phase(
                evaluator.evaluate, rays + phase2_rays, args.n_cuts,
                rng=rng, min_box_samples=args.min_box_samples,
                floor_step=args.adaptive_floor_step, n_points=args.n_points,
                eps=eps, min_seg=args.min_seg, tol=args.tol, lo_full=lo_full,
                out_dir=out_dir, stem=stem, floor_max=args.floor_max)
            summary["fill_n_rays"] = len(fill_rays)
            print(f"  Adaptive fill added {len(fill_rays)} rays.", flush=True)
    finally:
        compressor.remove_hooks()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write(rays, phase2_rays, em, phase2_em, final=True, fill_rays=fill_rays)
    print(f"Done. Wrote {stem}_rays.json / _summary.json / _rays.png to {out_dir}")


# ──────────────────────────────────────────────────────────────────────────
# `fill` subcommand: standalone adaptive-fill pass on an existing *_rays.json.
# Loads a model, tops up under-sampled floor levels only (skips floors that
# already have >= min_box_samples usable rays, so it's safe to resubmit this
# same job repeatedly / chain it across multiple walltime windows).
# ──────────────────────────────────────────────────────────────────────────

def cmd_fill(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rays_path = Path(args.rays_json)
    data = json.loads(rays_path.read_text())
    args.n_cuts = int(data["n_cuts"])
    base_rays = data["rays"] + data.get("phase2_rays", []) + data.get("fill_rays", [])

    slug = args.model.replace("/", "__")
    stem = f"{slug}__{args.dataset}__{args.metric}__{args.strategy}__cuts{args.n_cuts}"
    fill_cp = out_dir / f"{stem}_fill.partial.json"
    prior_fill: List[dict] = []
    if fill_cp.exists():
        prior_fill = json.loads(fill_cp.read_text())
        base_rays = base_rays + prior_fill
        print(f"  resuming: {len(prior_fill)} fill rays already on disk", flush=True)

    # De-duplicate before deciding coverage: the persisted fill_rays and the
    # _fill.partial.json overlap (both hold the resumed fill pass), and a ray
    # can be re-stored identically across merges. Counting duplicates inflates
    # the usable-per-floor tally and makes a genuinely under-filled task look
    # satisfied, so the fill would no-op. Key on (start, end, y) — matches
    # jensen_concavity._ray_hash; exact dupes also share the eta path.
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
        print(f"  de-duplicated base rays: {len(base_rays)} -> {len(_deduped)} "
              f"unique (dropped {len(base_rays) - len(_deduped)})", flush=True)
    base_rays = _deduped

    eps = args.eps
    lo_full = np.full(args.n_cuts, eps)
    floors = np.round(np.arange(0.0, args.floor_max + 1e-9, args.adaptive_floor_step), 3)
    under_filled = []
    for c in floors:
        _, usable = concavity_fraction(base_rays, np.full(args.n_cuts, float(c)), args.tol)
        if usable < args.min_box_samples:
            under_filled.append((float(c), usable))
    if not under_filled:
        print(f"  All floors already have >= {args.min_box_samples} usable rays "
              f"({len(base_rays)} base rays on disk). Skipping model load.", flush=True)
        merged = dict(data)
        merged["fill_rays"] = data.get("fill_rays", []) + prior_fill
        (out_dir / f"{stem}_rays.json").write_text(json.dumps(merged, indent=2))
        return
    print(f"  Under-filled floors: {under_filled}", flush=True)

    device, tokenizer, model, compressor, cuts = load_model_and_compressor(args)
    rng = np.random.default_rng(args.seed)
    try:
        evaluator = build_evaluator(
            args, model=model, tokenizer=tokenizer, compressor=compressor,
            device=device, n_cuts=args.n_cuts)
        print(f"  base rays={len(base_rays)}  target min_box_samples="
              f"{args.min_box_samples}  floor_step={args.adaptive_floor_step}",
              flush=True)
        new_fill = adaptive_fill_phase(
            evaluator.evaluate, base_rays, args.n_cuts,
            rng=rng, min_box_samples=args.min_box_samples,
            floor_step=args.adaptive_floor_step, n_points=args.n_points,
            eps=eps, min_seg=args.min_seg, tol=args.tol, lo_full=lo_full,
            out_dir=out_dir, stem=stem, floor_max=args.floor_max)
    finally:
        compressor.remove_hooks()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_fill = prior_fill + new_fill
    for i, r in enumerate(all_fill):
        r["id"] = i
    merged = dict(data)
    merged["fill_rays"] = data.get("fill_rays", []) + all_fill
    merged["n_cuts"] = args.n_cuts
    out_path = out_dir / f"{stem}_rays.json"
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"  Added {len(new_fill)} new fill rays this run "
          f"({len(all_fill)} total fill rays). Wrote {out_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────────
# `etamin` subcommand: model-free re-search on an existing *_rays.json
# ──────────────────────────────────────────────────────────────────────────

def cmd_etamin(args) -> None:
    data = json.loads(Path(args.rays_json).read_text())
    rays = data["rays"] + data.get("phase2_rays", []) + data.get("fill_rays", [])
    n = int(data["n_cuts"])
    em = find_eta_min(rays, n, tol=args.tol, target=args.target,
                      min_usable=args.min_usable)
    print(json.dumps(em, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(em, indent=2))
        print(f"Wrote {args.out}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="load model, run MC phase(s), save profiles")
    r.add_argument("--model", required=True)
    r.add_argument("--dataset", required=True, choices=["wikitext", "sharegpt", "mmlu"])
    r.add_argument("--metric", required=True, choices=["perplexity", "accuracy"])
    r.add_argument("--strategy", default="topk_per_token",
                   choices=["topk_per_token", "llmint8", "magnitude", "random",
                            "quantization"])
    r.add_argument("--n_cuts", type=int, default=4)
    r.add_argument("--n_rays", type=int, default=150, help="Phase-1 random rays.")
    r.add_argument("--n_points", type=int, default=7, help="Points per ray (uniform t).")
    r.add_argument("--phase2_rays", type=int, default=80,
                   help="Phase-2 targeted rays inside the candidate sub-cube (0 to skip).")
    r.add_argument("--eps", type=float, default=1e-3, help="Lower clamp to avoid eta<=0.")
    r.add_argument("--min_seg", type=float, default=0.05, help="Reject shorter segments.")
    r.add_argument("--tol", type=float, default=1e-3,
                   help="D2 concavity tolerance (use larger for noisy accuracy).")
    r.add_argument("--target", type=float, default=0.95)
    r.add_argument("--min_usable", type=int, default=30)
    # perplexity opts
    r.add_argument("--max_texts", type=int, default=128)
    r.add_argument("--max_length", type=int, default=512)
    r.add_argument("--batch_size", type=int, default=1)
    # mmlu opts
    r.add_argument("--mmlu_subjects", default="",
                   help="Comma-separated MMLU subjects (default: 6-subject mix).")
    r.add_argument("--samples_per_subject", type=int, default=20)
    r.add_argument("--n_shot", type=int, default=5)
    r.add_argument("--floor_max", type=float, default=0.9,
                    help="Highest eta-floor level the adaptive fill targets.")
    r.add_argument("--min_box_samples", type=int, default=100,
                   help="Min usable rays per floor level; triggers adaptive fill (0 to disable). "
                        "'Usable' = whole rays whose clipped segment inside [c,1]^n has >= 3 grid points.")
    r.add_argument("--adaptive_floor_step", type=float, default=0.1,
                   help="Floor-level grid spacing for adaptive fill check.")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "mc_concavity"))
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("fill", help="standalone adaptive-fill pass on an existing *_rays.json")
    f.add_argument("--rays_json", required=True,
                   help="Base *_rays.json to top up (also the output path).")
    f.add_argument("--model", required=True)
    f.add_argument("--dataset", required=True, choices=["wikitext", "sharegpt", "mmlu"])
    f.add_argument("--metric", required=True, choices=["perplexity", "accuracy"])
    f.add_argument("--strategy", default="topk_per_token",
                   choices=["topk_per_token", "llmint8", "magnitude", "random",
                            "quantization"])
    f.add_argument("--n_points", type=int, default=7)
    f.add_argument("--eps", type=float, default=1e-3)
    f.add_argument("--min_seg", type=float, default=0.05)
    f.add_argument("--tol", type=float, default=1e-3)
    f.add_argument("--max_texts", type=int, default=128)
    f.add_argument("--max_length", type=int, default=512)
    f.add_argument("--batch_size", type=int, default=1)
    f.add_argument("--mmlu_subjects", default="")
    f.add_argument("--samples_per_subject", type=int, default=20)
    f.add_argument("--n_shot", type=int, default=5)
    f.add_argument("--floor_max", type=float, default=0.9,
                    help="Highest eta-floor level the adaptive fill targets.")
    f.add_argument("--min_box_samples", type=int, default=100)
    f.add_argument("--adaptive_floor_step", type=float, default=0.1)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out_dir", required=True)
    f.set_defaults(func=cmd_fill)

    e = sub.add_parser("etamin", help="model-free eta_min search on a *_rays.json")
    e.add_argument("--rays_json", required=True)
    e.add_argument("--tol", type=float, default=1e-3)
    e.add_argument("--target", type=float, default=0.95)
    e.add_argument("--min_usable", type=int, default=30)
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_etamin)

    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
