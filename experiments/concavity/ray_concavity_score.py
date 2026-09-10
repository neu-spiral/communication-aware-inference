"""
Ray-based concavity test for a per-sample perplexity-ratio score across LLMs.

We install activation-compression hooks at N cut-points (default 5) inside the
decoder stack and evaluate a per-sample score along *rays* in eta-space.

A ray is defined by a start eta-vector and an end eta-vector (one entry per
cut-point) with end >= start component-wise (i.e. the end point compresses
less / not-more than the start in every position).  We evaluate a few equally
spaced points t in [0, 1] along the ray:

    eta(t) = (1 - t) * start + t * end

Metric (per held-out sample, on its evaluation span)
----------------------------------------------------
Let  Hbar = mean per-token negative log-likelihood (nats/token) on the span.
For each sample s we compute it under the compressed pipeline (Hbar_comp) and
under the uncompressed reference (Hbar_ref), and define

    score(s) = min{ 1, exp( Hbar_ref(s) - Hbar_comp(s) ) }
             = min{ 1, PPL_ref(s) / PPL_comp(s) }                  # per-span

The reported point score is the mean of score(s) over the held-out samples.
score is in (0, 1]: 1 means the compressed pipeline matches (or beats) the
reference on that span; lower means worse.

We then check concavity of mean-score as a function of the ray parameter t:
  (1) discrete second differences on the uniform t-grid (concave => D2 <= 0)
  (2) random chord / midpoint test along the ray.

Outputs (per model x strategy x ray) under --out_dir:
  - {model}__{strategy}__{ray}_sweep.csv
  - {model}__{strategy}_summary.json   (all rays for that model/strategy)
  - {model}__{strategy}_rays.png       (mean-score vs t, one line per ray)
  - all_summary.json                   (aggregate across everything)

Example:
  python experiments/ray_concavity_score.py \
      --models meta-llama/Llama-3.1-8B google/gemma-7b \
      --strategies topk_per_token llmint8 \
      --n_cuts 5 --n_points 6 --max_texts 64 --max_length 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

# Allow running as a top-level script and importing the GPU-native llmint8
# compressor (llm_compression.py does `from gpu_compressors import ...`).
HERE = Path(__file__).resolve().parent      # experiments/concavity
REPO_ROOT = HERE.parents[1]                 # repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "core"))

from src.core.llm_compression import (  # noqa: E402
    ActivationCompressionConfig,
    LLMActivationCompressor,
    detect_llm_decoder_layers,
)


# ──────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────

# WikiText loading and the perplexity score now live in src/core so the
# optimizer task instances and these runners cannot drift apart. Re-exported
# here because mc_concavity and resnet_mc_concavity import them from this
# module.
from src.core.llm_text_eval import (  # noqa: E402,F401
    load_wikitext_texts,
    make_cut_points,
    per_sample_mean_nll,
    sample_scores,
)


# ──────────────────────────────────────────────────────────────────────────
# Cut-points and rays
# ──────────────────────────────────────────────────────────────────────────



@dataclass
class Ray:
    name: str
    start: List[float]  # eta-vector (one per cut-point) at t=0
    end: List[float]    # eta-vector at t=1  (>= start component-wise)


def default_rays(n_cuts: int) -> List[Ray]:
    """A spread of rays going from a compressed `start` to a less-compressed `end`.

    Every ray has end >= start component-wise (some positions strictly higher,
    others equal), matching the requested ray geometry.
    """
    ones = [1.0] * n_cuts
    lo = 0.25

    # diagonal: uniform compression relaxing to baseline everywhere.
    diag = Ray("diagonal", start=[lo] * n_cuts, end=ones[:])

    # front-loaded: early cut-points compressed, relaxing to baseline.
    front_start = [lo if i < n_cuts // 2 else 1.0 for i in range(n_cuts)]
    front = Ray("front_loaded", start=front_start, end=ones[:])

    # back-loaded: late cut-points compressed, relaxing to baseline.
    back_start = [1.0 if i < n_cuts - n_cuts // 2 else lo for i in range(n_cuts)]
    back = Ray("back_loaded", start=back_start, end=ones[:])

    # single-first: only the first cut-point compressed.
    sf_start = ones[:]
    sf_start[0] = lo
    single_first = Ray("single_first", start=sf_start, end=ones[:])

    # single-last: only the last cut-point compressed.
    sl_start = ones[:]
    sl_start[-1] = lo
    single_last = Ray("single_last", start=sl_start, end=ones[:])

    return [diag, front, back, single_first, single_last]


def load_rays(path: str | None, n_cuts: int) -> List[Ray]:
    if not path:
        return default_rays(n_cuts)
    data = json.loads(Path(path).read_text())
    rays: List[Ray] = []
    for d in data:
        r = Ray(name=d["name"], start=[float(x) for x in d["start"]],
                end=[float(x) for x in d["end"]])
        if len(r.start) != n_cuts or len(r.end) != n_cuts:
            raise ValueError(
                f"Ray '{r.name}' has vectors of length "
                f"{len(r.start)}/{len(r.end)} but n_cuts={n_cuts}."
            )
        if any(e < s - 1e-9 for s, e in zip(r.start, r.end)):
            raise ValueError(f"Ray '{r.name}': end must be >= start component-wise.")
        rays.append(r)
    return rays


def ray_points(ray: Ray, n_points: int) -> List[np.ndarray]:
    """Equally spaced eta-vectors along the ray, t in [0, 1]."""
    s = np.asarray(ray.start, dtype=np.float64)
    e = np.asarray(ray.end, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, n_points)
    return [(1.0 - t) * s + t * e for t in ts]


# ──────────────────────────────────────────────────────────────────────────
# Sweep along a ray + concavity tests
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class RayPoint:
    t: float
    eta: List[float]
    mean_score: float
    median_score: float
    mean_hbar_comp: float
    seconds: float


def _eval_point(
    eta_vec: np.ndarray, *, model, tokenizer, compressor, texts,
    device, max_length, batch_size, hbar_ref: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Returns (per_sample_scores, mean_hbar_comp, seconds)."""
    compressor.set_eta(torch.tensor(eta_vec, dtype=torch.float32))
    t0 = time.time()
    hbar_comp = per_sample_mean_nll(
        model, tokenizer, texts,
        device=device, max_length=max_length, batch_size=batch_size,
    )
    scores = sample_scores(hbar_ref, hbar_comp)
    return scores, float(np.mean(hbar_comp)), time.time() - t0


def sweep_ray(
    ray: Ray, *, n_points: int, model, tokenizer, compressor, texts,
    device, max_length, batch_size, hbar_ref: np.ndarray,
) -> List[RayPoint]:
    pts: List[RayPoint] = []
    etas = ray_points(ray, n_points)
    ts = np.linspace(0.0, 1.0, n_points)
    for t, eta_vec in zip(ts, etas):
        scores, mean_hc, secs = _eval_point(
            eta_vec, model=model, tokenizer=tokenizer, compressor=compressor,
            texts=texts, device=device, max_length=max_length,
            batch_size=batch_size, hbar_ref=hbar_ref,
        )
        pts.append(RayPoint(
            t=float(t), eta=[float(x) for x in eta_vec],
            mean_score=float(np.mean(scores)),
            median_score=float(np.median(scores)),
            mean_hbar_comp=mean_hc, seconds=secs,
        ))
        print(f"      t={t:.3f}  eta={np.round(eta_vec, 3).tolist()}  "
              f"mean_score={pts[-1].mean_score:.4f}  ({secs:.1f}s)", flush=True)
    return pts


def second_difference_test(pts: List[RayPoint], tol: float = 1e-3) -> dict:
    """Concavity on the uniform t-grid: D2[j] = y[j-1] + y[j+1] - 2 y[j] <= 0."""
    ys = [p.mean_score for p in pts]
    d2 = []
    violations = 0
    worst = 0.0
    for j in range(1, len(ys) - 1):
        v = ys[j - 1] + ys[j + 1] - 2.0 * ys[j]  # > 0 => convex (non-concave) here
        d2.append(v)
        if v > worst:
            worst = v
        if v > tol:
            violations += 1
    return {
        "n_interior": max(0, len(ys) - 2),
        "violations": violations,
        "worst_d2": worst,  # > 0 means non-concave at that interior point
        "d2": d2,
        "tol": tol,
    }


def random_chord_test(
    ray: Ray, *, model, tokenizer, compressor, texts, device, max_length,
    batch_size, hbar_ref: np.ndarray, n_chords: int = 4, tol: float = 1e-3,
    rng_seed: int = 0,
) -> dict:
    """Midpoint chord test along the ray: f((a+b)/2) >= 0.5(f(a)+f(b)) - tol."""
    rng = np.random.default_rng(rng_seed)
    s = np.asarray(ray.start, dtype=np.float64)
    e = np.asarray(ray.end, dtype=np.float64)

    def mean_score_at(t: float) -> float:
        eta_vec = (1.0 - t) * s + t * e
        scores, _, _ = _eval_point(
            eta_vec, model=model, tokenizer=tokenizer, compressor=compressor,
            texts=texts, device=device, max_length=max_length,
            batch_size=batch_size, hbar_ref=hbar_ref,
        )
        return float(np.mean(scores))

    results = []
    passes = 0
    for _ in range(n_chords):
        a, b = sorted(rng.uniform(0.0, 1.0, size=2).tolist())
        if b - a < 0.05:
            b = min(1.0, a + 0.1)
        m = 0.5 * (a + b)
        fa, fb, fm = mean_score_at(a), mean_score_at(b), mean_score_at(m)
        chord = 0.5 * (fa + fb)
        margin = fm - chord
        ok = margin >= -tol
        passes += int(ok)
        results.append({"a": a, "b": b, "m": m, "fa": fa, "fb": fb, "fm": fm,
                        "chord": chord, "margin": margin, "concave": ok})
        print(f"      chord a={a:.3f} b={b:.3f} m={m:.3f}  margin={margin:+.4f}  "
              f"{'OK' if ok else 'VIOLATION'}", flush=True)
    return {"n_chords": n_chords, "passes": passes, "tol": tol, "results": results}


# ──────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────

def save_rays_plot(ray_pts: Dict[str, List[RayPoint]], path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip plot: {e})")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, pts in ray_pts.items():
        xs = [p.t for p in pts]
        ys = [p.mean_score for p in pts]
        ax.plot(xs, ys, marker="o", label=name)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.4)
    ax.set_xlabel("ray parameter t   (0 = start / more compressed, 1 = end)")
    ax.set_ylabel("mean per-sample score  min{1, PPL_ref/PPL_comp}")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Per-(model, strategy) driver
# ──────────────────────────────────────────────────────────────────────────

def run_model_strategy(
    model_name: str,
    strategy: str,
    *,
    texts: Sequence[str],
    rays: List[Ray],
    n_cuts: int,
    n_points: int,
    max_length: int,
    batch_size: int,
    n_chords: int,
    out_dir: Path,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n=== {model_name}  |  strategy={strategy} ===", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right-pad so masked next-token loss is clean

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    layers = detect_llm_decoder_layers(model)
    n_layers = len(layers)
    cuts = make_cut_points(n_layers, n_cuts)
    print(f"  layers={n_layers}  cuts={cuts}  strategy={strategy}", flush=True)

    compressor = LLMActivationCompressor(
        model,
        ActivationCompressionConfig(strategy=strategy, layer_indices=cuts, random_seed=0),
    )

    ray_pts: Dict[str, List[RayPoint]] = {}
    ray_sd: Dict[str, dict] = {}
    ray_chord: Dict[str, dict] = {}
    try:
        # Reference: uncompressed pipeline (eta = 1 everywhere).
        print("  -- reference (eta=1, uncompressed) --", flush=True)
        compressor.set_eta(torch.ones(len(cuts), dtype=torch.float32))
        t0 = time.time()
        hbar_ref = per_sample_mean_nll(
            model, tokenizer, texts,
            device=device, max_length=max_length, batch_size=batch_size,
        )
        print(f"     ref mean Hbar={np.mean(hbar_ref):+.4f} nats/tok over "
              f"{len(hbar_ref)} samples  ({time.time()-t0:.1f}s)", flush=True)

        for ray in rays:
            print(f"  -- ray '{ray.name}': start={ray.start} end={ray.end} --", flush=True)
            pts = sweep_ray(
                ray, n_points=n_points, model=model, tokenizer=tokenizer,
                compressor=compressor, texts=texts, device=device,
                max_length=max_length, batch_size=batch_size, hbar_ref=hbar_ref,
            )
            ray_pts[ray.name] = pts

            sd = second_difference_test(pts)
            ray_sd[ray.name] = sd
            print(f"     concavity: violations={sd['violations']}/{sd['n_interior']}  "
                  f"worst_d2={sd['worst_d2']:+.4f}", flush=True)

            if n_chords > 0:
                ray_chord[ray.name] = random_chord_test(
                    ray, model=model, tokenizer=tokenizer, compressor=compressor,
                    texts=texts, device=device, max_length=max_length,
                    batch_size=batch_size, hbar_ref=hbar_ref, n_chords=n_chords,
                )
            else:
                ray_chord[ray.name] = {"n_chords": 0, "passes": 0, "results": []}
    finally:
        compressor.remove_hooks()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_slug = model_name.replace("/", "__")
    strat_slug = strategy

    # Per-ray CSVs.
    for name, pts in ray_pts.items():
        csv_path = out_dir / f"{model_slug}__{strat_slug}__{name}_sweep.csv"
        with csv_path.open("w") as f:
            f.write("t,eta,mean_score,median_score,mean_hbar_comp,seconds\n")
            for p in pts:
                eta_str = "|".join(f"{x:.4f}" for x in p.eta)
                f.write(f"{p.t},{eta_str},{p.mean_score},{p.median_score},"
                        f"{p.mean_hbar_comp},{p.seconds}\n")

    summary = {
        "model": model_name,
        "strategy": strategy,
        "n_layers": n_layers,
        "cuts": cuts,
        "n_points": n_points,
        "ref_mean_hbar": float(np.mean(hbar_ref)),
        "n_samples": int(len(hbar_ref)),
        "rays": {
            name: {
                "start": next(r.start for r in rays if r.name == name),
                "end": next(r.end for r in rays if r.name == name),
                "sweep": [asdict(p) for p in pts],
                "second_difference": ray_sd[name],
                "chord": ray_chord[name],
            }
            for name, pts in ray_pts.items()
        },
    }
    (out_dir / f"{model_slug}__{strat_slug}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    save_rays_plot(
        ray_pts, out_dir / f"{model_slug}__{strat_slug}_rays.png",
        title=f"{model_name}  ({strategy}, {len(cuts)} cuts)",
    )
    return summary


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = ["meta-llama/Llama-3.1-8B", "google/gemma-7b"]
DEFAULT_STRATEGIES = ["topk_per_token", "llmint8"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="HF model ids (must be in your HF cache).")
    p.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES,
                   choices=["topk_per_token", "magnitude", "random",
                            "quantization", "llmint8"])
    p.add_argument("--n_cuts", type=int, default=5,
                   help="Number of activation cut-points (eta-vector dimension).")
    p.add_argument("--n_points", type=int, default=6,
                   help="Points evaluated along each ray (t in [0,1]).")
    p.add_argument("--rays_json", default=None,
                   help="Optional JSON file: [{name,start[],end[]}, ...]. "
                        "Defaults to a built-in spread of rays.")
    p.add_argument("--max_texts", type=int, default=64,
                   help="WikiText-2 paragraphs used per evaluation (held-out samples).")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_chords", type=int, default=3,
                   help="Random midpoint-chord tests per ray (0 to skip).")
    p.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "ray_concavity_score"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rays = load_rays(args.rays_json, args.n_cuts)
    print(f"Rays ({len(rays)}): {[r.name for r in rays]}")
    print(f"Loading WikiText-2 ({args.max_texts} paragraphs, max_length={args.max_length})…")
    texts = load_wikitext_texts(args.max_texts)
    print(f"  -> {len(texts)} paragraphs.")

    aggregate = {"args": vars(args), "runs": {}}
    for m in args.models:
        for strat in args.strategies:
            key = f"{m}::{strat}"
            try:
                summary = run_model_strategy(
                    m, strat,
                    texts=texts, rays=rays, n_cuts=args.n_cuts,
                    n_points=args.n_points, max_length=args.max_length,
                    batch_size=args.batch_size, n_chords=args.n_chords,
                    out_dir=out_dir,
                )
                aggregate["runs"][key] = {
                    "cuts": summary["cuts"],
                    "ref_mean_hbar": summary["ref_mean_hbar"],
                    "rays": {
                        name: {
                            "mean_scores": [p["mean_score"] for p in r["sweep"]],
                            "concavity_violations": r["second_difference"]["violations"],
                            "n_interior": r["second_difference"]["n_interior"],
                            "worst_d2": r["second_difference"]["worst_d2"],
                            "chord_passes": r["chord"]["passes"],
                            "n_chords": r["chord"]["n_chords"],
                        }
                        for name, r in summary["rays"].items()
                    },
                }
            except Exception as e:
                print(f"  !! {key} failed: {type(e).__name__}: {e}", flush=True)
                aggregate["runs"][key] = {"error": f"{type(e).__name__}: {e}"}

    (out_dir / "all_summary.json").write_text(json.dumps(aggregate, indent=2))
    print(f"\nDone. Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
