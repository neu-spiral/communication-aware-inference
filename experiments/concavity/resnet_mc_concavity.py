#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monte-Carlo random-ray concavity + adaptive fill for the ResNet56/CIFAR-10
backend, producing rays in the SAME format as mc_concavity.py so the resnet
groups reach >= min_box_samples usable rays per eta-floor (like the LLM/T5
groups) instead of only the 10 fixed directional rays from analyze_concavity.py.

The original resnet concavity (outputs/concavity_resnet/*/concavity_results.json)
samples 10 anchored rays (corner -> [1,1,1]); jensen_concavity folds those in via
resnet_entries. This script adds free random rays over [eps,1]^3 and an adaptive
fill over the high-floor sub-cubes, written as an mc_concavity-style *_rays.json
under outputs/mc_concavity/resnet/imagenet_accuracy/<strategy>/cuts3/. jensen
merges the two by (model,dataset,metric,n_cuts,strategy) content-hash, so the
directional rays and the new MC/fill rays combine into one group.

The evaluator is ResNetTaskCallables.accuracy_callable (top-1 over a few fixed
CIFAR-10 batches) — cheap, so base + fill run in one shot per compressor.

Backend config mirrors scripts/run_concavity_analysis.sh (cutpoints 8,14,21,
fast_batches from RESNET_FAST_BATCHES). Compressor is chosen with --compressor
(sets COMPRESSOR_TYPE for build_resnet_backend_from_env).

Usage:
  python experiments/concavity/resnet_mc_concavity.py --compressor topk \
      --checkpoint assets/resnet56-4bfd9763.th --data_root data \
      --n_rays 120 --min_box_samples 100 --floor_max 0.7 --out_dir <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

HERE = Path(__file__).resolve().parent      # experiments/concavity
REPO_ROOT = HERE.parents[1]                 # repo root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact MC helpers so the ray format + fill logic match the LLM runs.
from mc_concavity import (  # noqa: E402
    run_mc_phase,
    adaptive_fill_phase,
    concavity_fraction,
    concavity_of,
)

# _RESNET_STRAT in jensen_concavity maps the compressor name straight through;
# use the same strings so the new file merges with the directional-profile group.
_STRAT = {"topk": "topk", "quantization": "quantization", "llmint8": "llmint8"}


def _build_backend(args):
    """Construct the ResNet backend, mirroring run_concavity_analysis.sh env."""
    os.environ["INFERENCE_OPTIMIZER_BACKEND"] = "resnet"
    os.environ["RESNET_CHECKPOINT"] = str(args.checkpoint)
    os.environ["RESNET_DATA_ROOT"] = str(args.data_root)
    os.environ["RESNET_CUTPOINTS"] = args.cutpoints
    os.environ["RESNET_FAST_BATCHES"] = str(args.fast_batches)
    os.environ["RESNET_BATCH_SIZE"] = str(args.batch_size)
    os.environ["RESNET_DOWNLOAD"] = "1" if args.download else "0"
    os.environ["COMPRESSOR_TYPE"] = args.compressor
    os.environ["COMPRESSOR_BACKEND"] = args.compressor_backend
    from src.core.resnet_task_callables import build_resnet_backend_from_env

    return build_resnet_backend_from_env()


def cmd_enumerate(args) -> None:
    """Exhaustive quantization enumeration for the ResNet backend (5^n grid).

    Mirrors experiments/enumerate_quant_concavity.py but drives
    accuracy_callable(eta) instead of an LLM evaluator, writing axis-aligned
    rays in the same schema so jensen folds the ResNet quantization curve in."""
    from enumerate_quant_concavity import build_axis_rays, _key  # reuse helpers

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strategy = _STRAT.get(args.compressor, args.compressor)
    n_cuts = len([c for c in args.cutpoints.split(",") if c.strip()])
    stem = f"resnet__imagenet__accuracy__{strategy}__cuts{n_cuts}"
    levels = sorted(float(x) for x in args.levels.split(","))
    tol = float(args.tol)

    backend = _build_backend(args)
    if backend.num_links != n_cuts:
        raise ValueError(
            f"cutpoints imply {n_cuts} links but backend reports {backend.num_links}")

    n_repeat = max(1, int(args.n_repeat))

    def evaluate(eta_vec):
        e = np.asarray(eta_vec, dtype=np.float64)
        if n_repeat == 1:
            return float(backend.accuracy_callable(e))
        return float(np.mean([backend.accuracy_callable(e) for _ in range(n_repeat)]))

    baseline = evaluate(np.ones(n_cuts))
    print(f"  baseline accuracy (eta=1) = {baseline:.4f}", flush=True)

    grid = {}
    all_points = list(itertools.product(levels, repeat=n_cuts))
    print(f"  enumerating {len(all_points)} lattice points "
          f"({len(levels)} levels ^ {n_cuts} cuts) ...", flush=True)
    for i, pt in enumerate(all_points):
        t0 = time.time()
        grid[_key(pt)] = evaluate(pt)
        print(f"  [{i+1}/{len(all_points)}] eta={list(pt)} -> {grid[_key(pt)]:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    rays = build_axis_rays(levels, grid, n_cuts, tol)
    n_conc = sum(1 for r in rays if r["concave"])
    payload = {
        "model": "resnet", "dataset": "imagenet", "metric": "accuracy",
        "strategy": strategy, "compressor": args.compressor,
        "backend": "resnet_enum", "n_cuts": n_cuts, "cutpoints": args.cutpoints,
        "levels": levels, "n_points": len(levels), "seed": args.seed, "tol": tol,
        "fast_batches": args.fast_batches, "n_repeat": n_repeat,
        "baseline": baseline, "enumeration": True,
        "n_grid_points": len(all_points),
        "exact_concavity": (n_conc / len(rays)) if rays else None,
        "n_axis_rays": len(rays), "n_concave_axis_rays": n_conc,
        "rays": rays, "phase2_rays": [], "fill_rays": [],
    }
    (out_dir / f"{stem}_rays.json").write_text(json.dumps(payload, indent=2))
    ec = payload["exact_concavity"]
    print(f"Done. exact_concavity={ec:.3f} ({n_conc}/{len(rays)}). "
          f"Wrote {stem}_rays.json to {out_dir}", flush=True)


def cmd_run(args) -> None:
    if getattr(args, "enumerate", False):
        return cmd_enumerate(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy = _STRAT.get(args.compressor, args.compressor)
    n_cuts = len([c for c in args.cutpoints.split(",") if c.strip()])
    stem = f"resnet__imagenet__accuracy__{strategy}__cuts{n_cuts}"
    rays_path = out_dir / f"{stem}_rays.json"

    backend = _build_backend(args)
    if backend.num_links != n_cuts:
        raise ValueError(
            f"cutpoints imply {n_cuts} links but backend reports "
            f"{backend.num_links}"
        )

    n_repeat = max(1, int(args.n_repeat))

    def evaluate(eta_vec: np.ndarray) -> float:
        e = np.asarray(eta_vec, dtype=np.float64)
        if n_repeat == 1:
            return float(backend.accuracy_callable(e))
        return float(np.mean([backend.accuracy_callable(e) for _ in range(n_repeat)]))

    rng = np.random.default_rng(args.seed)
    eps = float(args.eps)
    lo_full = np.full(n_cuts, eps, dtype=np.float64)
    tol = float(args.tol)

    print(
        f"  baseline accuracy (eta=1) = "
        f"{evaluate(np.ones(n_cuts)):.4f}",
        flush=True,
    )

    # ── Resume: reuse existing base + fill rays if present ──────────────────
    base_rays: List[dict] = []
    prior_fill: List[dict] = []
    if rays_path.exists():
        prev = json.loads(rays_path.read_text())
        base_rays = prev.get("rays", [])
        prior_fill = prev.get("fill_rays", [])
        print(
            f"  resuming: {len(base_rays)} base + {len(prior_fill)} fill rays "
            f"already on disk",
            flush=True,
        )

    # ── Phase 1: random rays over [eps, 1]^n ────────────────────────────────
    need_base = max(0, args.n_rays - len(base_rays))
    if need_base > 0:
        print(f"  -- Phase 1: {need_base} random rays over [{eps}, 1]^{n_cuts} --",
              flush=True)
        cp = out_dir / f"{stem}_phase1.partial.json"
        new_base = run_mc_phase(
            evaluate, n=n_cuts, n_rays=need_base, n_points=args.n_points,
            rng=rng, lo=lo_full, eps=eps, min_seg=args.min_seg, tol=tol,
            label="phase1", checkpoint_path=cp)
        for k, r in enumerate(new_base):
            r["id"] = len(base_rays) + k
        base_rays = base_rays + new_base
        frac, n_used = concavity_of(base_rays)
        print(f"  Phase-1 concavity = {frac:.3f} (M={n_used})", flush=True)

    # ── Adaptive fill over high-floor sub-cubes ─────────────────────────────
    all_for_fill = base_rays + prior_fill
    new_fill = adaptive_fill_phase(
        evaluate, all_for_fill, n_cuts, rng=rng,
        min_box_samples=args.min_box_samples,
        floor_step=args.adaptive_floor_step, n_points=args.n_points,
        eps=eps, min_seg=args.min_seg, tol=tol, lo_full=lo_full,
        out_dir=out_dir, stem=stem, floor_max=args.floor_max)

    all_fill = prior_fill + new_fill
    for k, r in enumerate(all_fill):
        r["id"] = k

    payload = {
        "model": "resnet",
        "dataset": "imagenet",  # label matches resnet_entries in jensen_concavity
        "metric": "accuracy",
        "strategy": strategy,
        "compressor": args.compressor,
        "backend": "resnet_mc",
        "n_cuts": n_cuts,
        "cutpoints": args.cutpoints,
        "n_points": args.n_points,
        "seed": args.seed,
        "tol": tol,
        "fast_batches": args.fast_batches,
        "n_repeat": n_repeat,
        "rays": base_rays,
        "phase2_rays": [],
        "fill_rays": all_fill,
    }
    rays_path.write_text(json.dumps(payload, indent=2))

    # Per-floor coverage report.
    print("  per-floor usable rays (base+fill):", flush=True)
    merged = base_rays + all_fill
    for c in np.round(np.arange(0.0, args.floor_max + 1e-9, args.adaptive_floor_step), 3):
        _, u = concavity_fraction(merged, np.full(n_cuts, float(c)), tol)
        print(f"    floor {c:.1f}: usable={u}", flush=True)
    print(
        f"  Added {len(new_fill)} fill rays this run "
        f"({len(base_rays)} base + {len(all_fill)} fill total). Wrote {rays_path}",
        flush=True,
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compressor", default="topk",
                   choices=["topk", "quantization", "llmint8"])
    p.add_argument("--compressor_backend", default="gpu")
    p.add_argument("--checkpoint", default=str(REPO_ROOT / "assets" / "resnet56-4bfd9763.th"))
    p.add_argument("--data_root", default=str(REPO_ROOT / "data"))
    p.add_argument("--cutpoints", default="8,14,21")
    p.add_argument("--fast_batches", default="1",
                   help="int or 'all' (RESNET_FAST_BATCHES)")
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--download", type=int, default=1)
    p.add_argument("--n_repeat", type=int, default=1,
                   help="average this many accuracy_callable calls per point")
    p.add_argument("--n_rays", type=int, default=120)
    p.add_argument("--n_points", type=int, default=20)
    p.add_argument("--min_box_samples", type=int, default=100)
    p.add_argument("--adaptive_floor_step", type=float, default=0.1)
    p.add_argument("--floor_max", type=float, default=0.7)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--min_seg", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--enumerate", action="store_true",
                   help="Exhaustively enumerate the level lattice (quantization) "
                        "instead of Monte-Carlo random rays.")
    p.add_argument("--levels", default="0.0625,0.125,0.25,0.5,1.0",
                   help="Discrete eta levels for --enumerate (paper FP32 ladder).")
    p.set_defaults(func=cmd_run)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
