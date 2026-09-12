"""
Exhaustive enumeration of the QUANTIZATION compression profile + exact concavity.

Unlike top-k / llmint8 (continuous in eta), the quantization strategy snaps each
cut-point's eta to a HANDFUL of discrete levels -- for FP16/BF16 activations the
achievable ratios are {0.25, 0.5, 1.0} (= 4-bit / 8-bit / 16-bit-no-op; see
``LLMActivationCompressor._compress``'s ``[base, base//2, base//4]`` ladder).

With only L levels per cut and n cut-points there are just L**n distinct
operating points, so we do NOT need the Monte-Carlo random-ray campaign that
top-k / llmint8 require: we evaluate the task metric at EVERY lattice point
exactly. That is the whole "few levels -> all compression profiles quickly"
observation.

Output for the downstream pipeline
----------------------------------
We express the exact grid as a complete set of AXIS-ALIGNED straight-line rays
(one per (varying-axis, fixed-others) combination; L points each) written in the
same ``*_rays.json`` schema ``mc_concavity.py`` emits, so ``jensen_concavity.py``
and ``plot_concavity_per_method.py`` fold the quantization curve into the
existing plots with NO analysis changes.

Correctness of the concavity test on non-uniform levels: ``jensen_pass`` computes
its chord weight ``lam`` from the ray parameter ``t``. On a straight segment
``eta(t) = (1-t)*start + t*end``, ``t`` is affine in every eta coordinate, so the
chord test in t-space is identical to the correct eta-space test even though the
levels {0.25, 0.5, 1.0} are unevenly spaced. We set each ray's ``t`` to the true
fractional position of its level (``(level-min)/(max-min)``) to preserve this.

FLOOR-SWEEP CAVEAT: ``ray_jensen_trials`` needs >= 3 IN-FLOOR points per axis
line. With the paper's 5 levels {0.0625,...,1.0} an axis line keeps >= 3 points
only while the scalar eta-floor c <= the third-smallest level (0.25); above that
the line is dropped, so the quantization floor-sweep thins out fast and is
undefined for c > 0.5. The exact-grid concavity (this script's headline,
``exact_concavity`` in the summary) is the floor-0 value and is EXACT, not
Monte-Carlo -- that is the number to report for quantization.

Example (smoke, tiny grid on CPU-loadable model):
  python experiments/concavity/enumerate_quant_concavity.py \
      --model sshleifer/tiny-gpt2 --dataset wikitext --metric perplexity \
      --n_cuts 3 --max_texts 4 --max_length 64 \
      --out_dir outputs/mc_concavity/smoke_quant
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent      # experiments/concavity
REPO_ROOT = HERE.parents[1]                 # repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "core"))
sys.path.insert(0, str(HERE))

# Reuse the model / evaluator plumbing verbatim from the MC runner so the
# quantization arm is evaluated IDENTICALLY to the top-k / llmint8 arms.
from mc_concavity import (  # noqa: E402
    build_evaluator,
    load_model_and_compressor,
)


def _key(vec: Sequence[float]) -> str:
    """Stable dict key for a lattice point (rounded level vector)."""
    return json.dumps([round(float(x), 6) for x in vec])


def chord_concave(t: Sequence[float], y: Sequence[float], tol: float) -> Tuple[bool, float, int]:
    """Eta-aware concavity of one straight-line ray.

    Tests EVERY triple i<j<k (t already ascending): concave iff the middle point
    sits on/above the chord through the outer two,
        y_j >= lam*y_i + (1-lam)*y_k,   lam = (t_k - t_j)/(t_k - t_i).
    Returns (all_pass, worst_violation, n_triples). ``worst_violation`` is the
    largest (chord - y_j); positive means a convex dip somewhere.
    """
    t = [float(x) for x in t]
    y = [float(x) for x in y]
    n = len(t)

    # A non-finite utility means the evaluation itself failed (e.g. an
    # unclipped FP16 cast overflowing to inf, which then poisons the metric).
    # Without this guard NaN passes silently: `viol > tol` is False for NaN, so
    # `ok` stays True and a broken ray is scored CONCAVE. That inflated the T5
    # ppl_score result from 0.293 to 0.773 before the FP16 clipping fix.
    # Return worst=inf so the failure is loud and greppable in the rays JSON.
    if not all(math.isfinite(v) for v in y) or not all(math.isfinite(v) for v in t):
        return False, float("inf"), 0

    worst = 0.0
    n_tri = 0
    ok = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                span = t[k] - t[i]
                if span <= 0:
                    continue
                lam = (t[k] - t[j]) / span
                # Stable lerp — exact when y[i] == y[k]; see jensen_pass() in
                # jensen_concavity.py for why the two-term form breaks flat
                # rays. Inert at the tol=1e-3 this runs at (the error is 1 ULP),
                # fixed so the two chord tests cannot diverge at tol=0.
                chord = y[k] + lam * (y[i] - y[k])
                viol = chord - y[j]
                n_tri += 1
                if viol > worst:
                    worst = viol
                if viol > tol:
                    ok = False
    return ok, worst, n_tri


def build_axis_rays(levels: List[float], grid: Dict[str, float], n: int,
                    tol: float) -> List[dict]:
    """Assemble the full set of axis-aligned straight-line rays from the grid.

    For each axis k and each assignment of the other n-1 coords to lattice
    levels, one ray varies coord k across all ``levels`` (others fixed). Points
    are sorted ascending in the varying level so ``t`` is ascending.
    """
    lo, hi = min(levels), max(levels)
    span = (hi - lo) if hi > lo else 1.0
    ordered = sorted(levels)
    others_axes = list(range(n))
    rays: List[dict] = []
    rid = 0
    for k in range(n):
        rest = [a for a in others_axes if a != k]
        for combo in itertools.product(ordered, repeat=len(rest)):
            fixed = dict(zip(rest, combo))
            etas: List[List[float]] = []
            ts: List[float] = []
            ys: List[float] = []
            for lvl in ordered:
                vec = [0.0] * n
                for a in range(n):
                    vec[a] = float(lvl) if a == k else float(fixed[a])
                etas.append([float(x) for x in vec])
                ts.append((float(lvl) - lo) / span)
                ys.append(float(grid[_key(vec)]))
            start = list(etas[0])
            end = list(etas[-1])
            ok, worst, n_tri = chord_concave(ts, ys, tol)
            rays.append({
                "id": rid,
                "axis": k,
                "start": start,
                "end": end,
                "t": ts,
                "eta": etas,
                "y": ys,
                "concave": bool(ok),
                "worst_d2": float(worst),
                "violations": int(0 if ok else 1),
                "n_triples": n_tri,
            })
            rid += 1
    return rays


def _make_vec(n: int, k: int, lvl: float, others) -> List[float]:
    """eta-vector with axis k = lvl and the other axes = ``others`` (given in
    ascending axis order)."""
    vec = [0.0] * n
    oi = 0
    for a in range(n):
        if a == k:
            vec[a] = float(lvl)
        else:
            vec[a] = float(others[oi]); oi += 1
    return vec


def build_sampled_axis_rays(levels: List[float], grid: Dict[str, float], n: int,
                            specs, tol: float) -> List[dict]:
    """Axis-line rays for a SAMPLED subset of (axis, fixed-others) specs
    (used when the full L^n grid is infeasible, e.g. cuts7)."""
    lo, hi = min(levels), max(levels)
    span = (hi - lo) if hi > lo else 1.0
    ordered = sorted(levels)
    rays: List[dict] = []
    for rid, (k, others) in enumerate(specs):
        etas, ts, ys = [], [], []
        for lvl in ordered:
            vec = _make_vec(n, k, lvl, others)
            etas.append([float(x) for x in vec])
            ts.append((float(lvl) - lo) / span)
            ys.append(float(grid[_key(vec)]))
        ok, worst, n_tri = chord_concave(ts, ys, tol)
        rays.append({
            "id": rid, "axis": int(k),
            "start": list(etas[0]), "end": list(etas[-1]),
            "t": ts, "eta": etas, "y": ys,
            "concave": bool(ok), "worst_d2": float(worst),
            "violations": int(0 if ok else 1), "n_triples": n_tri,
        })
    return rays


def sampled_specs_and_points(levels: List[float], n: int, sample_lines: int,
                             seed: int):
    """Deterministic (specs, unique points) for a SAMPLED axis-line run.

    Every participant -- each enumerator shard, the status probe and the merger
    -- has to derive the identical sample, so this is seeded and shared rather
    than reimplemented in each. ``levels`` must already be sorted.
    """
    srng = np.random.default_rng(seed)
    max_lines = min(sample_lines, n * (len(levels) ** (n - 1)))
    specs, seen = [], set()
    while len(specs) < max_lines:
        k = int(srng.integers(0, n))
        others = tuple(float(srng.choice(levels)) for _ in range(n - 1))
        if (k, others) in seen:
            continue
        seen.add((k, others)); specs.append((k, others))
    pts, seen_pt = [], set()
    for k, others in specs:
        for lvl in levels:
            v = tuple(_make_vec(n, k, lvl, others))
            if v not in seen_pt:
                seen_pt.add(v); pts.append(v)
    return specs, pts


def full_grid_order(levels: List[float], n: int, priority_lines: int, seed: int):
    """Every lattice point, ordered so the SAMPLED subset is evaluated first.

    The exhaustive grid is the goal, but on a contended cluster a run may be cut
    short at any point. Leading with the points of ``priority_lines`` sampled
    axis lines means the sampled-estimate result becomes available as soon as
    that prefix is covered (~40% fewer evaluations), and the remaining points
    then upgrade it to the exact value if the capacity materialises. Ordering
    only -- the point SET is unchanged, so a completed run is bit-identical to
    a lexicographic one.
    """
    ordered = [tuple(p) for p in itertools.product(levels, repeat=n)]
    if priority_lines <= 0:
        return ordered
    _, head = sampled_specs_and_points(levels, n, priority_lines, seed)
    head_set = set(head)
    return head + [p for p in ordered if p not in head_set]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, choices=["wikitext", "sharegpt", "mmlu"])
    p.add_argument("--metric", required=True, choices=["perplexity", "accuracy"])
    # Fixed to quantization; exposed so the stem/meta read cleanly and so the
    # discrete ladder can be swapped if the compressor gains more levels.
    p.add_argument("--strategy", default="quantization", choices=["quantization"])
    p.add_argument("--levels", default="0.0625,0.125,0.25,0.5,1.0",
                   help="Comma-separated achievable eta levels per cut. Default "
                        "is the paper ladder relative to the FP32 native width: "
                        "{2,4,8,16,32}-bit = {0.0625,0.125,0.25,0.5,1.0}.")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "auto"],
                   help="Model load dtype. Quantization uses fp32 (paper native "
                        "reference); fp16/auto match the topk/llmint8 arms.")
    p.add_argument("--n_cuts", type=int, default=4)
    p.add_argument("--sample_lines", type=int, default=0,
                   help="If >0, evaluate this many RANDOM axis-lines over the "
                        "level lattice instead of the full L^n grid (for large n, "
                        "e.g. cuts7 where 5^7=78125 is infeasible). Mirrors how the "
                        "topk/llmint8 cuts7 arms sampled random rays. 0 = exhaustive.")
    p.add_argument("--tol", type=float, default=None,
                   help="Concavity chord tolerance (default 0.02 accuracy / 1e-3 perplexity).")
    # eval opts (mirror mc_concavity.run so the arm is comparable)
    p.add_argument("--max_texts", type=int, default=128)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--mmlu_subjects", default="")
    p.add_argument("--samples_per_subject", type=int, default=20)
    p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--shard_count", type=int, default=1,
                   help="split the point list round-robin across this many "
                        "workers (each writes its own *_grid.shardIofN.partial "
                        "checkpoint)")
    p.add_argument("--shard_index", type=int, default=0,
                   help="which slice this worker evaluates, in [0,shard_count)")
    p.add_argument("--points_file", default="",
                   help="JSON list (or {'points': [...]}) of eta-vectors to "
                        "evaluate -- FILL mode: extends the grid and skips ray "
                        "assembly. Pairs with quant_box_rays.py's "
                        "*_missing_points.json.")
    p.add_argument("--priority_lines", type=int, default=100,
                   help="exhaustive runs evaluate the points of this many "
                        "sampled axis lines FIRST, so an interrupted run still "
                        "yields the sampled estimate; 0 = plain lexicographic "
                        "order. Does not change the set of points evaluated.")
    p.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "mc_concavity"))
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.tol is None:
        args.tol = 0.02 if args.metric == "accuracy" else 1e-3
    # Seed torch so INT4 stochastic rounding is reproducible across a resumed
    # / re-run enumeration (the metric is otherwise noisy at the INT4 level).
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    levels = sorted(float(x) for x in args.levels.split(","))
    n = args.n_cuts
    n_points_grid = len(levels) ** n
    sampled = bool(args.sample_lines and args.sample_lines > 0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "__")
    stem = f"{slug}__{args.dataset}__{args.metric}__{args.strategy}__cuts{n}"

    # Which lattice points to evaluate: the full grid (exhaustive) or the unique
    # points touched by a random sample of axis-lines (for large n, e.g. cuts7).
    # FILL mode: evaluate an explicit list of lattice points (the ones a
    # random-box-ray resampling needs but the sparse grid is missing -- see
    # experiments/concavity/quant_box_rays.py, which writes *_missing_points.json). Ray
    # assembly is skipped; the point of the run is purely to extend the grid.
    fill_mode = bool(args.points_file)
    if fill_mode:
        raw = json.loads(Path(args.points_file).read_text())
        if isinstance(raw, dict):
            raw = raw.get("points", [])
        specs = None
        sampled = False
        points_to_eval = [tuple(float(x) for x in p) for p in raw]
    elif sampled:
        specs, points_to_eval = sampled_specs_and_points(
            levels, n, args.sample_lines, args.seed)
    else:
        specs = None
        points_to_eval = full_grid_order(levels, n, args.priority_lines, args.seed)
    # ── Optional sharding ──────────────────────────────────────────────────
    # A single worker is walltime-bound on the big grids (llama/mmlu is ~168 s
    # per lattice point in FP32, i.e. ~29 h for 5^4). Split the point list
    # round-robin across N workers, each checkpointing to its OWN file so there
    # is no write race. Slices are disjoint, so no point is ever evaluated
    # twice, and the shard checkpoints can be unioned afterwards.
    sharded = args.shard_count > 1
    if sharded and not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"--shard_index must be in [0,{args.shard_count})")
    if sharded:
        points_to_eval = [p for i, p in enumerate(points_to_eval)
                          if i % args.shard_count == args.shard_index]
    n_target = len(points_to_eval)

    base_cp = out_dir / f"{stem}_grid.partial.json"
    grid_cp = (out_dir / f"{stem}_grid.shard{args.shard_index}of"
                         f"{args.shard_count}.partial.json") if sharded else base_cp
    grid: Dict[str, float] = {}
    # Load our own checkpoint plus (when sharded) the shared serial-run
    # checkpoint, which is read-only here -- it may already hold points that
    # fall in our slice, and those must not be recomputed.
    for src in ([base_cp, grid_cp] if sharded else [grid_cp]):
        if not src.exists():
            continue
        try:
            grid.update(json.loads(src.read_text()))
            print(f"  resuming: {len(grid)} grid points on disk "
                  f"(after {src.name})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  (ignoring unreadable checkpoint {src}: {e})", flush=True)
    if sharded:
        todo = sum(1 for p in points_to_eval if _key(p) not in grid)
        print(f"  shard {args.shard_index}/{args.shard_count}: "
              f"{n_target} slice points, {todo} still to evaluate", flush=True)

    device, tokenizer, model, compressor, cuts = load_model_and_compressor(args)

    summary: dict = {
        "model": args.model, "dataset": args.dataset, "metric": args.metric,
        "strategy": args.strategy, "n_cuts": n, "cuts": cuts,
        "levels": levels, "n_points": len(levels), "seed": args.seed,
        "tol": args.tol, "enumeration": not sampled, "n_grid_points": n_points_grid,
        "sampled_axis_lines": (len(specs) if sampled else None),
        "n_evaluated": n_target,
        # Let the merge/status steps rebuild the same sample without a model.
        "sample_lines": args.sample_lines,
        "priority_lines": (0 if sampled else args.priority_lines),
    }

    try:
        evaluator = build_evaluator(
            args, model=model, tokenizer=tokenizer, compressor=compressor,
            device=device, n_cuts=n)
        summary["baseline"] = evaluator.baseline
        print(f"  baseline (eta=1) = {evaluator.baseline:.4f}", flush=True)
        mode = (f"sampling {len(specs)} axis-lines ({n_target} unique points)"
                if sampled else
                f"enumerating {n_target} lattice points ({len(levels)}^{n})")
        print(f"  {mode} ...", flush=True)

        for i, pt in enumerate(points_to_eval):
            kk = _key(pt)
            if kk in grid:
                continue
            t0 = time.time()
            y = float(evaluator.evaluate(np.asarray(pt, dtype=np.float64)))
            grid[kk] = y
            print(f"  [{i+1}/{n_target}] eta={list(pt)} -> {y:.5f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            if (i + 1) % args.checkpoint_every == 0 or (i + 1) == n_target:
                grid_cp.write_text(json.dumps(grid))
    finally:
        compressor.remove_hooks()
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # A shard only owns a slice of the grid, so it cannot build rays (an axis
    # line spans points held by other shards). Persist and stop; the merge step
    # unions every shard and does the ray assembly once.
    if sharded or fill_mode:
        grid_cp.write_text(json.dumps(grid))
        # Baseline / cut indices are only known to a worker that loaded the
        # model; stash them so the merge step can fill the summary.
        (out_dir / f"{stem}_meta.json").write_text(json.dumps(summary, indent=2))
        done = sum(1 for p in points_to_eval if _key(p) in grid)
        what = "fill" if fill_mode else f"Shard {args.shard_index}/{args.shard_count}"
        print(f"{what} done: {done}/{n_target} points evaluated. "
              f"Wrote {grid_cp.name}.", flush=True)
        return

    # ── Assemble axis-aligned rays + concavity ─────────────────────────────
    if sampled:
        rays = build_sampled_axis_rays(levels, grid, n, specs, args.tol)
    else:
        rays = build_axis_rays(levels, grid, n, args.tol)
    n_conc = sum(1 for r in rays if r["concave"])
    conc = (n_conc / len(rays)) if rays else None
    summary["exact_concavity"] = conc  # sampled: axis-line concavity estimate
    summary["n_axis_rays"] = len(rays)
    summary["n_concave_axis_rays"] = n_conc
    # Human-readable profile (evaluated points only).
    summary["profile"] = [
        {"eta": json.loads(kkey), "y": grid[kkey]}
        for kkey in sorted(grid, key=lambda s: json.loads(s))
    ]

    (out_dir / f"{stem}_rays.json").write_text(json.dumps({
        **{k: v for k, v in summary.items() if k != "profile"},
        "rays": rays, "phase2_rays": [], "fill_rays": [],
    }, indent=2))
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    # Complete -> drop the resume checkpoint.
    if grid_cp.exists() and all(_key(p) in grid for p in points_to_eval):
        grid_cp.unlink()

    ec = f"{conc:.3f}" if conc is not None else "n/a"
    kind = "sampled" if sampled else "exact"
    print(f"Done ({kind}). concavity={ec} ({n_conc}/{len(rays)} axis-lines concave). "
          f"Wrote {stem}_rays.json / _summary.json to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
