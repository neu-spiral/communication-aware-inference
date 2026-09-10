"""
Random-ray concavity for the QUANTIZATION arm, read straight off the lattice.

Why this needs no GPU
---------------------
Quantization snaps each cut's eta to a discrete level BEFORE compressing, so a
ray drawn anywhere through the eta-box only ever touches lattice points that the
exhaustive enumeration already evaluated. We can therefore sample random rays
exactly the way ``mc_concavity.py`` does for top-k / llmint8 -- two independent
uniform endpoints in [lo, 1]^n, ``n_points`` equally spaced samples along the
segment -- and look the utility up instead of running the model. That finally
makes the quantization curve methodologically identical to the other two arms,
which were always measured with random rays rather than axis-aligned lines.

The snap rule is per-backend and they are NOT the same
-----------------------------------------------------
Read out of the code, then confirmed against the grids:

  LLM   (gemma, llama)  src/core/llm_compression.py:177-188
        eta >= 1 -> FP32 passthrough; else FLOOR to the ladder
        (>=0.5 FP16, >=0.25 INT8, >=0.125 INT4, else INT2).
  T5    flant5_sst2_concavity_test/compressors.py:396-407
        eta >= 1 -> passthrough; else FLOOR, same thresholds.
  ResNet src/core/resnet_task_callables.py:249 + gpu_compressors.py:351-356
        eta >= 1 -> passthrough; else CEIL (k < L+1e-6 picks L), and
        everything in (0.5, 1) falls back to FP16, i.e. level 0.5.

CONSEQUENCE, and it is a big one: under the real rule the top rung (1.0) is
reachable ONLY at eta exactly 1.0. Continuous ray sampling hits that with
probability zero, so random rays explore a FOUR-level lattice
{0.0625, 0.125, 0.25, 0.5} and the entire eta=1 slice of the grid goes unused.
``--snap nearest`` implements the "closest level" reading instead (eta > 0.75
-> 1.0), which makes all five rungs reachable but does NOT correspond to what
the deployed compressor does. Both are provided; `code` is the default because
it is the only one that describes a real operating point.

Usage:
  python experiments/quant_box_rays.py --group_dir <dir> [--snap code|nearest]
  python experiments/quant_box_rays.py --all           # every quant group
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent      # experiments/concavity
REPO_ROOT = HERE.parents[1]                 # repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from enumerate_quant_concavity import _key, chord_concave  # noqa: E402

LADDER = [0.0625, 0.125, 0.25, 0.5, 1.0]


# ── snapping ───────────────────────────────────────────────────────────────
def snap_floor(e: float, ladder: Sequence[float]) -> float:
    """LLM / T5: passthrough at >=1, else largest level <= eta."""
    if e >= 1.0:
        return 1.0
    below = [L for L in ladder if L <= e and L < 1.0]
    return max(below) if below else min(ladder)


def snap_ceil(e: float, ladder: Sequence[float]) -> float:
    """ResNet: passthrough at >=1, else smallest level >= eta, capped at 0.5
    (the GPU dispatcher has no FP32 rung -- anything in (0.5,1) is FP16)."""
    if e >= 1.0:
        return 1.0
    above = [L for L in ladder if L >= e and L < 1.0]
    return min(above) if above else max(L for L in ladder if L < 1.0)


def snap_nearest(e: float, ladder: Sequence[float]) -> float:
    """The 'closest quantization level' reading. Not what the code does."""
    return min(ladder, key=lambda L: (abs(L - e), L))


def snapper_for(group_dir: Path, mode: str):
    """(fn, name) -- `code` picks the rule the real backend implements."""
    if mode == "nearest":
        return snap_nearest, "nearest"
    if mode == "floor":
        # One uniform rule for every quantization group. This differs from
        # `code` on ResNet, whose GPU dispatcher ceils, so for ResNet it is an
        # analysis convention rather than a description of the backend.
        return snap_floor, "floor(all)"
    is_resnet = "resnet" in str(group_dir).lower()
    return (snap_ceil, "ceil(resnet)") if is_resnet else (snap_floor, "floor(llm/t5)")


# ── grid loading ───────────────────────────────────────────────────────────
def load_grid(group_dir: Path) -> Tuple[Dict[str, float], dict]:
    """Every evaluated lattice point for a group, plus its metadata."""
    grid: Dict[str, float] = {}
    meta: dict = {}
    for f in sorted(group_dir.glob("*.json")):
        name = f.name
        if not (name.endswith("_rays.json") or name.endswith("_summary.json")
                or name.endswith("_meta.json") or ".partial.json" in name):
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001 -- a shard mid-write is expected
            continue
        if isinstance(d, dict):
            for k in ("model", "dataset", "metric", "strategy", "n_cuts",
                      "levels", "tol", "baseline", "cuts"):
                if k in d and k not in meta:
                    meta[k] = d[k]
            for e in d.get("profile") or []:
                grid[_key(e["eta"])] = e["y"]
            for key in ("rays", "phase2_rays", "fill_rays"):
                for r in d.get(key) or []:
                    for eta, y in zip(r.get("eta", []), r.get("y", [])):
                        grid[_key(eta)] = y
            # bare grid checkpoint: {"[...]": y}
            if name.endswith(".partial.json") and "grid" in name:
                for k2, v in d.items():
                    if isinstance(v, (int, float)):
                        grid[k2] = float(v)
    return grid, meta


# ── ray sampling (mirrors mc_concavity.sample_random_ray) ──────────────────
def sample_rays(rng, n: int, n_rays: int, lo: float, n_points: int,
                min_seg: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    out = []
    while len(out) < n_rays:
        start = lo + (1.0 - lo) * rng.random(n)
        end = lo + (1.0 - lo) * rng.random(n)
        if np.linalg.norm(end - start) >= min_seg or (1.0 - lo) < min_seg:
            out.append((start, end))
    return out


def build_rays(grid, meta, group_dir, *, n_rays, n_points, floors, snap_mode,
               seed, min_seg, tol):
    snap, snap_name = snapper_for(group_dir, snap_mode)
    ladder = sorted(float(x) for x in (meta.get("levels") or LADDER))
    n = int(meta["n_cuts"])
    rng = np.random.default_rng(seed)
    rays: List[dict] = []
    missing: Dict[str, int] = {}
    n_dropped = 0
    for c in floors:
        for start, end in sample_rays(rng, n, n_rays, c, n_points, min_seg):
            ts = np.linspace(0.0, 1.0, n_points)
            etas_raw = [(1.0 - t) * start + t * end for t in ts]
            etas = [[snap(float(v), ladder) for v in e] for e in etas_raw]
            ys, ok = [], True
            for e in etas:
                k = _key(e)
                if k not in grid:
                    missing[k] = missing.get(k, 0) + 1
                    ok = False
                else:
                    ys.append(float(grid[k]))
            if not ok:
                n_dropped += 1
                continue
            good, worst, n_tri = chord_concave(list(ts), ys, tol)
            # REQUESTED vs DELIVERED eta. The eta-floor constraint (eta_i >= c)
            # is a constraint on the compression ratio you ASK for; snapping to a
            # rung is what the backend then delivers. So `eta` holds the
            # requested (raw) vector -- that is what jensen's in_floor_indices
            # must filter on -- and `eta_snapped` holds the lattice coords the
            # utility was actually read from.
            #
            # This lets the floor sweep run to 0.7 under floor-snap: for
            # c > 0.5 every coordinate snaps to 0.5, so the ray is flat and
            # trivially concave rather than discarded for holding no in-floor
            # points. It also matches topk and llmint8, whose eta is continuous
            # and where requested equals delivered by construction.
            rays.append({
                "id": len(rays), "floor": float(c),
                "start": [float(v) for v in etas_raw[0]],
                "end": [float(v) for v in etas_raw[-1]],
                "t": [float(t) for t in ts],
                "eta": [[float(v) for v in e] for e in etas_raw],
                "eta_snapped": etas, "y": ys,
                "concave": bool(good), "worst_d2": float(worst),
                "violations": int(0 if good else 1), "n_triples": n_tri,
            })
    return rays, missing, n_dropped, snap_name


def process(group_dir: Path, args) -> Optional[dict]:
    grid, meta = load_grid(group_dir)
    if not grid or "n_cuts" not in meta:
        print(f"  skip {group_dir} (no grid/metadata)")
        return None
    tol = args.tol if args.tol is not None else float(
        meta.get("tol", 0.02 if meta.get("metric") == "accuracy" else 1e-3))
    floors = [round(x, 4) for x in np.arange(
        args.floor_min, args.floor_max + 1e-9, args.floor_step)]
    rays, missing, n_dropped, snap_name = build_rays(
        grid, meta, group_dir, n_rays=args.n_rays, n_points=args.n_points,
        floors=floors, snap_mode=args.snap, seed=args.seed,
        min_seg=args.min_seg, tol=tol)
    n_conc = sum(1 for r in rays if r["concave"])
    conc = (n_conc / len(rays)) if rays else None

    # DEGENERACY SPLIT. Snapping maps many of a ray's samples onto the same
    # lattice point, and a piecewise-constant ray passes the chord test
    # trivially -- the same way dead zones did. A ray needs >= 3 distinct
    # utility values before its curvature means anything, so report that
    # subset separately; it is the number worth quoting.
    for r in rays:
        r["n_distinct_y"] = len({round(y, 12) for y in r["y"]})
        # Distinct LATTICE points touched -> the snapped vector, not the request.
        r["n_distinct_pts"] = len({tuple(e) for e in r["eta_snapped"]})
    nondeg = [r for r in rays if r["n_distinct_y"] >= 3]
    conc_nd = (sum(1 for r in nondeg if r["concave"]) / len(nondeg)
               ) if nondeg else None
    frac_flat = (sum(1 for r in rays if r["n_distinct_y"] == 1) / len(rays)
                 ) if rays else None

    label = f"{meta.get('model')} {meta.get('dataset')}/{meta.get('metric')} " \
            f"cuts{meta['n_cuts']}"
    print(f"  {label}")
    print(f"     grid {len(grid)} pts | snap={snap_name} | rays {len(rays)} "
          f"kept, {n_dropped} dropped (missing lattice pts: {len(missing)})")
    print(f"     concavity all={f'{conc:.3f}' if conc is not None else 'n/a'}"
          f"  NON-DEGENERATE={f'{conc_nd:.3f}' if conc_nd is not None else 'n/a'}"
          f" ({len(nondeg)}/{len(rays)} rays)"
          f"  flat-ray frac={f'{frac_flat:.2f}' if frac_flat is not None else 'n/a'}")

    stem = f"{str(meta.get('model')).replace('/', '__')}__" \
           f"{meta.get('dataset')}__{meta.get('metric')}__quantization__" \
           f"cuts{meta['n_cuts']}"
    out_dir = Path(args.out_root) / group_dir.relative_to(args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {**{k: v for k, v in meta.items() if k != "profile"},
               "strategy": "quantization",
               "ray_sampling": "random_box", "snap": snap_name,
               "snap_mode": args.snap, "n_points": args.n_points,
               "n_rays_per_floor": args.n_rays, "floors": floors,
               "seed": args.seed, "tol": tol,
               "box_ray_concavity": conc,
               "box_ray_concavity_nondegenerate": conc_nd,
               "n_rays_nondegenerate": len(nondeg),
               "flat_ray_fraction": frac_flat,
               "n_rays": len(rays),
               "n_rays_dropped_missing": n_dropped,
               "n_missing_lattice_points": len(missing),
               "rays": rays, "phase2_rays": [], "fill_rays": []}
    (out_dir / f"{stem}_rays.json").write_text(json.dumps(payload, indent=2))
    if missing:
        (out_dir / f"{stem}_missing_points.json").write_text(json.dumps(
            {"n_missing": len(missing),
             "points": [json.loads(k) for k in sorted(missing)],
             "demand": {k: v for k, v in sorted(
                 missing.items(), key=lambda kv: -kv[1])}}, indent=2))
        print(f"     -> wrote {len(missing)} missing lattice points for filling")
    return {"label": label, "concavity": conc, "n_rays": len(rays),
            "missing": len(missing), "dir": str(out_dir)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO_ROOT / "outputs" / "mc_concavity"))
    p.add_argument("--out_root",
                   default=str(REPO_ROOT / "outputs" / "quant_boxrays"))
    p.add_argument("--group_dir", default="")
    p.add_argument("--all", action="store_true",
                   help="process every */quantization/cuts* group under --root")
    p.add_argument("--n_rays", type=int, default=200,
                   help="rays sampled PER eta-floor")
    p.add_argument("--n_points", type=int, default=7,
                   help="samples along each ray (7 matches the topk/llmint8 arms)")
    p.add_argument("--floor_min", type=float, default=0.0)
    p.add_argument("--floor_max", type=float, default=0.7)
    p.add_argument("--floor_step", type=float, default=0.05)
    p.add_argument("--snap", default="floor", choices=["code", "floor", "nearest"],
                   help="floor (default): one uniform rule for every group. "
                        "every backend. code: per-backend rule as implemented "
                        "(floor for LLM/T5, ceil for ResNet). nearest: closest "
                        "level.")
    p.add_argument("--min_seg", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root)
    args.root = root
    if args.group_dir:
        dirs = [Path(args.group_dir)]
    else:
        dirs = sorted({f.parent for f in root.rglob("*_rays.json")
                       if "quantization" in str(f.parent)})
    print(f"box-ray resampling ({len(dirs)} groups, snap={args.snap})")
    results = [r for r in (process(d, args) for d in dirs) if r]
    print(f"\n{len(results)} groups written under {args.out_root}")


if __name__ == "__main__":
    main()
