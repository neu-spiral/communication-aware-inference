"""
Union the per-shard checkpoints of a sharded quantization enumeration and emit
the standard ``*_rays.json`` / ``*_summary.json`` pair.

``enumerate_quant_concavity.py --shard_count N --shard_index I`` splits the
lattice round-robin across N workers, each writing its own
``<stem>_grid.shardIofN.partial.json`` (plus a shared read-only
``<stem>_grid.partial.json`` from any earlier serial run). This script unions
them and does the ray assembly the shards cannot do individually -- an axis line
spans points owned by different shards.

Two modes, chosen automatically:

* **complete** -- every lattice point is present: emit the full axis-ray set,
  identical to what an unsharded run would have written.
* **salvage** -- the grid is still partial: emit only the axis lines whose L
  points are ALL present. This is what makes a killed/queued campaign usable at
  any moment, but note the surviving lines are NOT a uniform sample of the grid
  (in ``itertools.product`` order the fastest-varying axis completes first), so
  a salvage concavity number is biased toward that axis and must be labelled as
  partial -- ``summary["complete"]`` records which mode produced the file.

Usage:
  python experiments/merge_quant_shards.py --out_dir <dir>          # auto stem
  python experiments/merge_quant_shards.py --out_dir <dir> --stem <stem>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent      # experiments/concavity
REPO_ROOT = HERE.parents[1]                 # repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from enumerate_quant_concavity import (  # noqa: E402
    _key, _make_vec, chord_concave, sampled_specs_and_points,
)


def load_grid(out_dir: Path, stem: str) -> Dict[str, float]:
    """Union the serial checkpoint and every shard checkpoint."""
    grid: Dict[str, float] = {}
    srcs = [out_dir / f"{stem}_grid.partial.json"]
    srcs += sorted(out_dir.glob(f"{stem}_grid.shard*.partial.json"))
    for src in srcs:
        if not src.exists():
            continue
        try:
            d = json.loads(src.read_text())
        except Exception as e:  # noqa: BLE001 -- a shard mid-write is expected
            print(f"  ! skipping unreadable {src.name}: {e}")
            continue
        new = sum(1 for k in d if k not in grid)
        grid.update(d)
        print(f"  + {src.name}: {len(d)} points ({new} new)")
    return grid


def build_rays_available(levels: List[float], grid: Dict[str, float], n: int,
                         tol: float) -> tuple:
    """Axis rays for every line fully covered by ``grid``.

    Mirrors ``enumerate_quant_concavity.build_axis_rays`` but skips any line
    with a missing point instead of raising KeyError.
    """
    lo, hi = min(levels), max(levels)
    span = (hi - lo) if hi > lo else 1.0
    ordered = sorted(levels)
    rays: List[dict] = []
    rid = n_total = 0
    for k in range(n):
        rest = [a for a in range(n) if a != k]
        for combo in itertools.product(ordered, repeat=len(rest)):
            n_total += 1
            fixed = dict(zip(rest, combo))
            vecs = []
            for lvl in ordered:
                vec = [float(lvl) if a == k else float(fixed[a]) for a in range(n)]
                vecs.append(vec)
            if any(_key(v) not in grid for v in vecs):
                continue
            ts = [(float(lvl) - lo) / span for lvl in ordered]
            ys = [float(grid[_key(v)]) for v in vecs]
            ok, worst, n_tri = chord_concave(ts, ys, tol)
            rays.append({
                "id": rid, "axis": k,
                "start": list(vecs[0]), "end": list(vecs[-1]),
                "t": ts, "eta": [list(v) for v in vecs], "y": ys,
                "concave": bool(ok), "worst_d2": float(worst),
                "violations": int(0 if ok else 1), "n_triples": n_tri,
            })
            rid += 1
    return rays, n_total


def build_rays_for_specs(levels: List[float], grid: Dict[str, float], n: int,
                         specs, tol: float) -> tuple:
    """Rays for the SAMPLED axis lines that ``grid`` fully covers."""
    lo, hi = min(levels), max(levels)
    span = (hi - lo) if hi > lo else 1.0
    ordered = sorted(levels)
    rays: List[dict] = []
    rid = 0
    for k, others in specs:
        vecs = [_make_vec(n, k, lvl, others) for lvl in ordered]
        if any(_key(v) not in grid for v in vecs):
            continue
        ts = [(float(lvl) - lo) / span for lvl in ordered]
        ys = [float(grid[_key(v)]) for v in vecs]
        ok, worst, n_tri = chord_concave(ts, ys, tol)
        rays.append({
            "id": rid, "axis": int(k),
            "start": list(vecs[0]), "end": list(vecs[-1]),
            "t": ts, "eta": [list(v) for v in vecs], "y": ys,
            "concave": bool(ok), "worst_d2": float(worst),
            "violations": int(0 if ok else 1), "n_triples": n_tri,
        })
        rid += 1
    return rays, len(specs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--stem", default="",
                   help="output stem; inferred from *_meta.json when omitted")
    p.add_argument("--tol", type=float, default=None,
                   help="override the tol recorded in *_meta.json")
    p.add_argument("--dry_run", action="store_true",
                   help="report coverage/concavity without writing files")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    stem = args.stem
    if not stem:
        metas = sorted(out_dir.glob("*_meta.json"))
        parts = sorted(out_dir.glob("*_grid*.partial.json"))
        if metas:
            stem = metas[0].name[: -len("_meta.json")]
        elif parts:
            stem = parts[0].name.split("_grid")[0]
        else:
            raise SystemExit(f"no *_meta.json or *_grid*.partial.json in {out_dir}")
        print(f"stem = {stem}")

    meta_path = out_dir / f"{stem}_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path} (a shard writes it once it has "
                         f"loaded the model); pass --stem or rerun a shard")
    summary = json.loads(meta_path.read_text())
    levels = [float(x) for x in summary["levels"]]
    n = int(summary["n_cuts"])
    tol = args.tol if args.tol is not None else float(summary["tol"])

    print(f"merging shards for {stem}")
    grid = load_grid(out_dir, stem)
    all_points = [tuple(p) for p in itertools.product(sorted(levels), repeat=n)]
    n_grid = len(all_points)
    have = sum(1 for pt in all_points if _key(list(pt)) in grid)
    full_complete = have == n_grid
    print(f"  grid coverage: {have}/{n_grid} points ({100.0*have/n_grid:.1f}%)")

    # The sampled target: either this run WAS sampled, or it is an exhaustive
    # run whose point order leads with these lines (--priority_lines), in which
    # case the sampled estimate becomes reportable well before the exact value.
    n_sample = int(summary.get("sample_lines") or 0) or \
        int(summary.get("priority_lines") or 0)
    specs = sampled_pts = None
    if n_sample > 0:
        specs, sampled_pts = sampled_specs_and_points(
            sorted(levels), n, n_sample, int(summary.get("seed", 0)))
        s_have = sum(1 for pt in sampled_pts if _key(list(pt)) in grid)
        print(f"  sampled subset ({n_sample} lines): {s_have}/{len(sampled_pts)} "
              f"points ({100.0*s_have/len(sampled_pts):.1f}%)")
        sampled_complete = s_have == len(sampled_pts)
    else:
        sampled_complete = False

    # Prefer the strongest result the grid can currently support.
    if full_complete:
        mode = "exact"
        rays, n_lines_total = build_rays_available(levels, grid, n, tol)
    elif sampled_complete:
        mode = "sampled"
        rays, n_lines_total = build_rays_for_specs(levels, grid, n, specs, tol)
    else:
        mode = "partial"
        rays, n_lines_total = build_rays_available(levels, grid, n, tol)
    complete = mode in ("exact", "sampled")

    n_conc = sum(1 for r in rays if r["concave"])
    conc = (n_conc / len(rays)) if rays else None
    print(f"  mode={mode.upper()}: {len(rays)}/{n_lines_total} axis lines "
          f"covered; concave {n_conc} -> concavity="
          f"{f'{conc:.3f}' if conc is not None else 'n/a'}")

    if args.dry_run:
        print("  (dry run -- nothing written)")
        return
    if not rays:
        print("  no complete axis line yet; nothing to write.")
        return

    summary["exact_concavity"] = conc
    summary["n_axis_rays"] = len(rays)
    summary["n_concave_axis_rays"] = n_conc
    summary["n_axis_lines_total"] = n_lines_total
    summary["n_evaluated"] = have
    summary["complete"] = complete
    summary["mode"] = mode
    # An exact grid is a true enumeration; the sampled deliverable is not, and
    # must not claim to be.
    summary["enumeration"] = (mode == "exact")
    summary["sampled_axis_lines"] = (len(specs) if mode == "sampled" else None)
    summary["merged_from_shards"] = True
    summary["profile"] = [{"eta": json.loads(k), "y": grid[k]}
                          for k in sorted(grid, key=lambda s: json.loads(s))]

    rays_path = out_dir / f"{stem}_rays.json"
    if mode == "partial":
        # Never let a partial land on the canonical name the plot/analysis
        # scripts glob -- they would silently treat it as a finished group.
        rays_path = out_dir / f"{stem}_partial_rays.json"
        print(f"  PARTIAL -> writing {rays_path.name} (not the canonical "
              f"{stem}_rays.json) so downstream globs ignore it")
    rays_path.write_text(json.dumps({
        **{k: v for k, v in summary.items() if k != "profile"},
        "rays": rays, "phase2_rays": [], "fill_rays": [],
    }, indent=2))
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {rays_path.name} + {stem}_summary.json")


if __name__ == "__main__":
    main()
