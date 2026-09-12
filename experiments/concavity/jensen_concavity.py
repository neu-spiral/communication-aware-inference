"""
Jensen-inequality concavity estimate vs a scalar eta-floor (model-free).

This is a POST-HOC re-analysis of the rays already computed by
``experiments/concavity/mc_concavity.py`` (the ``*_rays.json`` profiles). It does NOT load
any model — it only reads the saved (eta-vector, metric) samples along each ray.

Idea
----
A function is concave iff it satisfies Jensen's inequality: for a convex
combination of inputs, the function of the mean is >= the mean of the function.
Along a single ray we already evaluated the metric ``y`` at ``n_points``
equally-spaced eta-vectors. We test concavity by SAMPLING triples of those
points: for indices i < j < k (ordered by the ray parameter t), the middle point
j is a convex combination of i and k,

    t_j = lam * t_i + (1 - lam) * t_k,   lam = (t_k - t_j) / (t_k - t_i),

so Jensen's inequality for a concave metric requires

    y_j  >=  lam * y_i + (1 - lam) * y_k      (the chord through i, k).

Each sampled triple is one Jensen trial; ``concavity`` is the fraction of trials
that pass. Drawing many triples per ray, across all rays, gives a Monte-Carlo
estimate of how concave the metric is.

Scalar eta-floor
----------------
We restrict the test to the sub-cube where EVERY cut-point's eta is at least a
single scalar floor ``c`` (the same c on all cut points). Because each ray is a
line segment, the grid points with ``min_coord(eta) >= c`` form a contiguous
block, so the sampled triples stay valid. Sweeping ``c`` from 0 -> ~1 and
plotting concavity(c) shows how excluding the aggressive-compression (low-eta)
regime restores concavity. x-axis = eta floor, y-axis = concavity, one curve per
strategy, one plot per (model, task).

Usage
-----
  python experiments/concavity/jensen_concavity.py \
      --root outputs/mc_concavity --out_dir outputs/mc_concavity/jensen
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent      # experiments/concavity
REPO_ROOT = HERE.parents[1]                 # repo root


# ──────────────────────────────────────────────────────────────────────────
# Discovery + metadata
# ──────────────────────────────────────────────────────────────────────────

# Strategy keys parsed out of stored ray files. llmint8_reserve and
# llmint8_adjacent are aliases for llmint8 that appear in stored rays.
_KNOWN_STRATEGIES = (
    "topk_per_token", "llmint8_reserve", "llmint8_adjacent", "llmint8",
    "magnitude", "random", "quantization",
)
# Short labels used in merged-file names and for legends.
_STRATEGY_ALIASES = {"topk": "topk_per_token", "reserve": "llmint8_reserve"}

# DISPLAY-ONLY dataset renames. The ResNet results are keyed dataset="imagenet"
# throughout (paths, file stems, metadata) so that resnet_entries() and the
# output-tree globs line up, but the data is CIFAR-10 and always was. Correct the
# label everywhere a human reads it; NEVER rewrite the key itself.
_DATASET_DISPLAY = {"imagenet": "CIFAR-10"}

# THE y-axis wording, for every concavity figure in every script (imported by
# plot_concavity_grid.py and plot_concavity_per_method.py so the three families
# can never drift apart). The quantity is triples-passed / triples-drawn, i.e. a
# rate in [0,1] — any relabelling has to keep that reading; a count would be off
# by the number of triples per ray.
Y_AXIS_LABEL = "Jensen inequality pass rate"

# Shared y-range for every concavity figure. Measured
# curves live in [0.49, 1.0] (per-group sweeps do
# not go below 0.85), so a 0-based axis spent most of its height on empty space
# and flattened the differences between curves. 1.02, not 1.0, keeps the markers
# of the many curves that saturate at exactly 1 inside the axes. Every script
# WARNS instead of silently cropping a curve that falls below the floor — a curve
# leaving the axes must not be mistaken for a curve that ran out of usable rays.
Y_LIMITS = (0.4, 1.02)

# Every concavity figure ships as BOTH raster and vector (user request
# 2026-08-13). The PNG is for quick viewing; the PDF is what \includegraphics
# pulls, so lines and labels stay sharp at any zoom instead of being resampled
# into the page. Order matters only in that .pdf must exist for the paper build.
FIGURE_FORMATS = ("png", "pdf")


def save_figure(fig, out_path, dpi: int = 150) -> list:
    """Write `fig` once per FIGURE_FORMATS, swapping `out_path`'s extension.

    Callers pass the PNG path they already computed, so filenames stay in one
    place. pdf.fonttype 42 embeds TrueType instead of Type 3 outlines, which some
    venues reject outright at submission."""
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    out_path = Path(out_path)
    written = []
    for ext in FIGURE_FORMATS:
        p = out_path.with_suffix(f".{ext}")
        # bbox_inches="tight" is required, not cosmetic: these figures carry a
        # figure-level legend outside the axes, which a default bbox crops.
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    return written


def display_dataset(dataset: str) -> str:
    """Human-facing dataset name for titles/legends (see _DATASET_DISPLAY)."""
    return _DATASET_DISPLAY.get(dataset, dataset)


def _infer_meta(path: Path, data: dict) -> dict:
    """Fill model/dataset/metric/strategy/n_cuts from json fields, falling back
    to the file name / path (the merged MMLU files carry no metadata; partial
    checkpoints are bare lists with no metadata at all)."""
    if not isinstance(data, dict):
        data = {}
    meta = {
        "model": data.get("model"),
        "dataset": data.get("dataset"),
        "metric": data.get("metric"),
        "strategy": data.get("strategy"),
        "n_cuts": data.get("n_cuts"),
    }
    name = path.name
    parts = path.parts

    # Bare partial checkpoints (killed-run rescue lists) carry no embedded
    # metadata, so recover it from the run-file naming convention
    #   <model>__<dataset>__<metric>__<strategy>__cuts<N>...   (model may itself
    # contain '__' -> '/'). Parse BACKWARD from the cuts<N> token so a
    # multi-token model slug can't be mistaken for dataset/metric.
    toks = name.split("__")
    cut_i = next((i for i, t in enumerate(toks)
                  if re.match(r"cuts\d+", t)), None)
    if cut_i is not None and cut_i >= 4:
        if meta["strategy"] is None:
            meta["strategy"] = toks[cut_i - 1]
        if meta["metric"] is None:
            meta["metric"] = toks[cut_i - 2]
        if meta["dataset"] is None:
            meta["dataset"] = toks[cut_i - 3]
        if meta["model"] is None:
            meta["model"] = "/".join(toks[:cut_i - 3])

    if meta["strategy"] is None:
        for s in _KNOWN_STRATEGIES:
            if s in name or s in parts:
                meta["strategy"] = s
                break
        else:
            for alias, full in _STRATEGY_ALIASES.items():
                if name.startswith(alias) or f"{alias}_merged" in name:
                    meta["strategy"] = full
                    break

    # Path looks like .../<model_slug>/<dataset>_<metric>/<strategy>/<cutsN>/file
    if meta["model"] is None or meta["dataset"] is None or meta["metric"] is None:
        for i, p in enumerate(parts):
            if p == "mc_concavity" and i + 2 < len(parts):
                if meta["model"] is None:
                    meta["model"] = parts[i + 1].replace("__", "/").replace("_", "/", 0)
                tk = parts[i + 2].split("_")
                if len(tk) >= 2:
                    if meta["dataset"] is None:
                        meta["dataset"] = tk[0]
                    if meta["metric"] is None:
                        meta["metric"] = tk[-1]
                break

    if meta["model"] is None:
        m = re.match(r"([^_]+(?:__[^_]+)*)__", name)
        if m:
            meta["model"] = m.group(1).replace("__", "/")
    # Normalise model slug "google_gemma-7b" -> "google/gemma-7b".
    if meta["model"] and "/" not in meta["model"] and "__" not in meta["model"]:
        meta["model"] = meta["model"].replace("_", "/", 1)
    meta["model"] = (meta["model"] or "?").replace("__", "/")

    if meta["n_cuts"] is None:
        meta["n_cuts"] = data.get("n_cuts")
    if meta["n_cuts"] is None:  # bare partial lists carry no n_cuts -> parse name
        m = re.search(r"cuts(\d+)", "/".join(parts))
        if m:
            meta["n_cuts"] = int(m.group(1))
    return meta


def discover_ray_files(root: Path) -> List[Path]:
    """All ray-bearing files: completed *_rays.json plus *.partial.json
    checkpoints (the latter rescue rays from runs killed mid-phase). Overlap is
    handled by content-hash dedup at load time."""
    return sorted(
        p for p in root.rglob("*.json")
        if p.name.endswith("_rays.json") or p.name.endswith(".partial.json")
    )


# ResNet (directional-profile) backend: a *_concavity_results.json produced by
# analyze_concavity_directions_only.py stores, per link-count, a set of random
# rays sampled from a start point to a shared endpoint (default [1,...,1]) as
# eta_vec(s) = start + s * (ray_end - start). We reconstruct the eta-vector at
# each grid point so the SAME scalar-floor Jensen test applies unchanged.
_RESNET_STRAT = {"topk": "topk", "quantization": "quantization",
                 "llmint8": "llmint8"}


def resnet_entries(root: Path) -> List[Tuple[dict, List[dict], str]]:
    """Convert every resnet concavity_results.json under ``root`` into
    (meta, rays, source) tuples in the same ray format as the LLM runs."""
    entries: List[Tuple[dict, List[dict], str]] = []
    for f in sorted(root.rglob("concavity_results.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f}: {e}")
            continue
        if d.get("backend") != "resnet":
            continue
        dp = d.get("directional_profiles") or {}
        s_grid = dp.get("s_grid")
        starts = dp.get("ray_starts")
        profiles = dp.get("ray_profiles")
        if not (s_grid and starts and profiles):
            continue
        n_links = int(d.get("num_links", len(starts[0])))
        ray_end = dp.get("ray_end") or [1.0] * n_links
        rays = []
        for start, y in zip(starts, profiles):
            eta = [[start[l] + s * (ray_end[l] - start[l]) for l in range(n_links)]
                   for s in s_grid]
            rays.append({"t": list(s_grid), "y": list(y), "eta": eta,
                         "start": list(start), "end": list(ray_end)})
        strat = _RESNET_STRAT.get(d.get("compressor"), d.get("compressor") or "?")
        meta = {"model": "resnet", "dataset": "imagenet", "metric": "accuracy",
                "strategy": strat, "n_cuts": n_links}
        entries.append((meta, rays, str(f)))
    return entries


def _rays_from(data) -> List[dict]:
    """Ray dicts from either a run file (dict with rays/phase2_rays) or a bare
    partial checkpoint (a plain list of ray dicts)."""
    if isinstance(data, list):
        return list(data)
    return (list(data.get("rays", [])) + list(data.get("phase2_rays", []))
            + list(data.get("fill_rays", [])))


def _ray_hash(r: dict) -> str:
    """Content key for de-duplicating a ray across files (seed subsets, merged
    files, and partial checkpoints all re-store identical ray dicts)."""
    start = tuple(round(float(x), 6) for x in r.get("start", []))
    end = tuple(round(float(x), 6) for x in r.get("end", []))
    y = tuple(round(float(x), 6) for x in r.get("y", []))
    return json.dumps([start, end, y])


def parse_excludes(specs: List[str]) -> List[Tuple[str, Optional[str]]]:
    """Parse --exclude specs of the form ``MODEL_SUBSTR:METRIC`` (metric
    optional). Returns a list of (model_substr, metric_or_None) filters. E.g.
    ``flan-t5-base:ppl_score`` drops only that model's ppl_score curves;
    ``flan-t5-base`` (no colon) drops every metric for that model."""
    out: List[Tuple[str, Optional[str]]] = []
    for s in specs or []:
        s = s.strip()
        if not s:
            continue
        if ":" in s:
            m, met = s.split(":", 1)
            out.append((m.strip(), met.strip() or None))
        else:
            out.append((s, None))
    return out


def is_excluded(meta: dict, excludes: List[Tuple[str, Optional[str]]]) -> bool:
    """True if meta matches any (model_substr, metric) exclusion filter."""
    for msub, met in excludes:
        model_ok = (not msub) or (msub in str(meta.get("model", "")))
        metric_ok = (met is None) or (meta.get("metric") == met)
        if model_ok and metric_ok:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Jensen inequality test
# ──────────────────────────────────────────────────────────────────────────

def in_floor_indices(eta_vecs: List[List[float]], c: float) -> List[int]:
    """Indices of grid points whose eta-vector is >= c on EVERY cut point."""
    return [j for j, e in enumerate(eta_vecs) if min(e) >= c]


def jensen_pass(t: np.ndarray, y: np.ndarray, i: int, j: int, k: int,
                tol: float) -> bool:
    """True if middle point j sits on/above the chord through i, k (concave)."""
    span = t[k] - t[i]
    if span <= 0:
        return True
    lam = (t[k] - t[j]) / span          # weight on point i
    # Stable lerp, NOT lam*y[i] + (1-lam)*y[k]: on a FLAT block (y[i] == y[k])
    # the two-term form rounds to 1 ULP ABOVE y[i] for many values of lam, so at
    # tol=0 a perfectly constant ray was scored non-concave. This form is exact
    # when y[i] == y[k] (the delta is exactly 0), and identical to within a ULP
    # otherwise. Flat blocks are the common case here: under floor-snap 42-53%
    # of quantization rays are entirely flat, and EVERY in-floor block is flat
    # once c > 0.5, where the top rung is the only reachable level.
    chord = y[k] + lam * (y[i] - y[k])
    return bool(y[j] >= chord - tol)


def ray_jensen_trials(ray: dict, c: float, *, rng: np.random.Generator,
                      samples_per_ray: int, tol: float) -> Tuple[int, int]:
    """Sample triples from one ray's in-floor points. Returns (passes, trials)."""
    idx = in_floor_indices(ray["eta"], c)
    if len(idx) < 3:
        return 0, 0
    t = np.asarray(ray["t"], dtype=np.float64)
    y = np.asarray(ray["y"], dtype=np.float64)
    idx = np.asarray(idx)

    # If the block is small, enumerate all triples; else sample without dups.
    from itertools import combinations
    all_triples = list(combinations(range(len(idx)), 3))
    if len(all_triples) <= samples_per_ray:
        chosen = all_triples
    else:
        sel = rng.choice(len(all_triples), size=samples_per_ray, replace=False)
        chosen = [all_triples[s] for s in sel]

    passes = 0
    for a, b, d in chosen:
        i, j, k = int(idx[a]), int(idx[b]), int(idx[d])  # already t-ordered
        if jensen_pass(t, y, i, j, k, tol):
            passes += 1
    return passes, len(chosen)


def floor_sweep(rays: List[dict], *, floors: np.ndarray, rng: np.random.Generator,
                samples_per_ray: int, tol: float) -> List[dict]:
    out = []
    for c in floors:
        passes = trials = usable = 0
        for r in rays:
            p, n = ray_jensen_trials(r, float(c), rng=rng,
                                     samples_per_ray=samples_per_ray, tol=tol)
            if n > 0:
                usable += 1
            passes += p
            trials += n
        out.append({
            "floor": float(c),
            "concavity": (passes / trials) if trials > 0 else None,
            "passes": passes,
            "trials": trials,
            "usable_rays": usable,
        })
    return out


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────

_STRAT_STYLE = {
    "topk_per_token": ("tab:blue", "top-k per token"),
    # Aliases for llmint8 found in stored rays.
    "llmint8_reserve": ("tab:orange", "llmint8"),
    "llmint8_adjacent": ("tab:green", "llmint8 (adjacent)"),
    # ResNet compressors (directional-profile backend, see resnet_entries).
    "topk": ("tab:blue", "top-k"),
    # Distinct from llmint8_reserve's orange: quantization shares an axis with
    # the LLM strategies on the gemma/llama groups, so it needs its own colour.
    "quantization": ("tab:red", "quantization"),
    "llmint8": ("tab:green", "llmint8"),
}


def plot_group(group_key: Tuple[str, str, str, int], curves: Dict[str, List[dict]],
               out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, dataset, metric, n_cuts = group_key
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for strat, sweep in sorted(curves.items()):
        color, lbl = _STRAT_STYLE.get(strat, (None, strat))
        xs = [s["floor"] for s in sweep if s["concavity"] is not None]
        ys = [s["concavity"] for s in sweep if s["concavity"] is not None]
        ax.plot(xs, ys, marker="o", color=color, label=lbl, linewidth=1.6)
        # Shaded ±1 binomial SE band.
        se_xs, se_lo, se_hi = [], [], []
        for s in sweep:
            p = s["concavity"]
            t = s.get("trials", 0)
            if p is not None and t > 0:
                se = np.sqrt(p * (1.0 - p) / t)
                se_xs.append(s["floor"])
                se_lo.append(max(0.0, p - se))
                se_hi.append(min(1.0, p + se))
        if se_xs:
            ax.fill_between(se_xs, se_lo, se_hi, color=color, alpha=0.2)
        # Mark where the usable-ray count drops below a quarter of the rays
        # (the estimate gets noisy past here).
        for s in sweep:
            if s["concavity"] is not None and s["usable_rays"] < 8:
                ax.plot(s["floor"], s["concavity"], marker="x", color=color,
                        markersize=9, markeredgewidth=2)
        if ys and min(ys) < Y_LIMITS[0]:
            print(f"  ! {model}|{dataset}/{metric}|cuts{n_cuts}|{strat} dips to "
                  f"{min(ys):.3f}, below the y-axis floor {Y_LIMITS[0]} — it "
                  f"will be clipped")
    ax.set_xlabel("eta floor  c   (same scalar on all cut points;  eta_i >= c)",
                  fontsize=13)
    ax.set_ylabel(f"{Y_AXIS_LABEL}\n±1 σ binomial SE shaded", fontsize=13)
    ax.set_ylim(*Y_LIMITS)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.4)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=12)
    # No title: model/dataset/metric/cuts are all in
    # the output FILENAME and belong in the paper caption. display_dataset() is
    # still the only correct way to render `dataset` if a title is ever wanted
    # back here — ResNet must read as CIFAR-10.
    leg = ax.legend(title="strategy", fontsize=12)
    leg.get_title().set_fontsize(12)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def default_tol(metric: Optional[str], override: Optional[float]) -> float:
    if override is not None:
        return override
    # Accuracy is quantised/noisy (steps ~1/n_questions); perplexity is smooth.
    return 0.02 if metric == "accuracy" else 1e-3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT / "outputs" / "mc_concavity"),
                    help="Directory tree to scan for *_rays.json.")
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "mc_concavity" / "jensen"))
    ap.add_argument("--resnet_root",
                    default=str(REPO_ROOT / "outputs" / "concavity_resnet"),
                    help="Directory tree scanned for ResNet directional-profile "
                         "concavity_results.json (folded into the same plots).")
    ap.add_argument("--samples_per_ray", type=int, default=200,
                    help="Jensen triples sampled per ray per floor (capped at C(m,3)).")
    ap.add_argument("--floor_min", type=float, default=0.0)
    ap.add_argument("--floor_max", type=float, default=0.7,
                    help="Highest eta floor to plot. Defaults to 0.7: the "
                         "adaptive fill guarantees >=min_box_samples usable rays "
                         "only up to floor 0.7, so the sweep is unreliable above it.")
    ap.add_argument("--floor_step", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=None,
                    help="Chord tolerance (default: 0.02 accuracy / 1e-3 perplexity).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Drop matching curves, spec 'MODEL_SUBSTR:METRIC' "
                         "(metric optional). E.g. flan-t5-base:ppl_score.")
    args = ap.parse_args()

    excludes = parse_excludes(args.exclude)
    if excludes:
        print(f"Excluding curves: {excludes}")
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    floors = np.round(np.arange(args.floor_min, args.floor_max + 1e-9, args.floor_step), 3)

    files = discover_ray_files(root)
    print(f"Found {len(files)} ray files (completed + partial) under {root}")

    # MERGE all rays for each (model, dataset, metric, n_cuts, strategy) key,
    # de-duplicated by content hash. This accumulates every seed and every phase
    # (Phase-1 + Phase-2) across separate run directories, harvests partial
    # checkpoints from killed runs, and is immune to the merged-MMLU files
    # double-counting their own single-seed subsets (identical ray dicts collapse
    # to one). New runs written to fresh dirs are picked up automatically.
    best: Dict[Tuple, dict] = {}
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            continue
        rays = _rays_from(data)
        if not rays:
            continue
        meta = _infer_meta(f, data)
        if meta["n_cuts"] is None:
            print(f"  skip {f.name}: cannot determine n_cuts")
            continue
        if is_excluded(meta, excludes):
            continue
        n_cuts = int(meta["n_cuts"])
        key = (meta["model"], meta["dataset"], meta["metric"],
               n_cuts, meta["strategy"])
        # Drop rays whose eta dimensionality != n_cuts. Some merged files are
        # mislabeled (e.g. a cuts7_merged_rays.json that declares n_cuts=7 but
        # stores 4-dim rays); counting them silently corrupts the scalar-floor
        # min(eta) test and the whole concavity curve for that (task, cuts).
        good = [r for r in rays if len(r.get("start", [])) == n_cuts]
        if len(good) != len(rays):
            print(f"  {f.name}: dropped {len(rays) - len(good)} rays "
                  f"with dim != {n_cuts}")
        entry = best.setdefault(
            key, {"meta": meta, "rays": [], "seen": set(), "files": []})
        added = 0
        for r in good:
            h = _ray_hash(r)
            if h in entry["seen"]:
                continue
            entry["seen"].add(h)
            entry["rays"].append(r)
            added += 1
        if added:
            entry["files"].append(str(f))

    # Fold in ResNet directional-profile runs (different backend, same ray format).
    resnet_root = Path(args.resnet_root)
    if resnet_root.exists():
        r_entries = resnet_entries(resnet_root)
        print(f"Found {len(r_entries)} resnet concavity files under {resnet_root}")
        for meta, rays, src in r_entries:
            if is_excluded(meta, excludes):
                continue
            key = (meta["model"], meta["dataset"], meta["metric"],
                   int(meta["n_cuts"]), meta["strategy"])
            entry = best.setdefault(
                key, {"meta": meta, "rays": [], "seen": set(), "files": []})
            added = 0
            for r in rays:
                h = _ray_hash(r)
                if h in entry["seen"]:
                    continue
                entry["seen"].add(h)
                entry["rays"].append(r)
                added += 1
            if added:
                entry["files"].append(src)

    # Group by (model, dataset, metric, n_cuts) -> one plot, curve per strategy.
    groups: Dict[Tuple, Dict[str, List[dict]]] = {}
    report: Dict[str, dict] = {}
    for (model, dataset, metric, n_cuts, strat), entry in best.items():
        rng = np.random.default_rng(args.seed)
        tol = default_tol(metric, args.tol)
        sweep = floor_sweep(entry["rays"], floors=floors, rng=rng,
                            samples_per_ray=args.samples_per_ray, tol=tol)
        gkey = (model, dataset, metric, n_cuts)
        groups.setdefault(gkey, {})[strat] = sweep
        base = next((s["concavity"] for s in sweep if s["floor"] == 0.0), None)
        rkey = f"{model}|{dataset}/{metric}|cuts{n_cuts}|{strat}"
        report[rkey] = {
            "model": model, "dataset": dataset, "metric": metric,
            "n_cuts": n_cuts, "strategy": strat, "n_rays": len(entry["rays"]),
            "tol": tol, "sources": entry["files"],
            "concavity_at_floor0": base, "sweep": sweep,
        }
        b = f"{base:.3f}" if base is not None else "n/a"
        print(f"  {rkey:70s} rays={len(entry['rays']):4d} conc@0={b}")

    # Write the numeric results FIRST, so a plotting-backend failure (e.g. a
    # broken matplotlib/numpy combo in the env) never costs us the analysis.
    (out_dir / "jensen_concavity.json").write_text(json.dumps({
        "floors": floors.tolist(),
        "samples_per_ray": args.samples_per_ray,
        "seed": args.seed,
        "results": report,
    }, indent=2))
    print(f"Wrote {out_dir/'jensen_concavity.json'}  ({len(report)} curves)")

    # Write plots. Non-fatal: the JSON above is already saved.
    n_plots = 0
    for gkey, curves in sorted(groups.items()):
        model, dataset, metric, n_cuts = gkey
        slug = (f"{model.replace('/', '_')}__{dataset}__{metric}__cuts{n_cuts}"
                "__jensen_floor.png")
        try:
            plot_group(gkey, curves, out_dir / slug)
            n_plots += 1
            print(f"  wrote plot {slug}")
        except Exception as e:  # noqa: BLE001
            print(f"  skip plot {slug}: {e}")
    print(f"Wrote {n_plots} plots to {out_dir}")


if __name__ == "__main__":
    main()
