"""
Grid of Jensen-concavity-vs-eta-floor line charts, one subplot per task, with
a line per (strategy, tol) combination.

Reuses the ray discovery/merge/floor-sweep machinery from jensen_concavity.py
(same de-duped, all-phases-and-seeds ray pool) but sweeps a *range of tol*
values ("the little offset we allow to deviate from concavity") instead of
one fixed tol per metric, so each subplot shows how forgiving the concavity
test needs to be before each strategy's curve saturates.

Usage:
  python experiments/concavity/plot_concavity_grid.py \
      --root outputs/mc_concavity --out outputs/mc_concavity/jensen/concavity_grid.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from jensen_concavity import (
    discover_ray_files, _rays_from, _ray_hash, _infer_meta, floor_sweep,
    resnet_entries, parse_excludes, is_excluded, Y_AXIS_LABEL, Y_LIMITS,
    save_figure, display_dataset,
)

HERE = Path(__file__).resolve().parent      # experiments/concavity
REPO_ROOT = HERE.parents[1]                 # repo root

# Re-exported for plot_concavity_per_method, which already imports its display
# conventions from here. Definition + rationale: jensen_concavity.Y_LIMITS.
_Y_MIN, _Y_MAX = Y_LIMITS

# Fixed categorical color per strategy (dataviz palette slots 1 & 6).
# One color + label per strategy IDENTITY, shared across all tasks. top-k is
# always per-token and llmint8 is always the reserve scheme, so the LLM keys
# (topk_per_token / llmint8_reserve) and the ResNet keys (topk / llmint8) map to
# the SAME identity, color, and legend entry.
_STRAT_COLOR = {
    "topk_per_token": "#2a78d6",   # blue
    "topk": "#2a78d6",             # blue (same identity)
    "llmint8_reserve": "#e34948",  # red
    "llmint8": "#e34948",          # red (same identity)
    "quantization": "#e8873a",     # orange (ResNet only)
}
_STRAT_LABEL = {
    "topk_per_token": "topk",
    "topk": "topk",
    "llmint8_reserve": "llmint8",
    "llmint8": "llmint8",
    "quantization": "quantization",
}
# Ordinal line style per tol, lightest/densest dashing = smallest offset.
_TOL_STYLE = {
    0.00: "-",
    0.01: (0, (5, 1)),
    0.02: (0, (4, 2)),
    0.05: (0, (2, 2)),
    0.10: (0, (1, 1.6)),
}
_TASK_ORDER = [
    ("google/gemma-2b", "sharegpt", "perplexity", 4),
    ("google/gemma-2b", "sharegpt", "perplexity", 7),
    ("google/gemma-7b", "sharegpt", "perplexity", 4),
    ("meta-llama/Llama-3.1-8B", "wikitext", "perplexity", 4),
    ("meta-llama/Llama-3.1-8B", "mmlu", "accuracy", 4),
    ("resnet", "imagenet", "accuracy", 3),
]
# Scenario names as they appear in the paper's results table, so a curve maps
# onto a table row directly. task_label() appends the stage count L_k, which is
# required: G1-2B-SGPT is two curves, at L_k=5 and L_k=8.
# Keep in sync with the macro definitions in the paper preamble:
#   \taskLangThree  G1-2B-SGPT     \taskLangFour  G1-7B-SGPT
#   \taskLangFive   Ll3-8B-MMLU    \taskLangSix   Ll3-8B-WT
#   \taskLangSeven  FT5-SST2       \taskVisionTwo RN56-CF10
# The resnet key stays dataset="imagenet", the on-disk key, while the
# DISPLAY name is CIFAR-10 — RN56-CF10 already carries that. See
# jensen_concavity._DATASET_DISPLAY.
_TASK_NAME = {
    ("google/gemma-2b", "sharegpt", "perplexity", 4): "G1-2B-SGPT",
    ("google/gemma-2b", "sharegpt", "perplexity", 7): "G1-2B-SGPT",
    ("google/gemma-7b", "sharegpt", "perplexity", 4): "G1-7B-SGPT",
    ("meta-llama/Llama-3.1-8B", "wikitext", "perplexity", 4): "Ll3-8B-WT",
    ("meta-llama/Llama-3.1-8B", "mmlu", "accuracy", 4): "Ll3-8B-MMLU",
    ("resnet", "imagenet", "accuracy", 3): "RN56-CF10",
    # T5 is absent from _TASK_ORDER, this script's task allowlist, but named
    # here for the scripts that discover tasks dynamically.
    # The paper's FT5-SST2 scenario is the ACCURACY one; the ppl-score variant has
    # no macro (it is excluded by default), so it is marked explicitly.
    ("google/flan-t5-base", "sst2", "accuracy", 3): "FT5-SST2",
    ("google/flan-t5-base", "sst2", "ppl_score", 3): "FT5-SST2 ppl-score",
}


def task_label(task_key: Tuple, math: bool = True) -> str:
    """Scenario name + STAGE count, for legends, subplot titles and table keys.

    The label reports the paper's \\bm{L_k}, which is a count of pipeline STAGES,
    not of cuts: the paper defines the processing path as
    P_k = (v_{1,k}, ..., v_{L_k,k}), i.e. L_k devices, and compression acts on the
    L_k - 1 links BETWEEN them. Our `n_cuts` is that link count — it is the
    dimension of eta — so the displayed L_k is `n_cuts + 1`. The same fence-post
    appears in the paper's tables, where
    tau has L_k per-stage entries and `a` has L_k - 1 per-boundary entries.

    Figures can only approximate the symbol: matplotlib 3.5 mathtext has no \\bm
    or \\boldsymbol, and there is no LaTeX install to fall back on, so the
    rendered form is \\mathbf{L_k} — bold upright where the paper is bold italic.
    `math=False` gives the plain-text form used for JSON keys and console
    warnings, which must not carry mathtext markup.

    A task with no paper name falls back to its raw identity, which is loud
    enough to notice in a legend.
    """
    model, dataset, metric, n_cuts = task_key
    name = _TASK_NAME.get(task_key)
    if name is None:
        name = f"{model} · {display_dataset(dataset)} {metric}"
    n_stages = n_cuts + 1
    if math:
        return f"{name} ($\\mathbf{{L_k}}={n_stages}$)"
    return f"{name} (L_k={n_stages})"


def load_ray_pool(root: Path, resnet_root: Path = None,
                  excludes=None) -> Dict[Tuple, dict]:
    excludes = excludes or []
    files = discover_ray_files(root)
    print(f"Found {len(files)} ray files (completed + partial) under {root}")
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
            continue
        if is_excluded(meta, excludes):
            continue
        key = (meta["model"], meta["dataset"], meta["metric"],
               int(meta["n_cuts"]), meta["strategy"])
        entry = best.setdefault(key, {"meta": meta, "rays": [], "seen": set()})
        for r in rays:
            h = _ray_hash(r)
            if h in entry["seen"]:
                continue
            entry["seen"].add(h)
            entry["rays"].append(r)

    # Fold in ResNet directional-profile runs (same ray format, different backend).
    if resnet_root and resnet_root.exists():
        r_entries = resnet_entries(resnet_root)
        print(f"Found {len(r_entries)} resnet concavity files under {resnet_root}")
        for meta, rays, _src in r_entries:
            if is_excluded(meta, excludes):
                continue
            key = (meta["model"], meta["dataset"], meta["metric"],
                   int(meta["n_cuts"]), meta["strategy"])
            entry = best.setdefault(key, {"meta": meta, "rays": [], "seen": set()})
            for r in rays:
                h = _ray_hash(r)
                if h in entry["seen"]:
                    continue
                entry["seen"].add(h)
                entry["rays"].append(r)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "outputs" / "mc_concavity"))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "mc_concavity" / "jensen" / "concavity_grid.png"))
    ap.add_argument("--resnet_root",
                    default=str(REPO_ROOT / "outputs" / "concavity_resnet"),
                    help="ResNet directional-profile results, folded into the grid.")
    ap.add_argument("--tols", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--samples_per_ray", type=int, default=5,
                    help="Jensen triples sampled per ray per floor, per repeat "
                         "per repeat.")
    ap.add_argument("--n_repeats", type=int, default=5,
                    help="Independent resamples of samples_per_ray triples, used "
                         "to compute a mean +/- std band per curve.")
    ap.add_argument("--floor_min", type=float, default=0.0)
    ap.add_argument("--floor_max", type=float, default=0.7,
                    help="Highest eta floor to plot. Defaults to 0.7: the "
                         "adaptive fill guarantees >=min_box_samples usable rays "
                         "only up to floor 0.7, so the sweep is unreliable above it.")
    ap.add_argument("--floor_step", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Drop matching curves, spec 'MODEL_SUBSTR:METRIC' "
                         "(metric optional). E.g. flan-t5-base:ppl_score.")
    args = ap.parse_args()
    excludes = parse_excludes(args.exclude)
    if excludes:
        print(f"Excluding curves: {excludes}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    floors = np.round(np.arange(args.floor_min, args.floor_max + 1e-9, args.floor_step), 3)

    pool = load_ray_pool(Path(args.root), Path(args.resnet_root), excludes)

    # curves[(model,dataset,metric,n_cuts)][strategy][tol] = {"floors","mean","std"}
    curves: Dict[Tuple, Dict[str, Dict[float, dict]]] = {}
    for (model, dataset, metric, n_cuts, strat), entry in pool.items():
        if strat not in _STRAT_COLOR:
            continue
        task_key = (model, dataset, metric, n_cuts)
        for tol in args.tols:
            # n_repeats independent draws of samples_per_ray triples each ->
            # per-floor mean/std across repeats (band reflects small-sample
            # triple-selection noise, not just binomial trial-count noise).
            repeat_concavity = []  # list of arrays, one per repeat, len(floors)
            for rep in range(args.n_repeats):
                rng = np.random.default_rng(args.seed * 1000 + rep)
                sweep = floor_sweep(entry["rays"], floors=floors, rng=rng,
                                    samples_per_ray=args.samples_per_ray, tol=tol)
                repeat_concavity.append(
                    [s["concavity"] if s["concavity"] is not None else np.nan for s in sweep])
            arr = np.array(repeat_concavity, dtype=np.float64)  # (n_repeats, n_floors)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            curves.setdefault(task_key, {}).setdefault(strat, {})[tol] = {
                "floors": floors, "mean": mean, "std": std,
            }
            n_rays = len(entry["rays"])
        print(f"  {model}|{dataset}/{metric}|cuts{n_cuts}|{strat}: {n_rays} rays, "
              f"{len(args.tols)} tols x {args.n_repeats} repeats of "
              f"{args.samples_per_ray} triples/ray")

    tasks = [t for t in _TASK_ORDER if t in curves]
    if not tasks:
        raise SystemExit("No matching tasks found under root.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)

    for idx, task_key in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        strat_curves = curves[task_key]
        for strat in sorted(strat_curves):
            color = _STRAT_COLOR.get(strat, "#52514e")
            for tol in args.tols:
                sweep = strat_curves[strat][tol]
                xs = sweep["floors"]
                mean = sweep["mean"]
                std = sweep["std"]
                mask = ~np.isnan(mean)
                ax.plot(xs[mask], mean[mask], color=color, linewidth=1.6,
                        linestyle=_TOL_STYLE.get(tol, "-"))
                ax.fill_between(xs[mask], np.clip(mean[mask] - std[mask], 0, 1),
                                np.clip(mean[mask] + std[mask], 0, 1),
                                color=color, alpha=0.10, linewidth=0)
                if mask.any() and float(np.min(mean[mask])) < _Y_MIN:
                    print(f"  ! {task_label(task_key, math=False)} {strat} "
                          f"tol={tol:g} dips to {float(np.min(mean[mask])):.3f}, "
                          f"below the y-axis floor {_Y_MIN} — it will be clipped")
        ax.set_ylim(_Y_MIN, _Y_MAX)
        ax.set_xlim(args.floor_min, args.floor_max)
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.3, linewidth=1)
        ax.grid(True, alpha=0.25)
        ax.set_title(task_label(task_key), fontsize=10)
        ax.set_xlabel("eta floor  c", fontsize=9)
        ax.set_ylabel(Y_AXIS_LABEL, fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide unused axes.
    for j in range(len(tasks), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Two separate legends: strategy (color/identity) and tol (linestyle/ordinal).
    from matplotlib.lines import Line2D
    present_strats = sorted({s for t in tasks for s in curves[t]})
    # Dedup by display label so shared identities (topk_per_token/topk,
    # llmint8_reserve/llmint8) collapse to a single legend entry.
    strat_handles, seen_labels = [], set()
    for s in present_strats:
        lbl = _STRAT_LABEL.get(s, s)
        if lbl in seen_labels:
            continue
        seen_labels.add(lbl)
        strat_handles.append(
            Line2D([0], [0], color=_STRAT_COLOR.get(s, "#52514e"), lw=2.2, label=lbl))
    tol_handles = [Line2D([0], [0], color="#52514e", lw=1.6, linestyle=_TOL_STYLE[t],
                          label=f"tol={t:g}") for t in args.tols]
    leg1 = fig.legend(handles=strat_handles, title="strategy", loc="upper center",
                      bbox_to_anchor=(0.28, 1.04), ncol=len(strat_handles), fontsize=9)
    fig.legend(handles=tol_handles, title="tol (concavity offset)", loc="upper center",
              bbox_to_anchor=(0.75, 1.04), ncol=len(tol_handles), fontsize=9)
    fig.add_artist(leg1)

    fig.suptitle(f"Jensen inequality test vs eta-floor, by strategy and concavity tolerance\n"
                 f"({args.samples_per_ray} triples/ray, mean ± std over {args.n_repeats} resamples)",
                 fontsize=13, y=1.13)
    fig.tight_layout()
    written = save_figure(fig, out_path)
    plt.close(fig)
    print(f"Wrote {', '.join(str(p) for p in written)}")


if __name__ == "__main__":
    main()
