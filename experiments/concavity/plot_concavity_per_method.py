"""
Per-compression-method views of Jensen-concavity-vs-eta-floor.

Layout:

  * ONE FIGURE PER COMPRESSION METHOD (topk / llmint8 / quantization), so a
    figure never mixes methods in one axes ("oranges vs apples").
  * ONE SUBPLOT PER TOLERANCE within that figure, all in a SINGLE ROW so they
    read left-to-right as the concavity test loosens.
  * ONE LINE PER TASK within each subplot — so a panel answers "at this
    tolerance, how do the tasks compare under this method?"
  * T5 ppl_score is excluded BY DEFAULT (see --exclude), and tol=0.1 is out of
    the default sweep because every task saturates at 1.0 there.
  * Together the figures still cover the full cartesian product
    (compression method) x (tolerance) x (task).

Lines are labelled with the PAPER SCENARIO NAMES (FT5-SST2, G1-2B-SGPT, ... — the
\task* macro expansions) plus the stage count L_k = n_cuts + 1, so a curve maps
onto a results-table row directly; see plot_concavity_grid.task_label. The y axis is
cropped to
[0.4, 1.02] (plot_concavity_grid._Y_MIN/_Y_MAX) because no measured curve goes
lower, and any that does prints a clipping warning.

Task colour is pinned to the task IDENTITY via _TASK_SLOT, not to its position
in whatever subset a given method happens to have. A task is therefore the same
colour in all three figures and in every panel, and adding/dropping a task never
repaints the others. Colours are the validated 8-slot categorical palette in
fixed slot order; each task also carries a distinct marker, so identity survives
greyscale printing and colour-vision deficiency (three of the light-mode slots
sit under 3:1 contrast, so the relief rule applies — hence the markers plus the
companion JSON table this script writes).

--split_tols additionally writes one standalone PNG per (method, tolerance) if
you want the panels as separate figures rather than a grid.

Usage:
  python experiments/plot_concavity_per_method.py \
      --root outputs/mc_concavity --out_dir outputs/mc_concavity/jensen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from jensen_concavity import (
    floor_sweep, parse_excludes, Y_AXIS_LABEL, save_figure,
)
from plot_concavity_grid import load_ray_pool, task_label, _Y_MIN, _Y_MAX

HERE = Path(__file__).resolve().parent      # experiments/concavity
REPO_ROOT = HERE.parents[1]                 # repo root

# Strategy key -> compression method, one method per figure. The LLM runners
# write topk_per_token where the ResNet runner writes topk, and llmint8_reserve
# is an alias for llmint8.
_STRAT_IDENTITY = {
    "topk_per_token": "topk",
    "topk": "topk",
    "llmint8_reserve": "llmint8",
    "llmint8": "llmint8",
    "quantization": "quantization",
}
_METHOD_ORDER = ["topk", "llmint8", "quantization"]

# Canonical task order == categorical colour slot. Deliberately NOT reusing
# plot_concavity_grid._TASK_ORDER: that list doubles as the task ALLOWLIST for
# the combined grid figure and omits T5, whereas colour assignment here covers
# every task this script can discover.
_TASK_SLOT = [
    ("google/gemma-2b", "sharegpt", "perplexity", 4),
    ("google/gemma-2b", "sharegpt", "perplexity", 7),
    ("google/gemma-7b", "sharegpt", "perplexity", 4),
    ("meta-llama/Llama-3.1-8B", "wikitext", "perplexity", 4),
    ("meta-llama/Llama-3.1-8B", "mmlu", "accuracy", 4),
    ("google/flan-t5-base", "sst2", "accuracy", 3),
    ("google/flan-t5-base", "sst2", "ppl_score", 3),
    ("resnet", "imagenet", "accuracy", 3),   # displayed as CIFAR-10
]
# Validated categorical palette, fixed slot order (light mode). Eight slots is
# the cap: a 9th task is NOT given a generated hue, it falls back to neutral and
# the script says so.
_TASK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_TASK_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_OVERFLOW_COLOR = "#8a8880"

_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"


def _task_style(task_key: Tuple) -> Tuple[str, str]:
    """(colour, marker) pinned to task identity — never to enumeration order."""
    if task_key in _TASK_SLOT:
        i = _TASK_SLOT.index(task_key)
        return _TASK_COLORS[i], _TASK_MARKERS[i]
    return _OVERFLOW_COLOR, "."


def _draw_panel(ax, tol, tasks, task_curves, floors, args) -> None:
    """One tolerance panel: a line per task."""
    for task_key in tasks:
        sweep = task_curves[task_key].get(tol)
        if sweep is None:
            continue
        mean = sweep["mean"]
        mask = ~np.isnan(mean)
        if not mask.any():
            continue
        color, marker = _task_style(task_key)
        ax.plot(floors[mask], mean[mask], color=color, linewidth=2.0,
                marker=marker, markersize=5, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.6,
                label=task_label(task_key))
        # Never clip a curve without saying so: the y window is a display choice
        # (see _Y_MIN), and a curve leaving the axes must not look like a curve
        # that simply ran out of usable rays.
        if float(np.min(mean[mask])) < _Y_MIN:
            print(f"  ! {task_label(task_key, math=False)} tol={tol:g} dips to "
                  f"{float(np.min(mean[mask])):.3f}, below the y-axis floor "
                  f"{_Y_MIN} — it will be clipped")
    ax.set_ylim(_Y_MIN, _Y_MAX)
    ax.set_xlim(args.floor_min, args.floor_max)
    # y=1 is "every Jensen triple passed" — the concavity ceiling, not a target.
    ax.axhline(1.0, color=_INK_SECONDARY, linestyle=":", alpha=0.45, linewidth=1)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_SECONDARY)
        ax.spines[side].set_linewidth(0.8)
    # Panel label rather than a title: it is the only thing identifying the
    # tolerance. There is no figure-level title, so the figure drops into a
    # paper whose caption carries that text.
    ax.set_title(f"tol = {tol:g}", fontsize=15, color=_INK_PRIMARY)
    ax.set_xlabel("eta floor  c", fontsize=13, color=_INK_SECONDARY)
    # Wording is shared with the other two concavity scripts — change it in
    # jensen_concavity.Y_AXIS_LABEL, not here.
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=13, color=_INK_SECONDARY)
    ax.tick_params(labelsize=12, colors=_INK_SECONDARY)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "outputs" / "mc_concavity"))
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "mc_concavity" / "jensen"))
    ap.add_argument("--resnet_root",
                    default=str(REPO_ROOT / "outputs" / "concavity_resnet"))
    ap.add_argument("--tols", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05],
                    help="tol=0.1 is outside the default set: every "
                         "task saturates at 1.0 there, so the panel says nothing.")
    ap.add_argument("--exclude", nargs="*", default=["flan-t5-base:ppl_score"],
                    help="MODEL_SUBSTR[:METRIC] curves to drop. Defaults to "
                         "dropping T5 ppl_score; "
                         "pass --exclude with no values to keep everything.")
    ap.add_argument("--samples_per_ray", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--floor_min", type=float, default=0.0)
    ap.add_argument("--floor_max", type=float, default=0.7,
                    help="Adaptive fill only guarantees usable rays up to 0.7.")
    ap.add_argument("--floor_step", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_tols", action="store_true",
                    help="Also write one standalone PNG per (method, tolerance).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    floors = np.round(np.arange(args.floor_min, args.floor_max + 1e-9, args.floor_step), 3)

    excludes = parse_excludes(args.exclude)
    if excludes:
        print(f"  excluding curves: {args.exclude}")
    pool = load_ray_pool(Path(args.root), Path(args.resnet_root), excludes)

    # curves[method][(model,dataset,metric,n_cuts)][tol] = {"mean","std"}
    curves: Dict[str, Dict[Tuple, Dict[float, dict]]] = {}
    for (model, dataset, metric, n_cuts, strat), entry in pool.items():
        method = _STRAT_IDENTITY.get(strat)
        if method is None:
            continue
        task_key = (model, dataset, metric, n_cuts)
        for tol in args.tols:
            repeat_concavity = []
            for rep in range(args.n_repeats):
                rng = np.random.default_rng(args.seed * 1000 + rep)
                sweep = floor_sweep(entry["rays"], floors=floors, rng=rng,
                                    samples_per_ray=args.samples_per_ray, tol=tol)
                repeat_concavity.append(
                    [s["concavity"] if s["concavity"] is not None else np.nan
                     for s in sweep])
            arr = np.array(repeat_concavity, dtype=np.float64)
            with np.errstate(all="ignore"):
                curves.setdefault(method, {}).setdefault(task_key, {})[tol] = {
                    "mean": np.nanmean(arr, axis=0), "std": np.nanstd(arr, axis=0),
                }
        print(f"  {method:12s} {model}|{dataset}/{metric}|cuts{n_cuts}: "
              f"{len(entry['rays'])} rays")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table: Dict[str, dict] = {}

    for method in _METHOD_ORDER:
        if method not in curves:
            continue
        task_curves = curves[method]
        # Canonical slot order first, then anything new (which gets neutral ink).
        tasks = [t for t in _TASK_SLOT if t in task_curves]
        overflow = sorted(t for t in task_curves if t not in _TASK_SLOT)
        if overflow:
            print(f"  ! {method}: {len(overflow)} task(s) beyond the 8 colour "
                  f"slots, drawn in neutral grey: {overflow}")
        tasks += overflow

        # All tolerances in one row, so the panels read left to right as the
        # test loosens.
        ncols, nrows = len(args.tols), 1
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.8 * ncols, 4.8 * nrows),
                                 squeeze=False)
        for idx, tol in enumerate(args.tols):
            _draw_panel(axes[idx // ncols][idx % ncols], tol, tasks,
                        task_curves, floors, args)
        for j in range(len(args.tols), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        # One shared task legend for the whole figure (identity is never
        # colour-alone: every entry carries its marker too).
        handles, labels = axes[0][0].get_legend_handles_labels()
        # Reserve the top strip FIRST: tight_layout ignores figure legends, so
        # laying out the axes afterwards would let the "tol = ..." panel labels
        # ride up under the legend text (they collided at fontsize 12).
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        # No legend title: matplotlib centres it above the entries, at the top
        # of the figure, where it reads as a suptitle. The entries already name
        # the task.
        fig.legend(handles=handles, labels=labels,
                   loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=min(4, max(1, len(labels))), fontsize=12,
                   frameon=False)
        # No suptitle: the method is in the filename, and the layout and
        # sample counts belong in the paper caption. The numbers behind every
        # line are in concavity_per_method.json.
        out_path = out_dir / f"concavity_grid_{method}.png"
        written = save_figure(fig, out_path)
        plt.close(fig)
        print(f"Wrote {', '.join(str(p) for p in written)}")

        if args.split_tols:
            for tol in args.tols:
                f1, ax1 = plt.subplots(figsize=(6.4, 4.4))
                _draw_panel(ax1, tol, tasks, task_curves, floors, args)
                # Title-free here too; _draw_panel's "tol = ..." is kept.
                ax1.legend(fontsize=11, frameon=False, ncol=1,
                           loc="lower left")
                f1.tight_layout()
                p1 = out_dir / f"concavity_{method}_tol{tol:g}.png"
                w1 = save_figure(f1, p1)
                plt.close(f1)
                print(f"  Wrote {', '.join(str(p) for p in w1)}")

        # Table view — the relief rule for the low-contrast palette slots, and
        # the numbers behind every line. Plain-text keys: a JSON table is read as
        # data, so it must not carry the legend's mathtext markup.
        table[method] = {
            task_label(t, math=False): {
                f"tol={tol:g}": [None if np.isnan(v) else round(float(v), 4)
                                 for v in task_curves[t][tol]["mean"]]
                for tol in args.tols if tol in task_curves[t]
            } for t in tasks
        }

    table_path = out_dir / "concavity_per_method.json"
    table_path.write_text(json.dumps(
        {"floors": [float(f) for f in floors], "methods": table}, indent=2))
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
