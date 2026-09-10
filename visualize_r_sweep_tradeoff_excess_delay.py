"""
Aggregate R-sweep baseline runs into one trade-off plot.

For each algorithm, traces cumulative avg excess delay (x) vs accuracy at t=T-1 (y) as R varies.
Reads ``tradeoff_y`` as excess delay when ``simulation_metadata.json`` has delay_metric set;
otherwise treats ``tradeoff_y`` as legacy delay/(1/R) and converts to excess time.
Uses outputs from compare_to_baselines.py with --R_range (per-R folders R_<value>/).
The x-axis (excess delay) uses a symmetric log scale; tune linear range with x_symlog_linthresh / --x-symlog-linthresh.

If R_sweep_summary.json exists under the sweep directory, run order and paths are
taken from it; otherwise subdirectories matching R_* are scanned and sorted by R.

Example:
  python visualize_r_sweep_tradeoff_excess_delay.py --sweep_dir outputs/baseline_R_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def _markers_for_algos(algo_names: Sequence[str]) -> Dict[str, str]:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]
    return {name: markers[i % len(markers)] for i, name in enumerate(algo_names)}


def _parse_r_from_dirname(name: str) -> Optional[float]:
    m = re.fullmatch(r"R_([0-9.+-eE]+)", name)
    if not m:
        return None
    return float(m.group(1))


def _legend_label(name: str) -> str:
    lower = name.lower().strip()

    # Our methods (single- and multi-task variants).
    if "csi-aware (optimal)" in lower or "csi-aware optimal" in lower:
        return "OURS-CSI"
    if "no-csi" in lower and "alg" in lower and "lcb" in lower:
        return "OURS-NOCSI"

    prefix: Optional[str] = None
    desc = lower
    if ":" in name:
        raw_prefix, raw_desc = name.split(":", 1)
        prefix = raw_prefix.strip()
        desc = raw_desc.lower().strip()

    short: Optional[str] = None
    if "max compression" in desc or "min compression" in desc:
        short = "MAX"
    elif "no compression" in desc:
        short = "NONE"
    elif "uniform compression" in desc:
        short = "UNI"
    elif "myopic" in desc:
        short = "MYO"
    elif "conservative" in desc:
        short = "CONS"
    elif "moving average" in desc:
        short = "MA"
    elif "equal-share" in desc or "equal share" in desc:
        short = "EQ"
    elif "strict priority" in desc:
        short = "PRIO"
    elif "lambda-proportional" in desc or "lambda proportional" in desc:
        short = "D-L-PROP"
    elif "decoupled equal-split" in desc or "decoupled equal split" in desc:
        short = "D-EQUAL"
    elif "proportional" in desc:
        short = "PROP"
    elif "lcb" in desc:
        short = "LCB"

    if short is not None:
        return f"{prefix}: {short}" if prefix else short
    return name


def discover_runs(sweep_dir: str) -> List[Tuple[float, str]]:
    """Return sorted list of (R_k, run_dir_absolute_or_resolved)."""
    sweep_dir = os.path.abspath(sweep_dir)
    summary_path = os.path.join(sweep_dir, "R_sweep_summary.json")
    runs: List[Tuple[float, str]] = []

    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        for entry in data.get("runs", []):
            R_k = float(entry["R_k"])
            sub = str(entry.get("out_dir", ""))
            if os.path.isdir(sub):
                run_dir = os.path.abspath(sub)
            else:
                run_dir = os.path.join(sweep_dir, os.path.basename(sub))
            runs.append((R_k, run_dir))
    else:
        if not os.path.isdir(sweep_dir):
            raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
        for name in sorted(os.listdir(sweep_dir)):
            R = _parse_r_from_dirname(name)
            if R is None:
                continue
            path = os.path.join(sweep_dir, name)
            if os.path.isdir(path):
                runs.append((R, path))

    runs.sort(key=lambda x: x[0])
    return runs


def _load_tradeoff(
    run_dir: str,
) -> Tuple[float, np.ndarray, np.ndarray, List[str], Optional[str]]:
    meta_path = os.path.join(run_dir, "simulation_metadata.json")
    meta_path_multi = os.path.join(run_dir, "simulation_metadata_multi.json")
    npz_path = os.path.join(run_dir, "simulation_results.npz")
    npz_path_multi = os.path.join(run_dir, "simulation_results_multi.npz")
    if os.path.isfile(npz_path):
        pass
    elif os.path.isfile(npz_path_multi):
        npz_path = npz_path_multi
    else:
        raise FileNotFoundError(
            f"Neither {npz_path} nor {npz_path_multi} exists in {run_dir}"
        )

    R_k: float
    delay_metric: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    elif os.path.isfile(meta_path_multi):
        with open(meta_path_multi, encoding="utf-8") as f:
            meta = json.load(f)

    if meta is not None:
        if "R_k" in meta:
            R_k = float(meta["R_k"])
        else:
            rp = meta.get("R_k_per_task")
            if isinstance(rp, list) and len(rp) > 0:
                R_k = float(rp[0])
            else:
                base = os.path.basename(os.path.normpath(run_dir))
                parsed = _parse_r_from_dirname(base)
                if parsed is None:
                    raise ValueError(
                        f"Cannot infer R for {run_dir} (metadata missing R_k / R_k_per_task)"
                    )
                R_k = parsed
        delay_metric = meta.get("delay_metric")
    else:
        base = os.path.basename(os.path.normpath(run_dir))
        parsed = _parse_r_from_dirname(base)
        if parsed is None:
            raise ValueError(
                f"Cannot infer R for {run_dir} (no simulation_metadata*.json and not R_* dirname)"
            )
        R_k = parsed

    payload = dict(np.load(npz_path, allow_pickle=True))
    names = [str(x) for x in payload["algo_names"]]
    tradeoff_x = np.asarray(payload["tradeoff_x"], dtype=float).reshape(-1)
    tradeoff_y = np.asarray(payload["tradeoff_y"], dtype=float).reshape(-1)
    if tradeoff_x.shape != tradeoff_y.shape or tradeoff_x.shape[0] != len(names):
        raise ValueError(f"Inconsistent tradeoff arrays in {npz_path}")
    return R_k, tradeoff_x, tradeoff_y, names, delay_metric


def plot_r_sweep_tradeoff_excess_delay(
    *,
    sweep_dir: str,
    out_path: str,
    cmap: str = "viridis",
    x_symlog_linthresh: float = 1e-3,
) -> str:
    runs = discover_runs(sweep_dir)
    if not runs:
        raise RuntimeError(f"No R_* runs found under {sweep_dir}")

    # First pass: collect per-run points keyed by algorithm name.
    series: Dict[str, List[Tuple[float, float, float]]] = {}
    ref_order: Optional[List[str]] = None

    for _R_label, run_dir in runs:
        R_k, tx, ty, names, delay_metric = _load_tradeoff(run_dir)
        if ref_order is None:
            ref_order = list(names)
        else:
            if names != ref_order:
                raise ValueError(
                    f"Algorithm list mismatch between runs.\n"
                    f"  expected: {ref_order}\n"
                    f"  got in {run_dir}: {names}"
                )

        # New runs: tradeoff_y is cumulative average excess delay (D - 1/R_k).
        # Legacy: tradeoff_y was delay/(1/R); excess time = (ratio - 1) / R_k.
        if delay_metric == "excess_delay_D_minus_invR":
            exc_vec = ty
        else:
            exc_vec = (ty - 1.0) / R_k
        for name, acc, exc in zip(names, tx, exc_vec):
            series.setdefault(name, []).append((R_k, float(acc), float(exc)))

    if ref_order is None:
        raise RuntimeError("No data loaded")

    if x_symlog_linthresh <= 0:
        raise ValueError("x_symlog_linthresh must be positive for matplotlib symlog")

    R_all = sorted({p[0] for pts in series.values() for p in pts})
    R_min, R_max = float(R_all[0]), float(R_all[-1])
    if R_max <= R_min:
        R_max = R_min + 1.0
    norm = Normalize(vmin=R_min, vmax=R_max)
    cmap_obj = plt.get_cmap(cmap)

    # Larger defaults so axis labels, ticks, legend, and colorbar stay readable in saved PNGs.
    _plot_rc = {
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.title_fontsize": 13,
    }
    with plt.rc_context(_plot_rc):
        fig, ax = plt.subplots(figsize=(10, 7))
        markers = _markers_for_algos(ref_order)
        legend_handles: List[Line2D] = []

        for name in ref_order:
            pts = sorted(series[name], key=lambda p: p[0])
            R_vals = np.asarray([p[0] for p in pts], dtype=float)
            accs = np.asarray([p[1] for p in pts], dtype=float)
            excs = np.asarray([p[2] for p in pts], dtype=float)
            path = np.column_stack([excs, accs])

            if path.shape[0] >= 2:
                segments = np.stack([path[:-1], path[1:]], axis=1)
                seg_r = 0.5 * (R_vals[:-1] + R_vals[1:])
                colors = cmap_obj(norm(seg_r))
                # Style conventions:
                # - dotted: no compression + max/min compression baselines
                # - dashed: No-CSI variants
                # - solid: everything else
                lower_name = name.lower()
                # Keep No-CSI variants visually distinct from compression baselines.
                if "no-csi" in lower_name or "no csi" in lower_name:
                    line_style = "--"
                elif "compression" in lower_name and (
                    "no compression" in lower_name or "max" in lower_name or "min" in lower_name
                ):
                    line_style = ":"
                else:
                    line_style = "-"
                lc = LineCollection(
                    segments,
                    colors=colors,
                    linewidths=0.65,
                    linestyle=line_style,
                    zorder=1,
                )
                # White halo improves contrast against nearby lines/grid.
                lc.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
                ax.add_collection(lc)

            m = markers[name]
            ax.scatter(
                excs,
                accs,
                c=R_vals,
                cmap=cmap_obj,
                norm=norm,
                marker=m,
                s=55,
                edgecolors="black",
                linewidths=.4,
                zorder=3,
            )

            lower_name = name.lower()
            if "no-csi" in lower_name or "no csi" in lower_name:
                legend_linestyle = "--"
            elif "compression" in lower_name and (
                "no compression" in lower_name or "max" in lower_name or "min" in lower_name
            ):
                legend_linestyle = ":"
            else:
                legend_linestyle = "-"
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="0.35",
                    marker=m,
                    linestyle=legend_linestyle,
                    linewidth=1.4,
                    markersize=9,
                    label=_legend_label(name),
                )
            )

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        # Horizontal label under the colorbar (default set_label is side-mounted, rotated).
        cbar.ax.set_xlabel(r"$R_k$", fontsize=15, labelpad=10)
        cbar.ax.tick_params(labelsize=13)

        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
        ax.set_xscale("symlog", linthresh=x_symlog_linthresh)
        ax.set_xlabel("Excess Delay")
        ax.set_ylabel("Aggregate Utility")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(handles=legend_handles, ncol=1)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep_dir",
        type=str,
        default="outputs/baseline_R_sweep",
        help="Directory containing R_* subfolders (and optionally R_sweep_summary.json).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: <sweep_dir>/tradeoff_acc_vs_excess_delay_R_sweep.png).",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name for coloring by R_k (default: viridis).",
    )
    p.add_argument(
        "--x-symlog-linthresh",
        type=float,
        default=1e-3,
        help="Symmetric log x-axis: linear within ±this value (matplotlib linthresh; default: 1e-3).",
    )
    args = p.parse_args()
    out = args.out or os.path.join(
        os.path.abspath(args.sweep_dir),
        "tradeoff_acc_vs_excess_delay_R_sweep.png",
    )
    saved = plot_r_sweep_tradeoff_excess_delay(
        sweep_dir=args.sweep_dir,
        out_path=out,
        cmap=args.cmap,
        x_symlog_linthresh=args.x_symlog_linthresh,
    )
    print(f"Wrote {saved}")


if __name__ == "__main__":
    main()
