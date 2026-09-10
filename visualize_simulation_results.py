"""
Visualization utility for saved simulation runs.

Works with:
  - outputs/.../simulation_results.npz (single-task)
  - outputs/.../simulation_results_multi.npz (multi-task)

Example:
  python visualize_simulation_results.py --npz outputs/baseline_compare/simulation_results.npz
  python visualize_simulation_results.py --npz outputs/baseline_compare_multi/simulation_results_multi.npz --plot_eta
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# Use a non-interactive backend by default (safe for headless runs)
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _load_npz(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ file not found: {path}")
    return dict(np.load(path, allow_pickle=True))


def _markers_for_algos(algo_names: Sequence[str]) -> Dict[str, str]:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]
    return {name: markers[i % len(markers)] for i, name in enumerate(algo_names)}


def _cumulative_nanmean(hist: np.ndarray, time_axis: int = 1) -> np.ndarray:
    """
    Compute cumulative mean over `time_axis` while ignoring NaNs.

    For hist shaped like (A, T, ...), this returns array of the same shape.
    """
    mask = ~np.isnan(hist)
    # np.nancumsum treats NaNs as zero, but we divide by counts to ignore NaNs.
    cumsum = np.nancumsum(hist, axis=time_axis)
    counts = np.sum(mask, axis=time_axis)
    # Broadcast counts to match cumsum dims.
    # counts has shape where time axis is removed; we re-insert singleton dim for broadcasting.
    counts_expanded = np.expand_dims(counts, axis=time_axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cumsum / counts_expanded


def _savefig(out_dir: str, filename: str) -> str:
    path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_accuracy_over_time(
    *,
    times: np.ndarray,
    algo_names: Sequence[str],
    acc_avg: np.ndarray,
    out_dir: str,
) -> str:
    markers = _markers_for_algos(algo_names)
    T = int(times.shape[0])
    markevery = max(1, T // 10)

    plt.figure(figsize=(12, 6))
    for a, name in enumerate(algo_names):
        plt.plot(
            times,
            acc_avg[a],
            label=name,
            linewidth=1.5,
            marker=markers[name],
            markevery=markevery,
            markersize=5,
        )
    plt.xlabel("Time slot t")
    plt.ylabel("Cumulative average accuracy")
    plt.title("Cumulative average accuracy over time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    return _savefig(out_dir, "accuracy_over_time.png")


def plot_excess_delay_over_time(
    *,
    times: np.ndarray,
    algo_names: Sequence[str],
    excess_delay_avg: np.ndarray,
    out_dir: str,
) -> str:
    markers = _markers_for_algos(algo_names)
    T = int(times.shape[0])
    markevery = max(1, T // 10)

    plt.figure(figsize=(12, 6))
    for a, name in enumerate(algo_names):
        plt.plot(
            times,
            excess_delay_avg[a],
            label=name,
            linewidth=1.5,
            marker=markers[name],
            markevery=markevery,
            markersize=5,
        )
    plt.xlabel("Time slot t")
    plt.ylabel("Cumulative average excess delay (D - 1/R_k)")
    plt.title("Cumulative average excess delay over time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    return _savefig(out_dir, "excess_delay_over_time.png")


def plot_delay_ratio_over_time(
    *,
    times: np.ndarray,
    algo_names: Sequence[str],
    delay_ratio_avg: np.ndarray,
    out_dir: str,
) -> str:
    """Legacy NPZ: delay / (1/R) ratio."""
    markers = _markers_for_algos(algo_names)
    T = int(times.shape[0])
    markevery = max(1, T // 10)

    plt.figure(figsize=(12, 6))
    for a, name in enumerate(algo_names):
        plt.plot(
            times,
            delay_ratio_avg[a],
            label=name,
            linewidth=1.5,
            marker=markers[name],
            markevery=markevery,
            markersize=5,
        )
    plt.xlabel("Time slot t")
    plt.ylabel("Cumulative average delay / (1/R)")
    plt.title("Cumulative average delay / (1/R) over time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    return _savefig(out_dir, "delay_over_invR_over_time.png")


def plot_tradeoff(
    *,
    algo_names: Sequence[str],
    tradeoff_x: np.ndarray,
    tradeoff_y: np.ndarray,
    out_dir: str,
    inset: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    excess_delay: bool = False,
) -> str:
    markers = _markers_for_algos(algo_names)
    plt.figure(figsize=(8, 6))
    for a, name in enumerate(algo_names):
        plt.scatter(
            float(tradeoff_x[a]),
            float(tradeoff_y[a]),
            marker=markers[name],
            s=80,
            label=name,
        )
    if excess_delay:
        plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
        plt.xlabel("Cumulative average accuracy at t = T-1")
        plt.ylabel("Cumulative average excess delay at t = T-1")
        plt.title("Accuracy vs excess delay at t = T-1")
        fname = "tradeoff_acc_vs_excess_delay_Tminus1.png"
    else:
        plt.axhline(1.0, color="black", linewidth=1.0, alpha=0.8)
        plt.xlabel("Cumulative average accuracy at t = T-1")
        plt.ylabel("Cumulative average delay / (1/R) at t = T-1")
        plt.title("Accuracy vs feasibility (delay/(1/R)) at t = T-1")
        fname = "tradeoff_acc_vs_delay_over_invR_Tminus1.png"
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=8)

    if inset is not None:
        # Create a magnified view of the region of interest.
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        (x0, x1), (y0, y1) = inset
        ax_main = plt.gca()
        # Place inset away from the main point cloud for readability.
        # Slightly right of center in axes coordinates.
        ax_inset = inset_axes(
            ax_main,
            width=5,
            height=1.38,
            loc="center",
            bbox_to_anchor=(0.62, 0.50),
            bbox_transform=ax_main.transAxes,
            borderpad=0.0,
        )

        ref_y = 0.0 if excess_delay else 1.0
        for a, name in enumerate(algo_names):
            ax_inset.scatter(
                float(tradeoff_x[a]),
                float(tradeoff_y[a]),
                marker=markers[name],
                s=30,
            )
        ax_inset.axhline(ref_y, color="black", linewidth=0.8, alpha=0.8)
        ax_inset.set_xlim(x0, x1)
        ax_inset.set_ylim(y0, y1)
        ax_inset.grid(True, alpha=0.25)
        ax_inset.tick_params(axis="both", which="major", labelsize=8)

    return _savefig(out_dir, fname)


def plot_eta_over_time(
    *,
    times: np.ndarray,
    algo_names: Sequence[str],
    eta_hist: np.ndarray,
    out_dir: str,
    eta_mode: str,
    eta_tasks: Optional[Sequence[int]],
) -> Optional[str]:
    """
    Plot cumulative-average eta traces.

    Supported eta_hist shapes:
      - (A, T, L) for single-task
      - (A, T, K, L) for multi-task
    """
    if eta_hist.ndim not in (3, 4):
        return None

    markers = _markers_for_algos(algo_names)
    T = int(times.shape[0])
    markevery = max(1, T // 10)

    if eta_hist.ndim == 3:
        # (A, T, L)
        eta_cum_avg = _cumulative_nanmean(eta_hist, time_axis=1)  # (A,T,L)
        A, _, L = eta_cum_avg.shape

        plt.figure(figsize=(12, 6))
        for a, name in enumerate(algo_names):
            for link_local_idx in range(L):
                plt.plot(
                    times,
                    eta_cum_avg[a, :, link_local_idx],
                    label=f"{name} (eta[{link_local_idx}])",
                    linewidth=1.2,
                    marker=markers[name],
                    markevery=markevery,
                    markersize=4,
                )
        plt.xlabel("Time slot t")
        plt.ylabel("Cumulative average compression eta")
        plt.title("Cumulative average compression over time (single task)")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=7)
        return _savefig(out_dir, "eta_over_time.png")

    # eta_hist.ndim == 4
    # (A, T, K, L)
    eta_cum_avg = _cumulative_nanmean(eta_hist, time_axis=1)  # (A,T,K,L)
    A, _, K, L = eta_cum_avg.shape

    if eta_mode == "avg_tasks":
        # (A,T,L)
        eta_to_plot = np.nanmean(eta_cum_avg, axis=2)
        plt.figure(figsize=(12, 6))
        for a, name in enumerate(algo_names):
            for link_local_idx in range(L):
                plt.plot(
                    times,
                    eta_to_plot[a, :, link_local_idx],
                    label=f"{name} (eta_avg_tasks[{link_local_idx}])",
                    linewidth=1.2,
                    marker=markers[name],
                    markevery=markevery,
                    markersize=4,
                )
        plt.xlabel("Time slot t")
        plt.ylabel("Cumulative avg compression eta (avg over tasks)")
        plt.title("Cumulative average compression over time (multi-task)")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=7)
        return _savefig(out_dir, "eta_over_time_multi_avg_tasks.png")

    # per_task (plot subset)
    if eta_tasks is None:
        eta_tasks = [0]

    selected_tasks = [int(x) for x in eta_tasks]
    plt.figure(figsize=(12, 6))
    for a, name in enumerate(algo_names):
        for k_idx in selected_tasks:
            if k_idx < 0 or k_idx >= K:
                continue
            for link_local_idx in range(L):
                plt.plot(
                    times,
                    eta_cum_avg[a, :, k_idx, link_local_idx],
                    label=f"{name} (task {k_idx}, eta[{link_local_idx}])",
                    linewidth=1.0,
                    marker=markers[name],
                    markevery=markevery,
                    markersize=3,
                )
    plt.xlabel("Time slot t")
    plt.ylabel("Cumulative average compression eta")
    plt.title("Cumulative average compression over time (multi-task, subset)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=6)
    return _savefig(out_dir, "eta_over_time_multi_per_task_subset.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True, help="Path to simulation_results*.npz")
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots (defaults to <npz_dir>/visualizations)",
    )
    p.add_argument("--plot_eta", action="store_true", help="Also plot eta over time")
    p.add_argument(
        "--eta_mode",
        type=str,
        default="avg_tasks",
        choices=["avg_tasks", "per_task"],
        help="For multi-task eta plots, whether to average over tasks",
    )
    p.add_argument(
        "--eta_tasks",
        type=str,
        default=None,
        help="Comma-separated task indices to plot when --eta_mode=per_task",
    )

    args = p.parse_args()
    npz_data = _load_npz(args.npz)

    times = npz_data["times"]
    algo_names = [str(x) for x in npz_data["algo_names"]]

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(args.npz), "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    results: Dict[str, str] = {}

    acc_avg = npz_data.get("acc_avg")
    excess_delay_avg = npz_data.get("excess_delay_avg")
    delay_ratio_avg = npz_data.get("delay_ratio_avg")
    tradeoff_x = npz_data.get("tradeoff_x")
    tradeoff_y = npz_data.get("tradeoff_y")

    if acc_avg is not None:
        results["accuracy_over_time"] = plot_accuracy_over_time(
            times=times, algo_names=algo_names, acc_avg=acc_avg, out_dir=out_dir
        )
    if excess_delay_avg is not None:
        results["excess_delay_over_time"] = plot_excess_delay_over_time(
            times=times,
            algo_names=algo_names,
            excess_delay_avg=excess_delay_avg,
            out_dir=out_dir,
        )
    elif delay_ratio_avg is not None:
        results["delay_over_invR_over_time"] = plot_delay_ratio_over_time(
            times=times,
            algo_names=algo_names,
            delay_ratio_avg=delay_ratio_avg,
            out_dir=out_dir,
        )
    if tradeoff_x is not None and tradeoff_y is not None:
        # Multi-task NPZs include `K` and `eta_hist` with 4 dims.
        # Use that to decide whether to show the magnified inset.
        K_val = npz_data.get("K")
        is_multi_task = False
        if K_val is not None:
            try:
                is_multi_task = int(np.atleast_1d(K_val)[0]) > 1
            except Exception:
                is_multi_task = False

        use_excess = excess_delay_avg is not None
        inset = None
        if is_multi_task:
            inset = ((0.75, 0.88), (-0.02, 0.15)) if use_excess else ((0.75, 0.88), (0.8, 1.5))
        results["tradeoff"] = plot_tradeoff(
            algo_names=algo_names,
            tradeoff_x=tradeoff_x,
            tradeoff_y=tradeoff_y,
            out_dir=out_dir,
            inset=inset,
            excess_delay=use_excess,
        )

    if args.plot_eta:
        eta_hist = npz_data.get("eta_hist")
        if eta_hist is None:
            print("No `eta_hist` found in NPZ; skipping eta plots.")
        else:
            eta_tasks: Optional[Sequence[int]] = None
            if args.eta_tasks is not None:
                eta_tasks = [int(x.strip()) for x in args.eta_tasks.split(",") if x.strip()]
            eta_path = plot_eta_over_time(
                times=times,
                algo_names=algo_names,
                eta_hist=eta_hist,
                out_dir=out_dir,
                eta_mode=args.eta_mode,
                eta_tasks=eta_tasks,
            )
            if eta_path is not None:
                results["eta_over_time"] = eta_path

    print(f"Saved plots to: {out_dir}")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

