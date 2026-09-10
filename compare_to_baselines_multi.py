"""
Compare multi-task optimizers to multi-task baselines.

Runs a simulation for T time slots with i.i.d. uniform channel capacities and compares:
  - Average weighted accuracy over time
  - Average excess delay (D_k - 1/R_k) over tasks over time (under true c_t)

Algorithms included:
  - CSI-aware multi-task optimizer (convex solver, perfect c_t)
  - CSI-aware multi-task baselines:
      0) MaxCompressionMultiTaskBaseline, NoCompressionMultiTaskBaseline
      1) StaticEqualShareMultiTaskBaseline
      2) ProportionalResourceAllocationMultiTaskBaseline
      3) StrictPriorityGreedyMultiTaskBaseline
  - No-CSI multi-task optimizer (Algorithm 2) fed by an estimator (LCB by default)
  - No-CSI multi-task baselines:
      4) DecoupledEqualSplitStochasticDescentMultiTaskBaseline
      5) QueueProportionalHeuristicMultiTaskBaseline
      6) HistoricalAverageCertaintyEquivalenceMultiTaskBaseline (certainty equivalence)
      7) LCB certainty-equivalence baseline

Usage:
  python compare_to_baselines_multi.py --T 150 --K 2 --c_min 100.0 --c_max 350.0 --mu 10.0 --epsilon 0.1 --w_k 1,2 --R_k 10,15 --tasks toy_mlp_mnist,toy_mlp_mnist --M 3
  python compare_to_baselines_multi.py --T 200 --seed 0 --K 2
  python compare_to_baselines_multi.py --K 2 --tasks toy_mlp_mnist,toy_mlp_mnist --M 3
  python compare_to_baselines_multi.py --w_k 1,2 --R_k 10,15
  python compare_to_baselines_multi.py --R 20
  python compare_to_baselines_multi.py --R_range 10,60,10 --out_dir outputs/baseline_multi_R_sweep
  python compare_to_baselines_multi.py --baseline_algorithms equal_share,strict_priority
  python compare_to_baselines_multi.py --baseline_algorithms none

"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Repo root on sys.path so `from src...` works when running this script directly.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

# Use a non-interactive backend by default (safe for headless runs)
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.core.task import InferenceTask
from src.core.task_handler import get_task, registered_task_names
from src.optimizers.base import BaseOptimizer
from src.optimizers.csi_aware import CSIAwareMultiTaskOptimizer
from src.optimizers.no_csi import NoCSIMultiTaskOptimizer
from src.optimizers.estimators import MeanEstimator, MeanMinusZStdLCB
from src.optimizers.baseline import (
    MaxCompressionMultiTaskBaseline,
    NoCompressionMultiTaskBaseline,
    StaticEqualShareMultiTaskBaseline,
    ProportionalResourceAllocationMultiTaskBaseline,
    StrictPriorityGreedyMultiTaskBaseline,
    DecoupledEqualSplitStochasticDescentMultiTaskBaseline,
    QueueProportionalHeuristicMultiTaskBaseline,
    HistoricalAverageCertaintyEquivalenceMultiTaskBaseline,
    LCBCertaintyEquivalenceMultiTaskBaseline,
)


def _parse_task_names(s: str) -> List[str]:
    return [part.strip() for part in s.split(",") if part.strip()]


def _expand_task_names_for_k(names: List[str], K: int) -> List[str]:
    """
    One name -> repeat K times (same task family for every stream).
    Otherwise require exactly K names (one registered task name per stream).
    """
    if len(names) == 1:
        return [names[0]] * K
    if len(names) != K:
        raise ValueError(
            f"Provide one --tasks name (used for all K={K} tasks) or exactly K={K} "
            f"comma-separated names; got {len(names)}: {names!r}"
        )
    return names


def _parse_float_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Expected at least one comma-separated number; got {s!r}")
    return [float(p) for p in parts]


def _expand_floats_for_k(vals: List[float], K: int, argname: str) -> List[float]:
    if len(vals) == 1:
        return [vals[0]] * K
    if len(vals) != K:
        raise ValueError(
            f"{argname}: provide one value (repeated for all K={K} tasks) or exactly K={K} values; "
            f"got {len(vals)}: {vals!r}"
        )
    return vals


def _default_w_k_list(K: int) -> List[float]:
    return [float(k + 1) for k in range(K)]


def _weighted_accuracy(tasks: List[InferenceTask], allocations: Dict[int, Dict[str, np.ndarray]]) -> float:
    vals = []
    wsum = 0.0
    for task in tasks:
        eta = allocations[task.task_id]["eta"]
        vals.append(float(task.w_k) * float(task.A_k_true(eta)))
        wsum += float(task.w_k)
    return float(sum(vals) / wsum) if wsum > 0 else float(np.mean(vals))


def _avg_excess_delay(
    optimizer: BaseOptimizer,
    tasks: List[InferenceTask],
    t: int,
    allocations: Dict[int, Dict[str, np.ndarray]],
    c_t: np.ndarray,
) -> float:
    residuals = optimizer.compute_residual_delay(t, allocations, c_t)
    excesses = []
    for task in tasks:
        residual = float(residuals[task.task_id])
        R_k = float(task.get_R_k(t))
        inv_R = 1.0 / R_k
        delay = residual + inv_R
        excesses.append(delay - inv_R)
    return float(np.mean(excesses))


def run_compare(
    T: int,
    seed: int,
    K: int,
    c_min: float,
    c_max: float,
    out_dir: str,
    mu: float,
    epsilon: float,
    baseline_algorithms: List[str],
    w_k: List[float],
    R_k_per_task: List[float],
    task_names: Optional[List[str]] = None,
    M: int = 3,
) -> Tuple[str, str, str]:
    rng = np.random.default_rng(seed)

    if task_names is None:
        task_names = ["toy_mlp_mnist"] * K
    elif len(task_names) != K:
        raise ValueError(f"task_names length {len(task_names)} must match K={K}")

    if len(w_k) != K:
        raise ValueError(f"w_k length {len(w_k)} must match K={K}")
    if len(R_k_per_task) != K:
        raise ValueError(f"R_k_per_task length {len(R_k_per_task)} must match K={K}")

    num_links = M - 1

    tasks: List[InferenceTask] = []
    for k in range(K):
        tasks.append(
            get_task(
                task_names[k],
                task_id=k + 1,
                w_k=float(w_k[k]),
                R_k=float(R_k_per_task[k]),
            )
        )

    # Algorithms
    csi_opt = CSIAwareMultiTaskOptimizer(M=M, tasks=tasks)
    no_csi_opt = NoCSIMultiTaskOptimizer(M=M, tasks=tasks, mu=mu, epsilon=epsilon, J=5)

    algos: List[Tuple[str, BaseOptimizer]] = [
        ("CSI-Aware (optimal)", csi_opt),
        ("No-CSI (Alg2) + LCB", no_csi_opt),
    ]
    baseline_catalog = {
        "max_compression": (
            "CSI-Aware Baseline: max compression",
            MaxCompressionMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "no_compression": (
            "CSI-Aware Baseline: no compression",
            NoCompressionMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "equal_share": (
            "CSI-Aware Baseline: equal-share",
            StaticEqualShareMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "proportional": (
            "CSI-Aware Baseline: proportional",
            ProportionalResourceAllocationMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "strict_priority": (
            "CSI-Aware Baseline: strict priority",
            StrictPriorityGreedyMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "decoupled_equal_split": (
            "No-CSI Baseline: decoupled equal-split",
            DecoupledEqualSplitStochasticDescentMultiTaskBaseline(
                M=M, tasks=tasks, mu=mu, epsilon=epsilon
            ),
        ),
        "queue_proportional": (
            "No-CSI Baseline: lambda-proportional",
            QueueProportionalHeuristicMultiTaskBaseline(
                M=M, tasks=tasks, mu=mu, epsilon=epsilon
            ),
        ),
        "certainty_equivalence": (
            "No-CSI Baseline: moving average",
            HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "lcb": (
            "No-CSI Baseline: LCB",
            LCBCertaintyEquivalenceMultiTaskBaseline(M=M, tasks=tasks),
        ),
    }
    for baseline_name in baseline_algorithms:
        algo_entry = baseline_catalog.get(baseline_name)
        if algo_entry is None:
            valid = ", ".join(sorted(baseline_catalog.keys()))
            raise ValueError(
                f"Unknown baseline algorithm '{baseline_name}'. "
                f"Valid options: {valid}."
            )
        algos.append(algo_entry)

    # Channel estimators
    lcb = MeanMinusZStdLCB(num_links=num_links, warmup_value=c_min, z=1.2815515655446004, warmup=5)
    hist_mean = (
        MeanEstimator(num_links=num_links, warmup_value=c_min, warmup=5)
        if "certainty_equivalence" in set(baseline_algorithms)
        else None
    )

    times = np.arange(T)
    acc_hist: Dict[str, np.ndarray] = {name: np.full(T, np.nan, dtype=float) for name, _ in algos}
    excess_delay_hist: Dict[str, np.ndarray] = {
        name: np.full(T, np.nan, dtype=float) for name, _ in algos
    }
    c_t_hist = np.full((T, num_links), np.nan, dtype=float)

    # Per-task compression history, for later plotting/debugging.
    # Shape: (A, T, K, L) where L = L_k - 1 (links per task path).
    algo_names = [name for name, _ in algos]
    algo_index = {name: i for i, name in enumerate(algo_names)}
    L = int(tasks[0].L_k - 1)
    eta_hist_stack = np.full((len(algo_names), T, K, L), np.nan, dtype=float)
    task_ids = [int(t.task_id) for t in tasks]

    for t in range(T):
        print(f"Time slot: {t}")
        c_t = rng.uniform(low=c_min, high=c_max, size=num_links).astype(float)
        c_t_hist[t, :] = c_t

        for name, opt in algos:
            if name in {
                "No-CSI (Alg2) + LCB",
                "No-CSI Baseline: decoupled equal-split",
                "No-CSI Baseline: lambda-proportional",
                "No-CSI Baseline: LCB",
            }:
                c_hat = lcb.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            elif name == "No-CSI Baseline: moving average":
                if hist_mean is None:
                    raise RuntimeError("Historical-mean estimator is not initialized.")
                c_hat = hist_mean.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            else:
                allocations = opt.optimize(t=t, c_t=c_t)

            if allocations is None:
                print(f"algo: {name}, allocations: None (infeasible) -> skipping t={t}")
                continue
            print(f"algo: {name}, allocations: {allocations}")
            acc_hist[name][t] = _weighted_accuracy(tasks, allocations)
            print(f"algo: {name}, accuracy: {acc_hist[name][t]}")
            excess_delay_hist[name][t] = _avg_excess_delay(opt, tasks, t, allocations, c_t)
            print(f"algo: {name}, excess delay: {excess_delay_hist[name][t]}")

            # Store per-task eta values if present.
            algo_idx = algo_index[name]
            for k_idx, task in enumerate(tasks):
                eta = np.array(allocations[task.task_id]["eta"], dtype=float)
                if eta.size != L:
                    # Be robust if something changes in task path length.
                    eta = eta.reshape(-1)
                eta_hist_stack[algo_idx, t, k_idx, : min(L, eta.size)] = eta[: min(L, eta.size)]

            # Dual updates for algorithms that track queues
            if name in {
                "No-CSI (Alg2) + LCB",
                "No-CSI Baseline: decoupled equal-split",
                "No-CSI Baseline: lambda-proportional",
            }:
                # compute realized per-task delays under true c_t
                residuals = opt.compute_residual_delay(t, allocations, c_t)
                actual_delays = {}
                for task in tasks:
                    tid = task.task_id
                    actual_delays[tid] = float(residuals[tid]) + 1.0 / float(task.get_R_k(t))
                opt.update_dual(t=t, actual_delays=actual_delays)

        # Estimators observe c_t after actions
        lcb.update(c_t)
        if hist_mean is not None:
            hist_mean.update(c_t)

    os.makedirs(out_dir, exist_ok=True)

    acc_avg = {name: np.array([np.nanmean(acc_hist[name][: i + 1]) for i in range(T)], dtype=float) for name, _ in algos}
    excess_delay_avg = {
        name: np.array(
            [np.nanmean(excess_delay_hist[name][: i + 1]) for i in range(T)], dtype=float
        )
        for name, _ in algos
    }

    # ----------------------------
    # Save all simulation results for later plotting
    # ----------------------------
    acc_hist_stack = np.stack([acc_hist[name] for name in algo_names], axis=0)  # (A, T)
    excess_delay_hist_stack = np.stack(
        [excess_delay_hist[name] for name in algo_names], axis=0
    )  # (A, T)
    acc_avg_stack = np.stack([acc_avg[name] for name in algo_names], axis=0)  # (A, T)
    excess_delay_avg_stack = np.stack(
        [excess_delay_avg[name] for name in algo_names], axis=0
    )  # (A, T)

    tradeoff_x = np.array([float(acc_avg[name][-1]) for name in algo_names], dtype=float)
    tradeoff_y = np.array(
        [float(excess_delay_avg[name][-1]) for name in algo_names], dtype=float
    )

    data_path = os.path.join(out_dir, "simulation_results_multi.npz")
    np.savez_compressed(
        data_path,
        times=times,
        c_t_hist=c_t_hist,
        algo_names=np.array(algo_names, dtype=str),
        task_ids=np.array(task_ids, dtype=int),
        acc_hist=acc_hist_stack,
        excess_delay_hist=excess_delay_hist_stack,
        eta_hist=eta_hist_stack,
        acc_avg=acc_avg_stack,
        excess_delay_avg=excess_delay_avg_stack,
        tradeoff_x=tradeoff_x,
        tradeoff_y=tradeoff_y,
        K=np.array([int(K)], dtype=int),
        task_L_k=np.array([int(tasks[0].L_k)], dtype=int),
    )

    metadata_path = os.path.join(out_dir, "simulation_metadata_multi.json")
    metadata = {
        "T": int(T),
        "seed": int(seed),
        "K": int(K),
        "c_min": float(c_min),
        "c_max": float(c_max),
        "num_links": int(num_links),
        "M": int(M),
        "mu": float(mu),
        "epsilon": float(epsilon),
        "R_k_per_task": [float(x) for x in R_k_per_task],
        "w_k": [float(x) for x in w_k],
        "baseline_algorithms_used": list(baseline_algorithms),
        "algo_names": algo_names,
        "task_ids": task_ids,
        "L_k": int(tasks[0].L_k),
        "task_names": list(task_names),
        "delay_metric": "excess_delay_D_minus_invR",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Markers (distinct per algorithm) to make plots readable.
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]
    marker_map = {name: markers[i % len(markers)] for i, (name, _) in enumerate(algos)}
    markevery = max(1, T // 10)

    plt.figure(figsize=(12, 6))
    for name, _ in algos:
        plt.plot(
            times,
            acc_avg[name],
            label=name,
            linewidth=1.5,
            marker=marker_map[name],
            markevery=markevery,
            markersize=5,
        )
    plt.xlabel("Time slot t")
    plt.ylabel("Cumulative avg weighted accuracy")
    plt.title("Multi-task: cumulative average weighted accuracy over time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    acc_path = os.path.join(out_dir, "accuracy_over_time.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    for name, _ in algos:
        plt.plot(
            times,
            excess_delay_avg[name],
            label=name,
            linewidth=1.5,
            marker=marker_map[name],
            markevery=markevery,
            markersize=5,
        )
    plt.xlabel("Time slot t")
    plt.ylabel("Cumulative avg (1/K) * sum_k (D_k - 1/R_k)")
    plt.title("Multi-task: cumulative average excess delay over time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    res_path = os.path.join(out_dir, "excess_delay_over_time.png")
    plt.tight_layout()
    plt.savefig(res_path, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, _ in algos:
        # acc_avg[name] / excess_delay_avg[name] are cumulative averages over time,
        # so the last entry corresponds to t = T-1.
        x = float(acc_avg[name][-1])
        y = float(excess_delay_avg[name][-1])
        plt.scatter(
            x,
            y,
            marker=marker_map[name],
            s=80,
            label=name,
        )
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    plt.xlabel("Cumulative avg weighted accuracy at t = T-1")
    plt.ylabel("Cumulative avg (1/K) * sum_k (D_k - 1/R_k) at t = T-1")
    plt.title("Multi-task: accuracy vs excess delay at t = T-1")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=8)
    tradeoff_path = os.path.join(out_dir, "tradeoff_acc_vs_excess_delay_Tminus1.png")
    plt.tight_layout()
    plt.savefig(tradeoff_path, dpi=180)
    plt.close()

    return acc_path, res_path, tradeoff_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=150)
    p.add_argument("--seed", type=int, default=567)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--c_min", type=float, default=100.0)
    p.add_argument("--c_max", type=float, default=350.0)
    p.add_argument("--mu", type=float, default=10.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument(
        "--R",
        type=float,
        default=10.0,
        help="Default target throughput per task when --R_k is not set (same R for all tasks). Ignored if --R_range is set.",
    )
    p.add_argument(
        "--R_range",
        type=str,
        default=None,
        help=(
            "Optional integer range for sweeping R, as start,stop,step (stop exclusive). "
            "Each run sets R_k(t)=R for every task (same R for all K). "
            "Example: 10,60,10 -> R in {10,20,30,40,50}. Outputs under <out_dir>/R_<n>/."
        ),
    )
    p.add_argument(
        "--w_k",
        type=str,
        default=None,
        help=(
            "Comma-separated weights. One value repeats for all K tasks, or exactly K values. "
            "Default: 1,2,...,K."
        ),
    )
    p.add_argument(
        "--R_k",
        type=str,
        default=None,
        dest="R_k_arg",
        help=(
            "Comma-separated target throughputs R_k per task (constant in t). "
            "One value repeats for all K, or exactly K values. "
            "Default: each task uses --R. Ignored when --R_range is set."
        ),
    )
    p.add_argument(
        "--baseline_algorithms",
        type=str,
        default="max_compression,no_compression,equal_share,proportional,strict_priority,decoupled_equal_split,queue_proportional,certainty_equivalence",#lcb
        help=(
            "Comma-separated baseline algorithms to include. "
            "Use 'none' to disable all baselines. "
            "Options: max_compression, no_compression, equal_share, proportional, "
            "strict_priority, decoupled_equal_split, queue_proportional, "
            "certainty_equivalence, lcb."
        ),
    )
    p.add_argument("--out_dir", type=str, default="outputs/baseline_compare_multi1")
    p.add_argument(
        "--tasks",
        type=str,
        default="toy_mlp_mnist",
        help=(
            "Comma-separated registered task names. "
            "One name is reused for all K tasks; otherwise supply exactly K names. "
            f"Available: {', '.join(registered_task_names())}."
        ),
    )
    p.add_argument(
        "--M",
        type=int,
        default=3,
        dest="M",
        help="Number of nodes in the line network; must equal each task's L_k from the chosen task recipes.",
    )
    args = p.parse_args()
    task_names = _expand_task_names_for_k(_parse_task_names(args.tasks), args.K)
    if args.w_k is None:
        w_k_list = _default_w_k_list(args.K)
    else:
        w_k_list = _expand_floats_for_k(_parse_float_list(args.w_k), args.K, "--w_k")

    def _R_k_per_task_for_run(sweep_R: Optional[float] = None) -> List[float]:
        if sweep_R is not None:
            return [float(sweep_R)] * args.K
        if args.R_k_arg is not None:
            return _expand_floats_for_k(_parse_float_list(args.R_k_arg), args.K, "--R_k")
        return [float(args.R)] * args.K

    baseline_arg = args.baseline_algorithms.strip().lower()
    if baseline_arg == "none":
        baseline_algorithms: List[str] = []
    else:
        baseline_algorithms = [
            part.strip()
            for part in args.baseline_algorithms.split(",")
            if part.strip()
        ]

    if args.R_range is not None:
        parts = [int(x.strip()) for x in args.R_range.strip().split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"--R_range expects exactly 3 comma-separated ints: start,stop,step. Got: {args.R_range}"
            )
        start, stop, step = parts
        if step == 0:
            raise ValueError("--R_range step must be non-zero.")
        R_list = list(range(start, stop, step))
        if len(R_list) == 0:
            raise ValueError(
                f"--R_range produced an empty range: start={start}, stop={stop}, step={step}."
            )

        sweep_entries: List[dict] = []
        for R_sweep in R_list:
            R_k_run = _R_k_per_task_for_run(sweep_R=float(R_sweep))
            sub = os.path.join(args.out_dir, f"R_{R_sweep:g}")
            print(f"\n=== Running multi-task compare for R = {R_sweep:g} (all tasks) -> {sub} ===\n")
            acc_path, res_path, tradeoff_path = run_compare(
                T=args.T,
                seed=args.seed,
                K=args.K,
                c_min=args.c_min,
                c_max=args.c_max,
                out_dir=sub,
                mu=args.mu,
                epsilon=args.epsilon,
                baseline_algorithms=baseline_algorithms,
                w_k=w_k_list,
                R_k_per_task=R_k_run,
                task_names=task_names,
                M=args.M,
            )
            sweep_entries.append(
                {
                    "R_k": float(R_sweep),
                    "out_dir": sub,
                    "accuracy_plot": acc_path,
                    "excess_delay_plot": res_path,
                    "tradeoff_plot": tradeoff_path,
                    "simulation_results": os.path.join(sub, "simulation_results_multi.npz"),
                    "simulation_metadata": os.path.join(sub, "simulation_metadata_multi.json"),
                }
            )

        summary_path = os.path.join(args.out_dir, "R_sweep_summary_multi.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "R_range_spec": args.R_range.strip(),
                    "T": args.T,
                    "seed": args.seed,
                    "K": args.K,
                    "tasks": args.tasks,
                    "task_names": task_names,
                    "M": args.M,
                    "w_k": w_k_list,
                    "runs": sweep_entries,
                },
                f,
                indent=2,
            )
        print(f"\nR sweep complete. Summary written to: {summary_path}")
        for e in sweep_entries:
            print(f"  R={e['R_k']}: {e['out_dir']}")
    else:
        R_k_run = _R_k_per_task_for_run()
        acc_path, res_path, tradeoff_path = run_compare(
            T=args.T,
            seed=args.seed,
            K=args.K,
            c_min=args.c_min,
            c_max=args.c_max,
            out_dir=args.out_dir,
            mu=args.mu,
            epsilon=args.epsilon,
            baseline_algorithms=baseline_algorithms,
            w_k=w_k_list,
            R_k_per_task=R_k_run,
            task_names=task_names,
            M=args.M,
        )

        print(f"Saved accuracy plot to: {acc_path}")
        print(f"Saved excess delay plot to: {res_path}")
        print(f"Saved trade-off plot to: {tradeoff_path}")
        print(
            f"Saved raw simulation data to: {os.path.join(args.out_dir, 'simulation_results_multi.npz')}"
        )
        print(
            f"Saved simulation metadata to: {os.path.join(args.out_dir, 'simulation_metadata_multi.json')}"
        )


if __name__ == "__main__":
    main()

