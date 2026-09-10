import numpy as np
from scipy.optimize import minimize
from typing import Any, Dict, List, Optional, Tuple

from src.optimizers.base import BaseOptimizer
from src.core.task import InferenceTask


def _active_tasks_on_node(tasks: List[InferenceTask], node_i: int) -> List[InferenceTask]:
    return [t for t in tasks if t.is_active_at_node(node_i)]


def _active_tasks_on_link(tasks: List[InferenceTask], link_i: int) -> List[InferenceTask]:
    return [t for t in tasks if t.is_active_on_link(link_i)]


def _solve_single_task_eta_with_fixed_resources(
    *,
    task: InferenceTask,
    t: int,
    c_eff: np.ndarray,
    max_comp_delay: float,
    mu: float,
    lambda_t: float,
) -> np.ndarray:
    """
    Solves the single-task "Algorithm 1" configuration subproblem with fixed resources:

        min_{eta,z}  -A_k(eta) + mu * lambda_t * z
        s.t.         eta_i in [eta_min, 1]
                     z >= max_comp_delay
                     z >= (a_i / c_eff_i) * eta_i  for all links
    """

    links = list(range(task.b_k, task.e_k))
    if task.L_k <= 1:
        return np.array([], dtype=float)

    if np.any(c_eff <= 0):
        # Degenerate effective capacity -> must fall back to minimum compression.
        return np.array([task.eta_min[i] for i in links], dtype=float)

    def objective(x: np.ndarray) -> Tuple[float, np.ndarray]:
        eta_flat = x[:-1]
        z = float(x[-1])
        acc = float(task.A_k(eta_flat))
        grad_acc = task.grad_A_k(eta_flat)
        obj_val = -acc + mu * lambda_t * z
        grad = np.zeros_like(x)
        grad[:-1] = -grad_acc
        grad[-1] = mu * lambda_t
        return obj_val, grad

    # Bounds: eta in [eta_min, 1], z >= max_comp_delay
    bounds: List[Tuple[Optional[float], Optional[float]]] = [
        (task.eta_min[i], 1.0) for i in links
    ] + [(max_comp_delay, None)]
    x0 = np.array([b[0] for b in bounds], dtype=float)

    # Constraints: z - beta_i * eta_i >= 0
    constraints: List[Dict[str, Any]] = []
    for local_idx, link_i in enumerate(links):
        beta = float(task.a[link_i]) / float(c_eff[link_i])

        def make_constraint(idx_local=local_idx, beta_local=beta):
            return {
                "type": "ineq",
                "fun": lambda x: float(x[-1]) - beta_local * float(x[idx_local]),
                "jac": lambda x: np.array(
                    [
                        (-beta_local if j == idx_local else (1.0 if j == len(x) - 1 else 0.0))
                        for j in range(len(x))
                    ],
                    dtype=float,
                ),
            }

        constraints.append(make_constraint())

    res = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=constraints,
    )

    eta = np.array(res.x[:-1], dtype=float)
    # Safety clamp for numerical drift
    for local_idx, link_i in enumerate(links):
        eta[local_idx] = float(np.clip(eta[local_idx], task.eta_min[link_i], 1.0))
    return eta


class MaxCompressionSingleTaskBaseline(BaseOptimizer):
    """
    CSI-Aware Single-Task Baseline 1: Maximum Compression (Greedy Delay).

    Logic:
        Always apply the minimum allowable compression on every hop:
            eta_i(t) = eta_i^{min},  for all i in [L-1].

    This baseline does NOT perform any feasibility checks. Residual delay is
    intended to be evaluated separately via `BaseOptimizer.compute_residual_delay`.
    """

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        task: InferenceTask = self.tasks[0]

        # Compression fixed at the minimum allowable values.
        eta_vals = [task.eta_min[i] for i in range(task.b_k, task.e_k)]
        eta = np.array(eta_vals, dtype=float)

        # Resources are fully allocated to the single task.
        s_comp = np.ones(task.L_k, dtype=float)
        s_comm = np.ones(task.L_k - 1, dtype=float)

        return self.scale_allocations_to_unit_sum(
            {
                task.task_id: {
                    "eta": eta,
                    "s_comp": s_comp,
                    "s_comm": s_comm,
                }
            }
        )


class NoCompressionSingleTaskBaseline(BaseOptimizer):
    """
    CSI-Aware Single-Task Baseline 2: No Compression (Raw Transmission).

    Logic:
        Always transmit uncompressed data:
            eta_i(t) = 1.0,  for all i in [L-1].

    No feasibility check is performed.
    """

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        task: InferenceTask = self.tasks[0]

        eta = np.ones(task.L_k - 1, dtype=float)
        s_comp = np.ones(task.L_k, dtype=float)
        s_comm = np.ones(task.L_k - 1, dtype=float)

        return self.scale_allocations_to_unit_sum(
            {
                task.task_id: {
                    "eta": eta,
                    "s_comp": s_comp,
                    "s_comm": s_comm,
                }
            }
        )


class UniformCompressionSingleTaskBaseline(BaseOptimizer):
    """
    CSI-Aware Single-Task Baseline 3: Uniform Compression.

    Logic:
        Find a single scalar eta_fixed in [max_i eta_i^{min}, 1] such that
        it satisfies the per-link delay constraint when possible, then clamp
        to [max_i eta_i^{min}, 1]:

            eta_fixed = min(1, min_i c_i(t) / (R(t) * a_i)).
            eta_fixed = max(eta_fixed, max_i eta_i^{min}).

        Finally, set eta_i(t) = eta_fixed for all i.

    No feasibility check is performed; if the resulting configuration violates
    the true constraints, that will be detected via residual delay.
    """

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        task: InferenceTask = self.tasks[0]
        R_t = task.get_R_k(t)

        # Compute the unconstrained uniform compression level from instantaneous CSI.
        ratios = [
            c_t[i] / (R_t * task.a[i]) for i in range(task.b_k, task.e_k)
        ]
        eta_fixed = min(1.0, min(ratios))

        # Enforce minimum accuracy threshold across all links.
        eta_min_max = max(task.eta_min[i] for i in range(task.b_k, task.e_k))
        eta_fixed = max(eta_fixed, eta_min_max)

        eta = np.full(task.L_k - 1, eta_fixed, dtype=float)
        s_comp = np.ones(task.L_k, dtype=float)
        s_comm = np.ones(task.L_k - 1, dtype=float)

        return self.scale_allocations_to_unit_sum(
            {
                task.task_id: {
                    "eta": eta,
                    "s_comp": s_comp,
                    "s_comm": s_comm,
                }
            }
        )


class EstimatedCSICompressionSingleTaskBaseline(BaseOptimizer):
    """
    Online baseline that computes the same closed-form mapping as CSI-aware
    methods, but using *estimated* per-link capacities passed in as `c_hat`.

    For each link i in the task's active links:
        eta_i(t) = min(1, c_hat_i(t) / (R(t) * a_i))

    This class does NOT update/maintain any estimator state. The caller is
    responsible for producing `c_hat` online (e.g. via an estimator that is
    updated after observing true capacities).
    """

    def optimize(
        self, t: int, c_hat: np.ndarray
    ) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        task: InferenceTask = self.tasks[0]
        R_t = task.get_R_k(t)

        links = list(range(task.b_k, task.e_k))
        eta = np.array(
            [
                max(
                    float(task.eta_min[i]),
                    min(1.0, float(c_hat[i]) / (R_t * task.a[i])),
                )
                for i in links
            ],
            dtype=float,
        )

        s_comp = np.ones(task.L_k, dtype=float)
        s_comm = np.ones(task.L_k - 1, dtype=float)

        return self.scale_allocations_to_unit_sum(
            {
                task.task_id: {
                    "eta": eta,
                    "s_comp": s_comp,
                    "s_comm": s_comm,
                }
            }
        )


class MyopicSingleTaskBaseline(EstimatedCSICompressionSingleTaskBaseline):
    """
    Myopic baseline (Last-Observation) implemented by:
      1. Producing `c_hat` with a Last-Observation estimator (caller-owned).
      2. Calling `EstimatedCSICompressionSingleTaskBaseline.optimize(t, c_hat)`.
    """


class ConservativeSingleTaskBaseline(EstimatedCSICompressionSingleTaskBaseline):
    """
    Conservative baseline (Running Minimum) implemented by:
      1. Producing `c_hat` with a Running-Min estimator (caller-owned).
      2. Calling `EstimatedCSICompressionSingleTaskBaseline.optimize(t, c_hat)`.
    """


class MovingAverageSingleTaskBaseline(EstimatedCSICompressionSingleTaskBaseline):
    """
    Moving-average baseline implemented by:
      1. Producing `c_hat` with a Moving-Average estimator (caller-owned).
      2. Calling `EstimatedCSICompressionSingleTaskBaseline.optimize(t, c_hat)`.
    """


class LCBSingleTaskBaseline(EstimatedCSICompressionSingleTaskBaseline):
    """
    LCB baseline implemented by:
      1. Producing `c_hat` with a lower-confidence-bound estimator (caller-owned),
         e.g. `MeanMinusZStdLCB`.
      2. Calling `EstimatedCSICompressionSingleTaskBaseline.optimize(t, c_hat)`.
    """


# =============================================================================
# Multi-task baselines
# =============================================================================


class MaxCompressionMultiTaskBaseline(BaseOptimizer):
    """
    CSI-Aware Multi-Task: maximum compression with static equal resource shares.

    Uses the same per-node / per-link equal split of s_comp and s_comm as
    `StaticEqualShareMultiTaskBaseline`, but fixes compression on every hop to
    the minimum allowable value: eta_i,k = eta_{i,k}^{min}.

    Does not perform feasibility checks; residual delay is evaluated via
    `BaseOptimizer.compute_residual_delay`.
    """

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        allocations: Dict[int, Dict[str, np.ndarray]] = {}

        tasks_on_node = {i: _active_tasks_on_node(self.tasks, i) for i in range(self.M)}
        tasks_on_link = {i: _active_tasks_on_link(self.tasks, i) for i in range(self.M - 1)}

        for task in self.tasks:
            s_comp = np.zeros(task.L_k, dtype=float)
            for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1)):
                n = len(tasks_on_node[node_i])
                s_comp[local_idx] = 1.0 / n if n > 0 else 0.0

            s_comm = np.zeros(task.L_k - 1, dtype=float)
            eta = np.zeros(task.L_k - 1, dtype=float)
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                n = len(tasks_on_link[link_i])
                s_comm_val = 1.0 / n if n > 0 else 0.0
                s_comm[local_idx] = s_comm_val
                eta[local_idx] = float(task.eta_min[link_i])

            allocations[task.task_id] = {"eta": eta, "s_comp": s_comp, "s_comm": s_comm}

        return self.scale_allocations_to_unit_sum(allocations)


class NoCompressionMultiTaskBaseline(BaseOptimizer):
    """
    CSI-Aware Multi-Task: no compression with static equal resource shares.

    Same s_comp / s_comm splitting as `StaticEqualShareMultiTaskBaseline`, but
    fixes eta_i,k = 1 on every hop (uncompressed transmission).
    """

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        allocations: Dict[int, Dict[str, np.ndarray]] = {}

        tasks_on_node = {i: _active_tasks_on_node(self.tasks, i) for i in range(self.M)}
        tasks_on_link = {i: _active_tasks_on_link(self.tasks, i) for i in range(self.M - 1)}

        for task in self.tasks:
            s_comp = np.zeros(task.L_k, dtype=float)
            for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1)):
                n = len(tasks_on_node[node_i])
                s_comp[local_idx] = 1.0 / n if n > 0 else 0.0

            s_comm = np.zeros(task.L_k - 1, dtype=float)
            eta = np.ones(task.L_k - 1, dtype=float)
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                n = len(tasks_on_link[link_i])
                s_comm_val = 1.0 / n if n > 0 else 0.0
                s_comm[local_idx] = s_comm_val

            allocations[task.task_id] = {"eta": eta, "s_comp": s_comp, "s_comm": s_comm}

        return self.scale_allocations_to_unit_sum(allocations)


class StaticEqualShareMultiTaskBaseline(BaseOptimizer):
    """
    CSI-Aware Multi-Task Baseline 1: Static Equal-Share Allocation.
    """

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        allocations: Dict[int, Dict[str, np.ndarray]] = {}

        # Precompute per-node and per-link task sets
        tasks_on_node = {i: _active_tasks_on_node(self.tasks, i) for i in range(self.M)}
        tasks_on_link = {i: _active_tasks_on_link(self.tasks, i) for i in range(self.M - 1)}

        for task in self.tasks:
            R_k = task.get_R_k(t)

            # Compute shares per node on the task path
            s_comp = np.zeros(task.L_k, dtype=float)
            for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1)):
                n = len(tasks_on_node[node_i])
                s_comp[local_idx] = 1.0 / n if n > 0 else 0.0

            # Comm shares per link on the task path
            s_comm = np.zeros(task.L_k - 1, dtype=float)
            eta = np.zeros(task.L_k - 1, dtype=float)
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                n = len(tasks_on_link[link_i])
                s_comm_val = 1.0 / n if n > 0 else 0.0
                s_comm[local_idx] = s_comm_val

                # eta_i,k = min(1, s_comm * c / (R * a))
                raw = (s_comm_val * float(c_t[link_i])) / (R_k * float(task.a[link_i]))
                eta_val = min(1.0, raw)
                eta[local_idx] = max(float(task.eta_min[link_i]), float(eta_val))

            allocations[task.task_id] = {"eta": eta, "s_comp": s_comp, "s_comm": s_comm}

        return self.scale_allocations_to_unit_sum(allocations)


class ProportionalResourceAllocationMultiTaskBaseline(BaseOptimizer):
    """
    CSI-Aware Multi-Task Baseline 2: Proportional Resource Allocation.
    """

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        allocations: Dict[int, Dict[str, np.ndarray]] = {}

        tasks_on_node = {i: _active_tasks_on_node(self.tasks, i) for i in range(self.M)}
        tasks_on_link = {i: _active_tasks_on_link(self.tasks, i) for i in range(self.M - 1)}

        tau_sum = {
            i: sum(float(task.tau[i]) for task in tasks_on_node[i]) for i in range(self.M)
        }
        a_sum = {
            i: sum(float(task.a[i]) for task in tasks_on_link[i]) for i in range(self.M - 1)
        }

        for task in self.tasks:
            R_k = task.get_R_k(t)

            s_comp = np.zeros(task.L_k, dtype=float)
            for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1)):
                denom = float(tau_sum[node_i])
                s_comp[local_idx] = (float(task.tau[node_i]) / denom) if denom > 0 else 0.0

            s_comm = np.zeros(task.L_k - 1, dtype=float)
            eta = np.zeros(task.L_k - 1, dtype=float)
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                denom = float(a_sum[link_i])
                s_comm_val = (float(task.a[link_i]) / denom) if denom > 0 else 0.0
                s_comm[local_idx] = s_comm_val
                raw = (s_comm_val * float(c_t[link_i])) / (R_k * float(task.a[link_i]))
                eta_val = min(1.0, raw)
                eta[local_idx] = max(float(task.eta_min[link_i]), float(eta_val))

            allocations[task.task_id] = {"eta": eta, "s_comp": s_comp, "s_comm": s_comm}

        return self.scale_allocations_to_unit_sum(allocations)


class StrictPriorityGreedyMultiTaskBaseline(BaseOptimizer):
    """
    CSI-Aware Multi-Task Baseline 3: Strict Priority (Greedy).
    """

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        # Initialize with base allocations ensuring eta>=eta_min feasibility (when feasible).
        s_comp_map: Dict[int, np.ndarray] = {}
        s_comm_map: Dict[int, np.ndarray] = {}

        # Base compute: s_{i,k}^{comp} = tau_{i,k} * R_k(t)
        for task in self.tasks:
            R_k = task.get_R_k(t)
            s_comp_map[task.task_id] = np.array(
                [float(task.tau[i]) * float(R_k) for i in range(task.b_k, task.e_k + 1)],
                dtype=float,
            )

            if task.L_k > 1:
                s_comm_map[task.task_id] = np.array(
                    [
                        (float(task.a[i]) * float(task.eta_min[i]) * float(R_k)) / float(c_t[i])
                        for i in range(task.b_k, task.e_k)
                    ],
                    dtype=float,
                )
            else:
                s_comm_map[task.task_id] = np.array([], dtype=float)

        # Greedy allocation of remaining resources by w_k (done per node/link independently).
        tasks_by_weight = sorted(self.tasks, key=lambda x: float(x.w_k), reverse=True)

        # Compute: allocate remaining per node
        for node_i in range(self.M):
            active = _active_tasks_on_node(self.tasks, node_i)
            base_sum = sum(
                float(s_comp_map[task.task_id][node_i - task.b_k])
                for task in active
            )
            rem = max(0.0, 1.0 - base_sum)
            if rem <= 0 or not active:
                continue

            for task in tasks_by_weight:
                if not task.is_active_at_node(node_i) or rem <= 0:
                    continue
                # No explicit upper target; just pour remaining compute to priority tasks.
                idx = node_i - task.b_k
                s_comp_map[task.task_id][idx] += rem
                rem = 0.0

        # Communication: allocate remaining per link toward eta=1
        for link_i in range(self.M - 1):
            active = _active_tasks_on_link(self.tasks, link_i)
            base_sum = sum(
                float(s_comm_map[task.task_id][link_i - task.b_k])
                for task in active
            )
            rem = max(0.0, 1.0 - base_sum)
            if rem <= 0 or not active:
                continue

            for task in tasks_by_weight:
                if not task.is_active_on_link(link_i) or rem <= 0:
                    continue
                R_k = task.get_R_k(t)
                base = float(s_comm_map[task.task_id][link_i - task.b_k])
                # Extra needed to reach eta=1:
                target = (float(task.a[link_i]) * float(R_k)) / float(c_t[link_i])
                req = max(0.0, target - base)
                add = min(req, rem)
                s_comm_map[task.task_id][link_i - task.b_k] += add
                rem -= add

        # Pack final allocations (derive eta from assigned comm shares)
        allocations: Dict[int, Dict[str, np.ndarray]] = {}
        for task in self.tasks:
            R_k = task.get_R_k(t)
            s_comp = s_comp_map[task.task_id]
            s_comm = s_comm_map[task.task_id]
            eta = np.array(
                [
                    max(
                        float(task.eta_min[i]),
                        min(1.0, (float(s_comm[i - task.b_k]) * float(c_t[i])) / (float(R_k) * float(task.a[i]))),
                    )
                    for i in range(task.b_k, task.e_k)
                ],
                dtype=float,
            ) if task.L_k > 1 else np.array([], dtype=float)

            allocations[task.task_id] = {"eta": eta, "s_comp": s_comp, "s_comm": s_comm}

        return self.scale_allocations_to_unit_sum(allocations)


class DecoupledEqualSplitStochasticDescentMultiTaskBaseline(BaseOptimizer):
    """
    No-CSI Multi-Task Baseline 4: Decoupled Equal-Split Stochastic Descent.
    Caller provides c_hat; this baseline maintains per-task dual queues.
    """

    def __init__(self, M: int, tasks: List[InferenceTask], mu: float, epsilon: float = 0.1):
        super().__init__(M, tasks)
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.lambda_k: Dict[int, float] = {task.task_id: float(epsilon) for task in tasks}

    def optimize(self, t: int, c_hat: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        allocations: Dict[int, Dict[str, np.ndarray]] = {}

        tasks_on_node = {i: _active_tasks_on_node(self.tasks, i) for i in range(self.M)}
        tasks_on_link = {i: _active_tasks_on_link(self.tasks, i) for i in range(self.M - 1)}

        for task in self.tasks:
            # Equal splits
            s_comp = np.zeros(task.L_k, dtype=float)
            for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1)):
                n = len(tasks_on_node[node_i])
                s_comp[local_idx] = 1.0 / n if n > 0 else 0.0

            s_comm = np.zeros(task.L_k - 1, dtype=float)
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                n = len(tasks_on_link[link_i])
                s_comm[local_idx] = 1.0 / n if n > 0 else 0.0

            # Effective channel for task k on link i: c_eff = s_comm * c_hat
            c_eff = np.array(c_hat, copy=True, dtype=float)
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                c_eff[link_i] = float(s_comm[local_idx]) * float(c_hat[link_i])

            # Compute-delay lower bound: max_i tau_i / s_comp_i
            max_comp_delay = max(
                float(task.tau[node_i]) / max(float(s_comp[local_idx]), 1e-12)
                for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1))
            )

            eta = _solve_single_task_eta_with_fixed_resources(
                task=task,
                t=t,
                c_eff=c_eff,
                max_comp_delay=max_comp_delay,
                mu=self.mu,
                lambda_t=float(self.lambda_k[task.task_id]),
            )

            allocations[task.task_id] = {"eta": eta, "s_comp": s_comp, "s_comm": s_comm}

        return self.scale_allocations_to_unit_sum(allocations)

    def update_dual(self, t: int, actual_delays: Dict[int, float]):
        for task in self.tasks:
            tid = task.task_id
            if tid not in actual_delays:
                continue
            D_act = float(actual_delays[tid])
            R_t = float(task.get_R_k(t))
            self.lambda_k[tid] = max(self.epsilon, float(self.lambda_k[tid]) + D_act - (1.0 / R_t))


class QueueProportionalHeuristicMultiTaskBaseline(BaseOptimizer):
    """
    No-CSI Multi-Task Baseline 5: Queue-Proportional Heuristic.
    Caller provides c_hat; this baseline maintains per-task dual queues.
    """

    def __init__(self, M: int, tasks: List[InferenceTask], mu: float, epsilon: float = 0.1):
        super().__init__(M, tasks)
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.lambda_k: Dict[int, float] = {task.task_id: float(epsilon) for task in tasks}

    def optimize(self, t: int, c_hat: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        allocations: Dict[int, Dict[str, np.ndarray]] = {}

        for task in self.tasks:
            # will fill later
            allocations[task.task_id] = {"eta": None, "s_comp": None, "s_comm": None}  # type: ignore[assignment]

        # Coupled resource allocation based on lambda, with equal-split fallback
        tasks_on_node = {i: _active_tasks_on_node(self.tasks, i) for i in range(self.M)}
        tasks_on_link = {i: _active_tasks_on_link(self.tasks, i) for i in range(self.M - 1)}

        # Build s_comp / s_comm dicts for each task
        s_comp_map: Dict[int, np.ndarray] = {task.task_id: np.zeros(task.L_k, dtype=float) for task in self.tasks}
        s_comm_map: Dict[int, np.ndarray] = {task.task_id: np.zeros(task.L_k - 1, dtype=float) for task in self.tasks}

        for node_i in range(self.M):
            active = tasks_on_node[node_i]
            if not active:
                continue
            denom = sum(float(self.lambda_k[t.task_id]) for t in active)
            for task in active:
                idx = node_i - task.b_k
                if denom <= 0:
                    s_comp_map[task.task_id][idx] = 1.0 / len(active)
                else:
                    s_comp_map[task.task_id][idx] = float(self.lambda_k[task.task_id]) / denom

        for link_i in range(self.M - 1):
            active = tasks_on_link[link_i]
            if not active:
                continue
            denom = sum(float(self.lambda_k[t.task_id]) for t in active)
            for task in active:
                idx = link_i - task.b_k
                if denom <= 0:
                    s_comm_map[task.task_id][idx] = 1.0 / len(active)
                else:
                    s_comm_map[task.task_id][idx] = float(self.lambda_k[task.task_id]) / denom

        # Decoupled per-task compression optimization
        for task in self.tasks:
            s_comp = s_comp_map[task.task_id]
            s_comm = s_comm_map[task.task_id]

            c_eff = np.array(c_hat, copy=True, dtype=float)
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                c_eff[link_i] = float(s_comm[local_idx]) * float(c_hat[link_i])

            max_comp_delay = max(
                float(task.tau[node_i]) / max(float(s_comp[local_idx]), 1e-12)
                for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1))
            )

            eta = _solve_single_task_eta_with_fixed_resources(
                task=task,
                t=t,
                c_eff=c_eff,
                max_comp_delay=max_comp_delay,
                mu=self.mu,
                lambda_t=float(self.lambda_k[task.task_id]),
            )

            allocations[task.task_id] = {"eta": eta, "s_comp": s_comp, "s_comm": s_comm}

        return self.scale_allocations_to_unit_sum(allocations)

    def update_dual(self, t: int, actual_delays: Dict[int, float]):
        for task in self.tasks:
            tid = task.task_id
            if tid not in actual_delays:
                continue
            D_act = float(actual_delays[tid])
            R_t = float(task.get_R_k(t))
            self.lambda_k[tid] = max(self.epsilon, float(self.lambda_k[tid]) + D_act - (1.0 / R_t))


class HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(BaseOptimizer):
    """
    No-CSI Multi-Task Baseline 6: Historical Average (Certainty Equivalence).
    This baseline expects a pre-computed c_hat(t) (historical mean) and then
    simply runs the CSI-aware multi-task convex solver on that c_hat.
    """

    def __init__(self, M: int, tasks: List[InferenceTask]):
        super().__init__(M, tasks)
        from src.optimizers.csi_aware import CSIAwareMultiTaskOptimizer

        self._solver = CSIAwareMultiTaskOptimizer(M=M, tasks=tasks)

    def optimize(self, t: int, c_hat: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        return self._solver.optimize(t=t, c_t=c_hat)


class LCBCertaintyEquivalenceMultiTaskBaseline(BaseOptimizer):
    """
    No-CSI Multi-Task Baseline 7: LCB (Certainty Equivalence).
    This baseline expects a pre-computed c_hat(t) from an LCB estimator and then
    runs the CSI-aware multi-task convex solver on that c_hat.
    """

    def __init__(self, M: int, tasks: List[InferenceTask]):
        super().__init__(M, tasks)
        from src.optimizers.csi_aware import CSIAwareMultiTaskOptimizer

        self._solver = CSIAwareMultiTaskOptimizer(M=M, tasks=tasks)

    def optimize(self, t: int, c_hat: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        return self._solver.optimize(t=t, c_t=c_hat)
