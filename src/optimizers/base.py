from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np

from src.core.task import InferenceTask

class BaseOptimizer(ABC):
    """
    Abstract base class for all distributed inference optimizers.
    """
    def __init__(self, M: int, tasks: List[InferenceTask]):
        """
        Initializes the optimizer with the physical system topology and task list.
        
        Args:
            M (int): The number of computational nodes arranged in a linear topology. 
            tasks (List[InferenceTask]): The list of distinct inference tasks running on the system.
        """
        self.M = M
        self.tasks = tasks
        self.K = len(tasks)

    def check_feasibility(self, t: int, c_t: np.ndarray) -> bool:
        """
        Evaluates whether a valid resource allocation exists.
        
        By default, this returns True (used by No-CSI algorithms which handle 
        violations via long-term dual queues). 
        CSI-Aware subclasses MUST override this to implement strict per-slot checks.
        """
        return True

    @abstractmethod
    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """
        Determines the optimal resource allocations for a given time slot.
        Implementations MUST call `check_feasibility` before attempting to solve.
        
        Args:
            t (int): The current time slot.
            c_t (np.ndarray): The channel capacities at time slot t.
                              
        Returns:
            Optional[Dict[int, Dict[str, np.ndarray]]]: 
                A nested dictionary mapping each task_id to its specific configuration.
                Returns None if the problem is infeasible for the current time slot.
        """
        pass

    def scale_allocations_to_unit_sum(
        self, allocations: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Scale s_comp and s_comm so that at each node (resp. link) the sum across
        tasks equals 1. E.g. if two tasks each have s_comp=0.1 at a node (sum=0.2),
        both become 0.5 at that node. Leaves eta unchanged.
        """
        if not allocations:
            return allocations

        M = self.M
        comp_sum_per_node = np.zeros(M)
        comm_sum_per_link = np.zeros(M - 1)

        for task in self.tasks:
            tid = task.task_id
            if tid not in allocations:
                continue
            s_comp = allocations[tid]["s_comp"]
            s_comm = allocations[tid]["s_comm"]
            for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1)):
                comp_sum_per_node[node_i] += s_comp[local_idx]
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                comm_sum_per_link[link_i] += s_comm[local_idx]

        result = {}
        for task in self.tasks:
            tid = task.task_id
            if tid not in allocations:
                continue
            s_comp = np.array(allocations[tid]["s_comp"], dtype=float)
            s_comm = np.array(allocations[tid]["s_comm"], dtype=float)
            for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1)):
                total = comp_sum_per_node[node_i]
                if total > 0:
                    s_comp[local_idx] /= total
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                total = comm_sum_per_link[link_i]
                if total > 0:
                    s_comm[local_idx] /= total
            result[tid] = {
                **allocations[tid],
                "s_comp": s_comp,
                "s_comm": s_comm,
            }
        return result

    def update_dual(self, t: int, actual_delays: Dict[int, float]):
        """
        Updates the dual variables (Lagrange multipliers) based on the true 
        experienced bottleneck delay. 
        
        Args:
            t (int): The current time slot.
            actual_delays (Dict[int, float]): A mapping of task_id to the actual 
                                              bottleneck delay experienced.
        """
        pass

    def compute_residual_delay(
        self,
        t: int,
        allocations: Dict[int, Dict[str, np.ndarray]],
        c_t: np.ndarray,
    ) -> Dict[int, float]:
        """
        Computes the per-task residual delay for a given slot t.

        For each task k, this evaluates the bottleneck delay
            D_k(t) = max(
                { tau_{i,k} / s^{comp}_{i,k}(t) }_{i in stages},
                { a_{i,k} * eta_{i,k}(t) / (s^{comm}_{i,k}(t) * c_i(t)) }_{i in links}
            )
        and returns
            D_k(t) - 1 / R_k(t).

        Args:
            t (int): Current time slot.
            allocations (Dict[int, Dict[str, np.ndarray]]): Optimizer output mapping
                each task_id to its allocation dict with keys:
                - 'eta': compression vector over links (length L_k - 1)
                - 's_comp': compute share vector over nodes (length L_k)
                - 's_comm': comm share vector over links (length L_k - 1)
            c_t (np.ndarray): Vector of link capacities c_i(t) for i in [0, M-2].

        Returns:
            Dict[int, float]: Mapping from task_id to its residual delay
            D_k(t) - 1 / R_k(t).
        """
        residuals: Dict[int, float] = {}

        for task in self.tasks:
            tid = task.task_id
            R_t = task.get_R_k(t)

            if tid not in allocations:
                raise KeyError(f"Missing allocations for task_id={tid}.")
            task_alloc = allocations[tid]

            if "s_comp" not in task_alloc:
                raise KeyError(f"Missing 's_comp' in allocations[{tid}].")
            s_comp_vec = task_alloc["s_comp"]
            expected_s_comp_len = task.L_k
            if len(s_comp_vec) != expected_s_comp_len:
                raise ValueError(
                    f"allocations[{tid}]['s_comp'] has length {len(s_comp_vec)}, expected {expected_s_comp_len}."
                )

            # Computation delays across the assigned nodes: tau_{i,k} / s^{comp}_{i,k}(t).
            comp_delays = []
            for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1)):
                comp_share = float(s_comp_vec[local_idx])
                if comp_share <= 0:
                    raise ValueError(f"s_comp must be > 0, got {comp_share} for task_id={tid}.")
                comp_delays.append(task.tau[node_i] / comp_share)

            # Communication delays using provided eta_t and instantaneous capacities.
            comm_delays = []
            if task.L_k > 1:
                if "eta" not in task_alloc:
                    raise KeyError(f"Missing 'eta' in allocations[{tid}].")
                eta_vec = task_alloc["eta"]
                expected_len = task.L_k - 1
                if len(eta_vec) != expected_len:
                    raise ValueError(
                        f"allocations[{tid}]['eta'] has length {len(eta_vec)}, expected {expected_len}."
                    )

                if "s_comm" not in task_alloc:
                    raise KeyError(f"Missing 's_comm' in allocations[{tid}].")
                s_comm_vec = task_alloc["s_comm"]
                if len(s_comm_vec) != expected_len:
                    raise ValueError(
                        f"allocations[{tid}]['s_comm'] has length {len(s_comm_vec)}, expected {expected_len}."
                    )

                for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                    comm_share = float(s_comm_vec[local_idx])
                    if comm_share <= 0:
                        raise ValueError(f"s_comm must be > 0, got {comm_share} for task_id={tid}.")
                    comm_delay = (task.a[link_i] * eta_vec[local_idx]) / (comm_share * c_t[link_i])
                    comm_delays.append(comm_delay)

            D_k = max(comp_delays + comm_delays)
            residuals[tid] = D_k - 1.0 / R_t

        return residuals