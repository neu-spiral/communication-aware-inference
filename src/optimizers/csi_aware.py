import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any

from src.optimizers.base import BaseOptimizer
from src.core.task import InferenceTask


class CSIAwareSingleTaskOptimizer(BaseOptimizer):
    """
    Implements the closed-form optimal solution for a single task with perfect CSI (Problem 1).
    """
    def check_feasibility(self, t: int, c_t: np.ndarray) -> bool:
        """Implements Lemma 3.1 (Single-Task Feasibility Condition)[cite: 88]."""
        task = self.tasks[0]
        R_t = task.get_R_k(t)

        # [cite_start]Condition 3: Check computation delay bounds [cite: 90]
        if any(task.tau[i] > 1.0 / R_t for i in range(task.b_k, task.e_k + 1)):
            return False

        # [cite_start]Condition 4: Check communication delay bounds [cite: 90]
        if any((task.a[i] * task.eta_min[i]) / c_t[i] > 1.0 / R_t for i in range(task.b_k, task.e_k)):
            return False

        return True

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """Implements Theorem 3.2 (Closed-Form Optimal Solution)[cite: 105]."""
        if not self.check_feasibility(t, c_t):
            return None

        task = self.tasks[0]
        R_t = task.get_R_k(t)

        # [cite_start]Apply Theorem 3.2: \eta_i^*(t) = min(1, c_i(t) / (R(t) * a_i)) [cite: 106]
        eta = np.array([min(1.0, c_t[i] / (R_t * task.a[i])) for i in range(task.b_k, task.e_k)])
        
        # Resources are fully allocated to the single task
        s_comp = np.ones(task.L_k)
        s_comm = np.ones(task.L_k - 1)

        return self.scale_allocations_to_unit_sum(
            {task.task_id: {'eta': eta, 's_comp': s_comp, 's_comm': s_comm}}
        )


class CSIAwareMultiTaskOptimizer(BaseOptimizer):
    """
    Implements the convex optimization solver for multiple tasks with perfect CSI (Problem 2).
    Updated to use the reformulated smooth problem bounded strictly by s_max.
    """
    def check_feasibility(self, t: int, c_t: np.ndarray) -> bool:
        """Implements Lemma 3.3 (Multi-Task Feasibility Condition)[cite: 157]."""
        # Condition (14): Check computation sums at every node [cite: 158]
        for i in range(self.M):
            comp_sum = sum(task.tau[i] * task.get_R_k(t) for task in self.tasks if task.is_active_at_node(i))
            if comp_sum > 1.0:
                return False

        # Condition (15): Check communication sums at every link [cite: 159]
        for i in range(self.M - 1):
            comm_sum = sum(
                (task.a[i] * task.eta_min[i] * task.get_R_k(t)) / c_t[i]
                for task in self.tasks if task.is_active_on_link(i)
            )
            if comm_sum > 1.0:
                return False

        return True

    # ==========================================
    # FUNCTIONAL HELPERS FOR SCI-PY SOLVER
    # ==========================================

    def _build_context(self, t: int, c_t: np.ndarray) -> List[Dict[str, Any]]:
        """Maps active task-link pairs to a flat list to interface with SciPy's 1D arrays."""
        context = []
        for task in self.tasks:
            R_k = task.get_R_k(t)
            for i in range(task.b_k, task.e_k):
                beta = c_t[i] / (R_k * task.a[i]) # Derivative term [cite: 208]
                
                # Minimum QoS Constraint (Lower Bound) [cite: 205]
                s_min = (task.a[i] * task.eta_min[i] * R_k) / c_t[i] 
                
                # Maximum Compression Constraint (New Upper Bound)
                # s <= R_k * a_i / c_i (which is exactly 1.0 / beta). 
                # We also cap it at 1.0 since a share of a link cannot exceed 100%.
                s_max = min(1.0, 1.0 / beta) 
                
                context.append({
                    'task': task,
                    'link_i': i,
                    'beta': beta,
                    's_min': s_min,
                    's_max': s_max
                })
        return context

    def _calculate_eta_and_derivative(self, s_val: float, beta: float) -> Tuple[float, float]:
        """
        Evaluates the compression factor and its derivative.
        Due to the new upper bounds, the piecewise logic is removed. The relationship 
        is strictly linear within the solver's feasible region.
        """
        # A tiny min() wrapper is kept purely to prevent float precision rounding 
        # from handing an eta like 1.0000000001 to your black-box callable.
        eta_val = min(1.0, beta * s_val)
        
        # The derivative is now constantly beta! The solver's boundary constraints
        # naturally handle the stopping condition instead of a 0.0 gradient.
        d_eta_ds = beta 
        
        return eta_val, d_eta_ds

    def _evaluate_objective(self, s_flat: np.ndarray, context: List[Dict[str, Any]]) -> Tuple[float, np.ndarray]:
        """Calculates the objective function and flat gradient array using the chain rule[cite: 133, 208]."""
        total_obj = 0.0
        grad_flat = np.zeros_like(s_flat)

        for task in self.tasks:
            task_indices = [idx for idx, ctx in enumerate(context) if ctx['task'] == task]
            
            if not task_indices:
                continue

            eta_vec = []
            d_eta_ds_vec = []

            for idx in task_indices:
                s_val = s_flat[idx]
                beta = context[idx]['beta']
                eta_val, d_eta_ds = self._calculate_eta_and_derivative(s_val, beta)
                
                eta_vec.append(eta_val)
                d_eta_ds_vec.append(d_eta_ds)

            eta_array = np.array(eta_vec)

            acc = task.A_k(eta_array)
            grad_acc = task.grad_A_k(eta_array)

            total_obj += task.w_k * acc

            for local_idx, flat_idx in enumerate(task_indices):
                chain_rule_grad = task.w_k * grad_acc[local_idx] * d_eta_ds_vec[local_idx]
                grad_flat[flat_idx] = chain_rule_grad

        return -total_obj, -grad_flat

    def _build_constraints(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generates the linear capacity constraints for each link: sum(s) <= 1[cite: 138]."""
        constraints = []
        for link_i in range(self.M - 1):
            indices_on_link = [idx for idx, ctx in enumerate(context) if ctx['link_i'] == link_i]
            
            if indices_on_link:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda s_flat, idxs=indices_on_link: 1.0 - sum(s_flat[i] for i in idxs)
                })
        return constraints

    def _pack_results(self, t: int, c_t: np.ndarray, s_flat: np.ndarray, context: List[Dict[str, Any]]) -> Dict[int, Dict[str, np.ndarray]]:
        """Unpacks the optimized 1D array back into the final operational dictionary."""
        final_allocations = {}
        for task in self.tasks:
            R_k = task.get_R_k(t)
            
            # Compute shares are decoupled and set directly based on QoS [cite: 189]
            s_comp = np.array([task.tau[i] * R_k for i in range(task.b_k, task.e_k + 1)])
            
            task_indices = [idx for idx, ctx in enumerate(context) if ctx['task'] == task]
            
            s_comm = []
            eta = []
            
            for idx in task_indices:
                s_val = s_flat[idx]
                beta = context[idx]['beta']
                eta_val, _ = self._calculate_eta_and_derivative(s_val, beta)
                
                s_comm.append(s_val)
                eta.append(eta_val)

            final_allocations[task.task_id] = {
                'eta': np.array(eta),
                's_comp': s_comp,
                's_comm': np.array(s_comm)
            }
            
        return final_allocations

    # ==========================================
    # MAIN OPTIMIZE METHOD
    # ==========================================

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """Reduces to a convex optimization problem over s^{comm}(t) (Theorem 3.4)[cite: 179]."""
        if not self.check_feasibility(t, c_t):
            return None

        context = self._build_context(t, c_t)
        
        # UPDATED: Setup bounds using the newly formulated s_max
        bounds = [(ctx['s_min'], ctx['s_max']) for ctx in context]
        x0 = np.array([ctx['s_min'] for ctx in context])

        constraints = self._build_constraints(context)

        result = minimize(
            fun=self._evaluate_objective,
            x0=x0,
            args=(context,),
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=constraints
        )

        return self.scale_allocations_to_unit_sum(
            self._pack_results(t, c_t, result.x, context)
        )