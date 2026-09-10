import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any

from src.optimizers.base import BaseOptimizer
from src.core.task import InferenceTask

class NoCSISingleTaskOptimizer(BaseOptimizer):
    """
    Implements Algorithm 1: Estimated Stochastic Dual Descent for a single task [cite: 244-260].
    """
    def __init__(self, M: int, tasks: List[InferenceTask], mu: float, epsilon: float = 0.1):
        super().__init__(M, tasks)
        self.mu = mu
        self.lambda_t = epsilon
        self.epsilon = epsilon


    def _build_bounds(self, task: InferenceTask, links: List[int], max_tau: float) -> List[Tuple[Optional[float], Optional[float]]]:
        """Enforces \eta_i^{min} <= \eta_i <= 1 and z >= max(\tau)."""
        bounds = [(task.eta_min[i], 1.0) for i in links]
        bounds.append((max_tau, None)) # Auxiliary variable z
        return bounds

    def _objective_function(self, x: np.ndarray, task: InferenceTask) -> Tuple[float, np.ndarray]:
        """Evaluates min -A(\eta) + \mu * \lambda_t * z."""
        eta_flat = x[:-1]
        z = x[-1]
        
        acc = task.A_k(eta_flat)
        grad_acc = task.grad_A_k(eta_flat)
        
        obj_val = -acc + self.mu * self.lambda_t * z
        
        grad = np.zeros_like(x)
        grad[:-1] = -grad_acc
        grad[-1] = self.mu * self.lambda_t
        
        return obj_val, grad

    def _build_constraints(self, task: InferenceTask, links: List[int], c_hat: np.ndarray) -> List[Dict[str, Any]]:
        """Enforces z >= (a_i / \hat{c}_i) * \eta_i for all links."""
        constraints = []
        for idx, i in enumerate(links):
            beta = task.a[i] / c_hat[i]
            
            # Closure to capture current loop variables securely
            def make_constraint(idx_local=idx, beta_local=beta):
                return {
                    'type': 'ineq',
                    'fun': lambda x: x[-1] - beta_local * x[idx_local],
                    'jac': lambda x: np.array([-beta_local if j == idx_local else (1.0 if j == len(x)-1 else 0.0) for j in range(len(x))])
                }
            constraints.append(make_constraint())
        return constraints

    # ==========================================
    # MAIN OPTIMIZE METHOD
    # ==========================================

    def optimize(self, t: int, c_hat: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        task = self.tasks[0]
        links = list(range(task.b_k, task.e_k))
        nodes = list(range(task.b_k, task.e_k + 1))
        
        max_tau = max([task.tau[i] for i in nodes])
        bounds = self._build_bounds(task, links, max_tau)
        x0 = np.array([b[0] for b in bounds]) # Start at lower bounds

        constraints = self._build_constraints(task, links, c_hat)

        result = minimize(
            fun=self._objective_function,
            x0=x0,
            args=(task,),
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=constraints
        )

        return self.scale_allocations_to_unit_sum({
            task.task_id: {
                'eta': result.x[:-1],
                's_comp': np.ones(task.L_k),
                's_comm': np.ones(len(links))
            }
        })

    def update_dual(self, t: int, actual_delays: Dict[int, float]):
        """Dual Step: Update the multiplier [cite: 258-259]."""
        task = self.tasks[0]
        if task.task_id in actual_delays:
            D_act = actual_delays[task.task_id]
            R_t = task.get_R_k(t)
            self.lambda_t = max(self.epsilon, self.lambda_t + D_act - (1.0 / R_t))


class NoCSIMultiTaskOptimizer(BaseOptimizer):
    """
    Implements Algorithm 2: Multi-Task Estimated Stochastic Dual Descent using BCD [cite: 326-348].
    """
    def __init__(self, M: int, tasks: List[InferenceTask], mu: float, epsilon: float = 0.1, J: int = 10):
        super().__init__(M, tasks)
        self.mu = mu
        self.epsilon = epsilon
        self.J = J
        self.lambda_k = {task.task_id: epsilon for task in tasks}

    # ==========================================
    # FUNCTIONAL HELPERS: PHASE A (RESOURCES)
    # ==========================================

    def _phase_a_objective(self, z_flat: np.ndarray, active_tasks: List[InferenceTask]) -> Tuple[float, np.ndarray]:
        """Objective: min \sum \mu * \lambda_k * z_k[cite: 352]."""
        obj_val = sum(self.mu * self.lambda_k[task.task_id] * z_flat[idx] for idx, task in enumerate(active_tasks))
        grad = np.array([self.mu * self.lambda_k[task.task_id] for task in active_tasks])
        return obj_val, grad

    def _phase_a_constraints(self, active_tasks: List[InferenceTask], current_eta: Dict[int, np.ndarray], c_hat: np.ndarray) -> List[Dict[str, Any]]:
        """Node and link capacity constraints: sum(tau/z) <= 1 and sum(a*eta/(c*z)) <= 1 [cite: 358-361]."""
        constraints = []
        
        # Node constraints
        for i in range(self.M):
            tasks_on_node = [(idx, t_obj) for idx, t_obj in enumerate(active_tasks) if t_obj.is_active_at_node(i)]
            if tasks_on_node:
                def make_node_constraint(node_tasks=tasks_on_node, node_i=i):
                    return {'type': 'ineq', 'fun': lambda z: 1.0 - sum(t_obj.tau[node_i] / z[idx] for idx, t_obj in node_tasks)}
                constraints.append(make_node_constraint())

        # Link constraints
        for i in range(self.M - 1):
            tasks_on_link = [(idx, t_obj) for idx, t_obj in enumerate(active_tasks) if t_obj.is_active_on_link(i)]
            if tasks_on_link:
                def make_link_constraint(link_tasks=tasks_on_link, link_i=i):
                    return {'type': 'ineq', 'fun': lambda z: 1.0 - sum(
                        (t_obj.a[link_i] * current_eta[t_obj.task_id][link_i - t_obj.b_k]) / (c_hat[link_i] * z[idx]) 
                        for idx, t_obj in link_tasks
                    )}
                constraints.append(make_link_constraint())
        return constraints

    def _phase_a_optimize_resources(self, t: int, c_hat: np.ndarray, current_eta: Dict[int, np.ndarray]) -> Dict[int, Dict[str, np.ndarray]]:
        active_tasks = self.tasks
        if not active_tasks: return {}

        bounds = [(max(task.tau.values()), None) for task in active_tasks]
        x0 = np.array([b[0] * 1.1 for b in bounds]) # Start slightly above strict min for solver stability

        constraints = self._phase_a_constraints(active_tasks, current_eta, c_hat)
        result = minimize(fun=self._phase_a_objective, x0=x0, args=(active_tasks,), method='SLSQP', jac=True, bounds=bounds, constraints=constraints)

        # Recover resource allocations from optimal z [cite: 353]
        updated_s = {}
        for idx, task in enumerate(active_tasks):
            z_k = result.x[idx]
            s_comp = np.array([task.tau[i] / z_k for i in range(task.b_k, task.e_k + 1)])
            s_comm = np.array([
                (task.a[i] * current_eta[task.task_id][i - task.b_k]) / (c_hat[i] * z_k) 
                for i in range(task.b_k, task.e_k)
            ])
            updated_s[task.task_id] = {'s_comp': s_comp, 's_comm': s_comm}

        return updated_s

    # ==========================================
    # FUNCTIONAL HELPERS: PHASE B (CONFIGURATION)
    # ==========================================

    def _phase_b_objective(self, x: np.ndarray, task: InferenceTask) -> Tuple[float, np.ndarray]:
        """Objective: min -w_k * A_k + \mu * \lambda_k * z_k."""
        eta_flat = x[:-1]
        z = x[-1]
        acc = task.A_k(eta_flat)
        grad_acc = task.grad_A_k(eta_flat)
        
        obj_val = -task.w_k * acc + self.mu * self.lambda_k[task.task_id] * z
        grad = np.zeros_like(x)
        grad[:-1] = -task.w_k * grad_acc
        grad[-1] = self.mu * self.lambda_k[task.task_id]
        
        return obj_val, grad

    def _phase_b_optimize_configuration(self, t: int, c_hat: np.ndarray, current_s: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, np.ndarray]:
        updated_eta = {}
        
        for task in self.tasks:
            links = list(range(task.b_k, task.e_k))
            s_comm_fixed = current_s[task.task_id]['s_comm']
            s_comp_fixed = current_s[task.task_id]['s_comp']
            
            # Bound z by the max fixed compute delay
            max_comp_delay = max([task.tau[i] / s_comp_fixed[idx] for idx, i in enumerate(range(task.b_k, task.e_k + 1))])
            bounds = [(task.eta_min[i], 1.0) for i in links] + [(max_comp_delay, None)]
            x0 = np.array([b[0] for b in bounds])

            # Constraints: z >= (a_i / (c_hat_i * s_comm_i)) * eta_i [cite: 353]
            constraints = []
            for idx, i in enumerate(links):
                beta = task.a[i] / (c_hat[i] * s_comm_fixed[idx]) 
                def make_constraint(idx_local=idx, beta_local=beta):
                    return {
                        'type': 'ineq',
                        'fun': lambda x: x[-1] - beta_local * x[idx_local],
                        'jac': lambda x: np.array([-beta_local if j == idx_local else (1.0 if j == len(x)-1 else 0.0) for j in range(len(x))])
                    }
                constraints.append(make_constraint())

            result = minimize(fun=self._phase_b_objective, x0=x0, args=(task,), method='SLSQP', jac=True, bounds=bounds, constraints=constraints)
            updated_eta[task.task_id] = result.x[:-1]

        return updated_eta

    # ==========================================
    # MAIN OPTIMIZE METHOD
    # ==========================================

    def optimize(self, t: int, c_hat: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """Executes the BCD loop [cite: 340-343]."""
        current_eta = {task.task_id: np.array([task.eta_min[i] for i in range(task.b_k, task.e_k)]) for task in self.tasks}
        current_s = None

        for _ in range(self.J):
            current_s = self._phase_a_optimize_resources(t, c_hat, current_eta)
            current_eta = self._phase_b_optimize_configuration(t, c_hat, current_s)

        final_allocations = {}
        for task in self.tasks:
            final_allocations[task.task_id] = {
                'eta': current_eta[task.task_id],
                's_comp': current_s[task.task_id]['s_comp'],
                's_comm': current_s[task.task_id]['s_comm']
            }
        return self.scale_allocations_to_unit_sum(final_allocations)

    def update_dual(self, t: int, actual_delays: Dict[int, float]):
        """Dual Step: Update queues for active tasks [cite: 346-349]."""
        for task in self.tasks:
            if task.task_id in actual_delays:
                D_act = actual_delays[task.task_id]
                R_t = task.get_R_k(t)
                self.lambda_k[task.task_id] = max(self.epsilon, self.lambda_k[task.task_id] + D_act - (1.0 / R_t))