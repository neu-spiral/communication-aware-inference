from typing import Callable, Dict
import numpy as np

class InferenceTask:
    """
    Represents a distributed inference task k spanning a sequence of interconnected nodes.
    """
    def __init__(
        self,
        task_id: int,
        b_k: int,
        L_k: int,
        tau: Dict[int, float],
        a: Dict[int, float],
        eta_min: Dict[int, float],
        R_k_callable: Callable[[int], float],  
        w_k: float,
        accuracy_callable: Callable[[np.ndarray], float],
        accuracy_callable_true: Callable[[np.ndarray], float],
        gradient_callable: Callable[[np.ndarray], np.ndarray] # NEW: Dedicated gradient callable
    ):
        # Topology Assignments
        self.task_id = task_id   
        self.b_k = b_k           
        self.L_k = L_k           
        self.e_k = b_k + L_k - 1 
        
        # Resource Requirements 
        self.tau = tau           
        self.a = a               
        self.eta_min = eta_min   
        
        # Objective and QoS
        self._R_k_callable = R_k_callable 
        self.w_k = w_k           
        
        # Black-Box Callables
        self._accuracy_callable = accuracy_callable
        self._gradient_callable = gradient_callable
        self._accuracy_callable_true = accuracy_callable_true
        
    def get_R_k(self, t: int) -> float:
        """
        Retrieves the target throughput R_k(t) for a specific time slot t.
        """
        return self._R_k_callable(t)

    def A_k(self, eta_vec: np.ndarray) -> float:
        """
        Evaluates and returns ONLY the scalar accuracy value.
        """
        return self._accuracy_callable(eta_vec)
        
    def grad_A_k(self, eta_vec: np.ndarray) -> np.ndarray:
        """
        Evaluates and returns ONLY the gradient vector with respect to \eta.
        """
        return self._gradient_callable(eta_vec)
    
    def A_k_true(self, eta_vec: np.ndarray) -> float:
        """
        Evaluates and returns ONLY the scalar accuracy value.
        """
        return self._accuracy_callable_true(eta_vec)
        
    def is_active_at_node(self, i: int) -> bool:
        """
        Checks if task k has a computation stage at node i.
        """
        return self.b_k <= i <= self.e_k
        
    def is_active_on_link(self, i: int) -> bool:
        """
        Checks if task k transmits intermediate data over link i.
        """
        return self.b_k <= i < self.e_k