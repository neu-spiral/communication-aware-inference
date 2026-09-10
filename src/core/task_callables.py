"""
Builds accuracy_callable and gradient_callable for InferenceTask,
wrapping AccuracyFunction and the Stein gradient oracle.

Both callables accept and return np.ndarray as required by InferenceTask.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .stein_oracles import MLP, get_flat_params
from .accuracy_fn import CompressedModel, AccuracyFunction
from .toy_A import grad_oracle


class InferenceTaskCallables:
    """
    Builds accuracy_callable and gradient_callable for use in InferenceTask.

    Parameters
    ----------
    cm          : CompressedModel  -- model with hooks installed
    test_loader : DataLoader       -- for accuracy evaluation
    sigma       : float            -- Stein smoothing scale for gradient oracle
    N           : int              -- number of Stein samples
    n_samples   : int or None      -- test subset size for accuracy_callable
                                      (None = full test set)
    n_samples_true : int or None   -- test subset for accuracy_callable_true
                                      (None = full test set)
    flat_theta  : Tensor or None   -- fixed theta to evaluate at.
                                      If None, uses model's current weights.
    device      : torch.device

    Usage
    -----
    api = InferenceTaskCallables(cm, test_loader, sigma=0.05, N=50)

    task = InferenceTask(
        ...,
        accuracy_callable  = api.accuracy_callable,
        gradient_callable  = api.gradient_callable,
    )

    # or access directly:
    acc  = api.accuracy_callable(eta_np)    # np.ndarray -> float
    grad = api.gradient_callable(eta_np)    # np.ndarray -> np.ndarray
    """

    def __init__(self, cm: CompressedModel, test_loader: DataLoader,
                 sigma: float = 0.05, N: int = 50,
                 n_samples: int = 512,
                 n_samples_true: Optional[int] = None,
                 flat_theta: torch.Tensor = None,
                 device=None):
        self.cm          = cm
        self.sigma       = sigma
        self.N           = N
        self.flat_theta  = flat_theta
        self.device      = device or next(cm.parameters()).device

        self.A = AccuracyFunction(cm, test_loader,
                                  n_samples=n_samples, device=self.device)

        self.A_true = AccuracyFunction(cm, test_loader,
                                  n_samples=n_samples_true, device=self.device)

    # ── internal torch-level calls ────────────────────────────────

    def _acc_torch(self, eta: torch.Tensor) -> torch.Tensor:
        return self.A(eta, flat_theta=self.flat_theta)

    def _grad_torch(self, eta: torch.Tensor) -> torch.Tensor:
        return grad_oracle(self._acc_torch, eta, sigma=self.sigma, N=self.N)

    def _acc_true_torch(self, eta: torch.Tensor) -> torch.Tensor:
        return self.A_true(eta, flat_theta=self.flat_theta)

    # ── public callables (np.ndarray interface) ───────────────────

    def accuracy_callable(self, eta_vec: np.ndarray) -> float:
        """
        Callable for InferenceTask.accuracy_callable.
        np.ndarray -> float
        """
        eta = torch.tensor(eta_vec, dtype=torch.float32)
        return self._acc_torch(eta).item()

    def gradient_callable(self, eta_vec: np.ndarray) -> np.ndarray:
        """
        Callable for InferenceTask.gradient_callable.
        np.ndarray -> np.ndarray
        """
        eta = torch.tensor(eta_vec, dtype=torch.float32)
        return self._grad_torch(eta).numpy()

    def accuracy_callable_true(self, eta_vec: np.ndarray) -> float:
        """
        Callable for InferenceTask.accuracy_callable_true.
        np.ndarray -> float
        """
        eta = torch.tensor(eta_vec, dtype=torch.float32)
        return self._acc_true_torch(eta).item()


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = torch.device('cpu')

    # -- data --
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True,  download=True, transform=tf),
        batch_size=128, shuffle=True)
    test_loader  = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=tf),
        batch_size=256, shuffle=False)

    # -- train --
    model = MLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training MLP (3 epochs)...")
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

    flat_theta = get_flat_params(model)

    # -- build callables --
    eta_init = torch.ones(2)   # 2 cut-points in MLP
    cm  = CompressedModel(model, eta_init)
    api = InferenceTaskCallables(cm, test_loader,
                                 sigma=0.05, N=50, n_samples=512,
                                 flat_theta=flat_theta, device=device)

    # -- wire into InferenceTask --
    from task import InferenceTask   # the provided class

    task = InferenceTask(
        task_id          = 0,
        b_k              = 0,
        L_k              = 2,
        tau              = {0: 1.0, 1: 1.0},
        a                = {0: 0.5, 1: 0.5},
        eta_min          = {0: 0.1, 1: 0.1},
        R_k_callable     = lambda t: 1.0,
        w_k              = 1.0,
        accuracy_callable  = api.accuracy_callable,
        gradient_callable  = api.gradient_callable,
    )

    # -- test via InferenceTask API --
    eta_np = np.array([0.8, 0.8])
    print(f"\neta = {eta_np}")
    print(f"A_k(eta)      = {task.A_k(eta_np):.4f}")
    print(f"grad_A_k(eta) = {task.grad_A_k(eta_np).round(4)}")

    # -- sanity: direct vs task API --
    acc_direct  = api.accuracy_callable(eta_np)
    acc_via_task = task.A_k(eta_np)
    print(f"\nDirect call:    {acc_direct:.4f}")
    print(f"Via task API:   {acc_via_task:.4f}")
    diff = abs(acc_direct - acc_via_task)
    print(f"Difference:     {diff:.4f}  (expected small but nonzero due to random subsampling)")
    assert diff < 0.05, f"Large mismatch ({diff:.4f}) suggests a bug, not just sampling noise"
    print("Sanity check passed.")


if __name__ == "__main__":
    main()