"""
CompressedModel: nn.Module with persistent hooks, eta updated in-place.
AccuracyFunction: measures accuracy on a (compressed) model, optionally
                  swapping theta in-place and restoring afterwards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .stein_oracles import MLP, get_flat_params, set_flat_params
from .toy_A import grad_oracle


# ─────────────────────────────────────────────
# 1.  Top-k sparsification
# ─────────────────────────────────────────────

def topk_sparsify(x: torch.Tensor, eta: float) -> torch.Tensor:
    if eta >= 1.0:
        return x
    if eta <= 0.0:
        return torch.zeros_like(x)
    flat   = x.flatten()
    k      = max(1, int(eta * flat.numel()))
    thresh = flat.abs().topk(k).values.min()
    return (flat * (flat.abs() >= thresh)).reshape(x.shape)


# ─────────────────────────────────────────────
# 2.  CompressedModel
#     Registers hooks once at construction.
#     eta is updated in-place via set_eta().
# ─────────────────────────────────────────────

class CompressedModel(nn.Module):
    """
    Wraps base_model with persistent top-k sparsification hooks.
    Hooks are registered once and never removed — update eta in-place.

    Parameters
    ----------
    base_model  : nn.Module
    eta         : Tensor (d,) -- initial compression ratios
    cut_modules : list of nn.Module -- layers after which to sparsify.
                  Defaults to all ReLU/Tanh/GELU/Sigmoid layers.

    Usage
    -----
    cm = CompressedModel(model, eta)
    cm.set_eta(new_eta)     # update in-place, no new hooks
    out = cm(x)
    """

    def __init__(self, base_model: nn.Module, eta: torch.Tensor,
                 cut_modules=None):
        super().__init__()
        self.base_model = base_model
        self.eta        = eta.clone()

        if cut_modules is None:
            cut_modules = [m for m in base_model.modules()
                           if isinstance(m, (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid))]
        self.cut_modules = cut_modules
        self.d           = len(cut_modules)
        assert len(eta) == self.d, \
            f"eta length {len(eta)} does not match {self.d} cut-points"

        # register hooks once — they read from self.eta at forward time
        for i, m in enumerate(self.cut_modules):
            m.register_forward_hook(self._make_hook(i))

    def _make_hook(self, i: int):
        def hook(module, input, output):
            return topk_sparsify(output, self.eta[i].item())
        return hook

    def set_eta(self, eta: torch.Tensor):
        assert len(eta) == self.d
        self.eta = eta.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


# ─────────────────────────────────────────────
# 3.  AccuracyFunction
# ─────────────────────────────────────────────

class AccuracyFunction:
    """
    Measures classification accuracy of a CompressedModel on a test set.

    Parameters
    ----------
    cm          : CompressedModel -- model with hooks already installed
    test_loader : DataLoader
    n_samples   : int or None -- random subset size for speed (None = full set)
    device      : torch.device

    Usage
    -----
    A = AccuracyFunction(cm, test_loader, n_samples=512)

    A(eta)                   # evaluate at eta, current theta
    A(eta, flat_theta=th)    # evaluate at eta and th, restores theta after
    """

    def __init__(self, cm: CompressedModel, test_loader: DataLoader,
                 n_samples=None, device=None):
        self.cm          = cm
        self.test_loader = test_loader
        self.n_samples   = n_samples
        self.device      = device or next(cm.parameters()).device

        if n_samples is not None:
            xs, ys = zip(*list(test_loader))
            self._all_x = torch.cat(xs)
            self._all_y = torch.cat(ys)

    def _sample_batch(self):
        idx = torch.randperm(len(self._all_x))[:self.n_samples]
        return self._all_x[idx], self._all_y[idx]

    def _eval(self) -> torch.Tensor:
        self.cm.eval()
        with torch.no_grad():
            if self.n_samples is not None:
                x, y    = self._sample_batch()
                x, y    = x.to(self.device), y.to(self.device)
                correct = (self.cm(x).argmax(1) == y).sum().item()
                total   = len(y)
            else:
                correct = total = 0
                for x, y in self.test_loader:
                    x, y    = x.to(self.device), y.to(self.device)
                    correct += (self.cm(x).argmax(1) == y).sum().item()
                    total   += y.size(0)
        return torch.tensor(correct / total)

    def __call__(self, eta: torch.Tensor,
                 flat_theta: torch.Tensor = None) -> torch.Tensor:
        self.cm.set_eta(eta)

        if flat_theta is not None:
            original = get_flat_params(self.cm.base_model)
            set_flat_params(self.cm.base_model, flat_theta)
            try:
                acc = self._eval()
            finally:
                set_flat_params(self.cm.base_model, original)
        else:
            acc = self._eval()

        return acc


# ─────────────────────────────────────────────
# 4.  Demo
# ─────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = torch.device('cpu')

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
    print("Training MLP on MNIST (3 epochs)...")
    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

    flat_theta = get_flat_params(model)

    # construct once
    eta_full = torch.ones(2)   # MLP has 2 activation layers
    cm       = CompressedModel(model, eta_full)
    A        = AccuracyFunction(cm, test_loader, device=device)
    A_fast   = AccuracyFunction(cm, test_loader, n_samples=512, device=device)
    d        = cm.d
    print(f"Cut-points: {d}")

    # -- basic checks --
    eta_heavy = torch.full((d,), 0.1)
    print(f"\nNo compression    (eta=1.0): {A(eta_full):.4f}")
    print(f"Heavy compression (eta=0.1): {A(eta_heavy):.4f}")

    # verify theta restore
    _ = A(eta_full, flat_theta=torch.zeros_like(flat_theta))
    print(f"After dummy theta, acc restored: {A(eta_full):.4f}")

    # -- sweep eta_0 --
    print("\nSweeping eta_0 (others=1.0):")
    for v in torch.linspace(0, 1, 11).tolist():
        eta = eta_full.clone(); eta[0] = v
        acc = A_fast(eta)
        bar = "█" * int(acc * 30)
        print(f"  eta_0={v:.1f}  acc={acc:.4f}  {bar}")

    # -- Stein gradient w.r.t. eta --
    print("\nStein gradient of A w.r.t. eta (sigma=0.05, N=50):")
    def f_eta(e):   return A_fast(e, flat_theta=flat_theta)
    g = grad_oracle(f_eta, eta_full * 0.8, sigma=0.05, N=50)
    print(f"  grad_eta A = {g.numpy().round(4)}")
"""
    # -- Stein gradient w.r.t. theta --
    print("\nStein gradient of A w.r.t. theta (sigma=0.01, N=100):")
    eta_mid = torch.full((d,), 0.8)
    def f_theta(th): return A_fast(eta_mid, flat_theta=th)
    g_th = grad_oracle(f_theta, flat_theta, sigma=0.01, N=100)
    print(f"  ||grad_theta A|| = {g_th.norm():.4f}")
"""

if __name__ == "__main__":
    main()
