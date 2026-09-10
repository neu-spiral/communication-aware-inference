"""
Stein Lemma-based Gradient and Hessian Oracles
Applied to a small MLP on MNIST.

Oracles operate in a zeroth-order (function-value-only) fashion,
and are validated against PyTorch autograd.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# 1. Oracles
# ─────────────────────────────────────────────

def gradient_oracle(f, x: torch.Tensor, sigma: float = 0.01, N: int = 100) -> torch.Tensor:
    """
    Zeroth-order gradient estimate via Stein's lemma (antithetic pairs).

    grad f_sigma(x) ≈ (1 / 2Nσ) Σ z_i [f(x + σz_i) - f(x - σz_i)]

    Args:
        f      : scalar-valued callable  f: R^d -> R
        x      : point at which to estimate gradient, shape (d,)
        sigma  : smoothing scale
        N      : number of antithetic sample pairs
    Returns:
        grad estimate, shape (d,)
    """
    d = x.numel()
    g = torch.zeros_like(x)
    for _ in range(N):
        z = torch.randn(d, device=x.device)
        fp = f(x + sigma * z)
        fm = f(x - sigma * z)
        g += z * (fp - fm)
    return g / (2 * N * sigma)


def jacobian_oracle(F_vec, x: torch.Tensor, m: int, sigma: float = 0.01, N: int = 100) -> torch.Tensor:
    """
    Zeroth-order Jacobian estimate via Stein's lemma (antithetic pairs).

    J_F(x) ≈ (1 / 2Nσ) Σ z_i [F(x + σz_i) - F(x - σz_i)]^T  ∈ R^{d x m}

    Args:
        F_vec  : vector-valued callable  F: R^d -> R^m
        x      : point at which to estimate Jacobian, shape (d,)
        m      : output dimension of F_vec
        sigma  : smoothing scale
        N      : number of antithetic sample pairs
    Returns:
        Jacobian estimate, shape (d, m)
    """
    d = x.numel()
    J = torch.zeros(d, m, device=x.device)
    for _ in range(N):
        z = torch.randn(d, device=x.device)
        Fp = F_vec(x + sigma * z)   # shape (m,)
        Fm = F_vec(x - sigma * z)   # shape (m,)
        J += torch.outer(z, Fp - Fm)
    return J / (2 * N * sigma)


def hessian_oracle(f, x: torch.Tensor, sigma: float = 0.01, N: int = 50) -> torch.Tensor:
    """
    Zeroth-order Hessian estimate via nested Stein (antithetic, symmetrized).

    H_f(x) ≈ (1 / 4N²σ1σ2) Σ_i Σ_j z_i u_j^T (f++ - f+- - f-+ + f--)

    Uses σ1 = σ2 = σ/√2 to match target smoothing scale σ.

    Args:
        f      : scalar-valued callable  f: R^d -> R
        x      : point, shape (d,)
        sigma  : target smoothing scale (σ1 = σ2 = σ/√2)
        N      : samples per level (total evals = 4N²)
    Returns:
        symmetrized Hessian estimate, shape (d, d)
    """
    d = x.numel()
    s = sigma / (2 ** 0.5)   # σ1 = σ2
    H = torch.zeros(d, d, device=x.device)
    for _ in range(N):
        z = torch.randn(d, device=x.device)
        for _ in range(N):
            u = torch.randn(d, device=x.device)
            fpp = f(x + s*z + s*u)
            fpm = f(x + s*z - s*u)
            fmp = f(x - s*z + s*u)
            fmm = f(x - s*z - s*u)
            H += torch.outer(z, u) * (fpp - fpm - fmp + fmm)
    H = H / (4 * N * N * s * s)
    return (H + H.T) / 2   # symmetrize


# ─────────────────────────────────────────────
# 2. Toy MLP
# ─────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 3. Helpers: flatten / unflatten params
# ─────────────────────────────────────────────

def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_flat_params(model: nn.Module, flat: torch.Tensor):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view(p.shape))
        idx += n


# ─────────────────────────────────────────────
# 4. Loss oracle: R^p -> R
# ─────────────────────────────────────────────

def make_loss_fn(model: nn.Module, batch):
    """Returns a scalar loss function of flat parameter vector."""
    x, y = batch
    def loss_fn(flat_params: torch.Tensor) -> torch.Tensor:
        set_flat_params(model, flat_params)
        with torch.no_grad():
            logits = model(x)
            return F.cross_entropy(logits, y)
    return loss_fn


# ─────────────────────────────────────────────
# 5. Validation against autograd
# ─────────────────────────────────────────────

def autograd_gradient(model: nn.Module, batch) -> torch.Tensor:
    x, y = batch
    model.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def compare_gradients(model, batch, sigma=0.05, N=200):
    flat = get_flat_params(model)
    loss_fn = make_loss_fn(model, batch)

    # Stein estimate
    g_stein = gradient_oracle(loss_fn, flat, sigma=sigma, N=N)

    # Autograd (restore params first)
    set_flat_params(model, flat)
    for p in model.parameters():
        p.requires_grad_(True)
    g_auto = autograd_gradient(model, batch)

    cos_sim = F.cosine_similarity(g_stein.unsqueeze(0), g_auto.unsqueeze(0)).item()
    rel_err = (g_stein - g_auto).norm() / (g_auto.norm() + 1e-8)
    print(f"Gradient | cosine similarity: {cos_sim:.4f} | relative error: {rel_err:.4f}")
    return g_stein, g_auto


# ─────────────────────────────────────────────
# 6. Main experiment
# ─────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    # Data
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    loader   = DataLoader(train_ds, batch_size=64, shuffle=True)
    batch    = next(iter(loader))

    model = MLP()
    flat  = get_flat_params(model)
    print(f"Model parameter count: {flat.numel():,}")

    # --- Gradient oracle validation ---
    print("\n[Experiment 1] Gradient oracle vs autograd")
    compare_gradients(model, batch, sigma=0.05, N=300)

    # --- Sigma sensitivity ---
    print("\n[Experiment 2] Gradient oracle sigma sensitivity")
    loss_fn = make_loss_fn(model, batch)
    set_flat_params(model, flat)
    for p in model.parameters(): p.requires_grad_(True)
    g_auto = autograd_gradient(model, batch)

    for sigma in [0.001, 0.01, 0.05, 0.1, 0.5]:
        set_flat_params(model, flat)
        lf = make_loss_fn(model, batch)
        g_s = gradient_oracle(lf, flat, sigma=sigma, N=200)
        cos = F.cosine_similarity(g_s.unsqueeze(0), g_auto.unsqueeze(0)).item()
        print(f"  sigma={sigma:.3f} | cosine sim={cos:.4f}")

    # --- Hessian oracle on a tiny sub-network ---
    # Use only first-layer bias (10 params) for tractability
    print("\n[Experiment 3] Hessian oracle on small subspace (first-layer bias)")
    bias0 = model.net[1].bias.detach().clone()   # shape (128,) — use first 10 dims
    sub = bias0[:10]

    def scalar_loss_sub(v: torch.Tensor) -> torch.Tensor:
        """Loss as function of the first 10 dims of first-layer bias."""
        b = bias0.clone()
        b[:10] = v
        model.net[1].bias.data.copy_(b)
        with torch.no_grad():
            return F.cross_entropy(model(batch[0]), batch[1])

    H_stein = hessian_oracle(scalar_loss_sub, sub, sigma=0.05, N=20)
    print(f"  Hessian shape: {H_stein.shape}")
    print(f"  Hessian diagonal (curvature): {H_stein.diag().numpy().round(4)}")
    print(f"  Symmetry error: {(H_stein - H_stein.T).abs().max().item():.2e}")

    eigenvalues = torch.linalg.eigvalsh(H_stein)
    print(f"  Eigenvalues: {eigenvalues.numpy().round(4)}")
    print(f"  Positive definite: {(eigenvalues > 0).all().item()}")


if __name__ == "__main__":
    main()