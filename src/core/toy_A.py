"""
Toy accuracy function A(theta, eta) for testing Stein oracles.

A(theta, eta) = acc(theta) * prod_i (1 - alpha_i * (1 - eta_i)^2)

  - acc(theta) = exp(-gamma * MSE(theta))  in (0, 1)
  - alpha_i in (0,1): per-cut sensitivity to compression
  - eta_i in [0,1]:   compression ratio at cut-point i (1=no compression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# 1.  Small poly-fitting network (~97 params)
# ─────────────────────────────────────────────

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),  nn.Tanh(),
            nn.Linear(8, 8),  nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────
# 2.  Dataset: known polynomial
# ─────────────────────────────────────────────

def poly(x):
    return 2*x**3 - x**2 + 0.5*x - 1.0

def make_batch(n=128, noise=0.05):
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = poly(x) + noise * torch.randn_like(x)
    return x, y


# ─────────────────────────────────────────────
# 3.  Param helpers
# ─────────────────────────────────────────────

def get_flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_flat(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view(p.shape))
        idx += n


# ─────────────────────────────────────────────
# 4.  Stein oracles
# ─────────────────────────────────────────────

def grad_oracle(f, x, sigma=0.01, N=300):
    g = torch.zeros_like(x)
    for _ in range(N):
        z  = torch.randn_like(x)
        g += z * (f(x + sigma*z) - f(x - sigma*z))
    return g / (2 * N * sigma)


def jacobian_oracle(F_vec, x, m, sigma=0.01, N=100):
    d = x.numel()
    J = torch.zeros(d, m)
    for _ in range(N):
        z  = torch.randn(d)
        Fp = F_vec(x + sigma*z)
        Fm = F_vec(x - sigma*z)
        J += torch.outer(z, Fp - Fm)
    return J / (2 * N * sigma)


def hessian_oracle_eta(f, x, sigma=0.05, N=50):
    """Nested antithetic Stein Hessian, symmetrized."""
    d = x.numel()
    s = sigma / (2 ** 0.5)
    H = torch.zeros(d, d)
    for _ in range(N):
        z = torch.randn(d)
        for _ in range(N):
            u   = torch.randn(d)
            fpp = f(x + s*z + s*u)
            fpm = f(x + s*z - s*u)
            fmp = f(x - s*z + s*u)
            fmm = f(x - s*z - s*u)
            H  += torch.outer(z, u) * (fpp - fpm - fmp + fmm)
    H = H / (4 * N * N * s * s)
    return (H + H.T) / 2


def cross_jacobian_oracle(A_fn, flat_theta, eta,
                          sigma_t=0.05, sigma_e=0.05,
                          N_outer=200, N_inner=20):
    """
    Single nested Stein estimator for  nabla_eta nabla_theta A  in R^{p x d}.

      J ≈ (1/4 N_o N_i s_t s_e) Σ_i Σ_j z_i u_j^T (f++ - f+- - f-+ + f--)

    z ~ N(0,I_p),  u ~ N(0,I_d).
    No nested oracles — single smoothing level, no compounding bias.
    N_outer controls variance in p-dim theta space (typically needs to be larger).
    N_inner controls variance in d-dim eta space  (can be small since d << p).
    """
    p, d = flat_theta.numel(), eta.numel()
    J = torch.zeros(p, d)
    for _ in range(N_outer):
        z = torch.randn(p)
        for _ in range(N_inner):
            u   = torch.randn(d)
            fpp = A_fn(flat_theta + sigma_t*z, eta + sigma_e*u)
            fpm = A_fn(flat_theta + sigma_t*z, eta - sigma_e*u)
            fmp = A_fn(flat_theta - sigma_t*z, eta + sigma_e*u)
            fmm = A_fn(flat_theta - sigma_t*z, eta - sigma_e*u)
            J  += torch.outer(z, u) * (fpp - fpm - fmp + fmm)
    set_flat(A_fn.model, flat_theta)
    return J / (4 * N_outer * N_inner * sigma_t * sigma_e)


# ─────────────────────────────────────────────
# 5.  Toy A(theta, eta)
# ─────────────────────────────────────────────

class ToyA:
    """
    A(theta, eta) = acc(theta) * prod_i (1 - alpha_i*(1-eta_i)^2)
    acc(theta)    = exp(-gamma * MSE(theta))
    """
    def __init__(self, model, batch, alpha, gamma=10.0):
        self.model = model
        self.batch = batch
        self.alpha = alpha
        self.d     = alpha.numel()
        self.gamma = gamma

    def _acc(self, flat_theta):
        set_flat(self.model, flat_theta)
        x, y = self.batch
        with torch.no_grad():
            mse = F.mse_loss(self.model(x), y)
        return torch.exp(-self.gamma * mse)

    def _C(self, eta):
        return torch.prod(1.0 - self.alpha * (1.0 - eta)**2)

    def __call__(self, flat_theta, eta):
        return self._acc(flat_theta) * self._C(eta)

    def grad_eta(self, flat_theta, eta):
        acc   = self._acc(flat_theta)
        terms = 1.0 - self.alpha * (1.0 - eta)**2
        C     = torch.prod(terms)
        return acc * (C / terms) * 2.0 * self.alpha * (1.0 - eta)

    def grad_theta(self, flat_theta, eta, ):
        set_flat(self.model, flat_theta)
        for p in self.model.parameters():
            p.requires_grad_(True)
        x, y = self.batch
        mse  = F.mse_loss(self.model(x), y)
        acc  = torch.exp(-self.gamma * mse)
        acc.backward()
        g = torch.cat([p.grad.flatten() for p in self.model.parameters()])
        for p in self.model.parameters():
            p.requires_grad_(False)
        return g * self._C(eta)

    def hessian_eta(self, flat_theta, eta):
        acc   = self._acc(flat_theta)
        terms = 1.0 - self.alpha * (1.0 - eta)**2
        C     = torch.prod(terms)
        dC    = (C / terms) * 2.0 * self.alpha * (1.0 - eta)
        H     = acc * torch.outer(dC, dC) / C
        H    -= torch.diag(acc * (C / terms) * 2.0 * self.alpha)
        return H

    def cross_jacobian(self, flat_theta, eta):
        """Analytical  nabla_eta nabla_theta A  in R^{p x d}."""
        terms      = 1.0 - self.alpha * (1.0 - eta)**2
        C          = torch.prod(terms)
        grad_eta_C = (C / terms) * 2.0 * self.alpha * (1.0 - eta)   # (d,)

        set_flat(self.model, flat_theta)
        for p in self.model.parameters():
            p.requires_grad_(True)
        x, y  = self.batch
        mse   = F.mse_loss(self.model(x), y)
        acc   = torch.exp(-self.gamma * mse)
        acc.backward()
        d_acc = torch.cat([p.grad.flatten() for p in self.model.parameters()])
        for p in self.model.parameters():
            p.requires_grad_(False)
        set_flat(self.model, flat_theta)
        return torch.outer(d_acc, grad_eta_C)   # (p, d)


# ─────────────────────────────────────────────
# 6.  Experiments
# ─────────────────────────────────────────────

def exp_fit(model, batch, n_steps=2000, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = batch
    for _ in range(n_steps):
        opt.zero_grad()
        F.mse_loss(model(x), y).backward()
        opt.step()
    print(f"  Pre-training done — MSE: {F.mse_loss(model(x), y).item():.5f}")


def exp_grad_eta(A_fn, flat_theta, eta, sigmas, N=500):
    true = A_fn.grad_eta(flat_theta, eta)
    rows = []
    for sigma in sigmas:
        def f(e): return A_fn(flat_theta, e)
        est = grad_oracle(f, eta, sigma=sigma, N=N)
        cos = F.cosine_similarity(est.unsqueeze(0), true.unsqueeze(0)).item()
        rel = (est - true).norm() / (true.norm() + 1e-8)
        rows.append((sigma, cos, rel.item()))
        print(f"  sigma={sigma:.3f}  cos={cos:.4f}  rel_err={rel:.4f}")
    return true, rows


def exp_grad_theta(A_fn, flat_theta, eta, sigmas, N=500):
    true = A_fn.grad_theta(flat_theta, eta)
    set_flat(A_fn.model, flat_theta)
    rows = []
    for sigma in sigmas:
        def f(th): return A_fn(th, eta)
        est = grad_oracle(f, flat_theta, sigma=sigma, N=N)
        cos = F.cosine_similarity(est.unsqueeze(0), true.unsqueeze(0)).item()
        rel = (est - true).norm() / (true.norm() + 1e-8)
        rows.append((sigma, cos, rel.item()))
        print(f"  sigma={sigma:.3f}  cos={cos:.4f}  rel_err={rel:.4f}")
    return true, rows


def exp_hessian_eta(A_fn, flat_theta, eta, sigmas, N=30):
    H_true = A_fn.hessian_eta(flat_theta, eta)
    print(f"\n  Analytical Hessian:\n{H_true.numpy().round(4)}")
    rows = []
    for sigma in sigmas:
        def f(e): return A_fn(flat_theta, e)
        H_est   = hessian_oracle_eta(f, eta, sigma=sigma, N=N)
        fro_err = (H_est - H_true).norm() / (H_true.norm() + 1e-8)
        idx     = torch.triu_indices(A_fn.d, A_fn.d)
        cos     = F.cosine_similarity(H_est[idx[0],idx[1]].unsqueeze(0),
                                      H_true[idx[0],idx[1]].unsqueeze(0)).item()
        rows.append((sigma, cos, fro_err.item()))
        print(f"  sigma={sigma:.4f}  cos={cos:.5f}  frob_err={fro_err:.5f}")
    best_s  = sigmas[max(range(len(rows)), key=lambda i: rows[i][1])]
    def f(e): return A_fn(flat_theta, e)
    H_best  = hessian_oracle_eta(f, eta, sigma=best_s, N=50)
    return H_true, rows, H_best


def exp_cross_jacobian(A_fn, flat_theta, eta, sigmas, N_outer=200, N_inner=20):
    """Validate cross-Jacobian oracle against analytical reference."""
    J_true = A_fn.cross_jacobian(flat_theta, eta)
    p, d   = J_true.shape
    print(f"\n  Cross-Jacobian shape: ({p}, {d})")
    print(f"  Frobenius norm (analytical): {J_true.norm():.5f}")

    rows = []
    for sigma in sigmas:
        set_flat(A_fn.model, flat_theta)
        J_est   = cross_jacobian_oracle(A_fn, flat_theta, eta,
                                        sigma_t=sigma, sigma_e=sigma,
                                        N_outer=N_outer, N_inner=N_inner)
        fro_err = (J_est - J_true).norm() / (J_true.norm() + 1e-8)
        cos     = F.cosine_similarity(J_est.flatten().unsqueeze(0),
                                      J_true.flatten().unsqueeze(0)).item()
        rows.append((sigma, cos, fro_err.item(), J_est))
        print(f"  sigma={sigma:.4f}  cos={cos:.5f}  frob_err={fro_err:.5f}")

    best_idx = max(range(len(rows)), key=lambda i: rows[i][1])
    return J_true, rows, rows[best_idx][3]


def exp_ordering_comparison(A_fn, flat_theta, eta, sigma=0.05, Ns=None):
    if Ns is None:
        Ns = [10, 50, 100, 200, 500]
    J_true = A_fn.cross_jacobian(flat_theta, eta)
    rows   = []
    for N in Ns:
        # Order 1: outer-theta, inner-eta
        def F_vec_theta(th):
            def f_eta(e): return A_fn(th, e)
            return grad_oracle(f_eta, eta, sigma=sigma/2, N=20)
        J1 = jacobian_oracle(F_vec_theta, flat_theta, m=A_fn.d, sigma=sigma, N=N)
        set_flat(A_fn.model, flat_theta)

        # Order 2: outer-eta, inner-theta (transpose back to p x d)
        def F_vec_eta(e):
            def f_th(th): return A_fn(th, e)
            return grad_oracle(f_th, flat_theta, sigma=sigma/2, N=20)
        J2 = jacobian_oracle(F_vec_eta, eta, m=flat_theta.numel(), sigma=sigma, N=N).T
        set_flat(A_fn.model, flat_theta)

        cos1 = F.cosine_similarity(J1.flatten().unsqueeze(0), J_true.flatten().unsqueeze(0)).item()
        cos2 = F.cosine_similarity(J2.flatten().unsqueeze(0), J_true.flatten().unsqueeze(0)).item()
        rows.append((N, cos1, cos2))
        print(f"  N={N:4d}  outer-theta cos={cos1:.4f}  outer-eta cos={cos2:.4f}")
    return rows


def exp_N_sweep(A_fn, flat_theta, eta, sigma=0.05, Ns=None):
    if Ns is None:
        Ns = [10, 50, 100, 200, 500, 1000]
    true_eta   = A_fn.grad_eta(flat_theta, eta)
    true_theta = A_fn.grad_theta(flat_theta, eta)
    set_flat(A_fn.model, flat_theta)
    rows = []
    for N in Ns:
        def fe(e):  return A_fn(flat_theta, e)
        def ft(th): return A_fn(th, eta)
        ge    = grad_oracle(fe, eta,        sigma=sigma, N=N)
        gt    = grad_oracle(ft, flat_theta, sigma=sigma, N=N)
        cos_e = F.cosine_similarity(ge.unsqueeze(0), true_eta.unsqueeze(0)).item()
        cos_t = F.cosine_similarity(gt.unsqueeze(0), true_theta.unsqueeze(0)).item()
        rows.append((N, cos_e, cos_t))
        print(f"  N={N:5d}  cos_eta={cos_e:.5f}  cos_theta={cos_t:.5f}")
    return rows


# ─────────────────────────────────────────────
# 7.  Plot
# ─────────────────────────────────────────────

def plot_results(sigmas, eta_rows, theta_rows, hess_rows, cross_rows,
                 ordering_rows, N_rows, batch, model, H_true, H_est_best,
                 J_true, J_est_best):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # gradient sigma sweep
    axes[0,0].semilogx(sigmas, [r[1] for r in eta_rows],   'o-', label='grad_eta')
    axes[0,0].semilogx(sigmas, [r[1] for r in theta_rows], 's--', label='grad_theta')
    axes[0,0].set_xlabel('sigma'); axes[0,0].set_ylabel('cosine similarity')
    axes[0,0].set_title('Gradient oracle vs sigma'); axes[0,0].legend(); axes[0,0].grid(True)

    # 2nd-order sigma sweep
    axes[0,1].semilogx(sigmas, [r[1] for r in hess_rows],  'o-',  color='C2', label='Hessian_eta')
    axes[0,1].semilogx(sigmas, [r[1] for r in cross_rows], 's--', color='C4', label='cross-Jacobian')
    axes[0,1].set_xlabel('sigma'); axes[0,1].set_ylabel('cosine similarity')
    axes[0,1].set_title('2nd-order oracles vs sigma'); axes[0,1].legend(); axes[0,1].grid(True)

    # ordering comparison
    Ns = [r[0] for r in ordering_rows]
    axes[0,2].semilogx(Ns, [r[1] for r in ordering_rows], 'o-',  label=f'outer-θ (p={J_true.shape[0]})')
    axes[0,2].semilogx(Ns, [r[2] for r in ordering_rows], 's--', label=f'outer-η (d={J_true.shape[1]})')
    axes[0,2].set_xlabel('N'); axes[0,2].set_ylabel('cosine similarity')
    axes[0,2].set_title('Cross-Jacobian: ordering comparison'); axes[0,2].legend(); axes[0,2].grid(True)

    # N sweep
    Ns_n = [r[0] for r in N_rows]
    axes[1,0].semilogx(Ns_n, [r[1] for r in N_rows], 'o-', label='grad_eta')
    axes[1,0].semilogx(Ns_n, [r[2] for r in N_rows], 's--', label='grad_theta')
    axes[1,0].set_xlabel('N'); axes[1,0].set_ylabel('cosine similarity')
    axes[1,0].set_title('Gradient oracle vs N'); axes[1,0].legend(); axes[1,0].grid(True)

    # Hessian heatmaps
    vmin = min(H_true.min().item(), H_est_best.min().item())
    vmax = max(H_true.max().item(), H_est_best.max().item())
    im0  = axes[1,1].imshow(H_true.numpy(),     vmin=vmin, vmax=vmax, cmap='RdBu_r')
    axes[1,1].set_title('Hessian_eta (analytical)'); plt.colorbar(im0, ax=axes[1,1])
    im1  = axes[1,2].imshow(H_est_best.numpy(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
    axes[1,2].set_title('Hessian_eta (oracle)'); plt.colorbar(im1, ax=axes[1,2])

    plt.tight_layout()
    plt.savefig('toy_A_poly.png', dpi=120)
    plt.show()


# ─────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    model = TinyMLP()
    print(f"Parameter count: {count_params(model)}")
    batch = make_batch(n=128, noise=0.05)

    print("\nPre-training on polynomial...")
    exp_fit(model, batch, n_steps=2000)

    flat_theta = get_flat(model)
    alpha  = torch.tensor([0.3, 0.5, 0.7, 0.2])
    eta    = torch.tensor([0.8, 0.6, 0.5, 0.9])
    A_fn   = ToyA(model, batch, alpha)
    sigmas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03]

    print(f"\nA(theta, eta) = {A_fn(flat_theta, eta).item():.5f}")

    print("\n[1] grad_eta sigma sweep")
    _, eta_rows = exp_grad_eta(A_fn, flat_theta, eta, sigmas)

    print("\n[2] grad_theta sigma sweep")
    set_flat(model, flat_theta)
    _, theta_rows = exp_grad_theta(A_fn, flat_theta, eta, sigmas)

    print("\n[3] Hessian over eta sigma sweep")
    set_flat(model, flat_theta)
    H_true, hess_rows, H_est_best = exp_hessian_eta(A_fn, flat_theta, eta, sigmas, N=30)

    print("\n[4] Cross-Jacobian sigma sweep (single nested Stein)")
    set_flat(model, flat_theta)
    J_true, cross_rows, J_est_best = exp_cross_jacobian(
        A_fn, flat_theta, eta, sigmas, N_outer=200, N_inner=20)

    print("\n[5] Cross-Jacobian ordering comparison")
    set_flat(model, flat_theta)
    ordering_rows = exp_ordering_comparison(A_fn, flat_theta, eta, sigma=0.0005)

    print("\n[6] N sweep (sigma=0.05)")
    set_flat(model, flat_theta)
    N_rows = exp_N_sweep(A_fn, flat_theta, eta, sigma=0.0005)

    plot_results(sigmas, eta_rows, theta_rows, hess_rows, cross_rows,
                 ordering_rows, N_rows, batch, model,
                 H_true, H_est_best, J_true, J_est_best)


if __name__ == "__main__":
    main()