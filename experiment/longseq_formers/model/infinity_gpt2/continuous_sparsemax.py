import torch
import torch.nn as nn


class ContinuousSparsemaxFunction(torch.autograd.Function):
    @classmethod
    def _integrate_phi_times_psi(cls, ctx, a, b):
        """Compute integral int_a^b phi(t) * psi(t).T."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        V = torch.zeros((a.shape[0], 2, total_basis), dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.tensor(num_basis, dtype=torch.int, device=ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            V[:, 0, start : offsets[j]] = basis_functions.integrate_t_times_psi(a, b)
            V[:, 1, start : offsets[j]] = basis_functions.integrate_t2_times_psi(a, b)
            start = offsets[j]
        return V

    @classmethod
    def _integrate_psi(cls, ctx, a, b):
        """Compute integral int_a^b psi(t)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        v = torch.zeros(a.shape[0], total_basis, dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.tensor(num_basis, dtype=torch.int, device=ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            v[:, start : offsets[j]] = basis_functions.integrate_psi(a, b)
            start = offsets[j]
        return v

    @classmethod
    def _integrate_phi(cls, ctx, a, b):
        """Compute integral int_a^b phi(t)."""
        v = torch.zeros(a.shape[0], 2, dtype=ctx.dtype, device=ctx.device)
        v[:, 0] = ((b**2 - a**2) / 2).squeeze(1)
        v[:, 1] = ((b**3 - a**3) / 3).squeeze(1)
        return v

    @classmethod
    def forward(cls, ctx, theta, psi):
        # We assume a truncated parabola.
        # We have:
        # theta = [mu/sigma**2, -1/(2*sigma**2)],
        # phi(t) = [t, t**2],
        # p(t) = [theta.dot(phi(t)) - A]_+,
        # supported on [mu - a, mu + a].
        ctx.dtype = theta.dtype
        ctx.device = theta.device
        ctx.psi = psi
        sigma = torch.sqrt(-0.5 / theta[:, 1])
        mu = theta[:, 0] * sigma**2
        A = -0.5 * (3.0 / (2 * sigma)) ** (2.0 / 3)
        a = torch.sqrt(-2 * A) * sigma
        A += mu**2 / (2 * sigma**2)
        left = (mu - a).unsqueeze(1)
        right = (mu + a).unsqueeze(1)
        V = cls._integrate_phi_times_psi(ctx, left, right)
        u = cls._integrate_psi(ctx, left, right)
        r = torch.matmul(theta.unsqueeze(1), V).squeeze(1) - A.unsqueeze(1) * u
        ctx.save_for_backward(mu, a, V, u)
        return r

    @classmethod
    def backward(cls, ctx, grad_output):
        mu, a, V, u = ctx.saved_tensors
        # J.T = int_{-a}^{+a} phi(t+mu)*psi(t+mu).T
        # - (int_{-a}^{+a} phi(t+mu)) * (int_{-a}^{+a} psi(t+mu).T) / (2*a)
        left = (mu - a).unsqueeze(1)
        right = (mu + a).unsqueeze(1)
        i_phi = cls._integrate_phi(ctx, left, right)
        ger = torch.bmm(i_phi.unsqueeze(2), u.unsqueeze(1))
        # ger = torch.einsum('bi,bj->bij', (i_phi, u))
        J = V - ger / (2 * a.unsqueeze(1).unsqueeze(2))
        grad_input = torch.matmul(J, grad_output.unsqueeze(2)).squeeze(2)
        return grad_input, None


class ContinuousSparsemax(nn.Module):
    def __init__(self, psi=None):
        super(ContinuousSparsemax, self).__init__()
        self.psi = psi

    def forward(self, theta):
        return ContinuousSparsemaxFunction.apply(theta, self.psi)
