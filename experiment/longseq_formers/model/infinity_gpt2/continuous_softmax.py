from typing import List, Optional

import torch
import torch.nn as nn
from torch.autograd.function import FunctionCtx

from .basis_functions import GaussianBasisFunctions


class ContinuousSoftmaxFunction(torch.autograd.Function):
    @classmethod
    def _expectation_phi_psi(cls, ctx: FunctionCtx, mu: torch.FloatTensor, sigma_sq: torch.FloatTensor):
        """Compute expectation of phi(t) * psi(t).T under N(mu, sigma_sq)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        V = torch.zeros((mu.shape[0], 2, total_basis), dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.tensor(num_basis, dtype=torch.int, device=ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            V[:, 0, start : offsets[j]] = basis_functions.integrate_t_times_psi_gaussian(mu, sigma_sq)
            V[:, 1, start : offsets[j]] = basis_functions.integrate_t2_times_psi_gaussian(mu, sigma_sq)
            start = offsets[j]
        return V

    @classmethod
    def _expectation_psi(
        cls, ctx: FunctionCtx, mu: torch.FloatTensor, sigma_sq: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute expectation of psi under N(mu, sigma_sq).

        Args:
            mu: mu of distribution shaped [BatchSize, 1]
            sigma_sq: sigma_sq of distribution shaped [BatchSize, 1]
        Return:
            integraded result shaped [BatchSize, TotalBasis]
        """
        psi: list[GaussianBasisFunctions] = ctx.psi
        num_basis = [len(basis_functions) for basis_functions in psi]
        total_basis = sum(num_basis)
        r = torch.zeros(mu.shape[0], total_basis, dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.tensor(num_basis, dtype=torch.int, device=ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(psi):
            r[:, start : offsets[j]] = basis_functions.integrate_psi_gaussian(mu, sigma_sq)
            start = offsets[j]
        return r

    @classmethod
    def _expectation_phi(cls, ctx: FunctionCtx, mu: torch.FloatTensor, sigma_sq: torch.FloatTensor):
        """Compute expectation of phi under N(mu, sigma_sq)."""
        v = torch.zeros(mu.shape[0], 2, dtype=ctx.dtype, device=ctx.device)
        v[:, 0] = mu.squeeze(1)
        v[:, 1] = (mu**2 + sigma_sq).squeeze(1)
        return v

    @classmethod
    def forward(
        cls, ctx: FunctionCtx, theta: torch.FloatTensor, psi: list[GaussianBasisFunctions]
    ) -> torch.FloatTensor:
        """
        We assume a Gaussian.
        We have:
            theta = [mu/sigma**2, -1/(2*sigma**2)],
            phi(t) = [t, t**2],
            p(t) = Gaussian(t; mu, sigma**2).

        Args:
            theta: shaped [BatchSize, 2]
            psi: list of basis functions
        """
        ctx.dtype = theta.dtype
        ctx.device = theta.device
        ctx.psi = psi
        # sigma_sq, mu: [BatchSize, 1]
        sigma_sq = (-0.5 / theta[:, 1]).unsqueeze(1)
        mu = theta[:, 0].unsqueeze(1) * sigma_sq

        r = cls._expectation_psi(ctx, mu, sigma_sq)
        ctx.save_for_backward(mu, sigma_sq, r)
        return r

    @classmethod
    def backward(cls, ctx: FunctionCtx, grad_output):
        mu, sigma_sq, r = ctx.saved_tensors
        J = cls._expectation_phi_psi(ctx, mu, sigma_sq)
        e_phi = cls._expectation_phi(ctx, mu, sigma_sq)
        e_psi = cls._expectation_psi(ctx, mu, sigma_sq)
        J -= torch.bmm(e_phi.unsqueeze(2), e_psi.unsqueeze(1))
        grad_input = torch.matmul(J, grad_output.unsqueeze(2)).squeeze(2)
        return grad_input, None


class ContinuousSoftmax(nn.Module):
    def __init__(self, psi: Optional[list[GaussianBasisFunctions]] = None):
        super(ContinuousSoftmax, self).__init__()
        self.psi = psi

    def forward(self, theta):
        return ContinuousSoftmaxFunction.apply(theta, self.psi)
