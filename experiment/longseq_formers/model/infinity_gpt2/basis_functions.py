import math

import torch


class BasisFunctions(object):
    def __init__(self):
        pass

    def __len__(self):
        """Number of basis functions."""
        pass

    def evaluate(self, t):
        pass

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        pass

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        pass

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        pass


class PowerBasisFunctions(BasisFunctions):
    """Function phi(t) = t**degree."""

    def __init__(self, degree):
        self.degree = degree.unsqueeze(0)

    def __len__(self):
        """Number of basis functions."""
        return self.degree.size(1)

    def evaluate(self, t):
        return t**self.degree

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        return (b ** (self.degree + 3) - a ** (self.degree + 3)) / (self.degree + 3)

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        return (b ** (self.degree + 2) - a ** (self.degree + 2)) / (self.degree + 2)

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        return (b ** (self.degree + 1) - a ** (self.degree + 1)) / (self.degree + 1)

    def __repr__(self):
        return f"PowerBasisFunction(degree={self.degree})"


class SineBasisFunctions(BasisFunctions):
    """Function phi(t) = sin(omega*t)."""

    def __init__(self, omega):
        self.omega = omega.unsqueeze(0)

    def __repr__(self):
        return f"SineBasisFunction(omega={self.omega})"

    def __len__(self):
        """Number of basis functions."""
        return self.omega.size(1)

    def evaluate(self, t):
        return torch.sin(self.omega * t)

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        # The antiderivative of (t**2)*sin(omega*t) is
        # ((2-(t**2)*(omega**2))*cos(omega*t) + 2*omega*t*sin(omega*t)) / omega**3.  # noqa
        return (
            (2 - (b**2) * (self.omega**2)) * torch.cos(self.omega * b)
            + 2 * self.omega * b * torch.sin(self.omega * b)
            - (2 - (a**2) * (self.omega**2)) * torch.cos(self.omega * a)
            - 2 * self.omega * a * torch.sin(self.omega * a)
        ) / (self.omega**3)

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        # The antiderivative of t*sin(omega*t) is
        # (sin(omega*t) - omega*t*cos(omega*t)) / omega**2.
        return (
            torch.sin(self.omega * b)
            - self.omega * b * torch.cos(self.omega * b)
            - torch.sin(self.omega * a)
            + self.omega * a * torch.cos(self.omega * a)
        ) / (self.omega**2)

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        # The antiderivative of sin(omega*t) is -cos(omega*t)/omega.
        return (-torch.cos(self.omega * b) + torch.cos(self.omega * a)) / self.omega


class CosineBasisFunctions(BasisFunctions):
    """Function phi(t) = cos(omega*t)."""

    def __init__(self, omega):
        self.omega = omega.unsqueeze(0)

    def __repr__(self):
        return f"CosineBasisFunction(omega={self.omega})"

    def __len__(self):
        """Number of basis functions."""
        return self.omega.size(1)

    def evaluate(self, t):
        return torch.cos(self.omega * t)

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        # The antiderivative of (t**2)*cos(omega*t) is
        # (((t**2)*(omega**2)-2)*cos(omega*t) + 2*omega*t*sin(omega*t)) / omega**3.  # noqa
        return (
            ((b**2) * (self.omega**2) - 2) * torch.sin(self.omega * b)
            + 2 * self.omega * b * torch.cos(self.omega * b)
            - ((a**2) * (self.omega**2) - 2) * torch.sin(self.omega * a)
            - 2 * self.omega * a * torch.cos(self.omega * a)
        ) / (self.omega**3)

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        # The antiderivative of t*cos(omega*t) is
        # (cos(omega*t) + omega*t*sin(omega*t)) / omega**2.
        return (
            torch.cos(self.omega * b)
            + self.omega * b * torch.sin(self.omega * b)
            - torch.cos(self.omega * a)
            - self.omega * a * torch.sin(self.omega * a)
        ) / (self.omega**2)

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        # The antiderivative of cos(omega*t) is sin(omega*t)/omega.
        return (torch.sin(self.omega * b) - torch.sin(self.omega * a)) / self.omega


class GaussianBasisFunctions(BasisFunctions):
    """Function phi(t) = Gaussian(t; mu, sigma_sq)

    Attributes:
        mu: mu shaped [1, NumBasis]
        sigma: sigma shaped [1, NumBasis]
    """

    def __init__(self, mu, sigma):
        self.mu = mu.unsqueeze(0)
        self.sigma = sigma.unsqueeze(0)

    def __repr__(self):
        return f"GaussianBasisFunction(mu={self.mu}, sigma={self.sigma})"

    def __len__(self):
        """Number of basis functions."""
        return self.mu.size(1)

    def _phi(self, t):
        return 1.0 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * t**2)

    def _Phi(self, t):
        return 0.5 * (1 + torch.erf(t / math.sqrt(2)))

    def _integrate_product_of_gaussians(self, mu, sigma_sq):
        sigma = torch.sqrt(self.sigma**2 + sigma_sq)
        return self._phi((mu - self.mu) / sigma) / sigma

    def evaluate(self, t):
        """Return Gaussian Function value

        Args:
            t: [BatchSize, NumBasis] or [BatchSize, 1] or scalar
                considered same value for all basis if NumBasis shape is none.
        Return:
            Gaussian function value shaped [BatchSize, NumBasis]
        """
        return self._phi((t - self.mu) / self.sigma) / self.sigma

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        return (
            (self.mu**2 + self.sigma**2)
            * (self._Phi((b - self.mu) / self.sigma) - self._Phi((a - self.mu) / self.sigma))
            - (self.sigma * (b + self.mu) * self._phi((b - self.mu) / self.sigma))
            + (self.sigma * (a + self.mu) * self._phi((a - self.mu) / self.sigma))
        )

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        return self.mu * (
            self._Phi((b - self.mu) / self.sigma) - self._Phi((a - self.mu) / self.sigma)
        ) - self.sigma * (self._phi((b - self.mu) / self.sigma) - self._phi((a - self.mu) / self.sigma))

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        return self._Phi((b - self.mu) / self.sigma) - self._Phi((a - self.mu) / self.sigma)

    def integrate_t2_times_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * t**2 * psi(t)."""
        S_tilde = self._integrate_product_of_gaussians(mu, sigma_sq)
        mu_tilde = (self.mu * sigma_sq + mu * self.sigma**2) / (self.sigma**2 + sigma_sq)
        sigma_sq_tilde = ((self.sigma**2) * sigma_sq) / (self.sigma**2 + sigma_sq)
        return S_tilde * (mu_tilde**2 + sigma_sq_tilde)

    def integrate_t_times_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * t * psi(t)."""
        S_tilde = self._integrate_product_of_gaussians(mu, sigma_sq)
        mu_tilde = (self.mu * sigma_sq + mu * self.sigma**2) / (self.sigma**2 + sigma_sq)
        return S_tilde * mu_tilde

    def integrate_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * psi(t)."""
        return self._integrate_product_of_gaussians(mu, sigma_sq)
