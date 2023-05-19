"""
Modified from: https://github.com/deep-spin/infinite-former
"""
from typing import List, Literal, Optional, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn

from .basis_functions import GaussianBasisFunctions
from .continuous_softmax import ContinuousSoftmax
from .continuous_sparsemax import ContinuousSparsemax


class GaussianBasisFunctionsModule(nn.Module, GaussianBasisFunctions):
    """GaussianBasisFunctionsModule

    Attributes:
        mu: mu shaped [1, NumBasis]
        sigma: sigma shaped [1, NumBasis]
    """

    def __init__(self, mu, sigma) -> None:
        super(GaussianBasisFunctionsModule, self).__init__()

        self.register_buffer("mu", mu.unsqueeze(0))
        self.register_buffer("sigma", sigma.unsqueeze(0))


class LongTermAttention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        memory_length: int,
        num_basis: int,
        num_samples: int,
        normalize_function: Literal["softmax", "sparsemax"],
        longterm_attention_dropout: float,
        num_attention_heads: int,
        mask_type: Literal["affine", "cnn", "none"],
        mask_dropout: float,
        tau: float,
        mu_0: Optional[float],
        sigma_0: Optional[float],
        use_affines: bool,
        use_kl_regularizer: bool,
        use_infinite_memory: bool,
        use_sticky_memories: bool,
        **kwargs,
    ):
        """Longterm contious attention

        Args:
            head_dim: hidden dimension of each head
            memory_length: memory length (as same as target length)
            num_basis: the number of long term attention basis
            num_samples: number of samples used for update
            normalize_function: attention function softmax or sparsemax
            longterm_attention_dropout: attention dropout
            use_infinite_memory: use infinity memory if true
            num_attention_heads: the number of heads
            mask_type: masking type
            mask_dropout: masking dropout
            tau: ratio of existing inputs meaning tau on paper
            mu_0: used to calculate kl regularization value
            sigma_0: used to calculate kl regularization value
            use_affines: use affine transform
            use_kl_regularizer: use kl regularizer
            use_infinite_memory: use infinity memory
            use_sticky_memories: use sticky memory
        """
        super(LongTermAttention, self).__init__(**kwargs)

        self.memory_length = memory_length
        self.head_dim = head_dim
        self.num_basis = num_basis
        self.num_samples = num_samples
        self.normalize_function = normalize_function
        self.num_attention_heads = num_attention_heads
        self.sigma_0 = sigma_0
        self.mu_0 = mu_0
        self.mask_type = mask_type

        self.use_affines = use_affines  # whether mu, sigma should be computed using affine transformations
        self.use_kl_regularizer = use_kl_regularizer
        self.use_infinite_memory = use_infinite_memory
        self.use_sticky_memories = use_sticky_memories

        self.hidden_dim = num_attention_heads * head_dim
        self.proj_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.proj_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.proj_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.attn_out = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.longterm_attention_dropout = nn.Dropout(longterm_attention_dropout)

        if self.mask_type == "affine":
            self.mask_net = nn.Linear(memory_length, memory_length)
        elif self.mask_type == "cnn":
            self.mask_net = torch.nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        elif self.mask_type != "none":
            raise ValueError(f"`mask_type` should be one of (affine, cnn, none)")
        self.mask_dropout = nn.Dropout(mask_dropout)

        # TODO: Check the meaning of padding
        padding = True

        if use_affines:
            self.mu = nn.Linear(num_basis, 1, bias=False)
            self.sigma = nn.Linear(num_basis, 1, bias=False)
            self.softplus = torch.nn.Softplus()

        # set gaussian basis sigmas
        sigmas = [0.005, 0.01]
        if num_basis % len(sigmas):
            num_basis += len(sigmas) - num_basis % len(sigmas)

        # get positions for memory vectors
        # basis_mu, basis_sigma: [NumBasis]
        basis_mu, basis_sigma = self.get_gaussian_basis_functions(num_basis, sigmas)
        self.psi = GaussianBasisFunctionsModule(mu=basis_mu, sigma=basis_sigma)
        self.register_buffer("basis_mu", basis_mu)
        self.register_buffer("basis_sigma", basis_sigma)

        # normalizing function
        if normalize_function == "softmax":
            self.transform = ContinuousSoftmax(psi=[self.psi])
        elif normalize_function == "sparsemax":
            self.transform = ContinuousSparsemax(psi=[self.psi])
        else:
            raise ValueError(f"`normalize_function` cannot be '{normalize_function}'!")

        # compute basis functions
        # [PositionLength]
        positions = self.get_positions(memory_length, padding)
        # [MemoryLength, NumBasis]
        Gs = self.compute_G_matrix(self.psi, memory_length, positions, padding=padding)
        self.register_buffer("Gs", Gs)
        self.register_buffer("positions", positions[int(memory_length / 2) : -int(memory_length / 2)])

        # compute samples for memory update
        if self.use_infinite_memory:
            samples, G_inf = self.get_infinity_components(tau, padding)
            self.register_buffer("G_inf", G_inf)

            if self.use_sticky_memories:
                self.register_buffer("bins", torch.linspace(0, 1, 129))
            else:
                self.register_buffer("samples", samples)

    def get_infinity_components(self, tau: float, padding: bool):
        """
        Args:
            padding: padding
            tau: ratio of existing inputs meaning tau on paper
        Return:
            samples and G_inf
                samples shaped [NumSamples, NumBasis]
                G_inf shaped [MemoryLength + NumSamples, NumBasis]
        """
        # [NumSamples]
        positions_existing = torch.linspace(0.0, tau, self.num_samples + 1)[1:]
        # [MemoryLength]
        positions_new = torch.linspace(tau, 1.0, self.memory_length + 1)[1:]

        positions = torch.cat([positions_existing, positions_new], dim=0)

        if padding:
            if self.memory_length % 2:
                shift = 1 / (self.memory_length + self.num_samples)
                positions_pad_before = torch.linspace(-0.5 + shift, 0, 2 * (self.memory_length + self.num_samples) - 1)
            else:
                shift = 1 / (2 * self.memory_length + self.num_samples)
                positions_pad = torch.linspace(-0.5 + shift, 1.5 - shift, 2 * (self.memory_length + self.num_samples))
            positions_pad_before = torch.tensor([i for i in positions_pad if i < 0], dtype=torch.float)
            positions_pad_after = torch.tensor([i for i in positions_pad if i > 1], dtype=torch.float)
            positions = torch.cat([positions_pad_before, positions, positions_pad_after], dim=0)

        # [NumSamples, NumBasis]
        samples = torch.cat([self.psi.evaluate(t / tau) for t in positions_existing], dim=0)

        # [MemoryLength + NumSamples, NumBasis]
        G_inf = self.compute_G_matrix(self.psi, self.memory_length + self.num_samples, positions, padding=padding)

        return samples, G_inf

    @staticmethod
    def get_gaussian_basis_functions(num_basis: int, sigmas: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make gaussian basis mus with sigmas using grid

        Args:
            num_basis: the number of gaussian basis
            sigmas: sigma values to be used for each mu
        Return:
            tuple of mu and sigmas
            mu is initialized [0, 1] values linear spacing repeated over all sigmas
            sigmas is repeated whose counts are summed to num_basis
        """
        # mu, sigma: [NumBasis // LenSigmas, LenSigmas]
        mu, sigma = torch.meshgrid(torch.linspace(0, 1, num_basis // len(sigmas)), torch.tensor(sigmas))
        mu = mu.flatten()
        sigma = sigma.flatten()
        assert mu.size(0) == num_basis
        # mu, sigma: [NumBasis]
        return mu, sigma

    @staticmethod
    def get_positions(memory_length: int, padding: bool) -> torch.FloatTensor:
        """Return positions for sampling points

        Args:
            memory_length: memory length
            padding: whether using padding or not
        Return:
            positions shaped [MemoryLength] or [MemoryLength * 2] or [MemoryLength * 2 - 1]
        """
        if padding:
            if memory_length % 2:
                shift = 1 / memory_length
                positions = torch.linspace(-0.5 + shift, 1.5 - shift, memory_length * 2 - 1)
            else:
                shift = 1 / (memory_length * 2)
                positions = torch.linspace(-0.5 + shift, 1.5 - shift, memory_length * 2)
        else:
            shift = 1 / (memory_length * 2)
            positions = torch.linspace(shift, 1 - shift, memory_length)
        return positions

    def compute_G_matrix(
        self,
        psi: GaussianBasisFunctionsModule,
        length: int,
        positions: torch.FloatTensor,
        ridge_penalty: float = 0.5,
        padding=True,
    ) -> torch.FloatTensor:
        """Compute G Matrix with gaussian distribution and sampling positions
        Refer Eq.5 in paper

        Args:
            psi: gaussian distribution
            length: original length
            positions: sampling positions shaped [PositionLength]
            ridge_penalty: inverse penalty
        Return:
            G matrix which is inverse of gaussian sampled matrix with position
                shaped [Length, NumBasis]
        """
        F = torch.zeros(self.num_basis, positions.size(0))
        F[:, :] = psi.evaluate(positions.unsqueeze(1)).T

        penalty = torch.eye(self.num_basis) * ridge_penalty
        G = F.T @ (F @ F.T + penalty).inverse()

        if padding:
            if length % 2:
                G = G[((length - 1) // 2) : (-(length - 1) // 2), :]
            else:
                G = G[(length // 2) : -(length // 2), :]

        return G

    def calculate_attention_weight(self, query: torch.FloatTensor, key: torch.FloatTensor) -> torch.FloatTensor:
        """Get attention weight

        Args:
            query: query shaped [BatchSize, NumHeads, QueryLength, HeadDim]
            key: key shaped [BatchSize, NumHeads, NumBasis, HeadDim]
        Return:
            attention scores shaped [BatchSize, NumHeads, QueryLength, NumBasis]
        """
        query = query / (self.head_dim**0.5)
        key = key.transpose(-1, -2)
        scores = torch.matmul(query, key)
        return scores

    def transform_to_continous(self, key: torch.FloatTensor, infinity: bool = False) -> torch.FloatTensor:
        """Get continous signal of input, which is B matrix, from discret input key

        Args:
            key: permuted memory [BatchSize, HiddenDim, KeyLength] if not infinity
                [BatchSize, HiddenDim, NumSamples + KeyLength] if infinity is True
            infinity: whether received concatened input or raw input
        Return:
            B: shaped [BatchSize, NumBasis, HiddenDim]
                B x psi(t) = x̄(t)
        """
        if infinity:
            # [NumSamples + KeyLength, NumBasis]
            G = self.G_inf
        else:
            # [KeyLength, NumBasis]
            G = self.Gs

        # [BatchSize, NumBasis, HiddenDim]
        B = torch.matmul(key, G).transpose(1, 2)
        return B

    def regress_B_matrix(
        self,
        key: torch.FloatTensor,
        B_past: Optional[torch.FloatTensor],
        past_attention_mu: Optional[torch.FloatTensor],
        past_attention_sigma: Optional[torch.FloatTensor],
    ) -> torch.FloatTensor:
        """Regress B matrix using new key and points sampled with B past

        Args:
            key: permuted memory [BatchSize, HiddenDim, KeyLength]
            B_past: B value calculated in previous timestep [BatchSize, NumBasis, HiddenDim]
            past_attention_mu: attention past output mu [BatchSize, NumHeads * QueryLength]
            past_attention_sigma: attention past output sigma [BatchSize, NumHeads * QueryLength]
        Return:
            B: shaped [BatchSize, NumBasis, HiddenDim]
        """
        # Make B only with current key
        if B_past is None:
            # [BatchSize, NumBasis, HiddenDim]
            B = self.transform_to_continous(key)
            return B

        if self.use_sticky_memories:
            batch_size = key.size(0)
            normal_dist = dist.Normal(past_attention_mu, past_attention_sigma)

            # [NumBins]
            bins = self.bins.clone()
            bins[0] = -0.000001
            bins[-1] = 1.000001
            # [NumBins, 1, 1]
            bins = bins.view(bins.size(0), 1, 1)
            # [NumBins, BatchSize, NumHeads * QueryLength]
            bins_cumulative_probs = normal_dist.cdf(bins)

            # [BatchSize, NumBins - 1]
            section_probs = (bins_cumulative_probs[1:] - bins_cumulative_probs[:-1]).sum(dim=-1).T
            section_probs /= section_probs.sum(dim=-1, keepdim=True)

            section_categorical_dist = dist.Categorical(section_probs)

            # [BatchSize, NumSamples]
            section_indices = section_categorical_dist.sample([self.num_samples])
            sampled_section_positions = self.bins[section_indices].T
            sampled_section_positions, _ = torch.sort(sampled_section_positions, dim=-1)
            # TODO: Check whether it is okay that there are many same positions?

            samples = torch.zeros(batch_size, self.num_samples, self.num_basis, device=key.device)
            for i in range(batch_size):
                # [NumSamples, NumBasis]
                samples[i] = self.psi.evaluate(sampled_section_positions[i].unsqueeze(1))
        else:
            samples = self.samples

        # [BatchSize, HiddenDim, NumSamples]
        xm_tau = B_past.transpose(1, 2).matmul(samples.transpose(1, 2))
        # [BatchSize, HiddenDim, NumSamples + KeyLength]
        concatenated_key = torch.cat([xm_tau, key], dim=2)
        # [BatchSize, NumBasis, HiddenDim]
        B = self.transform_to_continous(concatenated_key, infinity=True)
        return B

    def forward(
        self,
        query: torch.FloatTensor,
        key: Optional[torch.FloatTensor],
        B_past: Optional[torch.FloatTensor] = None,
        past_attention_mu=None,
        past_attention_sigma=None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Args:
            query: query shaped [BatchSize, NumHeads, QueryLength, HeadDim]
            key: memory [BatchSize, KeyLength, HiddenDim]
            B_past: B value calculated in previous timestep [BatchSize, NumBasis, HiddenDim]
            past_attention_mu: [BatchSize, NumHeads * QueryLength]
            past_attention_sigma: [BatchSize, NumHeads * QueryLength]
        """
        # TODO: Check whether it is good that start from empty memory
        # We can use current input as key without previous timestep input
        if key is None:
            return (None,) * 5

        batch_size, query_length = query.size(0), query.size(2)

        # [BatchSize, HiddenDim, KeyLength]
        key = self.mask_dropout(key).transpose(1, 2)

        if self.mask_type != "none":
            # [BatchSize, HiddenDim, KeyLength]
            reg_mask = self.mask_net(key).sigmoid()
            key = key * reg_mask

        # Update memory
        if self.use_infinite_memory:
            # [BatchSize, NumBasis, HiddenDim]
            B = self.regress_B_matrix(key, B_past, past_attention_mu, past_attention_sigma)
            new_B_past = B.detach()
        else:
            # [BatchSize, NumBasis, HiddenDim]
            B = self.transform_to_continous(key)
            new_B_past = None

        # projected_key, projected_value: [BatchSize, NumBasis, HiddenDim]
        projected_key = self.proj_key(B)
        projected_value = self.proj_value(B)

        # [BatchSize, NumHeads, NumBasis, HeadDim]
        projected_key = projected_key.view(
            batch_size,
            self.num_basis,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)
        # [BatchSize, NumHeads, HeadDim, NumBasis]
        projected_value = projected_value.view(
            batch_size,
            self.num_basis,
            self.num_attention_heads,
            self.head_dim,
        ).permute(0, 2, 3, 1)

        # [BatchSize, NumHeads, QueryLength, NumBasis]
        qk_logits = self.calculate_attention_weight(query, projected_key)

        # compute mu and sigma
        if self.use_affines:
            # TODO: check why transformed to dim-1?
            # query-key logits should have information on time series
            # but one mu, sigma value is enough to represent values on time?

            # mu, variance: [BatchSize, NumHeads, QueryLength, 1]
            mu = self.mu(qk_logits).sigmoid()
            variance = self.softplus(self.sigma(qk_logits))

            mu = mu.view(-1)
            variance = variance.clamp(min=1e-4).view(-1)
        else:
            # mu, variance: [BatchSize, NumHeads, QueryLength]
            qk_logits = qk_logits.softmax(dim=-1)
            mu = qk_logits @ self.basis_mu
            variance = qk_logits @ (self.basis_mu**2 + self.basis_sigma**2) - mu**2

            mu = mu.view(-1)
            variance = variance.view(-1)

        # [BatchSize * NumHeads * QueryLength, 2]
        theta = torch.zeros(batch_size * self.num_attention_heads * query_length, 2, device=key.device)
        # Transform to gaussian form
        theta[:, 0] = mu / variance
        theta[:, 1] = -1.0 / (2.0 * variance)

        # compute basis functions expectation
        # [BatchSize * NumHeads * QueryLength, NumBasis]
        qk_weights = self.transform(theta)

        # [BatchSize, NumHeads, NumBasis, QueryLength]
        qk_weights = qk_weights.view(
            batch_size,
            self.num_attention_heads,
            query_length,
            self.num_basis,
        ).permute(0, 1, 3, 2)

        # [BatchSize, NumHeads, HeadDim, QueryLength]
        context = projected_value @ qk_weights

        # [QueryLength, BatchSize, HiddenDim]
        context = context.view(batch_size, self.hidden_dim, query_length).transpose(1, 2)
        context = self.attn_out(context)

        if self.use_sticky_memories:
            new_past_attention_mu = mu.view(batch_size, -1).detach()
            new_past_attention_sigma = variance.view(batch_size, -1).sqrt().detach()
        else:
            new_past_attention_mu, new_past_attention_sigma = None, None

        if self.use_kl_regularizer:
            variance_0 = self.sigma_0**2
            # fmt: off
            if self.mu_0 > 0:
                kl_reg = 1 / 2 * (variance.view(batch_size, -1) / variance_0 - torch.log(variance.view(batch_size, -1) / variance_0) - 1)
            else:
                kl_reg = 1 / 2 * (variance.view(batch_size, -1) / variance_0 - torch.log(variance.view(batch_size, -1) / variance_0) - 1 + (mu.view(batch_size, -1) - self.mu_0) ** 2 / variance_0)
            # fmt: on
        else:
            kl_reg = None

        return context, new_B_past, new_past_attention_mu, new_past_attention_sigma, kl_reg

    def __repr__(self):
        return "ContinuousAttention"
