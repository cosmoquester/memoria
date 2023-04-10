import torch
import torch.nn as nn


class Abstractor(nn.Module):
    """Abstract Module to summarize data"""

    def __init__(self, num_memories: int, hidden_dim: int, feedforward_dim: int) -> None:
        """
        Args:
            num_memories (int): Number of memories to be created
            hidden_dim (int): Hidden dimension of the model
            feedforward_dim (int): Feedforward dimension of the model
        """
        super().__init__()

        w = torch.empty(1, num_memories, hidden_dim)
        nn.init.normal_(w, std=0.02)
        self.query_embeddings = nn.Parameter(w)
        self.key_transform = nn.Linear(hidden_dim, hidden_dim)
        self.value_transform = nn.Linear(hidden_dim, hidden_dim)
        self.feedforward = nn.Linear(hidden_dim, feedforward_dim)
        self.output = nn.Linear(feedforward_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """

        Args:
            hidden_states (torch.Tensor): [Batch, N, HiddenDim]
        Returns:
            torch.Tensor: [Batch, NumMemories, HiddenDim]
        """
        query = self.query_embeddings
        key = self.key_transform(hidden_states)
        # [Batch, N, HiddemDim]
        value = self.value_transform(hidden_states)
        # [Batch, NumMemories, HiddenDim] x [Batch, N, HiddenDim] -> [Batch, NumMemories, N]
        attn = query @ key.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = attn @ value
        attn = self.feedforward(attn)
        attn = self.output(attn)
        return attn
