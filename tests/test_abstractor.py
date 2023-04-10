import torch

from memoria.abstractor import Abstractor


def test_abstractor():
    abstractor = Abstractor(num_memories=3, hidden_dim=4, feedforward_dim=5)
    hidden_states = torch.randn(2, 3, 4)
    output = abstractor(hidden_states)
    assert output.shape == (2, 3, 4)
