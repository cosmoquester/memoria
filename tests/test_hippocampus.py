import torch

from general_memory_network.engram import Engrams, EngramType
from general_memory_network.hippocampus import Hippocampus


def test_add_working_memory():
    hippocampus = Hippocampus(3, 0.5)
    hippocampus.add_working_memory(torch.randn(3, 10, 32))
    assert len(hippocampus.engrams) == 30


def test_calculate_wm_stm_weight():
    hippocampus = Hippocampus(3, 0.5)
    hippocampus.add_working_memory(torch.randn(3, 10, 32))

    wm = Engrams(torch.randn(3, 10, 32))
    stm = Engrams(torch.randn(3, 20, 32), engrams_types=EngramType.SHORTTERM)
    weight = hippocampus.calculate_wm_stm_weight(wm, stm)
    assert weight.shape == torch.Size([3, 10, 20])


def test_remind_shortterm_memory():
    hippocampus = Hippocampus(3, 0.5)

    weight = torch.tensor([[[0.51, 0.2, 0.2, 0.8]]])
    shortterm_memory_indices = torch.tensor([[1, 2, 3, 4]])
    reminded = hippocampus.remind_shortterm_memory(weight, shortterm_memory_indices)
    assert (reminded == torch.tensor([[1, -1, -1, 4]])).all()


def test_find_stm_nearest_to_ltm():
    hippocampus = Hippocampus(2, 0.5)

    weight = torch.tensor([[[0.51, 0.2, 0.9, 0.8], [0.9, 0.1, 0.2, 0.5]]])
    shortterm_memory_indices = torch.tensor([[1, 2, 3, 4]])

    nearest_stm_indices = hippocampus.find_stm_nearest_to_ltm(weight, shortterm_memory_indices)

    assert (nearest_stm_indices == torch.tensor([[-1, 1, 3, 4]])).all()


def test_find_initial_ltm():
    num_initial_ltm = 3
    num_stm = 5
    num_ltm = 4
    hippocampus = Hippocampus(num_initial_ltm, 0.5)

    stm = Engrams(torch.randn(1, num_stm, 32), engrams_types=EngramType.SHORTTERM)
    ltm = Engrams(torch.randn(1, num_ltm, 32), engrams_types=EngramType.LONGTERM)
    engrams = stm + ltm
    hippocampus.engrams = engrams
    hippocampus.engrams.induce_counts[:, :num_stm, num_stm:] = torch.tensor(
        [
            [
                [1, 1, 2, 1],
                [1, 1, 1, 2],
                [1, 10, 1, 1],
                [1, 1, 2, 1],
                [1, 5, 1, 1],
            ]
        ]
    )
    hippocampus.engrams.induce_counts[:, :num_stm, :num_stm] = 999
    nearest_stm_indices = torch.tensor([[0, 2, 3]])

    initial_ltm_indices = hippocampus.find_initial_ltm(nearest_stm_indices)
    assert (initial_ltm_indices == torch.tensor([[6, 7]])).all()
