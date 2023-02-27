import torch

from memoria.engram import Engrams, EngramType
from memoria.memoria import Memoria


def test_add_working_memory():
    hippocampus = Memoria(3, 0.5, 3, 100, 0)
    hippocampus.add_working_memory(torch.randn(3, 10, 32))
    assert len(hippocampus.engrams) == 30


def test_calculate_wm_stm_weight():
    hippocampus = Memoria(3, 0.5, 3, 100, 0)
    hippocampus.add_working_memory(torch.randn(3, 10, 32))

    wm = Engrams(torch.randn(3, 10, 32))
    stm = Engrams(torch.randn(3, 20, 32), engrams_types=EngramType.SHORTTERM)
    weight = hippocampus.calculate_wm_stm_weight(wm, stm)
    assert weight.shape == torch.Size([3, 10, 20])


def test_remind_shortterm_memory():
    hippocampus = Memoria(3, 0.5, 3, 100, 0)

    weight = torch.tensor([[[0.51, 0.2, 0.2, 0.8]]])
    shortterm_memory_indices = torch.tensor([[1, 2, 3, 4]])
    reminded = hippocampus.remind_shortterm_memory(weight, shortterm_memory_indices)
    assert (reminded == torch.tensor([[1, -1, -1, 4]])).all()


def test_find_stm_nearest_to_ltm():
    hippocampus = Memoria(2, 0.5, 3, 100, 0)

    weight = torch.tensor([[[0.51, 0.2, 0.9, 0.8], [0.9, 0.1, 0.2, 0.5]]])
    shortterm_memory_indices = torch.tensor([[1, 2, 3, 4]])

    nearest_stm_indices = hippocampus.find_stm_nearest_to_ltm(weight, shortterm_memory_indices)

    assert (nearest_stm_indices == torch.tensor([[-1, 1, 3, 4]])).all()


def test_find_initial_ltm():
    num_initial_ltm = 3
    num_stm = 5
    num_ltm = 4
    hippocampus = Memoria(num_initial_ltm, 0.5, 3, 100, 0)

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

    initial_ltm_indices = hippocampus.find_initial_longterm_memory(nearest_stm_indices)
    assert (initial_ltm_indices == torch.tensor([[6, 7]])).all()


def test_search_longterm_memories_with_initials():
    num_initial_ltm = 2
    num_stm = 5
    num_ltm = 4
    ltm_search_depth = 3
    hippocampus = Memoria(num_initial_ltm, 0.5, ltm_search_depth, 100, 0)

    stm = Engrams(torch.randn(1, num_stm, 32), engrams_types=EngramType.SHORTTERM)
    ltm = Engrams(torch.randn(1, num_ltm, 32), engrams_types=EngramType.LONGTERM)
    hippocampus.engrams = stm + ltm

    initial_ltm_indices = torch.tensor([[5, 7]])
    searched_ltm_indices = hippocampus.search_longterm_memories_with_initials(initial_ltm_indices, ltm)

    assert searched_ltm_indices.shape == torch.Size([1, ltm_search_depth + 1, initial_ltm_indices.size(1)])
    assert (searched_ltm_indices == torch.tensor([[[0, 2], [1, 1], [3, 3], [-1, -1]]])).all()


def test_memorize_working_memory_as_shortterm_memory():
    num_initial_ltm = 2
    batch_size = 3
    num_wm = 5
    num_stm = 4
    num_ltm = 2
    ltm_search_depth = 3
    hippocampus = Memoria(num_initial_ltm, 0.5, ltm_search_depth, 100, 0)

    wm = Engrams(torch.randn(batch_size, num_wm, 32), engrams_types=EngramType.WORKING)
    stm = Engrams(torch.randn(batch_size, num_stm, 32), engrams_types=EngramType.SHORTTERM)
    ltm = Engrams(torch.randn(batch_size, num_ltm, 32), engrams_types=EngramType.LONGTERM)
    hippocampus.engrams = wm + stm + ltm

    hippocampus.memorize_working_memory_as_shortterm_memory()

    assert hippocampus.engrams.get_shortterm_memory()[0].data.shape == torch.Size([batch_size, num_wm + num_stm, 32])


def test_memorize_shortterm_memory_as_longterm_memory_or_drop():
    num_initial_ltm = 2
    batch_size = 1
    num_stm = 5
    num_ltm = 3
    ltm_search_depth = 3
    ltm_min_fire_count = 2
    hippocampus = Memoria(num_initial_ltm, 0.5, ltm_search_depth, 2, ltm_min_fire_count)

    fire_count = torch.tensor([[0, 1, 2, 3, 0]])
    stm = Engrams(torch.randn(batch_size, num_stm, 32), engrams_types=EngramType.SHORTTERM, fire_count=fire_count)
    ltm = Engrams(torch.randn(batch_size, num_ltm, 32), engrams_types=EngramType.LONGTERM)
    hippocampus.engrams = stm + ltm

    hippocampus.memorize_shortterm_memory_as_longterm_memory_or_drop()

    assert len(hippocampus.engrams) == batch_size * (2 + 4)


def test_call():
    num_initial_ltm = 3
    threshold_stm = 0.1
    ltm_search_depth = 3
    stm_capacity = 100
    ltm_min_fire_count = 2
    hippocampus = Memoria(
        num_initial_ltm=num_initial_ltm,
        stm_threshold=threshold_stm,
        ltm_search_depth=ltm_search_depth,
        stm_capacity=stm_capacity,
        ltm_min_fire_count=ltm_min_fire_count,
    )

    batch_size = 3
    memory_length = 50
    hidden_dim = 32
    outputs = hippocampus(torch.randn(batch_size, memory_length, hidden_dim))
    assert len(hippocampus.engrams.get_shortterm_memory()[0]) == batch_size * memory_length
    assert outputs.size(1) == 0

    outputs = hippocampus(torch.randn(batch_size, memory_length, hidden_dim))
    assert len(hippocampus.engrams.get_shortterm_memory()[0]) == batch_size * memory_length * 2
    assert outputs.size(1) > 0

    outputs = hippocampus(torch.randn(batch_size, memory_length, hidden_dim))
    assert len(hippocampus.engrams.get_shortterm_memory()[0]) == batch_size * memory_length * 2
    assert outputs.size(1) > 0
