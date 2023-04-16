import torch

from memoria.engram import Engrams, EngramType
from memoria.memoria import Memoria


def test_add_working_memory():
    memoria = Memoria(
        num_reminded_stm=10,
        ltm_search_depth=3,
        stm_capacity=100,
        initial_lifespan=100,
        num_final_ltms=100,
    )
    memoria.add_working_memory(torch.randn(3, 10, 32))
    assert len(memoria.engrams) == 30


def test_calculate_wm_stm_weight():
    memoria = Memoria(
        num_reminded_stm=10,
        ltm_search_depth=3,
        stm_capacity=100,
        initial_lifespan=100,
        num_final_ltms=100,
    )
    memoria.add_working_memory(torch.randn(3, 10, 32))

    wm = Engrams(torch.randn(3, 10, 32))
    stm = Engrams(torch.randn(3, 20, 32), engrams_types=EngramType.SHORTTERM)
    weight = memoria._calculate_memory_weight(wm, stm)
    assert weight.shape == torch.Size([3, 10, 20])


def test_remind_shortterm_memory():
    memoria = Memoria(
        num_reminded_stm=2,
        ltm_search_depth=3,
        stm_capacity=100,
        initial_lifespan=100,
        num_final_ltms=100,
    )

    weight = torch.tensor([[[0.51, 0.2, 0.2, 0.8]]])
    shortterm_memory_indices = torch.tensor([[1, 2, 3, 4]])
    reminded = memoria._remind_shortterm_memory(weight, shortterm_memory_indices)
    assert (reminded == torch.tensor([[1, -1, -1, 4]])).all()


def test_find_initial_ltm():
    num_stm = 5
    num_ltm = 4
    memoria = Memoria(
        num_reminded_stm=10,
        ltm_search_depth=3,
        stm_capacity=100,
        initial_lifespan=100,
        num_final_ltms=100,
    )

    stm = Engrams(torch.randn(1, num_stm, 32), engrams_types=EngramType.SHORTTERM)
    ltm = Engrams(torch.randn(1, num_ltm, 32), engrams_types=EngramType.LONGTERM)
    engrams = stm + ltm
    memoria.engrams = engrams
    memoria.engrams.induce_counts[:, :num_stm, num_stm:] = torch.tensor(
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
    memoria.engrams.induce_counts[:, :num_stm, :num_stm] = 999
    nearest_stm_indices = torch.tensor([[0, 2, 3]])

    initial_ltm_indices = memoria._find_initial_longterm_memory(nearest_stm_indices)
    assert (initial_ltm_indices == torch.tensor([[6, 7]])).all()


def test_search_longterm_memories_with_initials():
    num_stm = 5
    num_ltm = 4
    ltm_search_depth = 3
    memoria = Memoria(
        num_reminded_stm=10,
        ltm_search_depth=ltm_search_depth,
        stm_capacity=100,
        initial_lifespan=100,
        num_final_ltms=100,
    )

    stm = Engrams(torch.randn(1, num_stm, 32), engrams_types=EngramType.SHORTTERM)
    ltm = Engrams(torch.randn(1, num_ltm, 32), engrams_types=EngramType.LONGTERM)
    memoria.engrams = stm + ltm

    initial_ltm_indices = torch.tensor([[5, 7]])
    searched_ltm_indices = memoria._search_longterm_memories_with_initials(initial_ltm_indices, ltm)

    assert (searched_ltm_indices == torch.tensor([[1, 3, 2, -1, 0]])).all()


def test_memorize_working_memory_as_shortterm_memory():
    batch_size = 3
    num_wm = 5
    num_stm = 4
    num_ltm = 2
    ltm_search_depth = 3
    memoria = Memoria(
        num_reminded_stm=10,
        ltm_search_depth=ltm_search_depth,
        stm_capacity=100,
        initial_lifespan=100,
        num_final_ltms=100,
    )

    wm = Engrams(torch.randn(batch_size, num_wm, 32), engrams_types=EngramType.WORKING)
    stm = Engrams(torch.randn(batch_size, num_stm, 32), engrams_types=EngramType.SHORTTERM)
    ltm = Engrams(torch.randn(batch_size, num_ltm, 32), engrams_types=EngramType.LONGTERM)
    memoria.engrams = wm + stm + ltm

    memoria._memorize_working_memory_as_shortterm_memory()

    assert memoria.engrams.get_shortterm_memory()[0].data.shape == torch.Size([batch_size, num_wm + num_stm, 32])


def test_memorize_shortterm_memory_as_longterm_memory_or_drop():
    batch_size = 1
    num_stm = 5
    num_ltm = 3
    ltm_search_depth = 3
    memoria = Memoria(
        num_reminded_stm=10,
        ltm_search_depth=ltm_search_depth,
        stm_capacity=2,
        initial_lifespan=100,
        num_final_ltms=100,
    )

    fire_count = torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.int32)
    stm = Engrams(torch.randn(batch_size, num_stm, 32), engrams_types=EngramType.SHORTTERM)
    stm.fire_count = fire_count
    ltm = Engrams(torch.randn(batch_size, num_ltm, 32), engrams_types=EngramType.LONGTERM)
    memoria.engrams = stm + ltm

    memoria._memorize_shortterm_memory_as_longterm_memory()

    assert len(memoria.engrams) == batch_size * (num_stm + num_ltm)


def test_remind():
    num_reminded_stm = 2
    ltm_search_depth = 3
    stm_capacity = 100
    memoria = Memoria(
        num_reminded_stm=num_reminded_stm,
        ltm_search_depth=ltm_search_depth,
        stm_capacity=stm_capacity,
        initial_lifespan=100,
        num_final_ltms=100,
    )

    batch_size = 3
    memory_length = 50
    hidden_dim = 32
    working_memory = torch.randn(batch_size, memory_length, hidden_dim)
    memoria.add_working_memory(working_memory)
    outputs, indices = memoria.remind()
    memoria.adjust_lifespan_and_memories(indices, torch.ones_like(indices, dtype=float))
    assert len(memoria.engrams.get_shortterm_memory()[0]) == batch_size * memory_length
    assert outputs.size(1) == 0

    working_memory = torch.randn(batch_size, memory_length, hidden_dim)
    memoria.add_working_memory(working_memory)
    outputs, indices = memoria.remind()
    memoria.adjust_lifespan_and_memories(indices, torch.ones_like(indices, dtype=float))
    assert len(memoria.engrams.get_shortterm_memory()[0]) == batch_size * memory_length * 2
    assert outputs.size(1) > 0

    working_memory = torch.randn(batch_size, memory_length, hidden_dim)
    memoria.add_working_memory(working_memory)
    outputs, indices = memoria.remind()
    memoria.adjust_lifespan_and_memories(indices, torch.ones_like(indices, dtype=float))
    assert len(memoria.engrams.get_shortterm_memory()[0]) == batch_size * memory_length * 2
    assert outputs.size(1) > 0

    memoria.enable_stm = False
    working_memory = torch.randn(batch_size, memory_length, hidden_dim)
    memoria.add_working_memory(working_memory)
    outputs, indices = memoria.remind()
    assert (memoria.engrams.select(indices).engrams_types != EngramType.SHORTTERM.value).all()

    memoria.enable_ltm = False
    working_memory = torch.randn(batch_size, memory_length, hidden_dim)
    memoria.add_working_memory(working_memory)
    outputs, indices = memoria.remind()
    assert indices.numel() == 0

    memoria.enable_stm = True
    working_memory = torch.randn(batch_size, memory_length, hidden_dim)
    memoria.add_working_memory(working_memory)
    outputs, indices = memoria.remind()
    assert (memoria.engrams.select(indices).engrams_types != EngramType.LONGTERM.value).all()


def test_reset_memory():
    memoria = Memoria(
        num_reminded_stm=10,
        ltm_search_depth=3,
        stm_capacity=100,
        initial_lifespan=100,
        num_final_ltms=100,
    )
    memoria.add_working_memory(torch.randn(3, 10, 32))
    memoria.reset_memory()
    assert memoria.engrams == Engrams.empty()
