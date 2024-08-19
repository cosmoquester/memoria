import pytest
import torch

from memoria.engram import Engrams, EngramType


def test_empty():
    empty = Engrams.empty()
    empty2 = Engrams.empty()

    assert len(empty) == 0
    assert empty == empty2

    empty_with_options = Engrams.empty(track_age=True, track_engram_id=True)
    assert empty_with_options.age is not None
    assert empty_with_options.engram_ids is not None


def test_init():
    batch_size = 3
    memory_length = 14
    hidden_dim = 32

    data = torch.randn(batch_size, memory_length, hidden_dim)
    engrams = Engrams(data)

    assert (engrams.fire_count == 0).all()
    assert (engrams.induce_counts == 0).all()
    assert (engrams.engrams_types == EngramType.WORKING.value).all()


def test_equals():
    batch_size = 3
    memory_length = 14
    hidden_dim = 32

    data = torch.randn(batch_size, memory_length, hidden_dim)
    engrams = Engrams(data)
    engrams2 = Engrams(data)

    assert engrams == engrams2

    engrams2.fire_count[0, 0] += 1
    assert engrams != engrams2


def test_engram_ids():
    batch_size = 3
    memory_length = 14
    hidden_dim = 32

    data = torch.randn(batch_size, memory_length, hidden_dim)
    engrams = Engrams(data, track_engram_id=True, engram_ids=0)
    assert (engrams.engram_ids == torch.arange(0, memory_length).unsqueeze(0).repeat(batch_size, 1)).all()

    data2 = torch.randn(batch_size, memory_length, hidden_dim)
    engrams2 = Engrams(data2, track_engram_id=True, engram_ids=14)
    assert (engrams2.engram_ids == torch.arange(14, 14 + memory_length).unsqueeze(0).repeat(batch_size, 1)).all()


def test_add():
    data = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    data2 = torch.tensor([[[4.0], [5.0], [6.0], [7.0]]])
    engrams = Engrams(data)
    engrams2 = Engrams(data2, engrams_types=EngramType.SHORTTERM)

    engram_added = engrams + engrams2

    assert len(engrams) + len(engrams2) == len(engram_added)
    assert (engram_added.data == torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]])).all()
    assert (
        engram_added.engrams_types
        == torch.tensor([[EngramType.WORKING.value] * 4 + [EngramType.SHORTTERM.value] * 4], dtype=torch.int)
    ).all()


def test_iter():
    batch_size = 3
    memory_length = 14
    hidden_dim = 32

    data = torch.randn(batch_size, memory_length, hidden_dim)
    engrams = Engrams(data)

    for i, engram in enumerate(engrams):
        assert engram.data.shape[0] == 1
        assert (engram.data == data[i]).all()


@pytest.mark.parametrize(
    "data,mask,expected",
    [
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[True, False, True, False], [True, True, False, False]]),
            torch.tensor([[0, 2], [0, 1]]),
        ),
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[True, False, True, False], [True, True, True, False]]),
            torch.tensor([[-1, 0, 2], [0, 1, 2]]),
        ),
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[False, False, False, False], [False, False, False, False]]),
            torch.tensor([[], []]),
        ),
    ],
)
def test_get_indices_with_mask(data: torch.Tensor, mask: torch.Tensor, expected: torch.Tensor):
    engrams = Engrams(data)
    indices = engrams.get_indices_with_mask(mask)

    assert (indices == expected).all()


@pytest.mark.parametrize(
    "data,indices,expected",
    [
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[0, 2], [0, 1]]),
            torch.tensor([[True, False, True, False], [True, True, False, False]]),
        ),
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[0, 2, -1], [0, 1, 2]]),
            torch.tensor([[True, False, True, False], [True, True, True, False]]),
        ),
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[], []], dtype=torch.long),
            torch.tensor([[False, False, False, False], [False, False, False, False]]),
        ),
    ],
)
def test_get_mask_with_indices(data: torch.Tensor, indices: torch.Tensor, expected: torch.Tensor):
    engrams = Engrams(data)
    mask = engrams.get_mask_with_indices(indices)

    assert (mask == expected).all()


@pytest.mark.parametrize(
    "data,indices,selected_data,selected_fire_count,selected_induce_counts,selected_engram_types",
    [
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([1, 3]),
            torch.tensor([[[1.0], [3.0]], [[5.0], [7.0]]]),
            torch.zeros([2, 2]),
            torch.zeros([2, 2, 2]),
            torch.tensor([[EngramType.WORKING.value] * 2] * 2),
        ),
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[1, 3], [0, 2]]),
            torch.tensor([[[1.0], [3.0]], [[4.0], [6.0]]]),
            torch.zeros([2, 2]),
            torch.zeros([2, 2, 2]),
            torch.tensor([[EngramType.WORKING.value] * 2] * 2),
        ),
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[1, 3], [0, -1]]),
            torch.tensor([[[1.0], [3.0]], [[4.0], [0.0]]]),
            torch.tensor([[0, 0], [0, -1]]),
            torch.tensor([[[0, 0], [0, 0]], [[0, -1], [-1, -1]]]),
            torch.tensor([[EngramType.WORKING.value] * 2, [EngramType.WORKING.value, EngramType.NULL.value]]),
        ),
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]], [[8.0], [9.0], [10.0], [11.0]]]),
            torch.tensor([[1, 3, 2], [0, -1, -1], [2, 3, -1]]),
            torch.tensor([[[1.0], [3.0], [2.0]], [[4.0], [0.0], [0.0]], [[10.0], [11.0], [0.0]]]),
            torch.tensor([[0, 0, 0], [0, -1, -1], [0, 0, -1]]),
            torch.tensor(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                    [[0, 0, -1], [0, 0, -1], [-1, -1, -1]],
                ]
            ),
            torch.tensor(
                [
                    [EngramType.WORKING.value] * 3,
                    [EngramType.WORKING.value, EngramType.NULL.value, EngramType.NULL.value],
                    [EngramType.WORKING.value, EngramType.WORKING.value, EngramType.NULL.value],
                ]
            ),
        ),
    ],
)
def test_select(
    data: torch.Tensor,
    indices: torch.Tensor,
    selected_data: torch.Tensor,
    selected_fire_count: torch.Tensor,
    selected_induce_counts: torch.Tensor,
    selected_engram_types: torch.Tensor,
):
    engrams = Engrams(data)

    selected_engrams = engrams.select(indices)
    assert (selected_engrams.data == selected_data).all()
    assert (selected_engrams.fire_count == selected_fire_count).all()
    assert (selected_engrams.induce_counts == selected_induce_counts).all()
    assert (selected_engrams.engrams_types == selected_engram_types).all()


def test_get_local_indices_from_global_indices():
    data = torch.tensor([[[1], [2], [3], [4], [5], [6]]])
    engrams_types = torch.tensor([[EngramType.SHORTTERM.value] * 3 + [EngramType.LONGTERM.value] * 3])
    engrams = Engrams(data, engrams_types=engrams_types)

    global_indices = [[0, 4, 5, -1]]
    local_indices = engrams.get_local_indices_from_global_indices(engrams.longterm_memory_mask, global_indices)

    assert (local_indices == torch.tensor([[1, 2]])).all()


def test_fire_together_wire_together():
    data = torch.tensor([[[1], [2], [3], [4], [5], [6]]])
    engrams = Engrams(data)

    engrams.fire_together_wire_together(torch.tensor([[0, 2, 4, -1]]))
    assert (engrams.fire_count == torch.tensor([[1, 0, 1, 0, 1, 0]])).all()
    assert (
        engrams.induce_counts
        == torch.tensor(
            [
                [1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    ).all()

    engrams.fire_together_wire_together(torch.tensor([[0, 1, 2, 2, 3, -1, -10]]))
    assert (engrams.fire_count == torch.tensor([[2, 1, 2, 1, 1, 0]])).all()
    assert (
        engrams.induce_counts
        == torch.tensor(
            [
                [2, 1, 2, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [2, 1, 2, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    ).all()


@pytest.mark.parametrize(
    "data,mask,selected_data,selected_fire_count,selected_induce_counts,selected_engram_types",
    [
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[False, True] * 2] * 2),
            torch.tensor([[[1.0], [3.0]], [[5.0], [7.0]]]),
            torch.zeros([2, 2]),
            torch.zeros([2, 2, 2]),
            torch.tensor([[EngramType.WORKING.value] * 2] * 2),
        ),
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[False, True, False, True], [True, False, True, False]]),
            torch.tensor([[[1.0], [3.0]], [[4.0], [6.0]]]),
            torch.zeros([2, 2]),
            torch.zeros([2, 2, 2]),
            torch.tensor([[EngramType.WORKING.value] * 2] * 2),
        ),
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]]]),
            torch.tensor([[False, True, False, True], [True, False, False, False]]),
            torch.tensor([[[1.0], [3.0]], [[0.0], [4.0]]]),
            torch.tensor([[0, 0], [-1, 0]]),
            torch.tensor([[[0, 0], [0, 0]], [[-1, -1], [-1, 0]]]),
            torch.tensor([[EngramType.WORKING.value] * 2, [EngramType.NULL.value, EngramType.WORKING.value]]),
        ),
        (
            torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0], [7.0]], [[8.0], [9.0], [10.0], [11.0]]]),
            torch.tensor([[False, True, True, True], [True, False, False, False], [False, False, True, True]]),
            torch.tensor([[[1.0], [2.0], [3.0]], [[0.0], [0.0], [4.0]], [[0.0], [10.0], [11.0]]]),
            torch.tensor([[0, 0, 0], [-1, -1, 0], [-1, 0, 0]]),
            torch.tensor(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[-1, -1, -1], [-1, -1, -1], [-1, -1, 0]],
                    [[-1, -1, -1], [-1, 0, 0], [-1, 0, 0]],
                ]
            ),
            torch.tensor(
                [
                    [EngramType.WORKING.value] * 3,
                    [EngramType.NULL.value, EngramType.NULL.value, EngramType.WORKING.value],
                    [EngramType.NULL.value, EngramType.WORKING.value, EngramType.WORKING.value],
                ]
            ),
        ),
    ],
)
def test_mask_select(
    data: torch.Tensor,
    mask: torch.Tensor,
    selected_data: torch.Tensor,
    selected_fire_count: torch.Tensor,
    selected_induce_counts: torch.Tensor,
    selected_engram_types: torch.Tensor,
):
    engrams = Engrams(data)

    selected_engrams = engrams.mask_select(mask)
    assert (selected_engrams.data == selected_data).all()
    assert (selected_engrams.fire_count == selected_fire_count).all()
    assert (selected_engrams.induce_counts == selected_induce_counts).all()
    assert (selected_engrams.engrams_types == selected_engram_types).all()


def test_get_memories():
    data = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            [[6.0], [7.0], [8.0], [9.0], [10.0]],
        ]
    )
    NULL = EngramType.NULL.value
    WORKING = EngramType.WORKING.value
    SHORTTERM = EngramType.SHORTTERM.value
    LONGTERM = EngramType.LONGTERM.value
    engrams_types = torch.tensor(
        [
            [WORKING, WORKING, LONGTERM, NULL, SHORTTERM],
            [SHORTTERM, NULL, LONGTERM, LONGTERM, WORKING],
        ]
    )
    engrams = Engrams(data=data, engrams_types=engrams_types)

    wm_engrams, wm_indices = engrams.get_working_memory()
    stm_engrams, stm_indices = engrams.get_shortterm_memory()
    ltm_engrams, ltm_indices = engrams.get_longterm_memory()

    assert wm_engrams == engrams.select(wm_indices)
    assert stm_engrams == engrams.select(stm_indices)
    assert ltm_engrams == engrams.select(ltm_indices)

    assert (engrams.select(wm_indices).data == torch.tensor([[[1.0], [2.0]], [[0.0], [10.0]]])).all()
    assert (engrams.select(stm_indices).data == torch.tensor([[[5.0]], [[6.0]]])).all()
    assert (engrams.select(ltm_indices).data == torch.tensor([[[0.0], [3.0]], [[8.0], [9.0]]])).all()

    assert ((wm_engrams.engrams_types == WORKING) | (wm_engrams.engrams_types == NULL)).all()
    assert ((stm_engrams.engrams_types == SHORTTERM) | (stm_engrams.engrams_types == NULL)).all()
    assert ((ltm_engrams.engrams_types == LONGTERM) | (ltm_engrams.engrams_types == NULL)).all()


def test_delete():
    data = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            [[6.0], [7.0], [8.0], [9.0], [10.0]],
        ]
    )
    engrams = Engrams(data=data)
    engrams.delete(torch.tensor([[1, 3], [4, -1]]))

    assert (
        engrams.data
        == torch.tensor(
            [
                [[0.0], [1.0], [3.0], [5.0]],
                [[6.0], [7.0], [8.0], [9.0]],
            ]
        )
    ).all()
    assert (engrams.fire_count == torch.tensor([[-1, 0, 0, 0], [0, 0, 0, 0]])).all()


def test_lifespan():
    data = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            [[6.0], [7.0], [8.0], [9.0], [10.0]],
        ]
    )
    lifespan = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )
    engrams = Engrams(data=data, lifespan=lifespan)

    indices = torch.tensor(
        [
            [4, -1, 0],
            [-1, 2, -1],
        ]
    )
    lifespan_delta = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
    )
    engrams.extend_lifespan(indices, lifespan_delta)

    assert (
        engrams.lifespan
        == torch.tensor(
            [
                [4.0, 2.0, 3.0, 4.0, 6.0],
                [2.0, 3.0, 7.0, 5.0, 6.0],
            ]
        )
    ).all()

    engrams.decrease_lifespan()
    assert (
        engrams.lifespan
        == torch.tensor(
            [
                [3.0, 1.0, 2.0, 3.0, 5.0],
                [1.0, 2.0, 6.0, 4.0, 5.0],
            ]
        )
    ).all()


def test_status_summary():
    data = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
        ]
    )
    NULL = EngramType.NULL.value
    WORKING = EngramType.WORKING.value
    SHORTTERM = EngramType.SHORTTERM.value
    LONGTERM = EngramType.LONGTERM.value
    engrams_types = torch.tensor(
        [
            [WORKING, WORKING, LONGTERM, NULL, SHORTTERM],
        ]
    )
    lifespan = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    age = torch.tensor([[0, 1, 2, 3, 4]])
    ids = torch.tensor([[0, 1, 2, 3, 4]])
    induce_counts = torch.tensor(
        [
            [5, 1, 1, 3, 1],
            [1, 4, 1, 0, 4],
            [1, 0, 3, 0, 3],
            [2, 2, 1, 2, 1],
            [1, 2, 3, 0, 7],
        ],
    ).unsqueeze(0)

    engrams = Engrams(
        data=data,
        engrams_types=engrams_types,
        induce_counts=induce_counts,
        lifespan=lifespan,
        age=age,
        engram_ids=ids,
        track_engram_id=True,
        track_age=True,
    )
    info = engrams.status_summary()[0]

    assert info.working == [0, 1]
    assert info.shortterm == [4]
    assert info.longterm == [2]
    assert len(info.edges) == 4 * 4 - 4 - 1
    assert len(info.engrams) == 4
    assert info.engrams[2].lifespan == 3.0
    assert info.engrams[0].type == "WORKING"
    assert info.engrams[1].age == 1
    assert [edge.weight for edge in info.engrams[4].outgoings] == (torch.tensor([1, 2, 3]) / 7).tolist()
    assert [edge.weight for edge in info.engrams[1].incoming] == (torch.tensor([1, 2]) / torch.tensor([5, 7])).tolist()
    assert info.engrams[0].outgoings[2].cofire_count == 1
    assert info.engrams[1].outgoings[2].cofire_count == 4
    assert info.engrams[4].fire_count == 7
