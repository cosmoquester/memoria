import torch

from memoria.sparse_tensor import SparseTensor


def test_from_tensor():
    tensor = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=torch.int32)
    sparse_tensor = SparseTensor.from_tensor(tensor)
    assert sparse_tensor.indices.tolist() == [[0, 0], [1, 1], [2, 2]]
    assert sparse_tensor.values.tolist() == [1, 2, 3]


def test_get_item():
    tensor = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=torch.int32)
    sparse_tensor = SparseTensor.from_tensor(tensor)

    assert sparse_tensor.indices.tolist() == [[0, 0], [1, 1], [2, 2]]
    assert sparse_tensor.values.tolist() == [1, 2, 3]

    selected = sparse_tensor[0, 0]
    assert isinstance(selected, torch.Tensor)
    assert selected.item() == 1

    selected = sparse_tensor[1]
    assert selected.shape == (3,)
    assert selected.indices.tolist() == [[1]]
    assert selected.values.tolist() == [2]
    assert selected.tolist() == [0, 2, 0]

    selected = sparse_tensor[:, 2]
    assert selected.shape == (3,)
    assert selected.indices.tolist() == [[2]]
    assert selected.values.tolist() == [3]
    assert selected.tolist() == [0, 0, 3]

    selected = sparse_tensor[torch.tensor([0, 2])]
    assert selected.shape == (2, 3)
    assert selected.indices.tolist() == [[0, 0], [1, 2]]
    assert selected.values.tolist() == [1, 3]
    assert selected.tolist() == [[1, 0, 0], [0, 0, 3]]

    selected = sparse_tensor[torch.tensor([0, 2]), 2]
    assert selected.shape == (2,)
    assert selected.indices.tolist() == [[1]]
    assert selected.values.tolist() == [3]
    assert selected.tolist() == [0, 3]
