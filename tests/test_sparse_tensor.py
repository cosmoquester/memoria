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

    selected = sparse_tensor[torch.tensor([[[0, 2]]])]
    assert selected.shape == (1, 1, 2, 3)
    assert selected.indices.tolist() == [[0, 0, 0, 0], [0, 0, 1, 2]]
    assert selected.values.tolist() == [1, 3]
    assert selected.tolist() == [[[[1, 0, 0], [0, 0, 3]]]]

    selected = sparse_tensor[torch.tensor([[0, 1], [2, 0]]), torch.tensor([[1, 2], [2, 0]])]
    assert selected.shape == (2, 2)
    assert selected.indices.tolist() == [[1, 0], [1, 1]]
    assert selected.values.tolist() == [3, 1]
    assert selected.tolist() == [[0, 0], [3, 1]]

    selected = sparse_tensor[0:2]
    assert selected.shape == (2, 3)
    assert selected.indices.tolist() == [[0, 0], [1, 1]]
    assert selected.values.tolist() == [1, 2]
    assert selected.tolist() == [[1, 0, 0], [0, 2, 0]]

    selected = sparse_tensor[0:1, 1:3]
    assert selected.shape == (1, 2)
    assert selected.indices.tolist() == []
    assert selected.values.tolist() == []
    assert selected.tolist() == [[0, 0]]

    selected = sparse_tensor[0:2, 1:3]
    assert selected.shape == (2, 2)
    assert selected.indices.tolist() == [[1, 0]]
    assert selected.values.tolist() == [2]
    assert selected.tolist() == [[0, 0], [2, 0]]

    selected = sparse_tensor[torch.tensor([0, 2]), 1:3]
    assert selected.shape == (2, 2)
    assert selected.indices.tolist() == [[1, 1]]
    assert selected.values.tolist() == [3]
    assert selected.tolist() == [[0, 0], [0, 3]]


def test_diagonal():
    tensor = torch.randn(2, 5, 3, 5)
    sparse_tensor = SparseTensor.from_tensor(tensor)

    assert (tensor.diagonal(dim1=1, dim2=3) == sparse_tensor.diagonal(dim1=1, dim2=3).to_dense()).all()


def test_equals():
    tensor = torch.randn(2, 5, 3, 5)
    sparse_tensor = SparseTensor.from_tensor(tensor)

    assert tensor == sparse_tensor
    assert (tensor == sparse_tensor.to_dense()).all()
    assert sparse_tensor != SparseTensor.from_tensor(torch.randn(2, 5, 3, 5))


def test_add():
    tensor = torch.randint(0, 5, [2, 5, 3, 5])
    tensor2 = torch.randint(0, 5, [2, 5, 3, 5])
    sparse_tensor = SparseTensor.from_tensor(tensor)
    sparse_tensor2 = SparseTensor.from_tensor(tensor2)

    assert (tensor + 1 == (sparse_tensor + 1).to_dense()).all()
    assert (tensor + tensor == (sparse_tensor + sparse_tensor).to_dense()).all()
    assert (tensor + tensor2 == (sparse_tensor + sparse_tensor2).to_dense()).all()


def test_unsqueeze():
    tensor = torch.randn(2, 5, 3, 5)
    sparse_tensor = SparseTensor.from_tensor(tensor)

    assert sparse_tensor.unsqueeze(0).shape == (1, 2, 5, 3, 5)
    assert sparse_tensor.unsqueeze(1).shape == (2, 1, 5, 3, 5)
    assert sparse_tensor.unsqueeze(2).shape == (2, 5, 1, 3, 5)
    assert sparse_tensor.unsqueeze(3).shape == (2, 5, 3, 1, 5)
    assert sparse_tensor.unsqueeze(4).shape == (2, 5, 3, 5, 1)
