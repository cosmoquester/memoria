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

    selected = sparse_tensor[0, 2]
    assert isinstance(selected, torch.Tensor)
    assert selected.item() == 0

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


def test_set_item():
    tensor = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=torch.int32)
    sparse_tensor = SparseTensor.from_tensor(tensor)

    sparse_tensor[0, 0] = torch.tensor(90)
    assert sparse_tensor.indices.tolist() == [[1, 1], [2, 2], [0, 0]]
    assert sparse_tensor.values.tolist() == [2, 3, 90]
    assert sparse_tensor.to_dense().tolist() == [[90, 0, 0], [0, 2, 0], [0, 0, 3]]

    sparse_tensor[0, 0] += 10
    assert sparse_tensor.indices.tolist() == [[1, 1], [2, 2], [0, 0]]
    assert sparse_tensor.values.tolist() == [2, 3, 100]
    assert sparse_tensor.to_dense().tolist() == [[100, 0, 0], [0, 2, 0], [0, 0, 3]]

    sparse_tensor = SparseTensor.from_tensor(tensor)
    sparse_tensor[0, 0] = 10
    assert sparse_tensor.indices.tolist() == [[1, 1], [2, 2], [0, 0]]
    assert sparse_tensor.values.tolist() == [2, 3, 10]
    assert sparse_tensor.to_dense().tolist() == [[10, 0, 0], [0, 2, 0], [0, 0, 3]]

    sparse_tensor[1] = 20
    assert sparse_tensor.indices.tolist() == [[2, 2], [0, 0], [1, 0], [1, 1], [1, 2]]
    assert sparse_tensor.values.tolist() == [3, 10, 20, 20, 20]
    assert sparse_tensor.to_dense().tolist() == [[10, 0, 0], [20, 20, 20], [0, 0, 3]]

    sparse_tensor[:, 2] = 30
    assert sparse_tensor.indices.tolist() == [[0, 0], [1, 0], [1, 1], [0, 2], [1, 2], [2, 2]]
    assert sparse_tensor.values.tolist() == [10, 20, 20, 30, 30, 30]
    assert sparse_tensor.to_dense().tolist() == [[10, 0, 30], [20, 20, 30], [0, 0, 30]]

    sparse_tensor = SparseTensor.from_tensor(tensor)
    sparse_tensor[torch.tensor([0, 2])] = 40
    assert sparse_tensor.indices.tolist() == [[1, 1], [0, 0], [0, 1], [0, 2], [2, 0], [2, 1], [2, 2]]
    assert sparse_tensor.values.tolist() == [2, 40, 40, 40, 40, 40, 40]
    assert sparse_tensor.to_dense().tolist() == [[40, 40, 40], [0, 2, 0], [40, 40, 40]]

    sparse_tensor = SparseTensor.from_tensor(tensor)
    sparse_tensor[torch.tensor([0, 2]), 2] = 50
    assert sparse_tensor.indices.tolist() == [[0, 0], [1, 1], [0, 2], [2, 2]]
    assert sparse_tensor.values.tolist() == [1, 2, 50, 50]
    assert sparse_tensor.to_dense().tolist() == [[1, 0, 50], [0, 2, 0], [0, 0, 50]]

    sparse_tensor = SparseTensor.from_tensor(tensor)
    sparse_tensor[torch.tensor([[[0, 2]]])] = 60
    assert sparse_tensor.indices.tolist() == [[1, 1], [0, 0], [0, 1], [0, 2], [2, 0], [2, 1], [2, 2]]
    assert sparse_tensor.values.tolist() == [2, 60, 60, 60, 60, 60, 60]
    assert sparse_tensor.to_dense().tolist() == [[60, 60, 60], [0, 2, 0], [60, 60, 60]]

    sparse_tensor = SparseTensor.from_tensor(tensor)
    sparse_tensor[torch.tensor([[0, 1], [2, 0]]), torch.tensor([[1, 2], [2, 0]])] = 70
    assert sparse_tensor.indices.tolist() == [[1, 1], [0, 1], [1, 2], [2, 2], [0, 0]]
    assert sparse_tensor.values.tolist() == [2, 70, 70, 70, 70]
    assert sparse_tensor.to_dense().tolist() == [[70, 70, 0], [0, 2, 70], [0, 0, 70]]

    sparse_tensor = SparseTensor.from_tensor(tensor)
    sparse_tensor[torch.tensor([[0, 1], [2, 0]]), torch.tensor([[1, 2], [2, 0]])] = torch.tensor([[70, 80], [90, 100]])
    assert sparse_tensor.indices.tolist() == [[1, 1], [0, 1], [1, 2], [2, 2], [0, 0]]
    assert sparse_tensor.values.tolist() == [2, 70, 80, 90, 100]
    assert sparse_tensor.to_dense().tolist() == [[100, 70, 0], [0, 2, 80], [0, 0, 90]]

    sparse_tensor = SparseTensor.from_tensor(tensor)
    sparse_tensor[0:2] = 80
    assert sparse_tensor.indices.tolist() == [[2, 2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    assert sparse_tensor.values.tolist() == [3, 80, 80, 80, 80, 80, 80]
    assert sparse_tensor.to_dense().tolist() == [[80, 80, 80], [80, 80, 80], [0, 0, 3]]


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


def test_to():
    tensor = torch.randn(2, 5, 3, 5)
    sparse_tensor = SparseTensor.from_tensor(tensor)

    assert sparse_tensor.to(torch.device("cpu")) == sparse_tensor
    assert (sparse_tensor.to(torch.device("cpu")).to_dense() == tensor).all()
