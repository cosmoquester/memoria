from itertools import product
from typing import Iterable, List, Optional, Tuple, Union

import torch


class SparseTensor:
    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        default_value: Union[int, float] = 0,
        shape: Optional[Tuple[int]] = None,
    ) -> None:
        self.shape = tuple((indices.max(dim=0).values + 1).tolist()) if shape is None else tuple(shape)
        self.device = values.device

        self.indices = indices.to(self.device)
        self.values = values
        self.default_value = default_value

    def dim(self) -> int:
        return len(self.shape)

    def __getitem__(self, raw_keys: Union[int, slice, torch.Tensor, Tuple, List]) -> "SparseTensor":
        keys = self.__get_keys_with_raw_keys(raw_keys)
        current_data_indices = self.indices.clone()
        current_data_values = self.values.clone()
        current_shape = self.shape

        # Process Scalar Index
        for dim, dim_keys in reversed(list(enumerate(keys))):
            if not isinstance(dim_keys, int):
                continue

            selected_mask = current_data_indices[:, dim] == dim_keys
            selected_values = current_data_values.masked_select(selected_mask)
            selected_indices = torch.cat([current_data_indices[:, :dim], current_data_indices[:, dim + 1 :]], dim=1)
            selected_indices = selected_indices.masked_select(selected_mask.unsqueeze(1)).view(-1, self.dim() - 1)

            current_data_indices = selected_indices
            current_data_values = selected_values
            keys = keys[:dim] + keys[dim + 1 :]
            current_shape = current_shape[:dim] + current_shape[dim + 1 :]

        # Process Slice Index
        for dim, dim_keys in enumerate(keys):
            if not isinstance(dim_keys, slice):
                continue

            dim_key_values = list(range(dim_keys.start, dim_keys.stop, dim_keys.step))
            selected_mask = (current_data_indices[:, dim : dim + 1] == torch.tensor([dim_key_values])).any(dim=1)
            selected_values = current_data_values.masked_select(selected_mask)
            selected_indices = current_data_indices.masked_select(selected_mask.unsqueeze(1)).view(
                -1, current_data_indices.size(1)
            )
            selected_indices[:, dim] -= dim_keys.start
            selected_indices[:, dim] //= dim_keys.step

            current_data_indices = selected_indices
            current_data_values = selected_values
            keys[dim] = None
            current_shape = current_shape[:dim] + (len(dim_key_values),) + current_shape[dim + 1 :]

        # Process Complex Index
        if current_shape == ():
            return current_data_values

        if all(x is None for x in keys):
            return SparseTensor(current_data_indices, current_data_values, self.default_value, current_shape)

        new_key_rank = max(key.dim() for key in keys if key is not None)
        keys = [v.view([*[1] * (new_key_rank - v.dim()), *v.shape]) if v is not None else None for v in keys]
        new_key_shape = tuple(max(key.shape[i] for key in keys if key is not None) for i in range(new_key_rank))
        new_indices = []
        new_values = []

        dim_keys = [(dim, key) for dim, key in enumerate(keys) if key is not None]
        new_indices_dim = dim_keys[0][0]
        for key_idx in product(*[range(s) for s in new_key_shape if s is not None]):
            selected_mask = torch.ones([current_data_indices.size(0)], dtype=torch.bool, device=self.device)
            for dim, key in dim_keys:
                # Boradcast size 1
                key_idx_boardcast = tuple(idx_ if sz != 1 else 0 for idx_, sz in zip(key_idx, key.shape))
                selected_mask &= current_data_indices[:, dim] == key[key_idx_boardcast]

            selected_indices = current_data_indices.masked_select(selected_mask.unsqueeze(1)).view(
                -1, current_data_indices.size(1)
            )
            dim_to_idx = {dim: idx for idx, (dim, _) in zip(key_idx, dim_keys)}
            selected_indices = torch.cat(
                [
                    selected_indices[:, :new_indices_dim],
                    torch.tensor([key_idx]).expand([selected_indices.size(0), len(key_idx)]),
                ]
                + [
                    selected_indices[:, i : i + 1]
                    for i in range(new_indices_dim + 1, selected_indices.size(1))
                    if i not in dim_to_idx
                ],
                dim=1,
            )
            selected_values = current_data_values.masked_select(selected_mask)
            new_indices.append(selected_indices)
            new_values.append(selected_values)

        new_indices = torch.cat(new_indices, dim=0)
        new_values = torch.cat(new_values, dim=0)
        new_shape = []
        for dim, key in enumerate(keys):
            if dim == new_indices_dim:
                new_shape.extend(new_key_shape)
            elif key is None:
                new_shape.append(current_shape[dim])

        return SparseTensor(new_indices, new_values, self.default_value, new_shape)

    def __setitem__(self, raw_keys: Union[int, slice, torch.Tensor, Tuple, List], value: Union[int, float]):
        keys = self.__get_keys_with_raw_keys(raw_keys)
        selected_mask = torch.ones([self.indices.size(0)], dtype=torch.bool, device=self.device)
        tensor_key_shape = []
        for dim, key in list(enumerate(keys)):
            if isinstance(key, int):
                keys[dim] = [key]
            elif isinstance(key, slice):
                keys[dim] = range(key.start, key.stop, key.step)
            elif isinstance(key, torch.Tensor):
                if key.dim() < len(tensor_key_shape):
                    key = key.view(*([1] * (len(tensor_key_shape) - key.dim())), *key.shape)
                elif key.dim() > len(tensor_key_shape):
                    tensor_key_shape = [1] * (key.dim() - len(tensor_key_shape)) + tensor_key_shape
                tensor_key_shape = [max(s, ks) for s, ks in zip(key.shape, tensor_key_shape)]
                continue
            elif key is None:
                keys[dim] = range(self.shape[dim])
                continue
            else:
                raise ValueError("key must be int, slice, torch.Tensor, or None")
            key = torch.tensor([keys[dim]], device=self.device, dtype=torch.int32)
            selected_mask &= (self.indices[:, dim : dim + 1] == key).any(dim=1)

        # Process Tensor Key
        for dim, key in list(enumerate(keys)):
            if isinstance(key, torch.Tensor):
                keys[dim] = key.expand(*tensor_key_shape).flatten()

        tensor_keys = [key for key in keys if isinstance(key, torch.Tensor)]
        tensor_key_dims = [dim for dim, key in enumerate(keys) if isinstance(key, torch.Tensor)]
        if tensor_keys:
            tensor_key_indices = self.indices[:, tensor_key_dims]
            key_indices = torch.tensor(list(zip(*tensor_keys)), device=self.device, dtype=torch.int32)
            selected_mask &= (tensor_key_indices.unsqueeze(2) == key_indices.T.unsqueeze(0)).all(dim=1).any(dim=1)

        # Deleted Selected Values
        self.indices = self.indices.masked_select(selected_mask.logical_not().unsqueeze(1)).view(
            -1, self.indices.size(1)
        )
        self.values = self.values.masked_select(selected_mask.logical_not())

        if self.default_value == value:
            return

        # Add new values
        new_indices = []
        non_tensor_keys = [key for key in keys if not isinstance(key, torch.Tensor)]
        for *new_index, tensor_key in product(*non_tensor_keys, zip(*tensor_keys) if tensor_keys else [()]):
            for dim, key in zip(tensor_key_dims, tensor_key):
                new_index.insert(dim, key)
            new_indices.append(new_index)
        new_indices = torch.tensor(sorted(new_indices), device=self.device, dtype=torch.int32)
        new_values = torch.tensor([value] * new_indices.size(0), device=self.device, dtype=self.values.dtype)

        self.indices = torch.cat([self.indices, new_indices], dim=0)
        self.values = torch.cat([self.values, new_values], dim=0)

    def __add__(self, other: Union[int, "SparseTensor"]) -> "SparseTensor":
        if isinstance(other, int):
            return SparseTensor(self.indices, self.values + other, self.default_value + other, self.shape)
        elif isinstance(other, SparseTensor):
            if self.default_value != other.default_value:
                raise ValueError("default_value must be the same")
            if self.shape != other.shape:
                raise ValueError("shape must be the same")

            added = self.clone()
            duplicated_mask = (self.indices.unsqueeze(-1) == other.indices.T).all(dim=1)
            added.values[duplicated_mask.any(dim=1).nonzero().squeeze(-1)] += other.values[
                duplicated_mask.any(dim=0).nonzero().squeeze(-1)
            ]
            exclusive_mask = duplicated_mask.any(dim=0).logical_not().nonzero().squeeze(-1)
            added.indices = torch.cat([added.indices, other.indices[exclusive_mask]], dim=0)
            added.values = torch.cat([added.values, other.values[exclusive_mask]], dim=0)
            return added
        else:
            raise ValueError("other must be int or SparseTensor")

    def unsqueeze(self, dim: int = -1) -> "SparseTensor":
        if dim < 0:
            dim += self.dim()

        indices = torch.cat(
            [
                self.indices[:, :dim],
                torch.zeros([self.indices.size(0), 1], dtype=torch.int32, device=self.device),
                self.indices[:, dim:],
            ],
            dim=1,
        )
        return SparseTensor(indices, self.values, self.default_value, self.shape[:dim] + (1,) + self.shape[dim:])

    @classmethod
    def empty(
        cls,
        size: Iterable[int],
        default_value: Union[int, float] = 0,
        dtype: Optional[torch.dtype] = torch.int32,
        device: Optional[torch.device] = None,
    ):
        indices = torch.empty((0, len(size)), dtype=torch.int32, device=device)
        values = torch.empty((0,), dtype=dtype, device=device)
        return cls(indices, values, default_value)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, default_value: Union[int, float] = 0):
        indices = torch.nonzero(tensor != default_value, as_tuple=True)
        values = tensor[indices]
        sparse_tensor = cls(torch.stack(indices, dim=1), values, default_value)
        return sparse_tensor

    def to_dense(self) -> torch.Tensor:
        tensor = torch.full(self.shape, self.default_value, dtype=self.values.dtype, device=self.device)
        tensor[self.indices.tensor_split(self.indices.size(1), dim=1)] = self.values.unsqueeze(1)
        return tensor

    def diagonal(self, dim1: int = 0, dim2: int = 1) -> torch.Tensor:
        if dim1 < 0:
            dim1 += self.dim()
        if dim2 < 0:
            dim2 += self.dim()
        dim1, dim2 = min(dim1, dim2), max(dim1, dim2)

        mask = self.indices[:, dim1] == self.indices[:, dim2]
        indices = self.indices.masked_select(mask.unsqueeze(1)).view(-1, self.indices.size(1))
        values = self.values.masked_select(mask)
        indices = indices[:, [dim for dim in range(indices.size(1)) if dim != dim1]]
        shape = self.shape[:dim1] + self.shape[dim1 + 1 :]
        return SparseTensor(indices, values, self.default_value, shape)

    def to(self, device: torch.device) -> "SparseTensor":
        return SparseTensor(self.indices.to(device), self.values.to(device), self.default_value, self.shape)

    def tolist(self) -> list:
        return self.to_dense().tolist()

    def clone(self) -> "SparseTensor":
        return SparseTensor(self.indices.clone(), self.values.clone(), self.default_value, self.shape)

    def __eq__(self, other: Union["SparseTensor", torch.Tensor]) -> bool:
        if isinstance(other, torch.Tensor):
            other = SparseTensor.from_tensor(other, self.default_value)
        if isinstance(other, SparseTensor):
            return (
                self.shape == other.shape
                and self.default_value == other.default_value
                and torch.equal(self.indices, other.indices)
                and torch.equal(self.values, other.values)
            )
        return False

    def __get_keys_with_raw_keys(
        self, raw_keys: Union[int, slice, torch.Tensor, Tuple, List]
    ) -> List[Union[int, slice, torch.Tensor]]:
        if not isinstance(raw_keys, tuple):
            raw_keys = (raw_keys,)
        raw_keys = raw_keys + (None,) * (self.dim() - len(raw_keys))

        keys = list(raw_keys)
        for dim, dim_raw_indices in enumerate(raw_keys):
            if isinstance(dim_raw_indices, int) or dim_raw_indices is None:
                pass
            elif isinstance(dim_raw_indices, slice):
                if dim_raw_indices.start is None and dim_raw_indices.stop is None and dim_raw_indices.step is None:
                    keys[dim] = None
                else:
                    start = dim_raw_indices.start or 0
                    stop = min(dim_raw_indices.stop or self.shape[dim], self.shape[dim])
                    step = dim_raw_indices.step or 1
                    keys[dim] = slice(start, stop, step)
            elif isinstance(dim_raw_indices, torch.Tensor):
                keys[dim] = dim_raw_indices
            elif isinstance(dim_raw_indices, list):
                keys[dim] = torch.tensor(dim_raw_indices, dtype=torch.int32, device=self.device)
            else:
                raise ValueError(f"Unsupported index type {type(keys)}")
        return keys

    def __repr__(self) -> str:
        return f"SparseTensor(shape={self.shape}, default_value={self.default_value}, indices={self.indices}, values={self.values})"
