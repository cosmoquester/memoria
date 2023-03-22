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

    def tolist(self) -> list:
        return self.to_dense().tolist()

    def __repr__(self) -> str:
        return f"SparseTensor(shape={self.shape}, default_value={self.default_value}, indices={self.indices}, values={self.values})"
