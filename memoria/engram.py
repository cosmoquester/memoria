from enum import Enum
from typing import Optional, Tuple, Union

import torch


class EngramType(Enum):
    NULL = 0
    WORKING = 1
    SHORTTERM = 2
    LONGTERM = 3


class Engrams:
    """Memory Informations

    Attributes:
        batch_size: batch size of data
        memory_length: the number of engrams per each batch element
        data: real memory data shaped [BatchSize, MemoryLength, HiddenDim]
        induce_counts: induce count of each engram about each engram shaped [BatchSize, MemoryLength, MemoryLength]
        engrams_types: engram types of each engram shaped [BatchSize, MemoryLength]
        lifespan: lifespan of each engram shaped [BatchSize, MemoryLength]
    """

    @torch.no_grad()
    def __init__(
        self,
        data: torch.Tensor,
        induce_counts: Optional[torch.Tensor] = None,
        engrams_types: Optional[Union[torch.Tensor, EngramType]] = None,
        lifespan: Union[torch.Tensor, int] = 0,
    ) -> None:
        self.batch_size, self.memory_length, self.hidden_dim = data.shape
        self.data: torch.Tensor = data.detach().float()
        self.induce_counts: torch.Tensor = (
            torch.zeros(
                [self.batch_size, self.memory_length, self.memory_length],
                dtype=torch.int32,
                requires_grad=False,
                device=data.device,
            )
            if induce_counts is None
            else induce_counts.detach().type(torch.int32)
        )
        default_engram_type = engrams_types if isinstance(engrams_types, EngramType) else EngramType.WORKING
        self.engrams_types = (
            torch.full(
                [self.batch_size, self.memory_length],
                default_engram_type.value,
                dtype=torch.uint8,
                requires_grad=False,
                device=data.device,
            )
            if not isinstance(engrams_types, torch.Tensor)
            else engrams_types.detach().type(torch.uint8)
        )
        self.lifespan = (
            torch.full(
                [self.batch_size, self.memory_length],
                lifespan,
                dtype=torch.float32,
                requires_grad=False,
                device=data.device,
            )
            if not isinstance(lifespan, torch.Tensor)
            else lifespan.detach().type(torch.float32)
        )

    @classmethod
    def empty(cls) -> "Engrams":
        return cls(torch.zeros([0, 0, 0], dtype=torch.float32, requires_grad=False))

    def __len__(self) -> int:
        return self.lifespan.numel()

    @torch.no_grad()
    def __eq__(self, other: "Engrams") -> bool:
        return (
            (self.data == other.data).all()
            and (self.induce_counts == other.induce_counts).all()
            and (self.engrams_types == other.engrams_types).all()
            and (self.lifespan == other.lifespan).all()
        )

    @torch.no_grad()
    def __add__(self, other: "Engrams") -> "Engrams":
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self

        concatenated_data = torch.cat([self.data, other.data], dim=1)
        concatenated_engrams_types = torch.cat([self.engrams_types, other.engrams_types], dim=1)
        concatenated_lifespan = torch.cat([self.lifespan, other.lifespan], dim=1)

        new_memory_length = self.memory_length + other.memory_length
        concatenated_induce_counts = torch.zeros(
            [self.batch_size, new_memory_length, new_memory_length],
            dtype=torch.int32,
            requires_grad=False,
            device=self.data.device,
        )
        concatenated_induce_counts[:, : self.memory_length, : self.memory_length] = self.induce_counts
        concatenated_induce_counts[:, self.memory_length :, self.memory_length :] = other.induce_counts

        return Engrams(
            concatenated_data,
            concatenated_induce_counts,
            concatenated_engrams_types,
            concatenated_lifespan,
        )

    @property
    @torch.no_grad()
    def working_memory_mask(self) -> torch.Tensor:
        return self.engrams_types == EngramType.WORKING.value

    @property
    @torch.no_grad()
    def shortterm_memory_mask(self) -> torch.Tensor:
        return self.engrams_types == EngramType.SHORTTERM.value

    @property
    @torch.no_grad()
    def longterm_memory_mask(self) -> torch.Tensor:
        return self.engrams_types == EngramType.LONGTERM.value

    @property
    @torch.no_grad()
    def fire_count(self) -> torch.Tensor:
        return self.induce_counts.diagonal(dim1=1, dim2=2)

    @fire_count.setter
    @torch.no_grad()
    def fire_count(self, value: torch.Tensor) -> None:
        index = torch.arange(value.size(1), device=value.device, dtype=torch.long, requires_grad=False)
        self.induce_counts[:, index, index] = value

    @torch.no_grad()
    def get_indices_with_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Get global indices with boolean mask

        Args:
            mask: bool type mask tensor shaped [BatchSize, MemoryLength]
        Return:
            global indices shaped [BatchSize, MaxNumSelectedMems]
                the value -1 means null.
        """
        # [BatchSize, MemoryLength]
        num_memories_per_batch = mask.sum(dim=1)
        max_num_memories = num_memories_per_batch.max()
        if (num_memories_per_batch == max_num_memories).all():
            _, indices = mask.nonzero(as_tuple=True)
            indices = indices.view(self.batch_size, max_num_memories)
            return indices

        valued_mask = (
            torch.arange(self.memory_length, device=mask.device, dtype=torch.long, requires_grad=False).unsqueeze(0)
            * mask
        )
        valued_mask.masked_fill_(~mask, -1)
        sorted_values, _ = torch.sort(valued_mask, dim=1)
        indices = sorted_values[:, -max_num_memories:]
        return indices

    @torch.no_grad()
    def get_mask_with_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get mask with indices

        Args:
            indices: indices, negative values ignored shaped [BatchSize, NumIndices]
        Return:
            mask: boolean mask shaped [BatchSize, MemoryLength]
        """
        indices[indices < 0] = -1
        mask = torch.zeros(
            [self.batch_size, self.memory_length + 1], dtype=torch.bool, device=indices.device, requires_grad=False
        )
        index_0 = torch.arange(
            self.batch_size,
            device=indices.device,
            dtype=torch.long,
            requires_grad=False,
        ).unsqueeze(1)
        mask[index_0, indices] = True
        mask = mask[:, :-1]
        return mask

    @torch.no_grad()
    def get_working_memory(self) -> Tuple["Engrams", torch.Tensor]:
        """Get working memory engrams and global indices

        Return:
            memory: working memory engrams
            indices: global indices for memory engrams shaped [BatchSize, NumWMems]
                -1 means null index
        """
        mask = self.working_memory_mask
        indices = self.get_indices_with_mask(mask)
        memory = self.mask_select(mask=mask)
        return memory, indices

    @torch.no_grad()
    def get_shortterm_memory(self) -> Tuple["Engrams", torch.Tensor]:
        """Get shortterm memory engrams and global indices

        Return:
            memory: shortterm memory engrams
            indices: global indices for memory engrams shaped [BatchSize, NumSTMems]
                -1 means null index
        """
        mask = self.shortterm_memory_mask
        indices = self.get_indices_with_mask(mask)
        memory = self.mask_select(mask=mask)
        return memory, indices

    @torch.no_grad()
    def get_longterm_memory(self) -> Tuple["Engrams", torch.Tensor]:
        """Get longterm memory engrams and global indices

        Return:
            memory: longterm memory engrams
            indices: global indices for memory engrams shaped [BatchSize, NumLTMems]
                -1 means null index
        """
        mask = self.longterm_memory_mask
        indices = self.get_indices_with_mask(mask)
        memory = self.mask_select(mask=mask)
        return memory, indices

    @torch.no_grad()
    def get_local_indices_from_global_indices(
        self, partial_mask: torch.Tensor, global_indices: torch.Tensor
    ) -> torch.Tensor:
        """Get local ltm indices from global ltm indices
                -1 means null index, ignored

        Args:
            partial_mask: mask to select sub-engrams shaped [BatchSize, MemoryLength]
            global_indices: indices for this engrams shaped [BatchSize, NumIndices]
        """
        partial_indices = self.get_indices_with_mask(partial_mask)
        partial_engrams = self.select(partial_indices)

        selected_partial_mask_shape = list(partial_mask.shape)
        selected_partial_mask_shape[1] += 1
        selected_partial_mask = torch.full(
            selected_partial_mask_shape, False, requires_grad=False, device=partial_mask.device, dtype=torch.bool
        )
        index_0 = torch.arange(
            self.batch_size, requires_grad=False, device=partial_mask.device, dtype=torch.long
        ).unsqueeze(1)
        selected_partial_mask[index_0, global_indices] = True
        selected_partial_mask[:, -1] = False

        # [BatchSize, NumLTMems]
        local_selected_partial_mask = selected_partial_mask[index_0, partial_indices]
        # [BatchSize, NumUniqueInitialLTMs]
        local_selected_ltm_indices = partial_engrams.get_indices_with_mask(local_selected_partial_mask)

        return local_selected_ltm_indices

    @torch.no_grad()
    def fire_together_wire_together(self, indices: torch.Tensor) -> None:
        """Fire & Wire engrams with indices

        Args:
            indices: global indices of firing engrams shaped [BatchSize, NumIndices]
                -1 means ignore, multiple same indices considered once.
        """
        mask = self.get_mask_with_indices(indices)
        self.induce_counts += (mask.float().unsqueeze(2) @ mask.float().unsqueeze(1)).int()

    @torch.no_grad()
    def mask_select(self, mask: torch.Tensor) -> "Engrams":
        """Select values with mask

        Args:
            mask: mask tensor select only true value indices shaped [BatchSize, MemoryLength]
                which is same as self.data
        """
        indices = self.get_indices_with_mask(mask)
        selected_engrams = self.select(indices)
        return selected_engrams

    @torch.no_grad()
    def select(self, indices: torch.Tensor) -> "Engrams":
        """Select indices of memory parts

        Args:
            indices: index tensor for data shaped [NumIndices] or [BatchSize, NumIndices]
                when the indices is 2d-tensor, it can contains -1,
                -1 means null value. it will be null engram types.
        Return:
            Engrams whose data will be shaped [BatchSize, NumIndices, HiddenDim]
        """
        if len(indices.shape) == 1:
            selected_data = self.data[:, indices]
            selected_induce_counts = self.induce_counts[:, indices][:, :, indices]
            selected_engrams_types = self.engrams_types[:, indices]
            selected_lifespan = self.lifespan[:, indices]
        elif len(indices.shape) == 2:
            index_0 = torch.arange(self.batch_size, device=self.induce_counts.device, requires_grad=False)
            selected_data = self.data[index_0.unsqueeze(1), indices]
            selected_induce_counts = self.induce_counts[
                index_0.unsqueeze(1).unsqueeze(2), indices.unsqueeze(2), indices.unsqueeze(1)
            ]
            selected_engrams_types = self.engrams_types[index_0.unsqueeze(1), indices]
            selected_lifespan = self.lifespan[index_0.unsqueeze(1), indices]

            null_indices_mask = indices < 0
            reverse_mask = (~null_indices_mask).float()
            selected_data.masked_fill_(null_indices_mask.unsqueeze(2), 0.0)
            selected_induce_counts.masked_fill_(~(reverse_mask.unsqueeze(2) @ reverse_mask.unsqueeze(1)).bool(), -1)
            selected_engrams_types.masked_fill_(null_indices_mask, EngramType.NULL.value)
            selected_lifespan.masked_fill_(null_indices_mask, -1.0)
        else:
            raise ValueError("indices must be 1d or 2d tensor")
        return Engrams(selected_data, selected_induce_counts, selected_engrams_types, selected_lifespan)

    @torch.no_grad()
    def delete(self, indices: torch.Tensor) -> None:
        """Delete selected indices engrams

        Args:
            indices: indices of engrams to be deleted shaped [BatchSize, NumIndices]
        """
        mask = self.get_mask_with_indices(indices)
        preserve_indices = self.get_indices_with_mask(~mask)
        selected = self.select(preserve_indices)

        self.batch_size = selected.batch_size
        self.memory_length = selected.memory_length
        self.data = selected.data
        self.induce_counts = selected.induce_counts
        self.engrams_types = selected.engrams_types
        self.lifespan = selected.lifespan

    @torch.no_grad()
    def extend_lifespan(self, indices: torch.Tensor, lifespan_delta: torch.Tensor):
        """Extend lifespan of selected indices engrams by lifespan_delta

        Args:
            indices: indices of engrams to extend lifespan shaped [BatchSize, NumIndices]
            lifespan_delta: the extended lifespan which will be added to engrams's lifespan
                shaped [BatchSize, NumIndices]
        """
        indices[indices < 0] = -1
        lifespan_delta_mask = torch.zeros(
            [self.batch_size, self.memory_length + 1], dtype=torch.float32, device=indices.device, requires_grad=False
        )
        index_0 = torch.arange(self.batch_size, device=indices.device, requires_grad=False)
        lifespan_delta_mask[index_0.unsqueeze(1), indices] += lifespan_delta
        self.lifespan += lifespan_delta_mask[:, :-1]

    @torch.no_grad()
    def decrease_lifespan(self) -> None:
        self.lifespan -= 1.0
