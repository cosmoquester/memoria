from enum import Enum
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


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
        fire_count: fire count of each engram shaped [BatchSize, MemoryLength]
        induce_counts: induce count of each engram about each engram shaped [BatchSize, MemoryLength, MemoryLength]
        engrams_types: engram types of each engram shaped [BatchSize, MemoryLength]
    """

    @torch.no_grad()
    def __init__(
        self,
        data: torch.Tensor,
        fire_count: Optional[torch.Tensor] = None,
        induce_counts: Optional[torch.Tensor] = None,
        engrams_types: Optional[Union[torch.Tensor, EngramType]] = None,
    ) -> None:
        self.batch_size, self.memory_length, self.hidden_dim = data.shape
        self.data: torch.Tensor = data.detach()
        self.fire_count: torch.Tensor = (
            torch.zeros([self.batch_size, self.memory_length], dtype=int, requires_grad=False, device=data.device)
            if fire_count is None
            else fire_count.detach()
        )
        self.induce_counts: torch.Tensor = (
            torch.zeros(
                [self.batch_size, self.memory_length, self.memory_length],
                dtype=int,
                requires_grad=False,
                device=data.device,
            )
            if induce_counts is None
            else induce_counts.detach()
        )
        default_engram_type = engrams_types if isinstance(engrams_types, EngramType) else EngramType.WORKING
        self.engrams_types = (
            torch.full_like(
                self.fire_count, default_engram_type.value, dtype=int, requires_grad=False, device=data.device
            )
            if not isinstance(engrams_types, torch.Tensor)
            else engrams_types.detach()
        )

    @classmethod
    def empty(cls) -> "Engrams":
        return cls(torch.zeros([0, 0, 0], dtype=torch.int, requires_grad=False))

    def __len__(self) -> int:
        return self.fire_count.numel()

    @torch.no_grad()
    def __eq__(self, other: "Engrams") -> bool:
        return (
            (self.data == other.data).all()
            and (self.fire_count == other.fire_count).all()
            and (self.induce_counts == other.induce_counts).all()
            and (self.engrams_types == other.engrams_types).all()
        )

    @torch.no_grad()
    def __add__(self, other: "Engrams") -> "Engrams":
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self

        concatenated_data = torch.cat([self.data, other.data], dim=1)
        concatenated_fire_count = torch.cat([self.fire_count, other.fire_count], dim=1)
        concatenated_engrams_types = torch.cat([self.engrams_types, other.engrams_types], dim=1)

        new_memory_length = self.memory_length + other.memory_length
        concatenated_induce_counts = torch.zeros(
            [self.batch_size, new_memory_length, new_memory_length],
            dtype=int,
            requires_grad=False,
            device=self.data.device,
        )
        concatenated_induce_counts[:, : self.memory_length, : self.memory_length] = self.induce_counts
        concatenated_induce_counts[:, self.memory_length :, self.memory_length :] = other.induce_counts

        return Engrams(
            concatenated_data, concatenated_fire_count, concatenated_induce_counts, concatenated_engrams_types
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

    @torch.no_grad()
    def get_indices_with_mask(self, mask) -> torch.Tensor:
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

        # TODO: Don't use for loop
        indices = torch.full(
            [self.batch_size, max_num_memories],
            -1,
            requires_grad=False,
            device=mask.device,
            dtype=torch.long,
        )
        column_index = 0
        prev_row_index = 0
        for row_index, value in zip(*mask.nonzero(as_tuple=True)):
            if row_index != prev_row_index:
                column_index = 0
            indices[row_index, column_index] = value
            column_index += 1
            prev_row_index = row_index
        return indices

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
    def mask_select(self, mask: torch.BoolTensor) -> "Engrams":
        """Select values with mask

        Args:
            mask: mask tensor select only true value indices shaped [BatchSize, MemoryLength]
                which is same as self.data
        """
        # [BatchSize]
        num_memories_per_batch = mask.sum(dim=1)
        max_num_memories = num_memories_per_batch.max()

        # [BatchSize]
        num_pad = max_num_memories - num_memories_per_batch

        # [BatchSize, PadLength]
        mask_pad = ~F.one_hot(num_pad).cumsum(dim=1).bool()
        pad_length = mask_pad.size(1)

        # [BatchSize, MemoryLength + PadLength]
        padded_mask = torch.cat([mask, mask_pad], dim=1)

        # [BatchSize, MemoryLength + PadLength, HiddenDim]
        padded_data = torch.zeros(
            [self.batch_size, self.memory_length + pad_length, self.hidden_dim],
            dtype=self.data.dtype,
            requires_grad=False,
            device=self.data.device,
        )
        padded_data[:, : self.memory_length] = self.data

        # [BatchSize, MemoryLength + PadLength]
        padded_fire_count = torch.full_like(
            padded_mask, -1, dtype=self.fire_count.dtype, requires_grad=False, device=self.fire_count.device
        )
        padded_fire_count[:, : self.memory_length] = self.fire_count

        # [BatchSize, MemoryLength + PadLength]
        padded_engram_types = torch.full_like(
            padded_mask,
            fill_value=EngramType.NULL.value,
            dtype=self.engrams_types.dtype,
            requires_grad=False,
            device=self.engrams_types.device,
        )
        padded_engram_types[:, : self.memory_length] = self.engrams_types

        # [BatchSize, MemoryLength + PadLength, MemoryLength + PadLength]
        padded_induce_counts = torch.full(
            [self.batch_size, self.memory_length + pad_length, self.memory_length + pad_length],
            -1,
            dtype=self.induce_counts.dtype,
            requires_grad=False,
            device=self.induce_counts.device,
        )
        padded_induce_counts[:, : self.memory_length, : self.memory_length] = self.induce_counts

        padded_engrams = Engrams(padded_data, padded_fire_count, padded_induce_counts, padded_engram_types)

        raw_indices = torch.arange(
            self.memory_length + pad_length, requires_grad=False, device=padded_mask.device
        ).unsqueeze(0)
        indices = raw_indices.masked_select(padded_mask).view(self.batch_size, max_num_memories)

        selected_engrams = padded_engrams.select(indices)
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
            selected_fire_count = self.fire_count[:, indices]
            selected_induce_counts = self.induce_counts[:, indices][:, :, indices]
            selected_engrams_types = self.engrams_types[:, indices]
        elif len(indices.shape) == 2:
            index_0 = torch.arange(self.batch_size, device=self.induce_counts.device, requires_grad=False)
            selected_data = self.data[index_0.unsqueeze(1), indices]
            selected_fire_count = self.fire_count[index_0.unsqueeze(1), indices]
            selected_induce_counts = self.induce_counts[
                index_0.unsqueeze(1).unsqueeze(2), indices.unsqueeze(2), indices.unsqueeze(1)
            ]
            selected_engrams_types = self.engrams_types[index_0.unsqueeze(1), indices]

            null_indices_mask = indices < 0
            reverse_mask = (~null_indices_mask).int()
            selected_data.masked_fill_(null_indices_mask.unsqueeze(2), 0.0)
            selected_fire_count.masked_fill_(null_indices_mask, -1)
            selected_induce_counts.masked_fill_(1 - reverse_mask.unsqueeze(2) @ reverse_mask.unsqueeze(1), -1)
            selected_engrams_types.masked_fill_(null_indices_mask, EngramType.NULL.value)
        return Engrams(selected_data, selected_fire_count, selected_induce_counts, selected_engrams_types)
