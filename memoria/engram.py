from typing import Optional, Tuple, Union

import torch

from .engram_type import EngramType
from .engrams_functional import *


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

    def __eq__(self, other: "Engrams") -> bool:
        return engram_equals(
            self.data,
            other.data,
            self.induce_counts,
            other.induce_counts,
            self.engrams_types,
            other.engrams_types,
            self.lifespan,
            other.lifespan,
        )

    def __add__(self, other: "Engrams") -> "Engrams":
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self

        concatenated_data, concatenated_induce_counts, concatenated_engrams_types, concatenated_lifespan = engram_add(
            self.data,
            other.data,
            self.induce_counts,
            other.induce_counts,
            self.engrams_types,
            other.engrams_types,
            self.lifespan,
            other.lifespan,
        )

        return Engrams(
            concatenated_data,
            concatenated_induce_counts,
            concatenated_engrams_types,
            concatenated_lifespan,
        )

    @property
    @torch.no_grad()
    def working_memory_mask(self) -> torch.Tensor:
        return engrams_working_memory_mask(self.engrams_types)

    @property
    @torch.no_grad()
    def shortterm_memory_mask(self) -> torch.Tensor:
        return engrams_shortterm_memory_mask(self.engrams_types)

    @property
    @torch.no_grad()
    def longterm_memory_mask(self) -> torch.Tensor:
        return engrams_longterm_memory_mask(self.engrams_types)

    @property
    @torch.no_grad()
    def fire_count(self) -> torch.Tensor:
        return engrams_fire_count(self.induce_counts)

    @fire_count.setter
    @torch.no_grad()
    def fire_count(self, value: torch.Tensor) -> None:
        self.induce_counts = engrams_fire_count_setter(self.induce_counts, value)

    @torch.no_grad()
    def get_indices_with_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Get global indices with boolean mask

        Args:
            mask: bool type mask tensor shaped [BatchSize, MemoryLength]
        Return:
            global indices shaped [BatchSize, MaxNumSelectedMems]
                the value -1 means null.
        """
        return engrams_get_indices_with_mask(self.batch_size, self.memory_length, mask)

    @torch.no_grad()
    def get_mask_with_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get mask with indices

        Args:
            indices: indices, negative values ignored shaped [BatchSize, NumIndices]
        Return:
            mask: boolean mask shaped [BatchSize, MemoryLength]
        """
        return engrams_get_mask_with_indices(self.batch_size, self.memory_length, indices)

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

        Args:
            partial_mask: mask to select sub-engrams shaped [BatchSize, MemoryLength]
            global_indices: indices for this engrams shaped [BatchSize, NumIndices]
        """
        return engrams_get_local_indices_from_global_indices(
            self.data, self.induce_counts, self.engrams_types, self.lifespan, partial_mask, global_indices
        )

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
        selected_data, selected_induce_counts, selected_engrams_types, selected_lifespan = engrams_select(
            self.data, self.induce_counts, self.engrams_types, self.lifespan, indices
        )
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
        self.lifespan += engrams_extend_lifespan(self.batch_size, self.memory_length, indices, lifespan_delta)

    @torch.no_grad()
    def decrease_lifespan(self) -> None:
        self.lifespan -= 1.0
