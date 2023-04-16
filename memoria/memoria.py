from typing import Optional, Tuple, Union

import torch

from .engram import Engrams, EngramType


class Memoria:
    """Memoria, general memory network for sequential modeling or processing"""

    def __init__(
        self,
        num_reminded_stm: float,
        stm_capacity: int,
        ltm_search_depth: int,
        ltm_min_fire_count: int,
        initial_lifespan: int,
        num_final_ltms: int,
        enable_stm: bool = True,
        enable_ltm: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """

        Args:
            num_reminded_stm: the number of stm reminded
            stm_capacity: the maximum memory length per batch for shortterm memory
            ltm_search_depth: the maximum number of depth for dfs memory search
            ltm_min_fire_count: the minimum fire count value to memorize shortterm memory into longterm memory.
            initial_lifespan: initial lifespan for each engrams
            num_final_ltms: the number of longterm memories to return
            enable_stm: whether to use shortterm memory.
                this module will not return shortterm memory indices, but keep shortterm memory
                to remind longterm memory. so when `enable ltm` is True.
            enable_ltm: whether to use longterm memory. this module will keep shortterm and longterm memories.
            device: memoria device to save engrams
        """
        self.engrams = Engrams.empty()

        self.num_reminded_stm: float = num_reminded_stm
        self.stm_capacity: int = stm_capacity
        self.ltm_search_depth: int = ltm_search_depth
        self.ltm_min_fire_count: int = ltm_min_fire_count
        self.initial_lifespan: int = initial_lifespan
        self.num_final_ltms: int = num_final_ltms
        self.enable_stm: bool = enable_stm
        self.enable_ltm: bool = enable_ltm
        self.device = torch.device(device) if device else None
        self.ext_device = None

    @torch.no_grad()
    def add_working_memory(self, data: torch.Tensor) -> None:
        """Add new working memories to engrams

        Args:
            data: input data which is working memory shaped [BatchSize, WorkingMemoryLength, HiddenDim]
        """
        self.ext_device = data.device
        self.device = self.device or data.device
        self.engrams += Engrams(data.to(self.device), engrams_types=EngramType.WORKING, lifespan=self.initial_lifespan)

    @torch.no_grad()
    def remind(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remind with current working memory and memorize, drop memories

        Return:
            reminded indices, data not containing working memory
            data shaped [BatchSize, RemindedMemoryLength, HiddenDim]
            indices shaped [BatchSize, RemindedMemoryLength]
        """
        wm_engrams, wm_indices = self.engrams.get_working_memory()
        stm_engrams, stm_indices = self.engrams.get_shortterm_memory()
        ltm_engrams, _ = self.engrams.get_longterm_memory()

        weight = self._calculate_memory_weight(wm_engrams, stm_engrams)
        reminded_stm_indices = self._remind_shortterm_memory(weight, stm_indices)
        initial_ltm_indices = self._find_initial_longterm_memory(reminded_stm_indices)
        searched_ltm_indices = self._search_longterm_memories_with_initials(initial_ltm_indices, ltm_engrams)
        reminded_ltm_indices = self._select_final_ltms(wm_engrams, searched_ltm_indices)

        reminded_indices = torch.cat([reminded_stm_indices, reminded_ltm_indices], dim=1)

        fire_indices = torch.cat([wm_indices, reminded_indices], dim=1)
        self.engrams.fire_together_wire_together(fire_indices)

        if not self.enable_stm and not self.enable_ltm:
            reminded_indices = torch.zeros(
                [self.engrams.batch_size, 0], requires_grad=False, device=self.engrams.data.device, dtype=torch.long
            )
        elif not self.enable_ltm:
            reminded_indices = reminded_stm_indices
        elif not self.enable_stm:
            reminded_indices = reminded_ltm_indices

        reminded_indices = reminded_indices.unique(dim=1)

        # [BatchSize, RemindedMemoryLength, HiddenDim]
        reminded_memories = self.engrams.select(reminded_indices).data.to(self.ext_device).contiguous()

        return reminded_memories, reminded_indices

    @torch.no_grad()
    def adjust_lifespan_and_memories(self, indices: torch.Tensor, lifespan_delta: torch.Tensor):
        """Adjust lifespan and memories"""
        lifespan_delta = lifespan_delta.to(self.device)
        self.engrams.extend_lifespan(indices, lifespan_delta)
        self.engrams.decrease_lifespan()

        self._memorize_working_memory_as_shortterm_memory()
        self._memorize_shortterm_memory_as_longterm_memory_or_drop()

        self.engrams = self.engrams.mask_select(self.engrams.lifespan > 0)

    @torch.no_grad()
    def _calculate_memory_weight(self, working_memory: Engrams, key_memory: Engrams) -> torch.Tensor:
        """Calculate attention weight from working memories over shotterm/longterm memories

        Returns:
            attention weights shaped [BatchSize, WorkingMemoryLength, ShorttermMemoryLength]
        """
        l2_square = (working_memory.data.unsqueeze(2) - key_memory.data.unsqueeze(1)).pow(2).sum(dim=3)
        return l2_square.neg().exp()

    @torch.no_grad()
    def _remind_shortterm_memory(self, weight: torch.Tensor, shortterm_memory_indices: torch.Tensor) -> torch.Tensor:
        """Remind shortterm memories by its working memories

        Args:
            weight: attention weights got from `calculate_wm_stm_weight` method
                shaped [BatchSize, WorkingMemoryLength, ShorttermMemoryLength]
            shortterm_memory_indices: global indices of shortterm memories shaped [BatchSize, ShorttermMemoryLength]
        Returns:
            indices selected shortterm memory indices shaped [BatchSize, SelectedMemoryLength]
            -1 means not reminded, non-negative intergers is reminded
        """
        # [BatchSize, ShorttermMemoryLength]
        stm_weight = weight.mean(dim=1)

        _, reminded_indices = stm_weight.topk(min(self.num_reminded_stm, stm_weight.size(1)), dim=1)
        reminded_mask = torch.zeros_like(
            shortterm_memory_indices, dtype=bool, device=weight.device, requires_grad=False
        )
        index_0 = torch.arange(weight.size(0), device=weight.device, requires_grad=False, dtype=torch.long)
        reminded_mask[index_0.unsqueeze(1), reminded_indices] = True
        return shortterm_memory_indices.masked_fill(~reminded_mask, -1)

    @torch.no_grad()
    def _find_initial_longterm_memory(self, nearest_shortterm_memory_indices: torch.Tensor) -> torch.Tensor:
        """Search nearest longterm memory indices

        Args:
            nearest_shortterm_memory_indices: shortterm memory indices shaped [BatchSize, NumInitialLTMs]
                got from `find_stm_nearest_to_ltm` method
        Return:
            initial longterm memory indices to be reminded shaped [BatchSize, NumInitialLTMs]
        """
        index_0 = torch.arange(
            nearest_shortterm_memory_indices.size(0),
            requires_grad=False,
            device=nearest_shortterm_memory_indices.device,
        ).unsqueeze(1)
        # [BatchSize, NumInitialLTMs, MemoryLength]
        induce_counts = self.engrams.induce_counts[index_0, nearest_shortterm_memory_indices]
        # [BatchSize, MemoryLength]
        ltm_mask = self.engrams.longterm_memory_mask

        induce_counts.masked_fill_(~ltm_mask.unsqueeze(1), -1)

        # [BatchSize, NumInitialLTMs]
        # Find initial ltm indices with random tie-breaking
        max_mask = induce_counts == induce_counts.max(dim=2).values.unsqueeze(dim=2)
        initial_ltm_indices = (torch.rand_like(max_mask.type(torch.float16)) * max_mask).argmax(dim=2)
        initial_ltm = self.engrams.select(initial_ltm_indices)
        initial_ltm_indices.masked_fill_(initial_ltm.engrams_types != EngramType.LONGTERM.value, -1)

        # [BatchSize, NumUniqueInitialLTMs]
        initial_ltm_indices = torch.unique(initial_ltm_indices, dim=1)
        return initial_ltm_indices

    @torch.no_grad()
    def _search_longterm_memories_with_initials(
        self, initial_longterm_memory_indices: torch.Tensor, longterm_memory: Engrams
    ) -> torch.Tensor:
        """Find ltm engrams with initila ltm indices by dfs method

        Args:
            initial_longterm_memory_indices: initial ltm indices shaped [BatchSize, NumUniqueInitialLTMs]
            longterm_memory: longterm memory engrams
        Return:
            searched ltm indices shaped [BatchSize, (SearchDepth + 1) * NumUniqueInitialLTMs]
        """
        batch_size, num_init_ltms = initial_longterm_memory_indices.shape
        local_initial_ltm_indices = self.engrams.get_local_indices_from_global_indices(
            self.engrams.longterm_memory_mask, initial_longterm_memory_indices
        )
        if initial_longterm_memory_indices.numel() == 0 or len(longterm_memory) == 0:
            return torch.zeros(
                [batch_size, (self.ltm_search_depth + 1) * num_init_ltms],
                dtype=torch.long,
                device=initial_longterm_memory_indices.device,
                requires_grad=False,
            )

        # [BatchSize, NumLTMems]
        unreachable = ~longterm_memory.longterm_memory_mask
        found_ltm_indices = torch.full(
            [batch_size, self.ltm_search_depth + 1, min(num_init_ltms, local_initial_ltm_indices.size(1))],
            -1,
            requires_grad=False,
            device=local_initial_ltm_indices.device,
            dtype=local_initial_ltm_indices.dtype,
        )

        index_0 = torch.arange(
            batch_size, device=local_initial_ltm_indices.device, requires_grad=False, dtype=torch.long
        ).unsqueeze(1)
        found_ltm_indices[:, 0] = local_initial_ltm_indices
        unreachable[index_0, local_initial_ltm_indices] = True

        for depth in range(self.ltm_search_depth):
            last_ltm_indices = found_ltm_indices[:, depth]
            reachable_induce_counts = longterm_memory.induce_counts.masked_fill(unreachable.unsqueeze(1), -1)
            # [BatchSize, NumUniqueInitialLTMs, NumLTMems]
            last_ltm_reachable_induce_counts = reachable_induce_counts[index_0, last_ltm_indices]

            current_ltm_indices = last_ltm_reachable_induce_counts.argmax(dim=2)
            current_ltm_indices.masked_fill_(unreachable[index_0, current_ltm_indices], -1)
            found_ltm_indices[:, depth + 1] = current_ltm_indices
            unreachable[index_0, current_ltm_indices] = True

        return found_ltm_indices.view(batch_size, -1).unique(dim=1)

    @torch.no_grad()
    def _select_final_ltms(self, working_memory: Engrams, found_longterm_memory_indices: torch.Tensor) -> torch.Tensor:
        """Select top k LTM indices by working memory weights

        Args:
            working_memory: working memory engrams
            found_longterm_memory_indices: found ltm indices shaped [BatchSize, NumFoundLTMs]
        Return:
            final ltm indices shaped [BatchSize, NumFinalLTMs]
        """
        if found_longterm_memory_indices.size(1) == 0:
            return found_longterm_memory_indices

        # [BatchSize, NumFoundLTMs]
        found_ltms = self.engrams.select(found_longterm_memory_indices)
        # [BatchSize, NumFoundLTMs]
        weight = self._calculate_memory_weight(working_memory, found_ltms).sum(dim=1)

        topk = min(self.num_final_ltms, weight.size(1))
        # [BatchSize, NumFinalLTMs]
        final_ltm_indices = weight.topk(topk, dim=1).indices
        final_ltm_indices = found_longterm_memory_indices.gather(dim=1, index=final_ltm_indices)
        return final_ltm_indices

    @torch.no_grad()
    def _memorize_working_memory_as_shortterm_memory(self):
        """Move working memory to shortterm memory"""
        self.engrams.engrams_types[self.engrams.working_memory_mask] = EngramType.SHORTTERM.value

    @torch.no_grad()
    def _memorize_shortterm_memory_as_longterm_memory_or_drop(self):
        """Move exceeded shortterm memory to longterm memory or drop"""
        stm_engrams, stm_indices = self.engrams.get_shortterm_memory()
        num_exceeded_stm = stm_engrams.memory_length - self.stm_capacity

        if num_exceeded_stm <= 0:
            return

        # [BatchSize, NumExceededSTMems]
        exceeded_stm_indices = stm_indices[:, :num_exceeded_stm]
        index_0 = torch.arange(
            self.engrams.batch_size, device=stm_indices.device, requires_grad=False, dtype=torch.long
        ).unsqueeze(1)
        # [BatchSize, NumExceededSTMems]
        exceeded_stm_fire_counts = self.engrams.fire_count[index_0, exceeded_stm_indices]

        memorize_mask = exceeded_stm_fire_counts < self.ltm_min_fire_count
        memorize_stm_indices = exceeded_stm_indices.masked_fill(memorize_mask, -1)
        memorize_stm_mask = self.engrams.get_mask_with_indices(memorize_stm_indices)
        self.engrams.engrams_types[memorize_stm_mask] = EngramType.LONGTERM.value

        forget_indices = exceeded_stm_indices.masked_fill(~memorize_mask, -1)
        self.engrams.delete(forget_indices)

    def reset_memory(self):
        """Reset memory"""
        self.engrams = Engrams.empty()
