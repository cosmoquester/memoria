import torch

from .engram import Engrams, EngramType


class Hippocampus:
    def __init__(
        self,
        num_initial_ltm: int,
        threshold_stm: float,
        ltm_search_depth: int,
        stm_capacity: int,
        ltm_min_fire_count: int,
    ) -> None:
        self.engrams = Engrams.empty()

        self.num_initial_ltm: int = num_initial_ltm
        self.threshold_stm: float = threshold_stm
        self.ltm_search_depth: int = ltm_search_depth
        self.stm_capacity: int = stm_capacity
        self.ltm_min_fire_count: int = ltm_min_fire_count

    @torch.no_grad()
    def add_working_memory(self, data: torch.Tensor) -> None:
        """Add new working memories to engrams

        Args:
            data: working memory tensor shaped [BatchSize, MemoryLength, HiddenDim]
        """
        self.engrams += Engrams(data, engrams_types=EngramType.WORKING)

    @torch.no_grad()
    def calculate_wm_stm_weight(self, working_memory: Engrams, shortterm_memory: Engrams) -> torch.Tensor:
        """Calculate attention weight from working memories over shotterm memories

        Returns:
            attention weights shaped [BatchSize, WorkingMemoryLength, ShorttermMemoryLength]
        """
        weight = working_memory.data @ shortterm_memory.data.transpose(1, 2)
        return weight

    @torch.no_grad()
    def remind_shortterm_memory(self, weight: torch.Tensor, shortterm_memory_indices: torch.Tensor) -> torch.Tensor:
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

        mask = stm_weight < self.threshold_stm
        reminded_shortterm_memory_indices = shortterm_memory_indices.masked_fill(mask, -1)
        return reminded_shortterm_memory_indices

    @torch.no_grad()
    def find_stm_nearest_to_ltm(self, weight: torch.Tensor, shortterm_memory_indices: torch.Tensor) -> torch.Tensor:
        """Get shortterm memory indices nearest to initial ltm by its working memories

        Args:
            weight: attention weights got from `calculate_wm_stm_weight` method
                shaped [BatchSize, WorkingMemoryLength, ShorttermMemoryLength]
            shortterm_memory_indices: global indices of shortterm memories shaped [BatchSize, ShorttermMemoryLength]
        Returns:
            indices selected shortterm memory indices shaped [BatchSize, NumInitialLTMs]
                -1 means unselected. other values mean selected
        """
        # [BatchSize, WorkingMemoryLength, FiringShorttermMemories]
        _, top_indices = weight.topk(k=self.num_initial_ltm, dim=2)

        # [BatchSize, WorkingMemoryLength * FiringShorttermMemories]
        top_indices = top_indices.view(weight.size(0), -1)

        # Get STM Indices Nearest to Initial LTM
        batch_size = weight.size(0)
        index_0 = torch.arange(batch_size, requires_grad=False, device=weight.device).unsqueeze(1)
        nearest_stm_mask = torch.zeros_like(
            shortterm_memory_indices, requires_grad=False, device=weight.device, dtype=torch.bool
        )
        nearest_stm_mask[index_0, top_indices] = True
        nearest_stm_indices = shortterm_memory_indices.masked_fill(~nearest_stm_mask, -1)
        nearest_stm_indices = torch.unique(nearest_stm_indices, dim=1)  # Not necessary
        return nearest_stm_indices

    @torch.no_grad()
    def find_initial_longterm_memory(self, nearest_shortterm_memory_indices: torch.Tensor) -> torch.Tensor:
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
        initial_ltm_indices = induce_counts.argmax(dim=2)
        initial_ltm = self.engrams.select(initial_ltm_indices)
        initial_ltm_indices.masked_fill_(initial_ltm.engrams_types != EngramType.LONGTERM.value, -1)

        # [BatchSize, NumUniqueInitialLTMs]
        initial_ltm_indices = torch.unique(initial_ltm_indices, dim=1)
        return initial_ltm_indices

    @torch.no_grad()
    def search_longterm_memories_with_initials(
        self, initial_longterm_memory_indices: torch.Tensor, longterm_memory: Engrams
    ) -> torch.Tensor:
        """Find ltm engrams with initila ltm indices by dfs method

        Args:
            initial_longterm_memory_indices: initial ltm indices shaped [BatchSize, NumUniqueInitialLTMs]
            longterm_memory: longterm memory engrams
        Return:
            searched ltm indices shaped [BatchSize, SearchDepth + 1, NumUniqueInitialLTMs]
        """
        batch_size, num_init_ltms = initial_longterm_memory_indices.shape
        local_initial_ltm_indices = self.engrams.get_local_indices_from_global_indices(
            self.engrams.longterm_memory_mask, initial_longterm_memory_indices
        )
        # [BatchSize, NumLTMems]
        unreachable = ~longterm_memory.longterm_memory_mask
        found_ltm_indices = torch.full(
            [batch_size, self.ltm_search_depth + 1, num_init_ltms],
            -1,
            requires_grad=False,
            device=local_initial_ltm_indices.device,
            dtype=local_initial_ltm_indices.dtype,
        )

        index_0 = torch.arange(batch_size, device=local_initial_ltm_indices.device, requires_grad=False).unsqueeze(1)
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

        return found_ltm_indices
