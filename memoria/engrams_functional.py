import torch

from .engram_type import EngramType


@torch.jit.script
def engram_equals(
    self_data: torch.Tensor,
    other_data: torch.Tensor,
    self_induce_counts: torch.Tensor,
    other_induce_counts: torch.Tensor,
    self_engrams_types: torch.Tensor,
    other_engrams_types: torch.Tensor,
    self_lifespan: torch.Tensor,
    other_lifespan: torch.Tensor,
):
    return (
        (self_data == other_data).all()
        and (self_induce_counts == other_induce_counts).all()
        and (self_engrams_types == other_engrams_types).all()
        and (self_lifespan == other_lifespan).all()
    )


@torch.jit.script
def engram_add(
    self_data: torch.Tensor,
    other_data: torch.Tensor,
    self_induce_counts: torch.Tensor,
    other_induce_counts: torch.Tensor,
    self_engrams_types: torch.Tensor,
    other_engrams_types: torch.Tensor,
    self_lifespan: torch.Tensor,
    other_lifespan: torch.Tensor,
):
    self_batch_size, self_memory_length, _ = self_data.shape
    other_memory_length = other_data.shape[1]
    concatenated_data = torch.cat([self_data, other_data], dim=1)
    concatenated_engrams_types = torch.cat([self_engrams_types, other_engrams_types], dim=1)
    concatenated_lifespan = torch.cat([self_lifespan, other_lifespan], dim=1)

    new_memory_length = self_memory_length + other_memory_length
    concatenated_induce_counts = torch.zeros(
        [self_batch_size, new_memory_length, new_memory_length], dtype=torch.int32, device=self_data.device
    )
    concatenated_induce_counts[:, :self_memory_length, :self_memory_length] = self_induce_counts
    concatenated_induce_counts[:, self_memory_length:, self_memory_length:] = other_induce_counts
    return (concatenated_data, concatenated_induce_counts, concatenated_engrams_types, concatenated_lifespan)


@torch.jit.script
def engrams_get_indices_with_mask(batch_size: int, memory_length: int, mask: torch.Tensor):
    # [BatchSize, MemoryLength]
    num_memories_per_batch = mask.sum(dim=1)
    max_num_memories = num_memories_per_batch.max()
    if (num_memories_per_batch == max_num_memories).all():
        indices = mask.nonzero()[:, 1]
        indices = indices.view(batch_size, max_num_memories)
        return indices

    valued_mask = torch.arange(memory_length, device=mask.device, dtype=torch.long).unsqueeze(0) * mask
    valued_mask.masked_fill_(~mask, -1)
    sorted_values, _ = torch.sort(valued_mask, dim=1)
    indices = sorted_values[:, -max_num_memories:]
    return indices


@torch.jit.script
def engrams_get_mask_with_indices(batch_size: int, memory_length: int, indices: torch.Tensor):
    indices[indices < 0] = -1
    mask = torch.zeros([batch_size, memory_length + 1], dtype=torch.bool, device=indices.device)
    index_0 = torch.arange(batch_size, device=indices.device, dtype=torch.long).unsqueeze(1)
    mask[index_0, indices] = True
    mask = mask[:, :-1]
    return mask


@torch.jit.script
def engrams_select(
    self_data: torch.Tensor,
    self_induce_counts: torch.Tensor,
    self_engrams_types: torch.Tensor,
    self_lifespan: torch.Tensor,
    indices: torch.Tensor,
):
    if len(indices.shape) == 1:
        selected_data = self_data[:, indices]
        selected_induce_counts = self_induce_counts[:, indices][:, :, indices]
        selected_engrams_types = self_engrams_types[:, indices]
        selected_lifespan = self_lifespan[:, indices]
    elif len(indices.shape) == 2:
        index_0 = torch.arange(self_data.size(0), device=self_induce_counts.device)
        selected_data = self_data[index_0.unsqueeze(1), indices]
        selected_induce_counts = self_induce_counts[
            index_0.unsqueeze(1).unsqueeze(2), indices.unsqueeze(2), indices.unsqueeze(1)
        ]
        selected_engrams_types = self_engrams_types[index_0.unsqueeze(1), indices]
        selected_lifespan = self_lifespan[index_0.unsqueeze(1), indices]

        null_indices_mask = indices < 0
        reverse_mask = (~null_indices_mask).float()
        selected_data.masked_fill_(null_indices_mask.unsqueeze(2), 0.0)
        selected_induce_counts.masked_fill_(~((reverse_mask.unsqueeze(2) @ reverse_mask.unsqueeze(1)) > 0), -1)
        selected_engrams_types.masked_fill_(null_indices_mask, EngramType.NULL.value)
        selected_lifespan.masked_fill_(null_indices_mask, -1.0)
    else:
        raise ValueError(f"Invalid indices shape: {indices.shape}")
    return selected_data, selected_induce_counts, selected_engrams_types, selected_lifespan


@torch.jit.script
def engrams_get_local_indices_from_global_indices(
    self_data: torch.Tensor,
    self_induce_counts: torch.Tensor,
    self_engrams_types: torch.Tensor,
    self_lifespan: torch.Tensor,
    partial_mask: torch.Tensor,
    global_indices: torch.Tensor,
):
    batch_size, memory_length = self_engrams_types.shape
    partial_indices = engrams_get_indices_with_mask(batch_size, memory_length, partial_mask)
    partial_data, *_ = engrams_select(self_data, self_induce_counts, self_engrams_types, self_lifespan, partial_indices)
    partial_batch_size, partial_memory_length, _ = partial_data.shape

    selected_partial_mask = torch.zeros_like(partial_mask, device=partial_mask.device) > 0
    index_0 = torch.arange(batch_size, device=partial_mask.device, dtype=torch.long).unsqueeze(1)
    selected_partial_mask[index_0, global_indices] = True

    # [BatchSize, NumLTMems]
    local_selected_partial_mask = selected_partial_mask[index_0, partial_indices]
    # [BatchSize, NumUniqueInitialLTMs]
    local_selected_ltm_indices = engrams_get_indices_with_mask(
        partial_batch_size, partial_memory_length, local_selected_partial_mask
    )

    return local_selected_ltm_indices


@torch.jit.script
def engrams_extend_lifespan(batch_size: int, memory_length: int, indices: torch.Tensor, lifespan_delta: torch.Tensor):
    indices[indices < 0] = -1
    lifespan_delta_mask = torch.zeros([batch_size, memory_length + 1], dtype=torch.float32, device=indices.device)
    index_0 = torch.arange(batch_size, device=indices.device)
    lifespan_delta_mask[index_0.unsqueeze(1), indices] += lifespan_delta
    return lifespan_delta_mask[:, :-1]


@torch.jit.script
def engrams_working_memory_mask(engrams_types: torch.Tensor):
    return engrams_types == EngramType.WORKING.value


@torch.jit.script
def engrams_shortterm_memory_mask(engrams_types: torch.Tensor):
    return engrams_types == EngramType.SHORTTERM.value


@torch.jit.script
def engrams_longterm_memory_mask(engrams_types: torch.Tensor):
    return engrams_types == EngramType.LONGTERM.value


@torch.jit.script
def engrams_fire_count(induce_counts: torch.Tensor):
    return induce_counts.diagonal(dim1=1, dim2=2)


@torch.jit.script
def engrams_fire_count_setter(induce_counts: torch.Tensor, value: torch.Tensor):
    index = torch.arange(value.size(1), device=value.device, dtype=torch.long)
    induce_counts[:, index, index] = value
    return induce_counts
