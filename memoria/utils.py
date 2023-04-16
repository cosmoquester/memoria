import torch


def super_unique(t: torch.Tensor, dim: int) -> torch.Tensor:
    min_value = t.min()
    t = t - min_value

    max_value = t.max()
    new_shape = list(t.shape)
    new_shape[dim] = max_value + 1
    unique_t_mask = torch.zeros(new_shape, dtype=torch.bool, device=t.device)
    unique_t_mask.scatter_(dim, t.long(), True)

    validity, unique_t = unique_t_mask.int().topk(t.size(dim), dim=dim)
    unique_t += min_value
    unique_t.masked_fill_(~validity.bool(), -1)
    return unique_t
