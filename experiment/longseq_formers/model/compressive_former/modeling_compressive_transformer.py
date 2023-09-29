"""From https://github.com/lucidrains/compressive-transformer-pytorch"""
import math
from dataclasses import dataclass
from functools import partial
from inspect import isfunction
from typing import List, Optional

import torch
import torch.nn.functional as F
from mogrifier import Mogrifier
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


class CompressiveFormerConfig(PretrainedConfig):
    model_type = "compressive_transformer"

    def __init__(
        self,
        vocab_size=20000,
        dim=512,
        seq_len=1024,
        depth=6,
        emb_dim=None,
        memory_layers=[5, 6],
        enhanced_recurrence=True,
        mem_len=1024,
        cmem_len=256,
        cmem_ratio=4,
        heads=8,
        gru_gated_residual=True,
        mogrify_gru=False,
        attn_dropout=0.0,
        ff_glu=False,
        ff_dropout=0.0,
        attn_layer_dropout=0.0,
        reconstruction_attn_dropout=0.0,
        reconstruction_loss_weight=1.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.depth = depth
        self.emb_dim = emb_dim
        self.memory_layers = memory_layers
        self.enhanced_recurrence = enhanced_recurrence
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        self.heads = heads
        self.gru_gated_residual = gru_gated_residual
        self.mogrify_gru = mogrify_gru
        self.attn_dropout = attn_dropout
        self.ff_glu = ff_glu
        self.ff_dropout = ff_dropout
        self.attn_layer_dropout = attn_layer_dropout
        self.reconstruction_attn_dropout = reconstruction_attn_dropout
        self.reconstruction_loss_weight = reconstruction_loss_weight

        super().__init__(**kwargs)


def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)


def default(x, val):
    if x is not None:
        return x
    return val if not isfunction(val) else val()


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim : dim + 1] = split_dims
    return t.reshape(shape)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def queue_fifo(*args, length, dim=-2):
    queue = torch.cat(args, dim=dim)
    if length > 0:
        return split_at_index(dim, -length, queue)

    device = queue.device
    shape = list(queue.shape)
    shape[dim] = 0
    return queue, torch.empty(shape, device=device)


def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), dtype=x.dtype, device=x.device)
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, dtype=x.dtype, device=x.device)
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1 :]


def iterate_tensor(t):
    length = t.shape[0]
    for ind in range(length):
        yield t[ind]


def full_attn(q, k, v, dropout_fn=None):
    """full attention for calculating auxiliary reconstruction loss"""
    *_, dim = q.shape
    dots = torch.einsum("bhid,bhjd->bhij", q, k) * (dim**-0.5)
    attn = dots.softmax(dim=-1)
    if dropout_fn is not None:
        attn = dropout_fn(attn)
    return torch.einsum("bhij,bhjd->bhid", attn, v)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        out = cast_tuple(out)
        ret = (out[0] + x), *out[1:]
        return ret


class GRUGating(nn.Module):
    def __init__(self, dim, fn, mogrify=False):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nn.GRUCell(dim, dim)
        self.mogrify = Mogrifier(dim, factorize_k=dim // 4) if mogrify else None

    def forward(self, x, **kwargs):
        batch, dim = x.shape[0], self.dim
        out = self.fn(x, **kwargs)
        (y, *rest) = cast_tuple(out)

        if self.mogrify is not None:
            y, x = self.mogrify(y, x)

        gated_output = self.gru(y.reshape(-1, dim), x.reshape(-1, dim))

        gated_output = gated_output.reshape(batch, -1, dim)
        ret = gated_output, *rest
        return ret


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class ConvCompress(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, ratio, stride=ratio)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)


class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, "GELU") else GELU_


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        mem_len,
        cmem_len,
        cmem_ratio=4,
        heads=8,
        attn_dropout=0.0,
        dropout=0.0,
        reconstruction_attn_dropout=0.0,
    ):
        super().__init__()
        assert (dim % heads) == 0, "dimension must be divisible by the number of heads"

        self.heads = heads
        self.dim_head = dim // heads
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        self.scale = self.dim_head ** (-0.5)

        self.compress_mem_fn = ConvCompress(dim, cmem_ratio)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

        self.reconstruction_attn_dropout = nn.Dropout(reconstruction_attn_dropout)

    def forward(self, x, mem=None, cmem=None, pos_emb=None, input_mask=None, calc_memory=True, **kwargs):
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head

        init_empty_mem = lambda: torch.empty(b, 0, e, dtype=x.dtype, device=x.device)
        mem = default(mem, init_empty_mem)
        cmem = default(cmem, init_empty_mem)

        mem_len = mem.shape[1]
        cmem_len = cmem.shape[1]

        q = self.to_q(x)

        kv_input = torch.cat((cmem, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        merge_heads = lambda x: reshape_dim(x, -1, (-1, dim_h)).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))

        k, v = map(lambda x: x.expand(-1, h, -1, -1), (k, v))

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = max_neg_value(dots)

        if pos_emb is not None:
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum("bhid,hjd->bhij", q, pos_emb) * self.scale
            pos_dots = shift(pos_dots)
            dots = dots + pos_dots

        if input_mask is not None:
            input_mask = input_mask.bool()
            mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            mask = F.pad(mask, (mem_len + cmem_len, 0), value=True)
            dots.masked_fill_(~mask, mask_value)

        total_mem_len = mem_len + cmem_len
        mask = torch.ones(t, t + total_mem_len, dtype=x.dtype, device=x.device).triu_(diagonal=1 + total_mem_len).bool()
        dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        logits = self.to_out(out)
        logits = self.dropout(logits)

        new_mem = mem
        new_cmem = cmem
        aux_loss = torch.zeros(1, requires_grad=True, dtype=q.dtype, device=q.device)

        if not calc_memory:
            return logits, new_mem, new_cmem, aux_loss

        # calculate memory and compressed memory

        old_mem, new_mem = queue_fifo(mem, x, length=self.mem_len, dim=1)
        old_mem_padding = old_mem.shape[1] % self.cmem_ratio

        if old_mem_padding != 0:
            old_mem = F.pad(old_mem, (0, 0, old_mem_padding, 0), value=0.0)

        if old_mem.shape[1] == 0 or self.cmem_len <= 0:
            return logits, new_mem, new_cmem, aux_loss

        compressed_mem = self.compress_mem_fn(old_mem.detach())
        old_cmem, new_cmem = split_at_index(1, -self.cmem_len, torch.cat((cmem, compressed_mem), dim=1))

        if not self.training:
            return logits, new_mem, new_cmem, aux_loss

        # calculate compressed memory auxiliary loss if training

        self.to_kv.weight.detach_()

        cmem_k, cmem_v = self.to_kv(compressed_mem).chunk(2, dim=-1)
        cmem_k, cmem_v = map(merge_heads, (cmem_k, cmem_v))
        cmem_k, cmem_v = map(lambda x: x.expand(-1, h, -1, -1), (cmem_k, cmem_v))

        old_mem_range = slice(-min(mem_len, self.mem_len) - self.seq_len, -self.seq_len)
        old_mem_k, old_mem_v = map(lambda x: x[:, :, old_mem_range].clone(), (k, v))

        q, old_mem_k, old_mem_v = map(torch.detach, (q, old_mem_k, old_mem_v))

        attn_fn = partial(full_attn, dropout_fn=self.reconstruction_attn_dropout)

        aux_loss = F.mse_loss(attn_fn(q, old_mem_k, old_mem_v), attn_fn(q, cmem_k, cmem_v))

        return logits, new_mem, new_cmem, aux_loss


class CompressiveFormerPreTrainedModel(PreTrainedModel):
    config_class = CompressiveFormerConfig
    base_model_prefix = "compressive_transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


@dataclass
class CompressiveFormerModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    mems: List[torch.FloatTensor] = None
    cmems: List[torch.FloatTensor] = None
    aux_loss: torch.Tensor = None


class CompressiveFormerModel(CompressiveFormerPreTrainedModel):
    def __init__(self, config: CompressiveFormerConfig):
        super().__init__(config)

        config.emb_dim = default(config.emb_dim, config.dim)
        config.mem_len = default(config.mem_len, config.seq_len)
        config.cmem_len = default(config.cmem_len, config.mem_len // config.cmem_ratio)
        config.memory_layers = default(config.memory_layers, list(range(1, config.depth + 1)))

        assert config.cmem_len >= (
            config.mem_len // config.cmem_ratio
        ), f"length of compressed memory should be at least the memory length divided by the compression ratio {int(config.mem_len // config.cmem_ratio)}"
        assert all(
            [layer > 0 and layer <= config.depth for layer in config.memory_layers]
        ), "one of the indicated memory layers is invalid"

        self.seq_len = config.seq_len

        self.depth = config.depth
        self.memory_layers = list(config.memory_layers)
        self.enhanced_recurrence = config.enhanced_recurrence

        self.token_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.to_model_dim = nn.Identity() if config.emb_dim == config.dim else nn.Linear(config.emb_dim, config.dim)

        seq_and_mem_len = config.seq_len + config.mem_len + config.cmem_len
        self.pos_emb = nn.Parameter(torch.zeros(config.heads, seq_and_mem_len, config.dim // config.heads))

        wrapper = partial(GRUGating, config.dim, mogrify=config.mogrify_gru) if config.gru_gated_residual else Residual

        self.attn_layers = nn.ModuleList(
            [
                wrapper(
                    PreNorm(
                        config.dim,
                        SelfAttention(
                            config.dim,
                            config.seq_len,
                            config.mem_len,
                            config.cmem_len,
                            config.cmem_ratio,
                            config.heads,
                            dropout=config.attn_layer_dropout,
                            attn_dropout=config.attn_dropout,
                            reconstruction_attn_dropout=config.reconstruction_attn_dropout,
                        ),
                    )
                )
                for _ in range(config.depth)
            ]
        )
        self.ff_layers = nn.ModuleList(
            [
                wrapper(PreNorm(config.dim, FeedForward(config.dim, dropout=config.ff_dropout, glu=config.ff_glu)))
                for _ in range(config.depth)
            ]
        )

        self.reconstruction_loss_weight = config.reconstruction_loss_weight

    def forward(self, input_ids, attention_mask=None, mems=None, cmems=None) -> CompressiveFormerModelOutput:
        x = self.token_emb(input_ids)
        x = self.to_model_dim(x)
        b, t, d = x.shape

        assert (
            t <= self.seq_len
        ), f"input contains a sequence length {t} that is greater than the designated maximum sequence length {self.seq_len}"

        num_memory_layers = len(self.memory_layers)
        init_empty_mem = lambda: torch.empty(num_memory_layers, b, 0, d, dtype=x.dtype, device=x.device)
        mems = default(mems, init_empty_mem)
        cmems = default(cmems, init_empty_mem)

        total_len = mems.shape[2] + cmems.shape[2] + self.seq_len
        pos_emb = self.pos_emb[:, (self.seq_len - t) : total_len]

        next_mems = []
        next_cmems = []
        aux_loss = torch.tensor(0.0, requires_grad=True, dtype=x.dtype, device=x.device)

        if self.enhanced_recurrence:
            mems = torch.roll(mems, -1, 0)
            cmems = torch.roll(cmems, -1, 0)

        for ind, (attn, ff) in enumerate(zip(self.attn_layers, self.ff_layers)):
            layer_num = ind + 1

            use_memory = layer_num in self.memory_layers
            mem = mems[self.memory_layers.index(layer_num)] if use_memory else None
            cmem = cmems[self.memory_layers.index(layer_num)] if use_memory else None

            x, mem_out, cmem_out, layer_aux_loss = attn(
                x, mem=mem, cmem=cmem, calc_memory=use_memory, input_mask=attention_mask, pos_emb=pos_emb
            )
            (x,) = ff(x)

            aux_loss = aux_loss + layer_aux_loss

            if not use_memory:
                continue

            next_mems.append(mem_out)
            next_cmems.append(cmem_out)

        next_mems, next_cmems = map(torch.stack, (next_mems, next_cmems))
        next_mems, next_cmems = map(torch.detach, (next_mems, next_cmems))

        aux_loss = aux_loss * self.reconstruction_loss_weight / num_memory_layers
        return CompressiveFormerModelOutput(last_hidden_state=x, mems=next_mems, cmems=next_cmems, aux_loss=aux_loss)


@dataclass
class CompressiveFormerLMHeadModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    lm_loss: Optional[torch.FloatTensor] = None
    aux_loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    mems: List[torch.FloatTensor] = None
    cmems: List[torch.FloatTensor] = None


class CompressiveFormerLMHeadModel(CompressiveFormerPreTrainedModel):
    def __init__(self, config: CompressiveFormerConfig):
        super().__init__(config)
        self.compressive_transformer = CompressiveFormerModel(config)
        self.to_logits = nn.Sequential(
            nn.Identity() if config.emb_dim == config.dim else nn.Linear(config.dim, config.emb_dim),
            nn.Linear(config.emb_dim, config.vocab_size),
        )

    def forward(self, input_ids, attention_mask=None, mems=None, cmems=None, labels=None):
        output = self.compressive_transformer(
            input_ids=input_ids, attention_mask=attention_mask, mems=mems, cmems=cmems
        )
        logits = self.to_logits(output.last_hidden_state)

        loss = output.aux_loss
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += lm_loss
        else:
            lm_loss = None

        return CompressiveFormerLMHeadModelOutput(
            loss=loss, lm_loss=lm_loss, aux_loss=output.aux_loss, logits=logits, mems=output.mems, cmems=output.cmems
        )


CompressiveFormerConfig.register_for_auto_class()
CompressiveFormerModel.register_for_auto_class("AutoModel")
CompressiveFormerLMHeadModel.register_for_auto_class("AutoModelForCausalLM")
AutoConfig.register(CompressiveFormerConfig.model_type, CompressiveFormerConfig)
AutoModel.register(CompressiveFormerConfig, CompressiveFormerModel)
AutoModelForCausalLM.register(CompressiveFormerConfig, CompressiveFormerLMHeadModel)
