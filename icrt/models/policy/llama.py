# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from timm.layers import use_fused_attn

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    w_bias: bool = True # use bias tuning
    w_lora: bool = True # use lora tuning
    lora_rank: int = 16


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"Invalid shape for freqs_cis: {freqs_cis.shape}, x: {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.fused_attn = use_fused_attn()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.w_bias
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.wq.bias.data, 0)
            nn.init.constant_(self.wo.bias.data, 0)

        self.w_lora = args.w_lora
        if args.w_lora:
           self.lora_wq_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wq_l2 = Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wk_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wk_l2 = Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wv_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wv_l2 = Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wo_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wo_l2 = Linear(args.lora_rank, args.dim, bias=False)
           nn.init.constant_(self.lora_wq_l2.weight.data, 0)
           nn.init.constant_(self.lora_wk_l2.weight.data, 0)
           nn.init.constant_(self.lora_wv_l2.weight.data, 0)
           nn.init.constant_(self.lora_wo_l2.weight.data, 0)

        self.cache_k = None
        self.cache_v = None

    def train(self, mode: bool = True):
        if mode:
            self.cache_k = None
            self.cache_v = None
        else:
            self.cache_k = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
        return super().train(mode)

    def forward_attention(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.w_lora:
           xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))
           xk = xk + self.lora_wk_l2(self.lora_wk_l1(x))
           xv = xv + self.lora_wv_l2(self.lora_wv_l1(x))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not self.training:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            if start_pos + seqlen > self.cache_k.shape[1]:
                # we linearly increase the size of the kv cache by self.args.max_seq_len
                # we don't double it because it might OOM
                print(f"Updating kv cache from length {self.cache_k.shape[1]} to length {self.cache_k.shape[1] * 2}")

                self.cache_k = self.cache_k.repeat(1, 2, 1, 1)
                self.cache_v = self.cache_v.repeat(1, 2, 1, 1)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.clone().detach()
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.clone().detach()

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            assert start_pos==0
            keys = xk
            values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)

        # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        return scores
        

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.w_lora:
           xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))
           xk = xk + self.lora_wk_l2(self.lora_wk_l1(x))
           xv = xv + self.lora_wv_l2(self.lora_wv_l1(x))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not self.training:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            if start_pos + seqlen > self.cache_k.shape[1]:
                # we linearly increase the size of the kv cache by self.args.max_seq_len
                # we don't double it because it might OOM
                print(f"Updating kv cache from length {self.cache_k.shape[1]} to length {self.cache_k.shape[1] * 2}")

                self.cache_k = self.cache_k.repeat(1, 2, 1, 1)
                self.cache_v = self.cache_v.repeat(1, 2, 1, 1)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.clone().detach()
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.clone().detach()

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            assert start_pos==0
            keys = xk
            values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        if self.fused_attn:
            is_causal = self.training
            attn_mask = None if is_causal else mask
            output = F.scaled_dot_product_attention(
                xq, keys, values,
                dropout_p=0., attn_mask=attn_mask, is_causal=is_causal,
            )
        else:
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        if self.w_lora:
           return self.wo(output) + self.lora_wo_l2(self.lora_wo_l1(output))
        else:
           return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        args: ModelArgs
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=args.w_bias
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.w1.bias.data, 0)
            nn.init.constant_(self.w2.bias.data, 0)
            nn.init.constant_(self.w3.bias.data, 0)

        self.w_lora = args.w_lora
        if args.w_lora:
           self.lora_w1_l1 = Linear(dim, args.lora_rank, bias=False)
           self.lora_w1_l2 = Linear(args.lora_rank, hidden_dim, bias=False)
           self.lora_w2_l1 = Linear(hidden_dim, args.lora_rank, bias=False)
           self.lora_w2_l2 = Linear(args.lora_rank, dim, bias=False)
           self.lora_w3_l1 = Linear(dim, args.lora_rank, bias=False)
           self.lora_w3_l2 = Linear(args.lora_rank, hidden_dim, bias=False)
           nn.init.constant_(self.lora_w1_l2.weight.data, 0)
           nn.init.constant_(self.lora_w2_l2.weight.data, 0)
           nn.init.constant_(self.lora_w3_l2.weight.data, 0)

    def forward(self, x):
        if self.w_lora:
           out = F.silu(self.w1(x) + self.lora_w1_l2(self.lora_w1_l1(x))) * (self.w3(x) + self.lora_w3_l2(self.lora_w3_l1(x)))
           return self.w2(out) + self.lora_w2_l2(self.lora_w2_l1(out))
        else:
           return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, args=args
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):

        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # self.output = Linear(
        #     params.dim, params.vocab_size, bias=False
        # )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward_inference(self, seq: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        _, seqlen, _ = seq.shape
        if start_pos + seqlen > self.freqs_cis.shape[0]: 
            # update positional embedding 
            print(f"Updating positional embedding from length {self.freqs_cis.shape[0]} to length {self.freqs_cis.shape[0] * 2}")
            self.freqs_cis = precompute_freqs_cis(
                self.params.dim // self.params.n_heads, self.freqs_cis.shape[0] * 2
            )
        self.freqs_cis = self.freqs_cis.to(seq.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        if seqlen > 1:
            if mask is None:
                mask = torch.full(
                    (seqlen, seqlen), float("-inf"), device=seq.device
                )

                mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=seq.device),
                mask
            ]).type_as(seq)
        else:
            mask = None

        for layer in self.layers:
            seq = layer(seq, start_pos, freqs_cis, mask)
        seq = self.norm(seq)
        # output = self.output(seq[:, -1, :])  # only compute last logits
        # return output.float()
        return seq # forward the entire sequence, then post process in icrt or dicrt

    def forward(self, seq : torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        seq: B, 2T-1, C
        """
        _, seqlen, _ = seq.shape
        self.freqs_cis = self.freqs_cis.to(seq.device) 
        freqs_cis = self.freqs_cis[:seqlen]

        if mask is None:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=seq.device)
            mask = torch.triu(mask, diagonal=1).type_as(seq)

        for layer in self.layers:
            seq = layer(seq, 0, freqs_cis, mask)

        seq = self.norm(seq)
        return seq
