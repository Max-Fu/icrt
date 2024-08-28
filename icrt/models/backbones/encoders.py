from typing import Optional

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import timm.models.vision_transformer
from timm.layers import Mlp
from timm.layers.config import use_fused_attn
from timm.layers.mlp import Mlp
from timm.layers.weight_init import trunc_normal_tf_
from timm.models.vision_transformer import checkpoint_filter_fn, build_model_with_cfg, VisionTransformer
from torch.jit import Final
import loralib as lora
import numpy as np
from .pos_embed import get_1d_sincos_pos_embed_from_grid

class Attention(nn.Module):
    fused_attn: Final[bool]
    lora_rank = 8
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = lora.Linear(dim, dim * 3, r=self.lora_rank, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    if "pretrained_strict" in kwargs:
        strict = kwargs["pretrained_strict"]
        kwargs.pop("pretrained_strict")
    else:
        strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )

class VisionEncoder(nn.Module):
    def __init__(self, name, pretrained=True, global_pool='', finetune=False, lora_rank=8):
        """
        Using timm vision encoder 
        by default, we do not pool visual features at this stage
        currently it only works with a single camera

        Params: 
            finetune has a few options
                1. False: we freeze the model
                2. (int) > 0: we unfreeze the last n blocks
                3. "all": we finetune the entire model
                4. "lora": lora finetuning 
        """
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.finetune = finetune
        if finetune == "lora":
            Attention.lora_rank = lora_rank
            timm.models.vision_transformer.Attention = Attention
            timm.models.vision_transformer._create_vision_transformer = _create_vision_transformer
            kwargs = {"pretrained_strict": False}
        else:
            kwargs = {}
        if "cross-mae-rtx" in name:
            self.model = timm.create_model("vit_base_patch16_224.mae", pretrained=pretrained, global_pool=global_pool, **kwargs)
            timm.models.load_checkpoint(self.model, name, strict=False)
        elif "dust3r" in name.lower():
            self.model = timm.create_model("vit_large_patch16_224", pretrained=False)
            ckpt = torch.load(name, map_location='cpu')
            # Extract encoder weights from the dust3r model
            encoder_weights = ckpt['model']['patch_embed.proj.weight']            
            # Load encoder weights into the corresponding part
            self.model.patch_embed.proj.weight.data.copy_(encoder_weights)
        else:
            self.model = timm.create_model(name, pretrained=pretrained, global_pool=global_pool, **kwargs)
        if self.finetune:
            self.model.train()
            if isinstance(self.finetune, int):
                self.unfreeze_last_n_blocks(self.finetune)
            elif self.finetune == "all":
                self.unfreeze() 
            elif self.finetune == "lora":
                # enable training of all biases to sequeze more performance out
                # https://github.com/microsoft/LoRA
                lora.mark_only_lora_as_trainable(self.model, bias='all')
        else:
            self.model.eval()
            self.freeze()

        self.model.norm = nn.Identity() # we apply this outside to make it trainable

        # if "cross-mae-rtx" in name:
        #     self.model.train()
        #     self.model.norm.weight.requires_grad = True
        #     self.model.norm.bias.requires_grad = True

    def out_dim(self): 
        return self.model.embed_dim

    def unfreeze_last_n_blocks(self, n):
        # we unfreeze the last n blocks
        assert isinstance(self.model, timm.models.vision_transformer.VisionTransformer), "unfreeze only works for vision transformer"
        if len(self.model.blocks) < n:
            print(f"can only unfreeze {len(self.model.blocks)} blocks instead of {n} blocks!")
            n = len(self.model.blocks)
        # first freeze everything
        self.freeze()
        for i in range(n):
            for param in self.model.blocks[-i].parameters():
                param.requires_grad = True
        print(f"unfreeze last {n} blocks")

    def unfreeze(self):
        # we unfreeze the model depends on finetune or not
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze(self):
        # we freeze the model depends on finetune or not
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        x : B, (T), (N), 3, H, W
        remember to move all data to batch axis first
        output: B, (T), (N), L, D
        """
        initial_shape = x.shape[:-3]
        trailing_shape = x.shape[-3:]
        x = x.view(-1, *trailing_shape)

        if self.finetune:
            feats = self.model.forward_features(x)
        else:
            with torch.no_grad():
                feats = self.model.forward_features(x)

        return feats.view(*initial_shape, *feats.shape[1:])


class VisionEncoderCNN(nn.Module):
    def __init__(self, name, pretrained=True, global_pool='', finetune=False, lora_rank=8):
        """
        Using timm vision encoder 
        by default, we do not pool visual features at this stage
        currently it only works with a single camera

        Params: 
            finetune has a few options
                1. False: we freeze the model
                2. (int) > 0: we unfreeze the last n blocks
                3. "all": we finetune the entire model
                4. "lora": lora finetuning 
        """
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.finetune = finetune
        kwargs = {"num_classes": 0}
        self.model = timm.create_model(name, pretrained=pretrained, global_pool=global_pool, **kwargs)
        if self.finetune:
            self.model.train()
            if self.finetune == "all":
                self.unfreeze() 
            elif self.finetune == "lora":
                # enable training of all biases to sequeze more performance out
                # https://github.com/microsoft/LoRA
                lora.mark_only_lora_as_trainable(self.model, bias='all')
        else:
            self.model.eval()
            self.freeze()

    def out_dim(self): 
        return self.model.num_features

    def unfreeze_last_n_blocks(self, n):
        # we unfreeze the last n blocks
        raise NotImplementedError("Not supported for CNN")
    
    def unfreeze(self):
        # we unfreeze the model depends on finetune or not
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze(self):
        # we freeze the model depends on finetune or not
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        x : B, (T), (N), 3, H, W
        remember to move all data to batch axis first
        output: B, (T), (N), L, D
        """
        initial_shape = x.shape[:-3]
        trailing_shape = x.shape[-3:]
        x = x.view(-1, *trailing_shape)

        if self.finetune:
            feats = self.model.forward_features(x)
            feats = feats.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            with torch.no_grad():
                feats = self.model.forward_features(x)
                feats = feats.permute(0, 2, 3, 1).flatten(1, 2)

        return feats.view(*initial_shape, *feats.shape[1:])


class AttentionPool(nn.Module):
    """ 
    Attention pooling w/ latent query
    modified from 
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool.py
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            latent_len: int = 1,
            latent_dim: int = None,
            pos_embed: str = '',
            pool_type: str = '',
            norm_layer: Optional[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            drop: float = 0.0,
            spatial_len : int = 7,
    ):
        super().__init__()
        self.in_features = in_features
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        self.out_features = out_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn()
        
        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))

        if (pos_embed == 'learned' or pos_embed == 'abs') and latent_len > 1:
            if pos_embed == "learned":
                self.pos_embed = nn.Parameter(torch.zeros(self.latent_len, in_features))
            self.pos_embed_type = pos_embed
        else:
            self.pos_embed = None
            self.pos_embed_type = None

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(embed_dim, out_features)
        self.proj_drop = nn.Dropout(drop)

        self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(
            in_features=out_features, 
            hidden_features=int(out_features * mlp_ratio), 
            out_features=out_features
        )

        self.init_weights()

    def init_weights(self):
        if self.pos_embed_type is not None:
            if self.pos_embed_type == 'learned':
                trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
            elif self.pos_embed_type == 'abs':
                self.pos_embed = nn.Parameter(torch.tensor(
                    get_1d_sincos_pos_embed_from_grid(self.in_features, np.arange(self.latent_len)), 
                    requires_grad=False
                ))
        trunc_normal_tf_(self.latent, std=self.latent_dim ** -0.5)

    def forward(self, x):
        B, N, C = x.shape

        if self.pos_embed_type is not None:
            x = x.repeat(1, self.latent_len, 1)
            N = self.latent_len
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = self.latent.expand(B, -1, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        elif self.pool == '':
            pass # we return the whole sequence
        return x
    
    def expand_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is of dimension: B, T, (input_dim), C
        output is of dimension B, T, latent_len, out_features
        """
        if x.ndim == 3:
            x = x[:, :, None, :]
        B, T, input_dim, C = x.shape
        x = x.view(B*T, input_dim, C)
        out = self.forward(x)
        return out.view(B, T, self.latent_len, self.out_features)
    
    def combine_forward(self, visual_tokens : torch.Tensor, proprio_tokens : torch.Tensor) -> torch.Tensor:
        """
        visual_tokens are of dimension: B, T, num_tokens, C
        proprio_tokens are of dimension: B, T, C
        output is B x T x latent_len x C
        """
        B, T, num_tokens, C = visual_tokens.shape
        visual_tokens = visual_tokens.view(B*T, num_tokens, C)
        proprio_tokens = proprio_tokens.view(B*T, 1, C)
        tokens = torch.cat([visual_tokens, proprio_tokens], dim=1)
        return self.forward(tokens).view(B, T, self.latent_len, self.out_features)

    def forward_visual(self, visual_tokens : torch.Tensor) -> torch.Tensor: 
        """
        visual_tokens are of dimension: B, T, num_tokens, C
        output is B x T x latent_len x C
        """
        B, T, num_tokens, C = visual_tokens.shape
        visual_tokens = visual_tokens.view(B*T, num_tokens, C)
        return self.forward(visual_tokens).view(B, T, self.latent_len, self.out_features)

    def combine_forward_discrete(self, visual_tokens : torch.Tensor, proprio_tokens: torch.Tensor) -> torch.Tensor:
        """
        used for dicrt exclusively
        visual_tokens are of dimension: B, T, num_tokens, C
        proprio_tokens are of dimension: B, T, num_prop_tokens, C
        output is B x T x latent_len x C
        """
        B, T, num_tokens, C = visual_tokens.shape
        B, T, num_prop_tokens, C = proprio_tokens.shape
        visual_tokens = visual_tokens.view(B*T, num_tokens, C)
        proprio_tokens = proprio_tokens.view(B*T, num_prop_tokens, C)
        tokens = torch.cat([visual_tokens, proprio_tokens], dim=1)
        return self.forward(tokens).view(B, T, self.latent_len, self.out_features)


class MultiKVAttentionPool(nn.Module):
    """ 
    Attention pooling w/ latent query and different key-value projections for different data
    modified from 
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool.py
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            num_modalities=2,
            embed_dim: int = None,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            latent_len: int = 1,
            latent_dim: int = None,
            pos_embed: str = '',
            pool_type: str = '',
            norm_layer: Optional[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            drop: float = 0.0,
            spatial_len : int = 7,
    ):
        super().__init__()
        self.in_features = in_features
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        self.out_features = out_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn()
        self.num_modalities = num_modalities
        
        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))

        if (pos_embed == 'learned' or pos_embed == 'abs') and latent_len > 1:
            if pos_embed == "learned":
                self.pos_embed = nn.Parameter(torch.zeros(self.latent_len, in_features))
            self.pos_embed_type = pos_embed
        else:
            self.pos_embed = None
            self.pos_embed_type = None

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.ModuleList([nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias) for _ in range(self.num_modalities)])
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(embed_dim, out_features)
        self.proj_drop = nn.Dropout(drop)

        self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(
            in_features=out_features, 
            hidden_features=int(out_features * mlp_ratio), 
            out_features=out_features
        )

        self.init_weights()

    def init_weights(self):
        if self.pos_embed_type is not None:
            if self.pos_embed_type == 'learned':
                trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
            elif self.pos_embed_type == 'abs':
                self.pos_embed = nn.Parameter(torch.tensor(
                    get_1d_sincos_pos_embed_from_grid(self.in_features, np.arange(self.latent_len)), 
                    requires_grad=False
                ))
        trunc_normal_tf_(self.latent, std=self.latent_dim ** -0.5)

    def forward(self, *modality_tokens):
        kv = []
        for i in range(self.num_modalities):
            tokens = modality_tokens[i]
            if len(tokens.shape) == 3:
                num_tokens = 1
                B, T, C = tokens.shape
            else:
                B, T, num_tokens, C = tokens.shape
            tokens = tokens.view(B * T, num_tokens, C)
            kv.append(self.kv[i](tokens).reshape(B * T, num_tokens, 2, self.num_heads, self.head_dim))

        kv = torch.cat(kv, dim=1).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)


        q_latent = self.latent.expand(B * T, -1, -1)
        q = self.q(q_latent).reshape(B * T, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B * T, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))
        return x.view(B, T, self.latent_len, self.out_features)

    def forward_attention(self, *modality_tokens):
        kv = []
        for i in range(self.num_modalities):
            tokens = modality_tokens[i]
            if len(tokens.shape) == 3:
                num_tokens = 1
                B, T, C = tokens.shape
            else:
                B, T, num_tokens, C = tokens.shape
            tokens = tokens.view(B * T, num_tokens, C)
            kv.append(self.kv[i](tokens).reshape(B * T, num_tokens, 2, self.num_heads, self.head_dim))

        kv = torch.cat(kv, dim=1).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)


        q_latent = self.latent.expand(B * T, -1, -1)
        q = self.q(q_latent).reshape(B * T, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        return attn

    
    def combine_forward(self, *modality_tokens):
        return self.forward(*modality_tokens)

if __name__ == "__main__":
    print("testing vision encoder")
    model = VisionEncoder('vit_small_patch16_224.dino', pretrained=True)
    print("dim: ", model.out_dim())
    x = torch.randn(2, 4, 5, 3, 224, 224)
    y = model(x) # torch.Size([2, 4, 5, 197, 384])
    print(y.shape)

    print("testing mlp")
    model = Mlp(in_features=7, out_features=768)
    x = torch.randn(2, 7)
    y = model(x)
    print(y.shape)

    print("testing attention pool")
    model = AttentionPool(768, out_features=1024, latent_len=4, pool_type='')
    x = torch.randn(2, 4, 768)
    y = model(x)
    print(y.shape)

    # robomimic
    print("testing robomimic")
    model = AttentionPool(768, out_features=1024, latent_len=1, pool_type='')
    x = torch.randn(2, 4, 768) # B, K, D
    y = model(x) # B, 1, C
    print(y.shape)

    x = torch.randn(2, 4, 768) # B, T, D
    y = model.expand_forward(x) # B, T, 1, 1024
    print(y.shape)

    print("testing combine forward")
    model = AttentionPool(768, latent_len=1, pool_type='')
    x = torch.randn(2, 4, 5, 768)
    y = torch.randn(2, 4, 768)
    z = model.combine_forward(x, y)
    print(z.shape)
