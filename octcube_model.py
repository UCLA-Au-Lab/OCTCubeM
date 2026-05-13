# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.7"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url  = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""OCTCube backbone — self-contained, flash-attn-free.

A 3D spatio-temporal Vision Transformer (ViT-L/16) for OCT volumes, ported
from `OCTCube/models_vit_st_flash_attn.py` so that it depends only on torch
and uses `torch.nn.functional.scaled_dot_product_attention` (which dispatches
to the cuDNN-fused / FA-2 backend on PyTorch >= 2.5).

Usage
-----
    from octcube_model import OCTCube

    model = OCTCube().cuda().eval()
    model.load_pretrained("OCTCube.pth")

    # volume: [B, 1, T, H, W] float tensor, T = num_frames, H = W = input_size
    # (a [B, T, H, W] tensor is also accepted; the channel dim is added.)
    features = model(volume)            # [B, embed_dim] global-pooled
    cls      = model(volume, "cls")     # [B, embed_dim] CLS token
    tokens   = model(volume, "tokens")  # [B, num_tokens, embed_dim]

Defaults match the released `OCTCube.pth` (ViT-L):
    patch_size=16, embed_dim=1024, depth=24, num_heads=16,
    t_patch_size=3, num_frames=60, input_size=256,
    sep_pos_embed=True, cls_embed=True.

`load_pretrained` handles three quirks of the released checkpoints:
  1. State-dict may be wrapped under a "model" key.
  2. Weights are stored in flash-attn layout (`blocks.X.mixer.Wqkv` and
     `mixer.out_proj`); they are reverse-remapped to plain `attn.{q,k,v,proj}`.
  3. Pos embeddings are interpolated when the target `num_frames` /
     `input_size` differs from pre-training.
Extra keys for a classifier head or an MAE decoder are silently dropped.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inlined helpers (timm equivalents)
# ---------------------------------------------------------------------------

class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class _Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------
# Patch embed (3D, in_chans -> embed_dim via Conv3d)
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 1024,
        frames: int = 60,
        t_patch_size: int = 3,
    ):
        super().__init__()
        assert img_size % patch_size == 0, f"img_size {img_size} not divisible by patch_size {patch_size}"
        assert frames % t_patch_size == 0, f"frames {frames} not divisible by t_patch_size {t_patch_size}"
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.frames = frames
        self.t_patch_size = t_patch_size
        self.grid_size = img_size // patch_size
        self.t_grid_size = frames // t_patch_size
        self.input_size = (self.t_grid_size, self.grid_size, self.grid_size)
        self.num_patches = self.t_grid_size * self.grid_size * self.grid_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(t_patch_size, patch_size, patch_size),
            stride=(t_patch_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] -> [B, T_p, (H_p*W_p), D]
        T, H, W = x.shape[2], x.shape[3], x.shape[4]
        assert (T, H, W) == (self.frames, self.img_size[0], self.img_size[1]), (
            f"input ({T},{H},{W}) doesn't match model "
            f"({self.frames},{self.img_size[0]},{self.img_size[1]})"
        )
        x = self.proj(x)                     # [B, D, T_p, H_p, W_p]
        x = x.flatten(3)                     # [B, D, T_p, H_p*W_p]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T_p, H_p*W_p, D]
        return x


# ---------------------------------------------------------------------------
# Attention via scaled_dot_product_attention (dispatches to FA-2 on PyTorch >= 2.5)
# ---------------------------------------------------------------------------

class SDPAAttention(nn.Module):
    """ViT attention with separate q/k/v Linears + final proj.

    Key layout matches `OCTCube/util/video_vit.py:Attention` so that, after
    the flash->plain remap in `load_pretrained`, weights load cleanly.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q(x).view(B, N, H, D).transpose(1, 2)  # [B, H, N, D]
        k = self.k(x).view(B, N, H, D).transpose(1, 2)
        v = self.v(x).view(B, N, H, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SDPAAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Pos-embed interpolation helpers
# ---------------------------------------------------------------------------

def _interpolate_spatial_pos_embed(model: "OCTCube", sd: dict) -> None:
    """Resize spatial pos embeddings (separate or combined) if input_size changed."""
    # sep_pos_embed: key is "pos_embed_spatial", shape [1, H_p*W_p, D]
    if "pos_embed_spatial" in sd:
        pe = sd["pos_embed_spatial"]
        D = pe.shape[-1]
        orig_n = pe.shape[-2]
        new_n = model.patch_embed.grid_size ** 2
        if orig_n != new_n:
            orig_s = int(orig_n ** 0.5)
            new_s = model.patch_embed.grid_size
            pe = pe.reshape(1, orig_s, orig_s, D).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(new_s, new_s), mode="bicubic", align_corners=False)
            pe = pe.permute(0, 2, 3, 1).reshape(1, new_s * new_s, D)
            sd["pos_embed_spatial"] = pe

    # combined pos_embed: shape [1, N(+cls), D]
    if "pos_embed" in sd and hasattr(model, "pos_embed"):
        pe = sd["pos_embed"]
        D = pe.shape[-1]
        num_extra = model.pos_embed.shape[-2] - model.patch_embed.num_patches
        orig_s = int((pe.shape[-2] - num_extra) ** 0.5)
        new_s = int(model.patch_embed.num_patches ** 0.5)
        if orig_s != new_s:
            extra = pe[:, :num_extra]
            tok = pe[:, num_extra:].reshape(1, orig_s, orig_s, D).permute(0, 3, 1, 2)
            tok = F.interpolate(tok, size=(new_s, new_s), mode="bicubic", align_corners=False)
            tok = tok.permute(0, 2, 3, 1).reshape(1, new_s * new_s, D)
            sd["pos_embed"] = torch.cat([extra, tok], dim=1)


def _interpolate_temporal_pos_embed(model: "OCTCube", sd: dict) -> None:
    """Resize temporal pos embedding (linear interp) if num_frames changed."""
    if "pos_embed_temporal" not in sd:
        return
    pe = sd["pos_embed_temporal"]
    orig_n = pe.shape[-2]
    new_n = model.patch_embed.t_grid_size
    if orig_n != new_n:
        pe = pe.permute(0, 2, 1)  # [1, D, T_p]
        pe = F.interpolate(pe, size=new_n, mode="linear", align_corners=False)
        pe = pe.permute(0, 2, 1)
        sd["pos_embed_temporal"] = pe


# ---------------------------------------------------------------------------
# Flash-attn -> plain state-dict remap
# ---------------------------------------------------------------------------

def _remap_flash_to_plain(sd: dict, depth: int) -> dict:
    """Convert flash-attn key layout to the layout this file's blocks expect.

    Per-block changes:
      blocks.{i}.mixer.Wqkv.weight  [3D, D]  -> blocks.{i}.attn.{q,k,v}.weight  [D, D]
      blocks.{i}.mixer.Wqkv.bias    [3D]     -> blocks.{i}.attn.{q,k,v}.bias    [D]
      blocks.{i}.mixer.out_proj.weight       -> blocks.{i}.attn.proj.weight
      blocks.{i}.mixer.out_proj.bias         -> blocks.{i}.attn.proj.bias
    All other keys pass through unchanged.
    """
    out = dict(sd)
    for i in range(depth):
        wqkv_w = f"blocks.{i}.mixer.Wqkv.weight"
        wqkv_b = f"blocks.{i}.mixer.Wqkv.bias"
        if wqkv_w in out:
            W = out.pop(wqkv_w)
            wq, wk, wv = W.chunk(3, dim=0)
            out[f"blocks.{i}.attn.q.weight"] = wq
            out[f"blocks.{i}.attn.k.weight"] = wk
            out[f"blocks.{i}.attn.v.weight"] = wv
        if wqkv_b in out:
            B = out.pop(wqkv_b)
            bq, bk, bv = B.chunk(3, dim=0)
            out[f"blocks.{i}.attn.q.bias"] = bq
            out[f"blocks.{i}.attn.k.bias"] = bk
            out[f"blocks.{i}.attn.v.bias"] = bv
        for kind in ("weight", "bias"):
            src = f"blocks.{i}.mixer.out_proj.{kind}"
            if src in out:
                out[f"blocks.{i}.attn.proj.{kind}"] = out.pop(src)
    return out


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class OCTCube(nn.Module):
    """OCTCube 3D spatio-temporal ViT encoder.

    Encoder-only — no classification head. Call `.forward(x)` to get
    embeddings; pass `return_mode="cls"` or `"tokens"` for alternatives.
    """

    def __init__(
        self,
        num_frames: int = 60,
        t_patch_size: int = 3,
        input_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        sep_pos_embed: bool = True,
        cls_embed: bool = True,
        global_pool: bool = True,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = lambda d: nn.LayerNorm(d, eps=1e-6)  # noqa: E731

        self.global_pool = global_pool
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed

        self.patch_embed = PatchEmbed3D(
            img_size=input_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, frames=num_frames, t_patch_size=t_patch_size,
        )
        input_grid = self.patch_embed.input_size  # (T_p, H_p, W_p)
        self.input_grid = input_grid

        if cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_grid[1] * input_grid[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, input_grid[0], embed_dim))
            if cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            n_tok = self.patch_embed.num_patches + (1 if cls_embed else 0)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tok, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            _Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        # Present in the released state_dict; left unapplied to match upstream
        # forward behavior. Users can call `model.norm(x)` themselves if needed.
        self.norm = norm_layer(embed_dim)

    # ---------------------------------------------------------- forward

    def _build_pos_embed(self) -> torch.Tensor:
        if self.sep_pos_embed:
            T_p, H_p, W_p = self.input_grid
            pe = self.pos_embed_spatial.repeat(1, T_p, 1) + torch.repeat_interleave(
                self.pos_embed_temporal, H_p * W_p, dim=1
            )
            if self.cls_embed:
                pe = torch.cat([self.pos_embed_class, pe], dim=1)
            return pe
        return self.pos_embed

    def forward(
        self,
        x: torch.Tensor,
        return_mode: Literal["pool", "cls", "tokens"] = "pool",
    ) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)  # [B, T, H, W] -> [B, 1, T, H, W]
        assert x.ndim == 5, f"expected [B, C, T, H, W], got shape {tuple(x.shape)}"

        x = self.patch_embed(x)              # [B, T_p, S, D]
        B, T_p, S, D = x.shape
        x = x.view(B, T_p * S, D)            # [B, N, D]

        if self.cls_embed:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)   # [B, 1+N, D]

        x = x + self._build_pos_embed()
        for blk in self.blocks:
            x = blk(x)

        if return_mode == "tokens":
            return self.norm(x)
        if return_mode == "cls":
            assert self.cls_embed, "cls_embed=False; CLS token not available"
            return x[:, 0]
        # pool: global mean over patch tokens (skips cls if present)
        start = 1 if self.cls_embed else 0
        return x[:, start:].mean(dim=1)

    # ---------------------------------------------------------- loading

    def load_pretrained(
        self,
        ckpt_path: str,
        *,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Load weights from an OCTCube.pth-style checkpoint.

        Returns (missing_keys, unexpected_keys) from the underlying
        `load_state_dict` call (with classifier-head / decoder keys
        already filtered out before the call).
        """
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        sd = _remap_flash_to_plain(sd, depth=len(self.blocks))
        _interpolate_spatial_pos_embed(self, sd)
        _interpolate_temporal_pos_embed(self, sd)

        # Drop keys that aren't part of this encoder-only model
        # (classifier head, MAE decoder, MAE mask token, etc.).
        target_keys = set(self.state_dict().keys())
        sd = {k: v for k, v in sd.items() if k in target_keys}

        msg = self.load_state_dict(sd, strict=strict)
        return list(msg.missing_keys), list(msg.unexpected_keys)
