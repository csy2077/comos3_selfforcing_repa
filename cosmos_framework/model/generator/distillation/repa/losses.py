# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""REPA pieces shared by the self-forcing DMD student.

Ported from causal_forcing_repa_robotics (cosmos_backbone/wrapper.py, wan/modules/causal_model_align.py,
model/dmd.py). The student's align-layer tokens are projected to the encoder width and compared, by
token-relation distillation, with frozen OmniStream features of the paired real video.

Grid: the Cosmos3 student has one token per 2x2 latent patch of the 16x Wan2.2 VAE, i.e. 32 pixels.
The real frames are resized to (h*16, w*16) so OmniStream's 16-pixel patches give exactly the student's
(h, w) token grid -- the same 32-pixel cells the reference reaches with its 2x2 align downsampler.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# ImageNet statistics used by OmniStream's DINOv3ViTImageProcessorFast (preprocessor_config.json).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ENCODER_PATCH = 16


def build_repa_projector(hidden_size: int, projector_dim: int, align_dim: int) -> nn.Sequential:
    """The reference ``build_mlp``: hidden -> proj -> proj -> align, SiLU between."""
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, align_dim),
    )


def build_align_encoder(encoder_path: str) -> nn.Module:
    """Frozen OmniStream encoder, kept outside every FSDP-wrapped network."""
    if not encoder_path:
        raise ValueError("REPA needs repa_encoder_path (the OmniStream checkpoint directory).")
    from cosmos_framework.model.generator.distillation.repa.omnistream import OmnistreamMultiFrameTransformer

    encoder = OmnistreamMultiFrameTransformer.from_pretrained(encoder_path)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_repa_target(
    encoder: nn.Module,
    raw_video: torch.Tensor,  # [C,T_pix,H,W] or [1,C,T_pix,H,W], uint8 or float in [-1,1]
    latent_frames: list[int],
    grid_hw: tuple[int, int],
    temporal_compression: int = 4,
) -> torch.Tensor:
    """Frozen-encoder features of the real frames behind ``latent_frames``.

    Latent frame ``k >= 1`` covers pixel frames ``[tcf*(k-1)+1, tcf*k]`` of the causal VAE, so those
    ``tcf`` frames are encoded and averaged (``_encode_repa_target`` in the reference).

    Returns:
        [len(latent_frames), h, w, D] float32 features.
    """
    if raw_video.ndim == 5:
        raw_video = raw_video[0]  # [C,T_pix,H,W]
    if min(latent_frames) < 1:
        raise ValueError(f"REPA targets cover generated latent frames >= 1, got {latent_frames}")
    h, w = grid_hw
    tcf = temporal_compression
    first, last = min(latent_frames), max(latent_frames)
    pix_start, pix_end = tcf * (first - 1) + 1, tcf * last + 1
    if raw_video.shape[1] < pix_end:
        raise ValueError(
            f"REPA needs {pix_end} pixel frames for latent frames up to {last}, got {raw_video.shape[1]}"
        )
    enc_param = next(encoder.parameters())
    video = raw_video[:, pix_start:pix_end].to(device=enc_param.device)  # [C,T,H,W]
    video = video.float() / 255.0 if video.dtype == torch.uint8 else (video.float() + 1.0) / 2.0
    video = video.permute(1, 0, 2, 3)  # [T,C,H,W]
    video = F.interpolate(
        video, size=(h * ENCODER_PATCH, w * ENCODER_PATCH), mode="bilinear", align_corners=False, antialias=True
    )  # [T,C,h*16,w*16]
    mean = video.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = video.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    video = ((video - mean) / std).unsqueeze(0).to(enc_param.dtype).contiguous()  # [1,T,C,h*16,w*16]
    # OmniStream: [B,T,C,H,W] -> last_hidden_state [B,T,h*w,D] (CLS/register tokens stripped).
    feats = encoder(pixel_values=video, return_dict=True)["last_hidden_state"][0].float()  # [T,h*w,D]
    if feats.shape[1] != h * w:
        raise ValueError(f"expected {h * w} encoder patches per frame, got {feats.shape[1]}")
    feats = feats.reshape(last - first + 1, tcf, h, w, -1).mean(dim=1)  # [T_lat,h,w,D]
    return feats[[k - first for k in latent_frames]]  # [len(latent_frames),h,w,D]


def token_relation_distillation(
    align: torch.Tensor,  # [T,h*w,C], L2-normalized student tokens
    target: torch.Tensor,  # [T,h*w,C], L2-normalized encoder tokens
    margin: float,
) -> torch.Tensor:
    """VideoREPA TRD: each frame's tokens vs all space-time tokens, hinge on the relation gap."""
    T = align.shape[0]

    def relation(x: torch.Tensor) -> torch.Tensor:  # [T,hw,C] -> [T,hw,T*hw]
        keys = x.flatten(0, 1).unsqueeze(0).expand(T, -1, -1)  # [T,T*hw,C]
        return torch.bmm(x, keys.transpose(1, 2))

    return F.relu((relation(align) - relation(target)).abs() - margin).mean()
