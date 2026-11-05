"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from typing import List

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'proto_head']

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        proto_head: nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.proto_head = proto_head

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)

        # ============ Proto-Contrast Head ============
        proto_contrast_loss = None
        if self.proto_head is not None and self.training and targets is not None:
            # 取 encoder 中间尺度特征 (List[Tensor])
            mid_feat = x[len(x) // 2]  # (B, C, H, W)
            B, C, H, W = mid_feat.shape
            feat_flat = mid_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

            # 随机采样减少计算量
            sample_per_img = min(256, H * W)
            sampled_feats = []
            sampled_labels = []
            for b in range(B):
                indices = torch.randperm(H * W, device=feat_flat.device)[:sample_per_img]
                sampled_feats.append(feat_flat[b][indices])
                # 有 defect GT → label=0(defect), 无 GT → label=1(normal)
                if targets is not None and b < len(targets):
                    has_defect = len(targets[b].get("labels", [])) > 0
                    lbl = 0 if has_defect else 1
                else:
                    lbl = 1
                sampled_labels.append(
                    torch.full((sample_per_img,), lbl, dtype=torch.long, device=feat_flat.device)
                )

            feat_sampled = torch.cat(sampled_feats, dim=0)   # (B*N, C)
            label_sampled = torch.cat(sampled_labels, dim=0)  # (B*N,)

            proto_targets = [{"labels": label_sampled}]
            proto_out = self.proto_head(feat_sampled, proto_targets)
            if "loss_contrast" in proto_out:
                proto_contrast_loss = proto_out["loss_contrast"]
        # =============================================

        x = self.decoder(x, targets)

        if proto_contrast_loss is not None:
            x['proto_contrast_loss'] = proto_contrast_loss

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self