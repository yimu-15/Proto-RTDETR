"""
Proto-Contrast Head for Defect Detection
集成到 RT-DETRv2 的原型对比学习模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from ..core import register


@register()
class ProtoContrastHead(nn.Module):
    """
    原型对比学习头：
    1. 维护每个类别的可学习原型向量 (defect=0, normal=1)
    2. 通过 InfoNCE 对比损失增强特征判别力
    3. 原型通过反向传播端到端优化（与论文一致，无 EMA）
    """
    def __init__(self, feat_dim: int = 256, num_classes: int = 2,
                 proj_dim: int = 128, temperature: float = 0.07,
                 loss_weight: float = 0.5):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.loss_weight = loss_weight

        # 可学习原型向量
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # 投影头
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim)
        )

    def forward(self, features: torch.Tensor,
                targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (N, D) - encoder 输出的展平特征
            targets: 训练时的 GT 标注列表
        Returns:
            dict with proto_logits, loss_contrast
        """
        proj_feat = self.proj(features)
        proj_feat = F.normalize(proj_feat, dim=-1)

        proto_proj = self.proj(self.prototypes)
        proto_proj = F.normalize(proto_proj, dim=-1)

        sim = torch.matmul(proj_feat, proto_proj.T) / self.temperature

        result = {"proto_logits": sim}

        if targets is not None and self.training:
            loss = self._info_nce_loss(proj_feat, proto_proj, targets)
            if loss is not None:
                result["loss_contrast"] = loss * self.loss_weight

        return result

    def _info_nce_loss(self, proj_feat, proto_proj, targets):
        labels = []
        for t in targets:
            if "labels" in t:
                labels.append(t["labels"])
        if not labels:
            return None
        labels = torch.cat(labels)
        if labels.numel() == 0:
            return None

        n = min(labels.numel(), proj_feat.shape[0])
        labels = labels[:n]
        proj_feat = proj_feat[:n]

        sim = torch.matmul(proj_feat, proto_proj.T) / self.temperature
        loss = F.cross_entropy(sim, labels.long())
        return loss