"""Minimum smoke test for the released code (no full training).

Verifies: import, config load, model instantiation, ProtoContrastHead forward/backward,
prototype gradient, and inference path (no proto loss). Uses synthetic data only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.core import YAMLConfig
from src.nn.proto_contrast_head import ProtoContrastHead


def main():
    # 1. config load + model instantiation
    cfg = YAMLConfig("configs/proto_rtdetrv2_r50vd_visa_abl2_lw1.yml", resume="")
    model = cfg.model
    print("model type:", type(model).__name__)
    print("proto_head present:", model.proto_head is not None)
    print("proto_head loss_weight:", float(model.proto_head.loss_weight))

    # 2. ProtoContrastHead isolated forward/backward
    head = model.proto_head
    head.train()
    feats = torch.randn(512, 256)  # (N, feat_dim)
    labels = torch.cat([torch.zeros(256, dtype=torch.long), torch.ones(256, dtype=torch.long)])
    out = head(feats, [{"labels": labels}])
    assert "loss_contrast" in out, "loss_contrast missing"
    loss = out["loss_contrast"]
    print("proto loss:", float(loss.detach()))
    loss.backward()
    g = head.prototypes.grad
    print("prototype grad finite:", bool(torch.isfinite(g).all()))
    print("prototype grad norm:", float(g.norm()))
    assert torch.isfinite(g).all() and g.norm() > 0

    # 3. full model forward (train) + inference path (eval)
    model.train()
    x = torch.randn(2, 3, 640, 640)
    targets = [
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]])},
        {"labels": torch.tensor([], dtype=torch.long), "boxes": torch.zeros(0, 4)},
    ]
    out = model(x, targets)
    print("train output keys:", sorted(out.keys()))
    print("proto_contrast_loss present (train):", "proto_contrast_loss" in out)
    if "proto_contrast_loss" in out:
        print("proto_contrast_loss finite:", bool(torch.isfinite(out["proto_contrast_loss"])))

    model.eval()
    with torch.no_grad():
        out_inf = model(x)
    print("inference proto_contrast_loss present:", "proto_contrast_loss" in out_inf)
    assert "proto_contrast_loss" not in out_inf, "inference must not compute proto loss"

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
