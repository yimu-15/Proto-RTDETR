"""
Proto-RTDETRv2 训练启动脚本 for VisA Defect Detection
"""
import os
import sys
import random

import numpy as np
import torch

# 确保 src 在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core import YAMLConfig


def setup_seed(seed: int):
    """固定全部随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 关闭 cuDNN 非确定性算法（仅影响少量 op，代价可忽略）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args, ):
    """main
    """
    if args.seed is not None:
        setup_seed(args.seed)

    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        import torch
        checkpoint = torch.load(args.resume, map_location='cpu')
        # Robust weight extraction: try EMA > model > raw checkpoint
        if 'ema' in checkpoint and isinstance(checkpoint['ema'], dict):
            state = checkpoint['ema'].get('module', checkpoint['ema'])
        elif 'model' in checkpoint:
            state = checkpoint['model']
        else:
            state = checkpoint
        cfg.model.load_state_dict(state, strict=False)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print(f"Model type: {type(cfg.model).__name__}", flush=True)
    print(f"Proto head: {cfg.model.proto_head}", flush=True)

    if args.test_only:
        import torch
        from src.solver import DetSolver
        solver = DetSolver(cfg)
        solver.val()
        return

    import torch
    from src.solver import DetSolver

    print("DEBUG: Initializing DetSolver...", flush=True)
    solver = DetSolver(cfg)
    
    # ⚠️ 核心修改：将 solver.train() 改为 solver.fit()！
    print("DEBUG: DetSolver initialized. Calling solver.fit() to start training...", flush=True)
    solver.fit()
    
    print("DEBUG: solver.fit() finished. Training complete.", flush=True)


if __name__ == '__main__':
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='configs/proto_rtdetrv2_r50vd_visa.yml')
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducible experiments (default: 42)')

    args = parser.parse_args()
    
    print("DEBUG: Entering main()...", flush=True)
    main(args)