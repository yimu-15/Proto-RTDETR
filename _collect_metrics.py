import json
import glob
import os


def split_blocks(rows):
    """按 epoch 回退切分为多段 run，返回最后一段（完整 run）。"""
    blocks = []
    cur = []
    for r in rows:
        if cur and r["epoch"] <= cur[-1]["epoch"]:
            blocks.append(cur)
            cur = []
        cur.append(r)
    if cur:
        blocks.append(cur)
    return blocks[-1] if blocks else []


for lp in sorted(glob.glob("output/*/log.txt")):
    name = os.path.basename(os.path.dirname(lp))
    rows = [json.loads(l) for l in open(lp, encoding="utf-8") if "test_coco_eval_bbox" in l]
    if not rows:
        continue
    run = split_blocks(rows)
    best = max(run, key=lambda r: r["test_coco_eval_bbox"][0])
    e = run[-1]
    b = best["test_coco_eval_bbox"]
    f = e["test_coco_eval_bbox"]
    print(
        "{:45s} blocks={:d} bestE={:>2d} mAP={:.3f} AP50={:.3f} AP75={:.3f} AR1={:.3f} AR100={:.3f} | finalE={:>2d} mAP={:.3f}".format(
            name, len(rows) - len(run), best["epoch"], b[0], b[1], b[2], b[6], b[8], e["epoch"], f[0]
        )
    )
