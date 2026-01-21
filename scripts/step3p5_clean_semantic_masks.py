#Step3.5 清洗 masks（碎片并入 + 细长保护）
# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np
import cv2
from tqdm import tqdm

# Absolute paths (edit for your server)
DEFAULT_MASK_DIR = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks"
DEFAULT_LABELS_JSON = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json"
DEFAULT_OUT_DIR = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks_clean"


def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_json(p):
    with open(p, "r") as f:
        return json.load(f)

def majority_label_in_ring(mask, comp_mask, ring_r=3, ignore_ids=(0,)):
    """
    mask: HxW uint8 semantic labels
    comp_mask: HxW uint8 {0,1} for current component
    ring: dilate(comp) - comp
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ring_r+1, 2*ring_r+1))
    dil = cv2.dilate(comp_mask, kernel, iterations=1)
    ring = (dil > 0) & (comp_mask == 0)
    if not np.any(ring):
        return None, 0
    vals = mask[ring].astype(np.int32)
    if vals.size == 0:
        return None, 0
    for iid in ignore_ids:
        vals = vals[vals != iid]
    if vals.size == 0:
        return None, 0
    bc = np.bincount(vals)
    lab = int(bc.argmax())
    cnt = int(bc[lab])
    return lab, cnt

def is_long_thin(comp_mask, min_len_px=25, max_thickness_px=6):
    """
    保护“真实细长物体”：足够长、足够细。
    用包围盒近似：长边 >= min_len，短边 <= max_thickness
    """
    ys, xs = np.where(comp_mask > 0)
    if xs.size == 0:
        return False
    x0,x1 = xs.min(), xs.max()
    y0,y1 = ys.min(), ys.max()
    w = (x1 - x0 + 1)
    h = (y1 - y0 + 1)
    long_side = max(w, h)
    short_side = min(w, h)
    return (long_side >= min_len_px) and (short_side <= max_thickness_px)

def majority_filter_3x3(lbl):
    """
    3x3 多数滤波（忽略0时更合理，但这里先简单实现：0也参与）
    """
    H,W = lbl.shape
    out = lbl.copy()
    # pad
    pad = np.pad(lbl, ((1,1),(1,1)), mode="edge")
    for y in range(H):
        for x in range(W):
            block = pad[y:y+3, x:x+3].reshape(-1).astype(np.int32)
            bc = np.bincount(block)
            out[y,x] = int(bc.argmax())
    return out

def clean_mask(mask, *,
               small_area_px=350,
               ring_r=3,
               ring_min_cnt=30,
               min_keep_area_px=80,
               protect_long_thin=True,
               long_min_len_px=25,
               long_max_thick_px=6,
               do_majority_filter=True):
    """
    核心：小碎片并入邻域主类；细长结构保护；最后多数滤波。
    """
    H,W = mask.shape
    out = mask.copy()
    protect_mask = np.zeros_like(out, dtype=bool)

    ids = [int(x) for x in np.unique(mask) if int(x) != 0]
    for cid in ids:
        binm = (out == cid).astype(np.uint8)
        ncc, cc = cv2.connectedComponents(binm, connectivity=4)
        for k in range(1, ncc):
            comp = (cc == k).astype(np.uint8)
            area = int(comp.sum())
            if area <= 0:
                continue

            # 细长保护：线缆/杆等不要被并掉
            if protect_long_thin and is_long_thin(comp, long_min_len_px, long_max_thick_px):
                continue

            # 小碎片：并入邻域主类（不删除）
            if area < small_area_px:
                lab, cnt = majority_label_in_ring(out, comp, ring_r=ring_r, ignore_ids=(0, cid))
                if (lab is not None) and (cnt >= ring_min_cnt):
                    out[comp > 0] = np.uint8(lab)
                else:
                    if area <= min_keep_area_px:
                        protect_mask |= (comp > 0)
                # 如果找不到邻域主类：保持原类（不删除），以满足“不想删室内常见物体”的要求

    # 最后一轮轻量多数滤波，清掉盐椒噪点
    if do_majority_filter:
        out = majority_filter_3x3(out.astype(np.uint8))
        if np.any(protect_mask):
            out[protect_mask] = mask[protect_mask]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", default=DEFAULT_MASK_DIR, help=".../masks (uint8 labels)")
    ap.add_argument("--labels_json", default=DEFAULT_LABELS_JSON, help="labels.json for reference/log only")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="output masks_clean dir")
    ap.add_argument("--small_area_px", type=int, default=350)
    ap.add_argument("--ring_r", type=int, default=3)
    ap.add_argument("--ring_min_cnt", type=int, default=30)
    ap.add_argument("--min_keep_area_px", type=int, default=80)
    ap.add_argument("--protect_long_thin", action="store_true")
    ap.add_argument("--long_min_len_px", type=int, default=25)
    ap.add_argument("--long_max_thick_px", type=int, default=6)
    ap.add_argument("--no_majority_filter", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    _ = read_json(args.labels_json)  # only to ensure file exists

    files = sorted([f for f in os.listdir(args.mask_dir) if f.endswith(".png")])
    for f in tqdm(files, desc="[Step3.5] clean masks"):
        ip = os.path.join(args.mask_dir, f)
        m = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        mc = clean_mask(
            m,
            small_area_px=args.small_area_px,
            ring_r=args.ring_r,
            ring_min_cnt=args.ring_min_cnt,
            min_keep_area_px=args.min_keep_area_px,
            protect_long_thin=bool(args.protect_long_thin),
            long_min_len_px=args.long_min_len_px,
            long_max_thick_px=args.long_max_thick_px,
            do_majority_filter=(not args.no_majority_filter),
        )
        op = os.path.join(args.out_dir, f)
        cv2.imwrite(op, mc)

    print("[Step3.5] done. out_dir =", args.out_dir)


if __name__ == "__main__":
    main()

'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step3p5_clean_semantic_masks.py \
  --mask_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks_clean \
  --small_area_px 350 --ring_r 3 --protect_long_thin --long_min_len_px 25 --long_max_thick_px 6
'''
