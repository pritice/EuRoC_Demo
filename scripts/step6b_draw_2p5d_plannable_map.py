# -*- coding: utf-8 -*-
import os, json, argparse
from typing import Dict, Any, Tuple, List

import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_json(p: str) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def color_for_id(i: int) -> Tuple[int,int,int]:
    # deterministic vivid-ish
    return (int((i*37) % 255), int((i*17) % 255), int((i*67) % 255))

def visualize_semantic(sem: np.ndarray, id_to_name: Dict[int,str]) -> np.ndarray:
    H,W = sem.shape
    rgb = np.zeros((H,W,3), dtype=np.uint8)
    for cid in np.unique(sem):
        cid = int(cid)
        if cid <= 0:
            continue
        rgb[sem == cid] = np.array(color_for_id(cid), dtype=np.uint8)
    return rgb

def make_floor_reference(sem: np.ndarray, ztop: np.ndarray, floor_id: int) -> float:
    valid = np.isfinite(ztop) & (ztop > -1e8)
    if floor_id > 0:
        m = valid & (sem == floor_id)
        if np.any(m):
            return float(np.median(ztop[m]))
    # fallback: 10th percentile of all valid
    if np.any(valid):
        return float(np.percentile(ztop[valid], 10))
    return 0.0

def compute_plannable_maps(
    sem: np.ndarray,
    ztop: np.ndarray,
    class_to_id: Dict[str,int],
    traversable_names: List[str],
    floor_name: str,
    step_height_thr: float,
    unknown_cost: float,
    base_cost: float,
    obstacle_cost: float,
    inflate_radius_m: float,
    cell_m: float,
):
    H,W = sem.shape
    sem = sem.astype(np.int32)

    # valid z
    valid_z = np.isfinite(ztop) & (ztop > -1e8)
    unknown = (sem == 0) | (~valid_z)

    # floor ref
    floor_id = int(class_to_id.get(floor_name, -1))
    z_floor = make_floor_reference(sem, ztop, floor_id)

    h_rel = np.full_like(ztop, np.nan, dtype=np.float32)
    h_rel[valid_z] = (ztop[valid_z] - z_floor).astype(np.float32)

    # traversable ids
    trav_ids = set()
    for n in traversable_names:
        if n in class_to_id:
            trav_ids.add(int(class_to_id[n]))
    if len(trav_ids) == 0 and floor_id > 0:
        trav_ids.add(floor_id)

    # free if (semantic is traversable) and not too high
    trav = np.zeros((H,W), dtype=bool)
    for tid in trav_ids:
        trav |= (sem == tid)

    free = (~unknown) & trav & (np.nan_to_num(h_rel, nan=1e9) < step_height_thr)
    occupied = (~unknown) & (~free)

    # cost map
    cost = np.full((H,W), unknown_cost, dtype=np.float32)
    cost[free] = base_cost
    cost[occupied] = obstacle_cost

    # obstacle inflation (safety radius)
    if inflate_radius_m > 1e-6:
        r = int(np.ceil(inflate_radius_m / max(cell_m, 1e-6)))
        if r > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            occ_u8 = occupied.astype(np.uint8) * 255
            occ_d = cv2.dilate(occ_u8, kernel, iterations=1) > 0
            # inflated cells become occupied unless they are unknown (unknown stays unknown)
            cost[(occ_d & (~unknown))] = obstacle_cost

    stats = {
        "z_floor_ref": float(z_floor),
        "cell_m": float(cell_m),
        "unknown_ratio": float(unknown.mean()),
        "free_ratio": float(free.mean()),
        "occupied_ratio": float(occupied.mean()),
        "traversable_names": traversable_names,
        "step_height_thr_m": float(step_height_thr),
        "inflate_radius_m": float(inflate_radius_m),
    }
    return h_rel, cost, free, occupied, unknown, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stmr_json", required=True, help=".../stmr_20x20/{t}.json")
    ap.add_argument("--labels_json", required=True, help=".../labels.json")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--floor_name", type=str, default="floor")
    ap.add_argument("--traversable", type=str, default="floor,road",
                    help="comma-separated traversable class names")

    ap.add_argument("--step_height_thr", type=float, default=0.25,
                    help="h_rel < thr -> free (meters)")
    ap.add_argument("--inflate_radius_m", type=float, default=0.30,
                    help="inflate occupied by UAV radius (meters)")

    ap.add_argument("--unknown_cost", type=float, default=10.0)
    ap.add_argument("--base_cost", type=float, default=1.0)
    ap.add_argument("--obstacle_cost", type=float, default=1e6)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    stmr = read_json(args.stmr_json)
    labels = read_json(args.labels_json)
    class_to_id = labels.get("class_to_id", {})

    sem = np.array(stmr["semantic_grid"], dtype=np.int32)
    ztop = np.array(stmr["ztop_grid"], dtype=np.float32)
    cell_m = float(stmr.get("scale_m_per_cell", 0.4))  # range_m / 20

    # NOTE: stmr grid may be list-of-lists with shape (20,20)
    if sem.ndim != 2:
        raise ValueError(f"semantic_grid shape invalid: {sem.shape}")
    if ztop.shape != sem.shape:
        raise ValueError(f"ztop_grid shape mismatch: {ztop.shape} vs {sem.shape}")

    id_to_name = {int(v): k for k,v in class_to_id.items()}
    id_to_name[0] = "unknown"

    traversable_names = parse_list(args.traversable)

    h_rel, cost, free, occ, unk, stats = compute_plannable_maps(
        sem=sem, ztop=ztop, class_to_id=class_to_id,
        traversable_names=traversable_names,
        floor_name=args.floor_name,
        step_height_thr=args.step_height_thr,
        unknown_cost=args.unknown_cost,
        base_cost=args.base_cost,
        obstacle_cost=args.obstacle_cost,
        inflate_radius_m=args.inflate_radius_m,
        cell_m=cell_m,
    )

    # --- save arrays
    np.save(os.path.join(args.out_dir, "semantic_grid.npy"), sem)
    np.save(os.path.join(args.out_dir, "height_rel.npy"), h_rel)
    np.save(os.path.join(args.out_dir, "cost.npy"), cost)
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # --- draw
    sem_rgb = visualize_semantic(sem, id_to_name)

    # height colormap: show only valid; invalid -> black
    h_show = np.nan_to_num(h_rel, nan=np.nan)
    h_mask = np.isfinite(h_show)
    h_img = np.zeros_like(h_show, dtype=np.float32)
    if np.any(h_mask):
        lo = np.percentile(h_show[h_mask], 5)
        hi = np.percentile(h_show[h_mask], 95)
        h_img[h_mask] = (np.clip(h_show[h_mask], lo, hi) - lo) / (hi - lo + 1e-6)
    h_u8 = (h_img * 255).astype(np.uint8)

    # cost visualization (cap)
    cost_cap = np.copy(cost)
    cost_cap[cost_cap > 50] = 50  # cap for display
    # normalize to 0..1
    lo, hi = float(np.min(cost_cap)), float(np.max(cost_cap))
    c_norm = (cost_cap - lo) / (hi - lo + 1e-6)

    # plot three panels
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(sem_rgb)
    ax1.set_title("Semantic grid (20x20)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(h_u8, cmap="gray")
    ax2.set_title("Height rel-to-floor (2.5D)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(c_norm, cmap="inferno")
    ax3.set_title("Cost map (plannable)")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "map_semantic_height_cost.png"), dpi=200)
    plt.close(fig)

    # also save separate masks for debugging
    cv2.imwrite(os.path.join(args.out_dir, "map_semantic.png"), cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.out_dir, "map_height_rel.png"), h_u8)

    # free/occ/unk debug
    cv2.imwrite(os.path.join(args.out_dir, "mask_free.png"), (free.astype(np.uint8)*255))
    cv2.imwrite(os.path.join(args.out_dir, "mask_occupied.png"), (occ.astype(np.uint8)*255))
    cv2.imwrite(os.path.join(args.out_dir, "mask_unknown.png"), (unk.astype(np.uint8)*255))

    print("[Step6b] saved to:", args.out_dir)
    print("[Step6b] stats:", stats)


if __name__ == "__main__":
    main()

'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step6b_draw_2p5d_plannable_map.py \
  --stmr_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stmr_maps_dino_sam2_200_v3/stmr_20x20/1413393887255760384.json \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/plannable_2p5d/1413393887255760384 \
  --floor_name floor \
  --traversable floor,road \
  --step_height_thr 0.25 \
  --inflate_radius_m 0.30
'''