# -*- coding: utf-8 -*-
import os, json, math, argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2
from tqdm import tqdm

try:
    import yaml
except Exception:
    yaml = None

import matplotlib.pyplot as plt


# -------------------------
# IO
# -------------------------
def read_json(p: str) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def read_jsonl(p: str) -> List[Dict[str, Any]]:
    out = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -------------------------
# Pose / geometry
# -------------------------
def quat_wxyz_to_R(q):
    qw,qx,qy,qz = [float(x) for x in q]
    n = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz) + 1e-12
    qw,qx,qy,qz = qw/n,qx/n,qy/n,qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3] = np.array(t, dtype=np.float64)
    return T

def apply_T(T, pts):
    pts_h = np.concatenate([pts, np.ones((pts.shape[0],1), dtype=np.float64)], axis=1)
    out = (T @ pts_h.T).T
    return out[:, :3]

def backproject_rect(Kr, xx, yy, Z):
    X = (xx - Kr[0,2]) / Kr[0,0] * Z
    Y = (yy - Kr[1,2]) / Kr[1,1] * Z
    return np.stack([X,Y,Z], axis=1)


# -------------------------
# Rectify map (make semantic align with rectified depth)
# -------------------------
def load_cam0_KD(mav0_dir: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int]]:
    if yaml is None:
        raise RuntimeError("pyyaml missing. Install: pip install pyyaml")
    ypath = os.path.join(mav0_dir, "cam0", "sensor.yaml")
    with open(ypath, "r") as f:
        y = yaml.safe_load(f)
    fx, fy, cx, cy = [float(x) for x in y["intrinsics"]]
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    D = np.array(y.get("distortion_coeffs", []), dtype=np.float64).reshape(-1,1)
    W,H = int(y["resolution"][0]), int(y["resolution"][1])
    return K, D, (W,H)

def build_rectify_map(mav0_dir: str, stereo_meta_path: str):
    st = read_json(stereo_meta_path)
    W,H = st["image_wh"]
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    R0 = np.array(st["rectify"]["R0"], dtype=np.float64)
    K0, D0, wh = load_cam0_KD(mav0_dir)
    assert wh == (W,H), f"resolution mismatch: sensor.yaml {wh} vs stereo_meta {(W,H)}"
    map0x, map0y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (W,H), cv2.CV_32FC1)
    return map0x, map0y, st


# -------------------------
# Visualization helpers
# -------------------------
def color_for_id(i: int) -> Tuple[int,int,int]:
    # deterministic vivid-ish
    return (int((i*37) % 255), int((i*17) % 255), int((i*67) % 255))

def semantic_to_rgb(sem: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    H,W = sem.shape
    rgb = np.zeros((H,W,3), dtype=np.uint8)
    u = np.unique(sem[valid_mask])
    for cid in u:
        cid = int(cid)
        if cid <= 0: 
            continue
        rgb[(sem==cid) & valid_mask] = np.array(color_for_id(cid), dtype=np.uint8)
    return rgb

def draw_traj_on_rgb(rgb: np.ndarray, traj_xy: np.ndarray, x0: float, y0: float, res: float, color=(255,255,255)):
    # traj_xy: (N,2) world coords
    H,W,_ = rgb.shape
    pts = []
    for x,y in traj_xy:
        ix = int((x - x0)/res)
        iy = int((y - y0)/res)
        if 0 <= ix < W and 0 <= iy < H:
            pts.append((ix, iy))
    if len(pts) >= 2:
        for i in range(1, len(pts)):
            cv2.line(rgb, pts[i-1], pts[i], color, 1, cv2.LINE_AA)
    return rgb


# -------------------------
# Core: fuse frames into global 2.5D map
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--stereo_meta", required=True)

    ap.add_argument("--mav0_dir", default="/root/autodl-tmp/sam-3d-objects/inputs/mav0",
                    help="needed for cam0/sensor.yaml to build rectify map")

    ap.add_argument("--out_dir", required=True)

    # map params
    ap.add_argument("--res", type=float, default=0.10, help="global grid resolution (m/cell). indoor: 0.05~0.10")
    ap.add_argument("--margin_m", type=float, default=4.0, help="expand bounds around trajectory (meters)")
    ap.add_argument("--pix_step", type=int, default=3, help="subsample pixels for speed (3~5 recommended)")
    ap.add_argument("--min_depth", type=float, default=0.10)
    ap.add_argument("--max_depth", type=float, default=12.0)

    # ground / cost
    ap.add_argument("--floor_name", type=str, default="floor")
    ap.add_argument("--traversable", type=str, default="floor,road")
    ap.add_argument("--step_height_thr", type=float, default=0.25, help="h_rel<thr -> traversable")
    ap.add_argument("--inflate_radius_m", type=float, default=0.30, help="inflate occupied for safety")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    rows = read_jsonl(args.keyframes)
    labels = read_json(args.labels_json)
    class_to_id = labels.get("class_to_id", {})
    floor_id = int(class_to_id.get(args.floor_name, -1))

    trav_names = [s.strip() for s in args.traversable.split(",") if s.strip()]
    trav_ids = set(int(class_to_id[n]) for n in trav_names if n in class_to_id)
    if len(trav_ids) == 0 and floor_id > 0:
        trav_ids.add(floor_id)

    map0x, map0y, st = build_rectify_map(args.mav0_dir, args.stereo_meta)
    W,H = st["image_wh"]
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    Kr = P0[:3,:3]
    R0 = np.array(st["rectify"]["R0"], dtype=np.float64)
    R_rect2orig = R0.T

    T_B_C0 = np.array(st["T_B_C0"], dtype=np.float64)
    T_C0_B = np.linalg.inv(T_B_C0)

    # ---- bounds from trajectory (camera/body positions)
    traj = np.array([r["p_WB"][:2] for r in rows], dtype=np.float64)
    xmin = float(traj[:,0].min() - args.margin_m)
    xmax = float(traj[:,0].max() + args.margin_m)
    ymin = float(traj[:,1].min() - args.margin_m)
    ymax = float(traj[:,1].max() + args.margin_m)

    res = float(args.res)
    GW = int(math.ceil((xmax - xmin)/res))
    GH = int(math.ceil((ymax - ymin)/res))

    # global grids
    g_ztop = np.full((GH, GW), -np.inf, dtype=np.float32)
    g_sem  = np.zeros((GH, GW), dtype=np.uint8)

    # optional: accumulate a global "floor z" reference from floor-labeled cells
    g_floor_ztop = np.full((GH, GW), np.nan, dtype=np.float32)

    # stats
    total_pts_in = 0
    total_cells_updated = 0

    # precompute pixel grid indices for sampling
    ys = np.arange(0, H, args.pix_step, dtype=np.int32)
    xs = np.arange(0, W, args.pix_step, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    for r in tqdm(rows, desc="[Step6c] fuse frames"):
        t = int(r["t_ns"])
        depth_path = os.path.join(args.depth_dir, f"{t}.npy")
        mask_path  = os.path.join(args.mask_dir, f"{t}.png")
        if not (os.path.isfile(depth_path) and os.path.isfile(mask_path)):
            continue

        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (H,W):
            continue

        sem = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        if sem.shape != (H,W):
            sem = cv2.resize(sem, (W,H), interpolation=cv2.INTER_NEAREST)

        # rectify semantic to align with rectified depth
        sem_r = cv2.remap(sem, map0x, map0y, interpolation=cv2.INTER_NEAREST)

        Z = depth[yy, xx].astype(np.float64)
        lab = sem_r[yy, xx].astype(np.int32)

        ok = np.isfinite(Z) & (Z > args.min_depth) & (Z < args.max_depth)
        if not np.any(ok):
            continue

        xxo = xx[ok].astype(np.float64)
        yyo = yy[ok].astype(np.float64)
        Zo  = Z[ok]
        labo = lab[ok]

        # backproject in rectified cam, rotate back to original cam0
        P_rect = backproject_rect(Kr, xxo, yyo, Zo)            # (N,3)
        P_c0   = (R_rect2orig @ P_rect.T).T                    # (N,3)

        # cam0 -> body -> world
        P_B = apply_T(T_C0_B, P_c0)
        T_W_B = make_T(quat_wxyz_to_R(r["q_WB_wxyz"]), r["p_WB"])
        P_W = apply_T(T_W_B, P_B)

        # map to global grid
        gx = np.floor((P_W[:,0] - xmin)/res).astype(np.int32)
        gy = np.floor((P_W[:,1] - ymin)/res).astype(np.int32)
        inb = (gx >= 0) & (gx < GW) & (gy >= 0) & (gy < GH)
        if not np.any(inb):
            continue

        gx = gx[inb]; gy = gy[inb]
        z  = P_W[inb, 2].astype(np.float32)
        lb = labo[inb].astype(np.uint8)

        total_pts_in += int(z.size)

        lin = (gy.astype(np.int64) * GW + gx.astype(np.int64))

        # per-cell max within this frame batch (avoid same-cell overwrite issues)
        order = np.lexsort((z, lin))
        lin_s = lin[order]
        z_s   = z[order]
        lb_s  = lb[order]

        u, first, counts = np.unique(lin_s, return_index=True, return_counts=True)
        last = first + counts - 1
        lin_u = u
        z_u = z_s[last]
        lb_u = lb_s[last]

        # update global if higher
        g_flat_z = g_ztop.reshape(-1)
        g_flat_s = g_sem.reshape(-1)

        prev = g_flat_z[lin_u]
        upd = z_u > prev
        if np.any(upd):
            idx = lin_u[upd]
            g_flat_z[idx] = z_u[upd]
            g_flat_s[idx] = lb_u[upd]
            total_cells_updated += int(np.sum(upd))

        # floor reference (optional)
        if floor_id > 0:
            is_floor = (lb_u == floor_id)
            if np.any(is_floor):
                idxf = lin_u[is_floor]
                # store the ztop for floor cells (later take global median)
                g_floor_ztop.reshape(-1)[idxf] = g_flat_z[idxf]

    # ---- build 2.5D height rel and cost
    valid = np.isfinite(g_ztop) & (g_ztop > -1e8)

    # global floor z reference (single scalar, robust and stable)
    if floor_id > 0:
        floor_vals = g_floor_ztop[np.isfinite(g_floor_ztop)]
        if floor_vals.size > 50:
            z_floor_ref = float(np.median(floor_vals))
        else:
            z_floor_ref = float(np.percentile(g_ztop[valid], 10)) if np.any(valid) else 0.0
    else:
        z_floor_ref = float(np.percentile(g_ztop[valid], 10)) if np.any(valid) else 0.0

    g_hrel = np.full_like(g_ztop, np.nan, dtype=np.float32)
    g_hrel[valid] = (g_ztop[valid] - z_floor_ref).astype(np.float32)

    unknown = (~valid) | (g_sem == 0)
    traversable = np.zeros_like(g_sem, dtype=bool)
    for tid in trav_ids:
        traversable |= (g_sem == tid)

    free = (~unknown) & traversable & (np.nan_to_num(g_hrel, nan=1e9) < args.step_height_thr)
    occupied = (~unknown) & (~free)

    # inflate occupied
    occ_infl = occupied.copy()
    if args.inflate_radius_m > 1e-6:
        r = int(math.ceil(args.inflate_radius_m / max(res, 1e-6)))
        if r > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            occ_u8 = (occupied.astype(np.uint8) * 255)
            occ_d = cv2.dilate(occ_u8, kernel, iterations=1) > 0
            occ_infl = occ_d & (~unknown)

    # cost map
    g_cost = np.full((GH, GW), 10.0, dtype=np.float32)  # unknown cost
    g_cost[free] = 1.0
    g_cost[occ_infl] = 1e6

    # ---- save arrays
    np.save(os.path.join(args.out_dir, "global_semantic.npy"), g_sem)
    np.save(os.path.join(args.out_dir, "global_ztop.npy"), g_ztop)
    np.save(os.path.join(args.out_dir, "global_height_rel.npy"), g_hrel)
    np.save(os.path.join(args.out_dir, "global_cost.npy"), g_cost)

    stats = {
        "frames_total": len(rows),
        "grid_size_hw": [int(GH), int(GW)],
        "bounds_xy_m": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
        "res_m": res,
        "z_floor_ref_m": z_floor_ref,
        "valid_ratio": float(valid.mean()),
        "unknown_ratio": float(unknown.mean()),
        "free_ratio": float(free.mean()),
        "occupied_ratio": float(occupied.mean()),
        "occupied_infl_ratio": float(occ_infl.mean()),
        "total_pts_in_bounds": int(total_pts_in),
        "total_cells_updated": int(total_cells_updated),
        "pix_step": int(args.pix_step),
        "depth_range_m": [float(args.min_depth), float(args.max_depth)],
        "traversable_names": trav_names,
        "step_height_thr_m": float(args.step_height_thr),
        "inflate_radius_m": float(args.inflate_radius_m),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ---- visualization
    sem_rgb = semantic_to_rgb(g_sem, valid)
    sem_rgb = draw_traj_on_rgb(sem_rgb, traj, xmin, ymin, res, color=(255,255,255))

    # height vis (percentile normalize on valid)
    h = g_hrel.copy()
    h_vis = np.zeros_like(h, dtype=np.float32)
    m = np.isfinite(h)
    if np.any(m):
        lo = np.percentile(h[m], 5)
        hi = np.percentile(h[m], 95)
        h_vis[m] = (np.clip(h[m], lo, hi) - lo) / (hi - lo + 1e-6)
    h_u8 = (h_vis * 255).astype(np.uint8)

    # cost vis (cap)
    c = g_cost.copy()
    c[c > 50] = 50
    c_norm = (c - c.min()) / (c.max() - c.min() + 1e-6)

    cv2.imwrite(os.path.join(args.out_dir, "global_semantic_vis.png"), cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.out_dir, "global_height_rel_vis.png"), h_u8)
    plt.figure(figsize=(14,5))
    plt.subplot(1,3,1); plt.imshow(sem_rgb); plt.title("Global semantic (traj overlay)"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(h_u8, cmap="gray"); plt.title("Global height rel (2.5D)"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(c_norm, cmap="inferno"); plt.title("Global cost map"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "global_triplet.png"), dpi=200)
    plt.close()

    # debug masks
    cv2.imwrite(os.path.join(args.out_dir, "mask_valid.png"), (valid.astype(np.uint8)*255))
    cv2.imwrite(os.path.join(args.out_dir, "mask_free.png"), (free.astype(np.uint8)*255))
    cv2.imwrite(os.path.join(args.out_dir, "mask_occupied_infl.png"), (occ_infl.astype(np.uint8)*255))
    cv2.imwrite(os.path.join(args.out_dir, "mask_unknown.png"), (unknown.astype(np.uint8)*255))

    print("[Step6c] done. Saved to:", args.out_dir)
    print("[Step6c] stats:", stats)


if __name__ == "__main__":
    main()


'''python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step6c_build_global_2p5d_map.py \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --depth_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/depth_npy \
  --mask_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --stereo_meta /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/stereo_meta.json \
  --mav0_dir /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/global_map_2p5d_200 \
  --res 0.10 --margin_m 4.0 --pix_step 3 --min_depth 0.10 --max_depth 12.0 \
  --floor_name floor --traversable floor,road --step_height_thr 0.25 --inflate_radius_m 0.30
'''