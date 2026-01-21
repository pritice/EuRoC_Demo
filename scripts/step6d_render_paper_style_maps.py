# -*- coding: utf-8 -*-
import os, json, math, argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None


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
# Geometry
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

def yaw_from_q_WB(q_wxyz):
    # R_WB: body x-axis points forward (EuRoC convention usually x-forward)
    R = quat_wxyz_to_R(q_wxyz)
    fwd = R[:,0]  # body x in world
    return float(math.atan2(fwd[1], fwd[0]))

def rot2d(theta):
    c = math.cos(theta); s = math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


# -------------------------
# Rectify map (align semantic with rectified depth)
# -------------------------
def load_cam0_KD(mav0_dir: str):
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
# Color + legend helpers
# -------------------------
def color_for_id(i: int) -> Tuple[int,int,int]:
    return (int((i*37) % 255), int((i*17) % 255), int((i*67) % 255))

def render_semantic_grid(semN: np.ndarray) -> np.ndarray:
    H,W = semN.shape
    rgb = np.zeros((H,W,3), dtype=np.uint8)
    for cid in np.unique(semN):
        cid = int(cid)
        if cid <= 0:
            continue
        rgb[semN == cid] = np.array(color_for_id(cid), dtype=np.uint8)
    return rgb

def mode_ignore_zero(block: np.ndarray) -> int:
    v = block.reshape(-1)
    v = v[v > 0]
    if v.size == 0:
        return 0
    bc = np.bincount(v.astype(np.int64))
    return int(bc.argmax())

def robust_p95(block: np.ndarray) -> float:
    v = block.reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, 95))

def compute_floor_ref(semN: np.ndarray, hN: np.ndarray, floor_id: int) -> float:
    if floor_id > 0:
        m = (semN == floor_id) & np.isfinite(hN)
        if np.any(m):
            return float(np.median(hN[m]))
    m = np.isfinite(hN)
    if np.any(m):
        return float(np.percentile(hN[m], 10))
    return 0.0


# -------------------------
# Main: local-window fuse -> pool to NxN -> draw
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--stereo_meta", required=True)
    ap.add_argument("--mav0_dir", default="/root/autodl-tmp/sam-3d-objects/inputs/mav0")

    ap.add_argument("--out_dir", required=True)

    # which frame to center the local map at
    ap.add_argument("--t_ns", type=str, default="", help="center frame timestamp; empty -> use last frame")

    # local map physical size and resolution
    ap.add_argument("--local_size_m", type=float, default=20.0,
                    help="physical window size (meters). paper-like cell=5m => local_size_m=100 with N=20")
    ap.add_argument("--N", type=int, default=20, help="grid size NxN (paper uses 20)")
    ap.add_argument("--pix_step", type=int, default=3)
    ap.add_argument("--min_depth", type=float, default=0.10)
    ap.add_argument("--max_depth", type=float, default=12.0)

    # fusion window over time
    ap.add_argument("--fuse_k", type=int, default=80,
                    help="accumulate last K frames around t (larger => more complete map, but may blur)")
    ap.add_argument("--ego_align", action="store_true",
                    help="rotate local map so UAV forward points up (recommended for paper style)")

    # annotation
    ap.add_argument("--annotate_topk", type=int, default=10,
                    help="annotate top-K largest regions with name+height")
    ap.add_argument("--min_region_cells", type=int, default=6,
                    help="min cells (in NxN) for a region to be annotated")

    # output style
    ap.add_argument("--show_height_in_matrix", action="store_true",
                    help="matrix cell shows id and height (2 lines)")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    rows = read_jsonl(args.keyframes)
    labels = read_json(args.labels_json)
    class_to_id = labels.get("class_to_id", {})
    id_to_name = {int(v): k for k,v in class_to_id.items()}
    id_to_name[0] = "unknown"
    floor_id = int(class_to_id.get("floor", -1))

    # pick center frame
    if args.t_ns:
        t0 = int(args.t_ns)
        idx0 = None
        for i,r in enumerate(rows):
            if int(r["t_ns"]) == t0:
                idx0 = i; break
        if idx0 is None:
            raise ValueError(f"t_ns not found in keyframes: {t0}")
    else:
        idx0 = len(rows) - 1
        t0 = int(rows[idx0]["t_ns"])

    # choose fusion indices
    k = int(args.fuse_k)
    i0 = max(0, idx0 - k + 1)
    use = rows[i0:idx0+1]

    # build rectify map and camera transforms
    map0x, map0y, st = build_rectify_map(args.mav0_dir, args.stereo_meta)
    W,H = st["image_wh"]
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    Kr = P0[:3,:3]
    R0 = np.array(st["rectify"]["R0"], dtype=np.float64)
    R_rect2orig = R0.T

    T_B_C0 = np.array(st["T_B_C0"], dtype=np.float64)
    T_C0_B = np.linalg.inv(T_B_C0)

    # center pose
    r0 = rows[idx0]
    p0 = np.array(r0["p_WB"], dtype=np.float64)
    yaw0 = yaw_from_q_WB(r0["q_WB_wxyz"])
    R_align = rot2d(-yaw0) if args.ego_align else np.eye(2, dtype=np.float64)

    # local grid in meters
    L = float(args.local_size_m)
    N = int(args.N)
    cell_m = L / N
    half = L / 2.0

    # we will pool points into a finer grid first (pixel grid in meters) to stabilize, then downsample to NxN
    # choose fine resolution = cell_m / 5 (at least)
    fine = max(cell_m / 5.0, 0.05)
    FN = int(math.ceil(L / fine))
    fine = L / FN  # adjust
    # fine grids
    f_ztop = np.full((FN, FN), -np.inf, dtype=np.float32)
    f_sem  = np.zeros((FN, FN), dtype=np.uint8)

    # sample pixels
    ys = np.arange(0, H, args.pix_step, dtype=np.int32)
    xs = np.arange(0, W, args.pix_step, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    # trajectory in local coords (for drawing)
    traj_xy = np.array([r["p_WB"][:2] for r in use], dtype=np.float64)
    traj_local = (R_align @ (traj_xy - p0[:2]).T).T

    # fuse frames
    pts_used = 0
    for r in use:
        t = int(r["t_ns"])
        dp = os.path.join(args.depth_dir, f"{t}.npy")
        mp = os.path.join(args.mask_dir, f"{t}.png")
        if not (os.path.isfile(dp) and os.path.isfile(mp)):
            continue

        depth = np.load(dp).astype(np.float32)
        if depth.shape != (H,W):
            continue

        sem = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        if sem.shape != (H,W):
            sem = cv2.resize(sem, (W,H), interpolation=cv2.INTER_NEAREST)

        # rectify semantic to match depth
        sem_r = cv2.remap(sem, map0x, map0y, interpolation=cv2.INTER_NEAREST)

        Z = depth[yy, xx].astype(np.float64)
        lb = sem_r[yy, xx].astype(np.int32)
        ok = np.isfinite(Z) & (Z > args.min_depth) & (Z < args.max_depth)
        if not np.any(ok):
            continue

        xxo = xx[ok].astype(np.float64)
        yyo = yy[ok].astype(np.float64)
        Zo  = Z[ok]
        lbo = lb[ok].astype(np.uint8)

        # rect backproject -> cam0 -> body -> world
        P_rect = backproject_rect(Kr, xxo, yyo, Zo)
        P_c0   = (R_rect2orig @ P_rect.T).T
        P_B    = apply_T(T_C0_B, P_c0)
        T_W_B  = make_T(quat_wxyz_to_R(r["q_WB_wxyz"]), r["p_WB"])
        P_W    = apply_T(T_W_B, P_B)

        # world -> local centered (and optionally ego-aligned)
        XY = P_W[:, :2] - p0[:2]
        XY = (R_align @ XY.T).T

        # keep points within local window
        inw = (XY[:,0] >= -half) & (XY[:,0] < half) & (XY[:,1] >= -half) & (XY[:,1] < half)
        if not np.any(inw):
            continue

        XY = XY[inw]
        ZW = P_W[inw, 2].astype(np.float32)
        LB = lbo[inw]
        pts_used += int(ZW.size)

        ix = np.floor((XY[:,0] + half) / fine).astype(np.int32)
        iy = np.floor((XY[:,1] + half) / fine).astype(np.int32)
        inb = (ix >= 0) & (ix < FN) & (iy >= 0) & (iy < FN)
        ix = ix[inb]; iy = iy[inb]
        ZW = ZW[inb]; LB = LB[inb]

        # per-cell "highest z wins" within fine grid (more stable than direct NxN)
        # this is still simple but works well for visualization
        for i in range(ix.shape[0]):
            x = ix[i]; y = iy[i]
            if ZW[i] > f_ztop[y,x]:
                f_ztop[y,x] = ZW[i]
                f_sem[y,x]  = LB[i]

    # downsample fine -> NxN by block pooling
    semN = np.zeros((N,N), dtype=np.uint8)
    zN   = np.full((N,N), np.nan, dtype=np.float32)

    block = FN / N
    for gy in range(N):
        y0b = int(round(gy * block))
        y1b = int(round((gy+1) * block))
        y1b = max(y1b, y0b+1)
        for gx in range(N):
            x0b = int(round(gx * block))
            x1b = int(round((gx+1) * block))
            x1b = max(x1b, x0b+1)
            sb = f_sem[y0b:y1b, x0b:x1b]
            zb = f_ztop[y0b:y1b, x0b:x1b]
            semN[gy,gx] = mode_ignore_zero(sb)
            # ztop robust: use p95 of finite z in block
            zb2 = zb[np.isfinite(zb) & (zb > -1e8)]
            if zb2.size > 0:
                zN[gy,gx] = float(np.percentile(zb2, 95))

    valid = (semN > 0) & np.isfinite(zN)
    z_floor = compute_floor_ref(semN, zN, floor_id)
    hrelN = np.full_like(zN, np.nan, dtype=np.float32)
    hrelN[valid] = (zN[valid] - z_floor).astype(np.float32)

    # render semantic grid as RGB
    rgbN = render_semantic_grid(semN)

    # prepare legend entries (present classes)
    present_ids = sorted([int(x) for x in np.unique(semN) if int(x) > 0])
    legend_items = []
    for cid in present_ids[:12]:  # avoid huge legend
        name = id_to_name.get(cid, f"id{cid}")
        col = np.array(color_for_id(cid), dtype=np.uint8) / 255.0
        legend_items.append((name, col))

    # ---- annotate largest regions (name + median height)
    # connected components on NxN for each class
    ann = []
    for cid in present_ids:
        if cid == floor_id:
            continue
        mask = (semN == cid).astype(np.uint8)
        ncc, lab = cv2.connectedComponents(mask, connectivity=4)
        for kcc in range(1, ncc):
            pts = np.argwhere(lab == kcc)
            if pts.shape[0] < args.min_region_cells:
                continue
            cy, cx = pts.mean(axis=0)
            # median height in region
            hs = hrelN[lab == kcc]
            hs = hs[np.isfinite(hs)]
            hmed = float(np.median(hs)) if hs.size > 0 else float("nan")
            ann.append((pts.shape[0], cid, cx, cy, hmed))
    ann.sort(reverse=True, key=lambda x: x[0])
    ann = ann[:int(args.annotate_topk)]

    # ---- draw: (a) topdown colored grid with legend + traj + pose + annotations
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1,2,1)

    # show as metric map
    extent = [-half, half, -half, half]  # x, y in meters
    ax.imshow(rgbN, origin="lower", extent=extent, interpolation="nearest")

    # grid lines
    for i in range(N+1):
        v = -half + i*cell_m
        ax.plot([v, v], [-half, half], linewidth=0.3, color="white", alpha=0.4)
        ax.plot([-half, half], [v, v], linewidth=0.3, color="white", alpha=0.4)

    # trajectory (local coords)
    ax.plot(traj_local[:,0], traj_local[:,1], color="white", linewidth=2.0, alpha=0.85)
    # UAV pose star at center
    ax.scatter([0],[0], marker="*", s=250, color="yellow", edgecolor="black", linewidth=1.0, zorder=5)

    # annotations
    for area, cid, cx, cy, hmed in ann:
        name = id_to_name.get(cid, f"id{cid}")
        # convert cell coords (cx,cy in [0..N)) to meters
        xm = -half + (cx + 0.5)*cell_m
        ym = -half + (cy + 0.5)*cell_m
        if np.isfinite(hmed):
            txt = f"{name}\n{hmed:.2f}m"
        else:
            txt = f"{name}"
        ax.text(
            xm, ym, txt, fontsize=8, color="black",
            bbox=dict(facecolor=(1, 1, 1, 0.65), edgecolor="none", boxstyle="round,pad=0.2"))


    ax.set_title(f"Top-down semantic map (local {L:.1f}m, N={N}, cell={cell_m:.2f}m)\ncenter t={t0}  fuse_k={len(use)}  pts_used={pts_used}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # legend
    if legend_items:
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=c, label=n) for n,c in legend_items]
        ax.legend(handles=patches, loc="lower left", fontsize=8, framealpha=0.85)

    # (b) matrix representation (paper-like)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(rgbN, origin="lower", extent=[0,N,0,N], interpolation="nearest")
    for i in range(N+1):
        ax2.plot([i,i],[0,N], color="black", linewidth=0.3, alpha=0.6)
        ax2.plot([0,N],[i,i], color="black", linewidth=0.3, alpha=0.6)

    # cell texts
    for y in range(N):
        for x in range(N):
            cid = int(semN[y,x])
            if cid == 0:
                txt = "0"
            else:
                txt = str(cid)
            if args.show_height_in_matrix and np.isfinite(hrelN[y,x]):
                txt = f"{cid}\n{hrelN[y,x]:.1f}"
            ax2.text(x+0.5, y+0.5, txt, ha="center", va="center",
                     fontsize=7, color="black")

    ax2.scatter([N/2],[N/2], marker="*", s=250, color="yellow", edgecolor="black", linewidth=1.0)
    ax2.set_xlim(0,N); ax2.set_ylim(0,N)
    ax2.set_title("Matrix representation (NxN)\n(cell text: semantic id [+ height])")
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, f"paper_style_localmap_t{t0}.png")
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    # also save raw grids for later step7
    np.save(os.path.join(args.out_dir, f"semN_t{t0}.npy"), semN)
    np.save(os.path.join(args.out_dir, f"hrelN_t{t0}.npy"), hrelN)

    meta = {
        "t_ns": int(t0),
        "local_size_m": L,
        "N": N,
        "cell_m": cell_m,
        "fine_cell_m": fine,
        "fuse_k": len(use),
        "ego_align": bool(args.ego_align),
        "yaw0_rad": float(yaw0),
        "z_floor_ref": float(z_floor),
        "pts_used": int(pts_used),
    }
    with open(os.path.join(args.out_dir, f"meta_t{t0}.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[Step6d] saved:", out_png)
    print("[Step6d] meta:", meta)


if __name__ == "__main__":
    main()


'''
版本 A（室内更合理：窗口 20m×20m，20×20 → 每格 1m）
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step6d_render_paper_style_maps.py \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --depth_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/depth_npy \
  --mask_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --stereo_meta /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/stereo_meta.json \
  --mav0_dir /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/paper_style_maps \
  --local_size_m 24 --N 24 --fuse_k 160 --ego_align
'''
#--fuse_k：从 120 提到 160（更完整），或降到 60（更清晰但不完整）

#--local_size_m：从 20 调到 30 或 40（覆盖更大，但每格更粗）

'''
版本 B（严格模仿论文“每格 5m”：local_size_m=100m，室内会很稀疏）
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step6d_render_paper_style_maps.py \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --depth_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/depth_npy \
  --mask_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --stereo_meta /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/stereo_meta.json \
  --mav0_dir /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/paper_style_maps \
  --local_size_m 100 --N 20 --fuse_k 200 --ego_align
'''