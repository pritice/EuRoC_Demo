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
# Color + overlay
# -------------------------
def color_for_id(i: int) -> Tuple[int,int,int]:
    return (int((i*37) % 255), int((i*17) % 255), int((i*67) % 255))

def semantic_rgb(sem: np.ndarray) -> np.ndarray:
    rgb = np.zeros((sem.shape[0], sem.shape[1], 3), dtype=np.uint8)
    for cid in np.unique(sem):
        cid = int(cid)
        if cid <= 0:
            continue
        rgb[sem == cid] = np.array(color_for_id(cid), dtype=np.uint8)
    return rgb

def overlay_on_gray(gray: np.ndarray, sem: np.ndarray, alpha=0.55) -> np.ndarray:
    if gray.ndim == 2:
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        base = gray.copy()
    col = semantic_rgb(sem)
    mask = (sem > 0)
    out = base.copy().astype(np.float32)
    out[mask] = (1-alpha)*out[mask] + alpha*col[mask].astype(np.float32)
    return out.clip(0,255).astype(np.uint8)


# -------------------------
# Local map fusion (paper-style)
# -------------------------
def mode_ignore_zero(block: np.ndarray) -> int:
    v = block.reshape(-1)
    v = v[v > 0]
    if v.size == 0:
        return 0
    bc = np.bincount(v.astype(np.int64))
    return int(bc.argmax())

def compute_floor_ref(semN: np.ndarray, zN: np.ndarray, floor_id: int) -> float:
    if floor_id > 0:
        m = (semN == floor_id) & np.isfinite(zN)
        if np.any(m):
            return float(np.median(zN[m]))
    m = np.isfinite(zN)
    if np.any(m):
        return float(np.percentile(zN[m], 10))
    return 0.0

def fuse_local_map(rows_use: List[Dict[str,Any]],
                   t_center: int,
                   idx_center: int,
                   depth_dir: str,
                   mask_dir: str,
                   map0x, map0y,
                   st: Dict[str,Any],
                   local_size_m: float,
                   N: int,
                   pix_step: int,
                   min_depth: float,
                   max_depth: float,
                   ego_align: bool,
                   floor_id: int):
    W,H = st["image_wh"]
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    Kr = P0[:3,:3]
    R0 = np.array(st["rectify"]["R0"], dtype=np.float64)
    R_rect2orig = R0.T
    T_B_C0 = np.array(st["T_B_C0"], dtype=np.float64)
    T_C0_B = np.linalg.inv(T_B_C0)

    r0 = rows_use[idx_center]
    p0 = np.array(r0["p_WB"], dtype=np.float64)
    yaw0 = yaw_from_q_WB(r0["q_WB_wxyz"])
    R_align = rot2d(-yaw0) if ego_align else np.eye(2, dtype=np.float64)

    L = float(local_size_m)
    cell_m = L / N
    half = L / 2.0

    fine = max(cell_m / 5.0, 0.05)
    FN = int(math.ceil(L / fine))
    fine = L / FN

    f_ztop = np.full((FN, FN), -np.inf, dtype=np.float32)
    f_sem  = np.zeros((FN, FN), dtype=np.uint8)

    ys = np.arange(0, H, pix_step, dtype=np.int32)
    xs = np.arange(0, W, pix_step, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.reshape(-1); yy = yy.reshape(-1)

    traj_xy = np.array([r["p_WB"][:2] for r in rows_use], dtype=np.float64)
    traj_local = (R_align @ (traj_xy - p0[:2]).T).T

    pts_used = 0

    for r in rows_use:
        t = int(r["t_ns"])
        dp = os.path.join(depth_dir, f"{t}.npy")
        mp = os.path.join(mask_dir, f"{t}.png")
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

        sem_r = cv2.remap(sem, map0x, map0y, interpolation=cv2.INTER_NEAREST)

        Z = depth[yy, xx].astype(np.float64)
        lb = sem_r[yy, xx].astype(np.int32)
        ok = np.isfinite(Z) & (Z > min_depth) & (Z < max_depth)
        if not np.any(ok):
            continue

        xxo = xx[ok].astype(np.float64)
        yyo = yy[ok].astype(np.float64)
        Zo  = Z[ok]
        lbo = lb[ok].astype(np.uint8)

        P_rect = backproject_rect(Kr, xxo, yyo, Zo)
        P_c0   = (R_rect2orig @ P_rect.T).T
        P_B    = apply_T(T_C0_B, P_c0)
        T_W_B  = make_T(quat_wxyz_to_R(r["q_WB_wxyz"]), r["p_WB"])
        P_W    = apply_T(T_W_B, P_B)

        XY = P_W[:, :2] - p0[:2]
        XY = (R_align @ XY.T).T

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

        # highest-z wins per fine-cell
        for i in range(ix.shape[0]):
            x = ix[i]; y = iy[i]
            if ZW[i] > f_ztop[y,x]:
                f_ztop[y,x] = ZW[i]
                f_sem[y,x]  = LB[i]

    # pool FN->N
    semN = np.zeros((N,N), dtype=np.uint8)
    zN   = np.full((N,N), np.nan, dtype=np.float32)
    block = FN / N
    for gy in range(N):
        y0b = int(round(gy * block)); y1b = int(round((gy+1) * block))
        y1b = max(y1b, y0b+1)
        for gx in range(N):
            x0b = int(round(gx * block)); x1b = int(round((gx+1) * block))
            x1b = max(x1b, x0b+1)
            sb = f_sem[y0b:y1b, x0b:x1b]
            zb = f_ztop[y0b:y1b, x0b:x1b]
            semN[gy,gx] = mode_ignore_zero(sb)
            zb2 = zb[np.isfinite(zb) & (zb > -1e8)]
            if zb2.size > 0:
                zN[gy,gx] = float(np.percentile(zb2, 95))

    valid = (semN > 0) & np.isfinite(zN)
    z_floor = compute_floor_ref(semN, zN, floor_id)
    hrelN = np.full_like(zN, np.nan, dtype=np.float32)
    hrelN[valid] = (zN[valid] - z_floor).astype(np.float32)

    meta = {
        "t_ns": int(t_center),
        "yaw0_rad": float(yaw0),
        "local_size_m": float(local_size_m),
        "N": int(N),
        "cell_m": float(cell_m),
        "fine_cell_m": float(fine),
        "pts_used": int(pts_used),
        "z_floor_ref": float(z_floor),
        "traj_local": traj_local,
    }
    return semN, hrelN, meta


# -------------------------
# Metrics for paper
# -------------------------
def isolated_cell_ratio(semN: np.ndarray) -> float:
    """a cell is isolated if its 8-neighborhood majority differs from itself (ignore zeros)"""
    H,W = semN.shape
    pad = np.pad(semN, ((1,1),(1,1)), mode="edge")
    iso = 0; cnt = 0
    for y in range(H):
        for x in range(W):
            c = int(semN[y,x])
            if c == 0: 
                continue
            nb = pad[y:y+3, x:x+3].reshape(-1)
            nb = nb[nb > 0]
            if nb.size == 0:
                continue
            bc = np.bincount(nb.astype(np.int64))
            maj = int(bc.argmax())
            cnt += 1
            if maj != c:
                iso += 1
    return float(iso / max(cnt, 1))

def component_count_total(semN: np.ndarray) -> int:
    """sum of connected components over all classes (excluding 0) on NxN grid"""
    total = 0
    ids = [int(x) for x in np.unique(semN) if int(x) > 0]
    for cid in ids:
        m = (semN == cid).astype(np.uint8)
        ncc, _ = cv2.connectedComponents(m, connectivity=4)
        total += (ncc - 1)
    return int(total)

def small_component_cells_ratio(semN: np.ndarray, thr_cells=2) -> float:
    """ratio of cells belonging to components smaller than thr_cells"""
    H,W = semN.shape
    tot = 0; small = 0
    ids = [int(x) for x in np.unique(semN) if int(x) > 0]
    for cid in ids:
        m = (semN == cid).astype(np.uint8)
        ncc, lab = cv2.connectedComponents(m, connectivity=4)
        for k in range(1, ncc):
            pts = np.argwhere(lab == k)
            tot += pts.shape[0]
            if pts.shape[0] <= thr_cells:
                small += pts.shape[0]
    return float(small / max(tot, 1))


# -------------------------
# Render helpers (paper-like)
# -------------------------
def render_sem_grid_rgb(semN: np.ndarray) -> np.ndarray:
    H,W = semN.shape
    rgb = np.zeros((H,W,3), dtype=np.uint8)
    for cid in np.unique(semN):
        cid = int(cid)
        if cid <= 0:
            continue
        rgb[semN == cid] = np.array(color_for_id(cid), dtype=np.uint8)
    return rgb

def draw_topdown(ax, semN, meta, id_to_name, legend_max=10):
    L = meta["local_size_m"]; N = meta["N"]; cell_m = meta["cell_m"]
    half = L/2.0
    rgbN = render_sem_grid_rgb(semN)
    ax.imshow(rgbN, origin="lower", extent=[-half, half, -half, half], interpolation="nearest")
    # grid lines
    for i in range(N+1):
        v = -half + i*cell_m
        ax.plot([v, v], [-half, half], linewidth=0.3, color="white", alpha=0.35)
        ax.plot([-half, half], [v, v], linewidth=0.3, color="white", alpha=0.35)
    # traj + pose
    traj_local = meta["traj_local"]
    ax.plot(traj_local[:,0], traj_local[:,1], color="white", linewidth=2.0, alpha=0.85)
    ax.scatter([0],[0], marker="*", s=220, color="yellow", edgecolor="black", linewidth=1.0, zorder=5)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # legend (present classes)
    present = sorted([int(x) for x in np.unique(semN) if int(x) > 0])
    present = present[:legend_max]
    if present:
        import matplotlib.patches as mpatches
        patches = []
        for cid in present:
            name = id_to_name.get(cid, f"id{cid}")
            col = np.array(color_for_id(cid), dtype=np.uint8)/255.0
            patches.append(mpatches.Patch(color=col, label=name))
        ax.legend(handles=patches, loc="lower left", fontsize=7, framealpha=0.85)

def draw_matrix(ax, semN, show_text=True):
    N = semN.shape[0]
    rgbN = render_sem_grid_rgb(semN)
    ax.imshow(rgbN, origin="lower", extent=[0,N,0,N], interpolation="nearest")
    for i in range(N+1):
        ax.plot([i,i],[0,N], color="black", linewidth=0.3, alpha=0.6)
        ax.plot([0,N],[i,i], color="black", linewidth=0.3, alpha=0.6)
    if show_text:
        for y in range(N):
            for x in range(N):
                ax.text(x+0.5, y+0.5, str(int(semN[y,x])),
                        ha="center", va="center", fontsize=6, color="black")
    ax.scatter([N/2],[N/2], marker="*", s=220, color="yellow", edgecolor="black", linewidth=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0,N); ax.set_ylim(0,N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--mask_dir_raw", required=True)
    ap.add_argument("--mask_dir_clean", required=True)
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--stereo_meta", required=True)
    ap.add_argument("--mav0_dir", default="/root/autodl-tmp/sam-3d-objects/inputs/mav0")

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--t_ns", type=str, default="", help="empty -> use last frame")
    ap.add_argument("--fuse_k", type=int, default=120)
    ap.add_argument("--local_size_m", type=float, default=20.0)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--pix_step", type=int, default=3)
    ap.add_argument("--min_depth", type=float, default=0.10)
    ap.add_argument("--max_depth", type=float, default=12.0)
    ap.add_argument("--ego_align", action="store_true")

    ap.add_argument("--overlay_alpha", type=float, default=0.55)
    ap.add_argument("--matrix_text", action="store_true", help="draw semantic ids in each cell")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    rows = read_jsonl(args.keyframes)
    labels = read_json(args.labels_json)
    class_to_id = labels.get("class_to_id", {})
    id_to_name = {int(v): k for k,v in class_to_id.items()}
    id_to_name[0] = "unknown"
    floor_id = int(class_to_id.get("floor", -1))

    if args.t_ns:
        t0 = int(args.t_ns)
        idx0 = None
        for i,r in enumerate(rows):
            if int(r["t_ns"]) == t0:
                idx0 = i; break
        if idx0 is None:
            raise ValueError(f"t_ns not found: {t0}")
    else:
        idx0 = len(rows) - 1
        t0 = int(rows[idx0]["t_ns"])

    i0 = max(0, idx0 - args.fuse_k + 1)
    use = rows[i0:idx0+1]

    map0x, map0y, st = build_rectify_map(args.mav0_dir, args.stereo_meta)
    W,H = st["image_wh"]

    # Load center RGB (raw)
    cam0_path = rows[idx0].get("cam0_path", "")
    if cam0_path and os.path.isfile(cam0_path):
        rgb0 = cv2.imread(cam0_path, cv2.IMREAD_GRAYSCALE)
        if rgb0 is None:
            rgb0 = np.zeros((H,W), dtype=np.uint8)
    else:
        rgb0 = np.zeros((H,W), dtype=np.uint8)

    # Load center masks and rectify for overlay consistency
    mp_raw = os.path.join(args.mask_dir_raw, f"{t0}.png")
    mp_cln = os.path.join(args.mask_dir_clean, f"{t0}.png")
    m_raw = cv2.imread(mp_raw, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mp_raw) else np.zeros((H,W), np.uint8)
    m_cln = cv2.imread(mp_cln, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mp_cln) else np.zeros((H,W), np.uint8)
    if m_raw.shape != (H,W): m_raw = cv2.resize(m_raw, (W,H), interpolation=cv2.INTER_NEAREST)
    if m_cln.shape != (H,W): m_cln = cv2.resize(m_cln, (W,H), interpolation=cv2.INTER_NEAREST)
    m_raw_r = cv2.remap(m_raw, map0x, map0y, interpolation=cv2.INTER_NEAREST)
    m_cln_r = cv2.remap(m_cln, map0x, map0y, interpolation=cv2.INTER_NEAREST)

    # Overlays
    ov_raw = overlay_on_gray(rgb0, m_raw_r, alpha=args.overlay_alpha)
    ov_cln = overlay_on_gray(rgb0, m_cln_r, alpha=args.overlay_alpha)

    # Local maps (raw vs clean)
    sem_raw, h_raw, meta_raw = fuse_local_map(
        use, t0, len(use)-1, args.depth_dir, args.mask_dir_raw, map0x, map0y, st,
        args.local_size_m, args.N, args.pix_step, args.min_depth, args.max_depth, args.ego_align, floor_id
    )
    sem_cln, h_cln, meta_cln = fuse_local_map(
        use, t0, len(use)-1, args.depth_dir, args.mask_dir_clean, map0x, map0y, st,
        args.local_size_m, args.N, args.pix_step, args.min_depth, args.max_depth, args.ego_align, floor_id
    )

    # Metrics
    stats = {
        "t_ns": int(t0),
        "fuse_k": int(len(use)),
        "local_size_m": float(args.local_size_m),
        "N": int(args.N),
        "pix_step": int(args.pix_step),
        "raw": {
            "pts_used": int(meta_raw["pts_used"]),
            "isolated_cell_ratio": isolated_cell_ratio(sem_raw),
            "component_count_total": component_count_total(sem_raw),
            "small_component_cells_ratio": small_component_cells_ratio(sem_raw, thr_cells=2),
        },
        "clean": {
            "pts_used": int(meta_cln["pts_used"]),
            "isolated_cell_ratio": isolated_cell_ratio(sem_cln),
            "component_count_total": component_count_total(sem_cln),
            "small_component_cells_ratio": small_component_cells_ratio(sem_cln, thr_cells=2),
        }
    }
    with open(os.path.join(args.out_dir, f"compare_stats_t{t0}.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Figure: 2 rows (raw/clean) x 3 cols (overlay/topdown/matrix)
    fig = plt.figure(figsize=(18, 10))

    ax1 = fig.add_subplot(2,3,1)
    ax1.imshow(ov_raw)
    ax1.set_title("RAW: Rectified RGB + Semantic overlay")
    ax1.axis("off")

    ax2 = fig.add_subplot(2,3,2)
    draw_topdown(ax2, sem_raw, meta_raw, id_to_name)
    ax2.set_title(
        f"RAW Top-down (isolated={stats['raw']['isolated_cell_ratio']:.3f}, "
        f"cc={stats['raw']['component_count_total']}, "
        f"small={stats['raw']['small_component_cells_ratio']:.3f})"
    )

    ax3 = fig.add_subplot(2,3,3)
    draw_matrix(ax3, sem_raw, show_text=bool(args.matrix_text))
    ax3.set_title("RAW Matrix (20x20)")

    ax4 = fig.add_subplot(2,3,4)
    ax4.imshow(ov_cln)
    ax4.set_title("CLEAN: Rectified RGB + Semantic overlay")
    ax4.axis("off")

    ax5 = fig.add_subplot(2,3,5)
    draw_topdown(ax5, sem_cln, meta_cln, id_to_name)
    ax5.set_title(
        f"CLEAN Top-down (isolated={stats['clean']['isolated_cell_ratio']:.3f}, "
        f"cc={stats['clean']['component_count_total']}, "
        f"small={stats['clean']['small_component_cells_ratio']:.3f})"
    )

    ax6 = fig.add_subplot(2,3,6)
    draw_matrix(ax6, sem_cln, show_text=bool(args.matrix_text))
    ax6.set_title("CLEAN Matrix (20x20)")

    fig.suptitle(f"Semantic Map Refinement Comparison (t={t0}, fuse_k={len(use)}, local={args.local_size_m}m, ego_align={args.ego_align})", fontsize=14)
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, f"final_comparison_t{t0}.png")
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    # save grids for paper appendix / later pipeline
    np.save(os.path.join(args.out_dir, f"sem_raw_t{t0}.npy"), sem_raw)
    np.save(os.path.join(args.out_dir, f"sem_clean_t{t0}.npy"), sem_cln)

    print("[Step6e] saved:", out_png)
    print("[Step6e] stats:", stats)


if __name__ == "__main__":
    main()


'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step6e_make_final_comparison_figure.py \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --depth_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/depth_npy \
  --mask_dir_raw /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks \
  --mask_dir_clean /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks_clean \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --stereo_meta /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/stereo_meta.json \
  --mav0_dir /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/final_compare \
  --local_size_m 24 --N 24 --fuse_k 160 --ego_align
'''