# -*- coding: utf-8 -*-
import os, json, math, argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None


# -------------------------
# IO helpers
# -------------------------
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -------------------------
# Geometry helpers
# -------------------------
def quat_wxyz_to_R(q):
    qw, qx, qy, qz = [float(x) for x in q]
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) + 1e-12
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array(t, dtype=np.float64)
    return T

def apply_T(T, pts):
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (T @ pts_h.T).T
    return out[:, :3]

def rotz(theta):
    c = math.cos(theta); s = math.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)

def backproject_rect(K, xx, yy, Z):
    X = (xx - K[0,2]) / K[0,0] * Z
    Y = (yy - K[1,2]) / K[1,1] * Z
    return np.stack([X, Y, Z], axis=1)

def compute_camera_yaw(T_W_B, T_C0_B, R_rect2orig):
    # world heading of rectified camera forward (+Z), projected to XY
    T_B_C0 = np.linalg.inv(T_C0_B)
    R_WB = T_W_B[:3, :3]
    R_BC0orig = T_B_C0[:3, :3]
    R_orig2rect = np.linalg.inv(R_rect2orig)
    R_WC0rect = R_WB @ R_BC0orig @ R_orig2rect
    fwd = R_WC0rect @ np.array([0, 0, 1.0], dtype=np.float64)
    return float(math.atan2(fwd[1], fwd[0]))


# -------------------------
# Rectify map (mask/image alignment)
# -------------------------
def load_cam0_KD(mav0_dir: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int]]:
    if yaml is None:
        raise RuntimeError("pyyaml missing. Run: pip install pyyaml")
    ypath = os.path.join(mav0_dir, "cam0", "sensor.yaml")
    with open(ypath, "r") as f:
        y = yaml.safe_load(f)
    fx, fy, cx, cy = [float(x) for x in y["intrinsics"]]
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    D = np.array(y.get("distortion_coeffs", []), dtype=np.float64).reshape(-1,1)
    W, H = int(y["resolution"][0]), int(y["resolution"][1])
    return K, D, (W,H)

def build_rectify_map(mav0_dir: str, stereo_meta_path: str):
    st = read_json(stereo_meta_path)
    W, H = st["image_wh"]
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    R0 = np.array(st["rectify"]["R0"], dtype=np.float64)
    K0, D0, wh = load_cam0_KD(mav0_dir)
    assert wh == (W, H), f"resolution mismatch: sensor.yaml {wh} vs stereo_meta {(W,H)}"
    map0x, map0y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (W, H), cv2.CV_32FC1)
    return map0x, map0y, st


# -------------------------
# Visual helpers
# -------------------------
def color_for_id(i: int) -> Tuple[int,int,int]:
    # deterministic vivid-ish
    return (int((i*37) % 255), int((i*17) % 255), int((i*67) % 255))

def overlay_semantic(rgb_u8: np.ndarray, sem_u8: np.ndarray, id_to_name: Dict[int,str], alpha=0.55):
    out = rgb_u8.copy()
    H,W = sem_u8.shape
    for cid in sorted(id_to_name.keys()):
        if cid <= 0: 
            continue
        m = (sem_u8 == cid)
        if not m.any():
            continue
        col = np.array(color_for_id(cid), dtype=np.uint8).reshape(1,1,3)
        out[m] = (out[m].astype(np.float32)*(1-alpha) + col.astype(np.float32)*alpha).astype(np.uint8)
    return out

def depth_to_vis(depth_m: np.ndarray) -> np.ndarray:
    d = depth_m.copy()
    ok = np.isfinite(d) & (d > 0)
    vis = np.zeros_like(d, dtype=np.float32)
    if ok.any():
        v = d[ok]
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        vis[ok] = (np.clip(d[ok], lo, hi) - lo) / (hi - lo + 1e-6)
    return (vis * 255).astype(np.uint8)

def draw_instances_on_topdown(ax, insts: List[Dict[str,Any]], pW: np.ndarray, yaw: float,
                              local_range_m: float, res: float):
    """
    Draw instance centers and their XY AABB projected to ego frame.
    """
    R_EW = rotz(-yaw)
    half = local_range_m / 2.0

    for ins in insts:
        c = np.array(ins["center_W"], dtype=np.float64)
        mn = np.array(ins["aabb_min_W"], dtype=np.float64)
        mx = np.array(ins["aabb_max_W"], dtype=np.float64)

        # center in ego
        ce = (R_EW @ (c - pW).reshape(3,1)).reshape(3)
        if not (-half <= ce[0] <= half and -half <= ce[1] <= half):
            continue

        # bbox corners in world -> ego
        corners = np.array([
            [mn[0], mn[1], c[2]],
            [mn[0], mx[1], c[2]],
            [mx[0], mx[1], c[2]],
            [mx[0], mn[1], c[2]],
            [mn[0], mn[1], c[2]],
        ], dtype=np.float64)
        corners_e = (R_EW @ (corners - pW).T).T

        ax.plot(corners_e[:,0], corners_e[:,1], linewidth=1.0)
        ax.scatter([ce[0]], [ce[1]], s=10)
        ax.text(ce[0], ce[1], f'{ins.get("label","")}', fontsize=7)

    # ego axes
    ax.scatter([0],[0], s=30, marker="+")
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("ego X (m)"); ax.set_ylabel("ego Y (m)")
    ax.grid(True, linewidth=0.3)


def build_singleframe_topdown(sem_rect: np.ndarray,
                              depth_rect: np.ndarray,
                              r: Dict[str,Any],
                              st: Dict[str,Any],
                              pix_step: int,
                              min_depth: float,
                              max_depth: float,
                              local_range_m: float,
                              res: float):
    """
    只用当前帧(depth+semantic)生成一张 ego topdown（不累积），用于对齐检查。
    """
    W,H = st["image_wh"]
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    Kr = P0[:3,:3]
    R0 = np.array(st["rectify"]["R0"], dtype=np.float64)
    R_rect2orig = R0.T
    T_B_C0 = np.array(st["T_B_C0"], dtype=np.float64)
    T_C0_B = np.linalg.inv(T_B_C0)

    # pose
    T_W_B = make_T(quat_wxyz_to_R(r["q_WB_wxyz"]), r["p_WB"])
    pW = np.array(r["p_WB"], dtype=np.float64)

    # sample pixels
    ys = np.arange(0, H, pix_step, dtype=np.int32)
    xs = np.arange(0, W, pix_step, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.reshape(-1); yy = yy.reshape(-1)

    Z = depth_rect[yy, xx].astype(np.float64)
    lab = sem_rect[yy, xx].astype(np.int32)

    ok = np.isfinite(Z) & (Z > min_depth) & (Z < max_depth)
    xx = xx[ok].astype(np.float64)
    yy = yy[ok].astype(np.float64)
    Z = Z[ok]
    lab = lab[ok]

    if Z.size < 500:
        return None, None, None

    P_rect = backproject_rect(Kr, xx, yy, Z)
    P_orig = (R_rect2orig @ P_rect.T).T
    P_B = apply_T(T_C0_B, P_orig)
    P_W = apply_T(T_W_B, P_B)

    yaw = compute_camera_yaw(T_W_B, T_C0_B, R_rect2orig)
    R_EW = rotz(-yaw)

    half = local_range_m / 2.0
    grid = int(round(local_range_m / res))
    sem = np.zeros((grid, grid), dtype=np.uint8)
    ztop = np.full((grid, grid), -np.inf, dtype=np.float32)

    # world -> ego -> bin
    V = (R_EW @ (P_W - pW).T).T  # (N,3)
    inb = (V[:,0] >= -half) & (V[:,0] < half) & (V[:,1] >= -half) & (V[:,1] < half)
    V = V[inb]
    lab = lab[inb]

    ix = ((V[:,0] + half) / res).astype(np.int32)
    iy = ((V[:,1] + half) / res).astype(np.int32)
    for i in range(V.shape[0]):
        x = ix[i]; y = iy[i]
        if 0 <= x < grid and 0 <= y < grid:
            if V[i,2] > ztop[y,x]:
                ztop[y,x] = V[i,2]
                sem[y,x] = np.uint8(lab[i] if lab[i] > 0 else 0)

    return sem, ztop, yaw


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mav0_dir", required=True)
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--stereo_meta", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--semantic_mask_dir", required=True)
    ap.add_argument("--instances_rect_dir", required=True)
    ap.add_argument("--instance_metrics_dir", required=True, help=".../per_frame")
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--t_ns", type=str, default="", help="timestamp to visualize. empty -> pick first")
    ap.add_argument("--montage", type=int, default=0, help="if >0, create montage with N frames (evenly spaced)")
    ap.add_argument("--pix_step", type=int, default=3)
    ap.add_argument("--min_depth", type=float, default=0.05)
    ap.add_argument("--max_depth", type=float, default=12.0)
    ap.add_argument("--local_range_m", type=float, default=8.0)
    ap.add_argument("--res", type=float, default=0.10)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    rows = read_jsonl(args.keyframes)
    map0x, map0y, st = build_rectify_map(args.mav0_dir, args.stereo_meta)
    W,H = st["image_wh"]

    labels = read_json(args.labels_json)
    class_to_id = labels.get("class_to_id", {})
    id_to_name = {int(v): k for k, v in class_to_id.items()}
    id_to_name[0] = "unknown"

    def render_one(r: Dict[str,Any]) -> Optional[str]:
        t = int(r["t_ns"])
        img_path = r["cam0_path"]
        depth_path = os.path.join(args.depth_dir, f"{t}.npy")
        sem_path = os.path.join(args.semantic_mask_dir, f"{t}.png")
        inst_path = os.path.join(args.instance_metrics_dir, f"{t}.json")

        if not (os.path.isfile(img_path) and os.path.isfile(depth_path) and os.path.isfile(sem_path) and os.path.isfile(inst_path)):
            return None

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.resize(img, (W,H), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (H,W):
            return None

        sem = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        if sem is None:
            return None
        sem = cv2.resize(sem, (W,H), interpolation=cv2.INTER_NEAREST)

        # rectify image + semantic to match depth (depth is rectified)
        img_rect = cv2.remap(img_rgb, map0x, map0y, interpolation=cv2.INTER_LINEAR)
        sem_rect = cv2.remap(sem, map0x, map0y, interpolation=cv2.INTER_NEAREST)

        sem_overlay = overlay_semantic(img_rect, sem_rect, id_to_name, alpha=0.55)
        depth_vis = depth_to_vis(depth)

        # build single-frame topdown from (depth+semantic) in ego frame
        sem_td, z_td, yaw = build_singleframe_topdown(
            sem_rect=sem_rect, depth_rect=depth, r=r, st=st,
            pix_step=args.pix_step, min_depth=args.min_depth, max_depth=args.max_depth,
            local_range_m=args.local_range_m, res=args.res
        )

        inst = read_json(inst_path)
        insts = inst.get("instances", [])
        pW = np.array(r["p_WB"], dtype=np.float64)

        # plot
        fig = plt.figure(figsize=(13, 9))

        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(img_rgb)
        ax1.set_title(f"RGB (raw) t={t}")
        ax1.axis("off")

        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(sem_overlay)
        ax2.set_title("Rectified RGB + Semantic overlay")
        ax2.axis("off")

        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(depth_vis, cmap="gray")
        ax3.set_title("Rectified Depth (percentile normalized)")
        ax3.axis("off")

        ax4 = fig.add_subplot(2,2,4)
        if sem_td is None:
            ax4.text(0.1, 0.5, "Topdown build failed (too few depth points)", fontsize=12)
            ax4.axis("off")
        else:
            # show semantic topdown as colored image
            grid = sem_td.shape[0]
            rgb_td = np.zeros((grid, grid, 3), dtype=np.uint8)
            for cid, name in id_to_name.items():
                if cid <= 0: 
                    continue
                rgb_td[sem_td == cid] = np.array(color_for_id(cid), dtype=np.uint8)
            ax4.imshow(rgb_td, origin="lower",
                       extent=[-args.local_range_m/2, args.local_range_m/2,
                               -args.local_range_m/2, args.local_range_m/2])
            ax4.set_title(f"Ego topdown (single-frame) yaw={yaw:.2f} rad")
            draw_instances_on_topdown(ax4, insts, pW, yaw, args.local_range_m, args.res)

        out_path = os.path.join(args.out_dir, f"viz_{t}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # choose frame(s)
    if args.montage and args.montage > 0:
        idxs = np.linspace(0, len(rows)-1, args.montage, dtype=int).tolist()
        outs = []
        for i in idxs:
            p = render_one(rows[i])
            if p:
                outs.append(p)
        print("[Step6] montage frames:", len(outs))
        print("Saved in:", args.out_dir)
        return

    # single frame
    if args.t_ns:
        t = int(args.t_ns)
        rr = None
        for r in rows:
            if int(r["t_ns"]) == t:
                rr = r; break
        if rr is None:
            raise ValueError(f"t_ns not found in keyframes: {t}")
        outp = render_one(rr)
    else:
        outp = render_one(rows[0])

    print("[Step6] saved:", outp)


if __name__ == "__main__":
    main()

'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step6_visualize_scene.py \
  --mav0_dir /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --stereo_meta /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/stereo_meta.json \
  --depth_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/depth_npy \
  --semantic_mask_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks \
  --instances_rect_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/instances_rect \
  --instance_metrics_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/instance_metrics_200/per_frame \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/viz_checks \
  --montage 6
'''