import os, json, math, argparse
from typing import Dict, Any, Tuple, Optional

import numpy as np
import cv2
from tqdm import tqdm

try:
    import yaml
except Exception:
    yaml = None

# Absolute paths (edit for your server)
DEFAULT_MAV0_DIR = "/root/autodl-tmp/sam-3d-objects/inputs/mav0"
DEFAULT_KEYFRAMES = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl"
DEFAULT_DEPTH_DIR = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/depth_npy"
DEFAULT_MASK_DIR = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks"
DEFAULT_STEREO_META = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/stereo_meta.json"
DEFAULT_LABELS_JSON = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json"
DEFAULT_OUT_DIR = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stmr_maps_dino_sam2_200_v3"


def read_jsonl(path: str):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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
    # rectified pinhole back-projection
    X = (xx - K[0, 2]) / K[0, 0] * Z
    Y = (yy - K[1, 2]) / K[1, 1] * Z
    return np.stack([X, Y, Z], axis=1)


def compute_camera_yaw(T_W_B, T_C0orig_to_B, R_rect2orig):
    """
    yaw: world-frame heading of rectified camera's +Z axis projected on XY plane
    """
    T_B_C0orig = np.linalg.inv(T_C0orig_to_B)
    R_WB = T_W_B[:3, :3]
    R_BC0orig = T_B_C0orig[:3, :3]
    R_orig2rect = np.linalg.inv(R_rect2orig)
    R_WC0rect = R_WB @ R_BC0orig @ R_orig2rect
    fwd = R_WC0rect @ np.array([0, 0, 1.0], dtype=np.float64)  # camera forward
    return float(math.atan2(fwd[1], fwd[0]))


def maxpool_block(arr, oh, ow, mode="max"):
    H, W = arr.shape
    assert H % oh == 0 and W % ow == 0
    sh = H // oh
    sw = W // ow
    x = arr.reshape(oh, sh, ow, sw)
    if mode == "max":
        return x.max(axis=(1, 3))
    else:
        return x.mean(axis=(1, 3))


def argmax_label_hist(hist: dict) -> int:
    best_l = 0
    best_c = -1
    for l, c in hist.items():
        l = int(l)
        if l <= 0:
            continue
        if c > best_c:
            best_c = c
            best_l = l
    return int(best_l)


def load_cam0_KD_from_euroc(mav0_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read cam0/sensor.yaml:
      intrinsics: [fx, fy, cx, cy]
      distortion_coeffs: [k1,k2,p1,p2(,k3...)]
    """
    if yaml is None:
        raise RuntimeError("pyyaml not installed. Run: pip install pyyaml")

    ypath = os.path.join(mav0_dir, "cam0", "sensor.yaml")
    with open(ypath, "r") as f:
        y = yaml.safe_load(f)

    fx, fy, cx, cy = [float(x) for x in y["intrinsics"]]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    dc = y.get("distortion_coeffs", [])
    D = np.array(dc, dtype=np.float64).reshape(-1, 1)
    return K, D


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--keyframes", default=DEFAULT_KEYFRAMES)
    ap.add_argument("--depth_dir", default=DEFAULT_DEPTH_DIR)
    ap.add_argument("--mask_dir", default=DEFAULT_MASK_DIR)
    ap.add_argument("--stereo_meta", default=DEFAULT_STEREO_META)
    ap.add_argument("--labels_json", default=DEFAULT_LABELS_JSON)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)

    # for mask-depth alignment (recommended)
    ap.add_argument("--mav0_dir", default=DEFAULT_MAV0_DIR,
                    help="EuRoC mav0 dir to read cam0 K/D for mask rectification")
    ap.add_argument("--mask_already_rectified", action="store_true",
                    help="If true, skip remap(mask)->rectified. Use only if your Step3 ran on rectified images.")

    # geometry/map params (EuRoC indoor friendly defaults)
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--pix_step", type=int, default=3)
    ap.add_argument("--min_depth", type=float, default=0.05)
    ap.add_argument("--depth_trunc", type=float, default=12.0)

    ap.add_argument("--local_range_m", type=float, default=8.0)
    ap.add_argument("--res", type=float, default=0.10)
    ap.add_argument("--stmr_hw", type=int, nargs=2, default=[20, 20])
    ap.add_argument("--min_voxel_count", type=int, default=2)
    ap.add_argument("--min_depth_pts", type=int, default=300)

    # optional: prevent “full map” by keeping only recent voxels
    ap.add_argument("--keep_recent", type=int, default=0,
                    help="keep only voxels seen in last N frames (0=keep all)")
    ap.add_argument("--prune_every", type=int, default=10)

    # optional: height gating relative to body z (to reduce ceiling/wall smear)
    ap.add_argument("--zrel_min", type=float, default=-1.5, help="relative z min (meters) w.r.t body z")
    ap.add_argument("--zrel_max", type=float, default=3.0, help="relative z max (meters) w.r.t body z")
    ap.add_argument("--drop_unknown_depth", action="store_true", help="ignore depth points with label=0")
    ap.add_argument("--min_mask_depth_overlap", type=float, default=0.02,
                    help="skip frame if overlap(mask, valid_depth)/mask_pixels < this (0 to disable)")
    ap.add_argument("--min_mask_pixels", type=int, default=200,
                    help="minimum mask pixels to enable overlap check")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_jsonl(args.keyframes)
    st = read_json(args.stereo_meta)
    lbl = read_json(args.labels_json)

    W, H = st["image_wh"]

    # rectification params (depth is in rectified coordinates)
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    K0r = P0[:3, :3]  # rectified intrinsics
    R0 = np.array(st["rectify"]["R0"], dtype=np.float64)
    R_rect2orig = R0.T

    # extrinsic: body -> cam0 (original camera frame)
    T_B_C0 = np.array(st["T_B_C0"], dtype=np.float64)
    T_C0_B = np.linalg.inv(T_B_C0)
    T_C0orig_to_B = T_C0_B

    # build remap for mask if needed
    map0x = map0y = None
    mask_rectify_used = False
    if (not args.mask_already_rectified):
        if args.mav0_dir:
            K0, D0 = load_cam0_KD_from_euroc(args.mav0_dir)
            map0x, map0y = cv2.initUndistortRectifyMap(
                K0, D0, R0, P0, (W, H), cv2.CV_32FC1
            )
            mask_rectify_used = True
        else:
            print("[WARN] mask_already_rectified is False but --mav0_dir not provided. "
                  "Mask will NOT be rectified -> may misalign with rectified depth.")

    # local map sizing
    oh, ow = args.stmr_hw
    grid_w = int(round(args.local_range_m / args.res))
    grid_h = int(round(args.local_range_m / args.res))
    if grid_h % oh != 0:
        grid_h = int(math.ceil(grid_h / oh) * oh)
    if grid_w % ow != 0:
        grid_w = int(math.ceil(grid_w / ow) * ow)

    range_x = grid_w * args.res
    range_y = grid_h * args.res
    half_x = range_x / 2.0
    half_y = range_y / 2.0

    stmr_dir = os.path.join(args.out_dir, f"stmr_{oh}x{ow}")
    vis_dir = os.path.join(args.out_dir, "topdown_vis")
    os.makedirs(stmr_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    stats_path = os.path.join(args.out_dir, "frame_stats.jsonl")
    if os.path.isfile(stats_path):
        os.remove(stats_path)

    voxel = float(args.voxel)

    # global voxel accumulators
    g_count: Dict[Tuple[int, int, int], int] = {}
    g_maxz: Dict[Tuple[int, int, int], float] = {}
    g_hist: Dict[Tuple[int, int, int], Dict[int, int]] = {}
    g_last: Dict[Tuple[int, int, int], int] = {}  # last seen frame index (for pruning)

    def get_T_W_B(r):
        R = quat_wxyz_to_R(r["q_WB_wxyz"])
        return make_T(R, r["p_WB"])

    # precompute sampling grid
    ys = np.arange(0, H, args.pix_step, dtype=np.int32)
    xs = np.arange(0, W, args.pix_step, dtype=np.int32)
    xx0, yy0 = np.meshgrid(xs, ys)
    xx0 = xx0.reshape(-1)
    yy0 = yy0.reshape(-1)

    processed = 0

    for fi, r in enumerate(tqdm(rows, desc="[Step4-v3] accumulate")):
        t = int(r["t_ns"])
        dpath = os.path.join(args.depth_dir, f"{t}.npy")
        mpath = os.path.join(args.mask_dir, f"{t}.png")
        if not os.path.isfile(dpath) or not os.path.isfile(mpath):
            continue

        depth = np.load(dpath).astype(np.float32)
        if depth.shape != (H, W):
            append_jsonl(stats_path, {"t_ns": t, "error": f"depth_shape={depth.shape} expected={(H,W)}"})
            continue

        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            append_jsonl(stats_path, {"t_ns": t, "error": "mask_read_failed"})
            continue
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # IMPORTANT: align mask to rectified frame if depth is rectified
        if (not args.mask_already_rectified) and (map0x is not None):
            mask = cv2.remap(mask, map0x, map0y, interpolation=cv2.INTER_NEAREST)

        # mask-depth overlap sanity check
        if args.min_mask_depth_overlap > 0:
            mask_bin = (mask > 0)
            mask_pixels = int(mask_bin.sum())
            if mask_pixels >= int(args.min_mask_pixels):
                depth_valid_full = (depth > args.min_depth) & (depth < args.depth_trunc) & np.isfinite(depth)
                overlap = int((mask_bin & depth_valid_full).sum())
                overlap_ratio = float(overlap / max(mask_pixels, 1))
                if overlap_ratio < float(args.min_mask_depth_overlap):
                    append_jsonl(stats_path, {
                        "t_ns": t,
                        "mask_pixels": mask_pixels,
                        "mask_depth_overlap": overlap,
                        "mask_depth_overlap_ratio": overlap_ratio,
                        "skipped": "low_mask_depth_overlap"
                    })
                    continue

        # sample
        Z = depth[yy0, xx0].astype(np.float64)
        lab = mask[yy0, xx0].astype(np.int32)

        valid_depth = (Z > args.min_depth) & (Z < args.depth_trunc) & np.isfinite(Z)
        if not np.any(valid_depth):
            append_jsonl(stats_path, {
                "t_ns": t,
                "depth_valid_pts": 0,
                "mask_nonzero_ratio": float((mask > 0).mean()),
                "skipped": "no_valid_depth"
            })
            continue

        xx = xx0[valid_depth].astype(np.float64)
        yy = yy0[valid_depth].astype(np.float64)
        Z_d = Z[valid_depth].astype(np.float64)
        lab_d = lab[valid_depth].astype(np.int32)

        if Z_d.size < args.min_depth_pts:
            append_jsonl(stats_path, {
                "t_ns": t,
                "depth_valid_pts": int(Z_d.size),
                "labeled_pts": int((lab_d > 0).sum()),
                "mask_nonzero_ratio": float((mask > 0).mean()),
                "skipped": "too_few_depth_pts"
            })
            continue

        # back-project in RECTIFIED camera coords
        P_C0rect = backproject_rect(K0r, xx, yy, Z_d)

        # convert to ORIGINAL camera coords for using T_B_C0 (which is for original cam)
        P_C0orig = (R_rect2orig @ P_C0rect.T).T

        # to body and to world
        P_B = apply_T(T_C0orig_to_B, P_C0orig)
        T_W_B = get_T_W_B(r)
        P_W = apply_T(T_W_B, P_B)

        # optional height gating (relative to body z)
        p = np.array(r["p_WB"], dtype=np.float64)
        if args.zrel_min < 1e8 or args.zrel_max < 1e8:
            zrel = P_W[:, 2] - p[2]
            keep = np.ones((P_W.shape[0],), dtype=bool)
            if args.zrel_min < 1e8:
                keep &= (zrel >= args.zrel_min)
            if args.zrel_max < 1e8:
                keep &= (zrel <= args.zrel_max)
            P_W = P_W[keep]
            lab_d = lab_d[keep]

        if args.drop_unknown_depth:
            kn = (lab_d > 0)
            if not np.any(kn):
                append_jsonl(stats_path, {
                    "t_ns": t,
                    "depth_valid_pts": int(P_W.shape[0]),
                    "labeled_pts": 0,
                    "mask_nonzero_ratio": float((mask > 0).mean()),
                    "skipped": "all_unknown_labels"
                })
                continue
            P_W = P_W[kn]
            lab_d = lab_d[kn]

        if P_W.shape[0] < args.min_depth_pts:
            append_jsonl(stats_path, {
                "t_ns": t,
                "depth_valid_pts": int(P_W.shape[0]),
                "labeled_pts": int((lab_d > 0).sum()),
                "mask_nonzero_ratio": float((mask > 0).mean()),
                "skipped": "too_few_after_zgate"
            })
            continue

        idx = np.floor(P_W / voxel).astype(np.int32)

        labeled_cnt = 0
        for i in range(P_W.shape[0]):
            k = (int(idx[i, 0]), int(idx[i, 1]), int(idx[i, 2]))
            g_count[k] = g_count.get(k, 0) + 1
            g_maxz[k] = max(g_maxz.get(k, -1e9), float(P_W[i, 2]))
            g_last[k] = fi

            li = int(lab_d[i])
            if li > 0:
                labeled_cnt += 1
                if k not in g_hist:
                    g_hist[k] = {}
                g_hist[k][li] = g_hist[k].get(li, 0) + 1

        # optional pruning (avoid “full map” saturation)
        if args.keep_recent > 0 and (fi % args.prune_every == 0):
            thr = fi - args.keep_recent
            to_del = [k for k, last in g_last.items() if last < thr]
            for k in to_del:
                g_last.pop(k, None)
                g_count.pop(k, None)
                g_maxz.pop(k, None)
                g_hist.pop(k, None)

        # render local topdown
        yaw = compute_camera_yaw(T_W_B, T_C0orig_to_B, R_rect2orig)
        R_EW = rotz(-yaw)

        sem = np.zeros((grid_h, grid_w), dtype=np.uint8)
        ztop = np.full((grid_h, grid_w), -np.inf, dtype=np.float32)

        wx0, wx1 = p[0] - half_x - 1.0, p[0] + half_x + 1.0
        wy0, wy1 = p[1] - half_y - 1.0, p[1] + half_y + 1.0

        for k, c in g_count.items():
            if c < args.min_voxel_count:
                continue
            vx = (k[0] + 0.5) * voxel
            vy = (k[1] + 0.5) * voxel
            if (vx < wx0) or (vx > wx1) or (vy < wy0) or (vy > wy1):
                continue

            vw = np.array([vx, vy, g_maxz[k]], dtype=np.float64)
            ve = (R_EW @ (vw - p).reshape(3, 1)).reshape(3)

            if not (-half_x <= ve[0] < half_x and -half_y <= ve[1] < half_y):
                continue

            ix = int((ve[0] + half_x) / args.res)
            iy = int((ve[1] + half_y) / args.res)
            if 0 <= ix < grid_w and 0 <= iy < grid_h:
                label = argmax_label_hist(g_hist.get(k, {})) if (k in g_hist) else 0
                if ve[2] > ztop[iy, ix]:
                    ztop[iy, ix] = float(ve[2])
                    sem[iy, ix] = np.uint8(label)

        sem20 = maxpool_block(sem, oh, ow, mode="max")
        ztop_clip = np.where(np.isfinite(ztop), ztop, -1e9).astype(np.float32)
        ztop20 = maxpool_block(ztop_clip, oh, ow, mode="max")

        stmr = {
            "t_ns": t,
            "center_world_xyz": [float(p[0]), float(p[1]), float(p[2])],
            "yaw_rad": float(yaw),
            "scale_m_per_cell": float(range_x / ow),
            "semantic_grid": sem20.astype(int).tolist(),
            "ztop_grid": ztop20.astype(float).tolist(),
            "legend": lbl.get("class_to_id", {}),
            "mask_rectified": bool(mask_rectify_used),
            "keep_recent": int(args.keep_recent),
        }
        write_json(os.path.join(stmr_dir, f"{t}.json"), stmr)

        # visualization
        vis = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for cls, cid in lbl.get("class_to_id", {}).items():
            cid = int(cid)
            color = (int((cid * 37) % 255), int((cid * 17) % 255), int((cid * 67) % 255))
            vis[sem == cid] = color
        cx, cy = grid_w // 2, grid_h // 2
        cv2.drawMarker(vis, (cx, cy), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        cv2.imwrite(os.path.join(vis_dir, f"{t}.png"), vis)

        append_jsonl(stats_path, {
            "t_ns": t,
            "depth_valid_pts": int(Z_d.size),
            "labeled_pts": int(labeled_cnt),
            "mask_nonzero_ratio": float((mask > 0).mean()),
            "written": True
        })

        processed += 1

    write_json(os.path.join(args.out_dir, "summary.json"), {
        "processed_frames": processed,
        "mask_rectified_used": bool(mask_rectify_used),
        "note": "若后期topdown过快“满图”，可设置 --keep_recent 80 或启用 --zrel_min/--zrel_max 限制高度。"
    })
    print("[Step4-v3] processed_frames =", processed)


if __name__ == "__main__":
    main()


'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step4_stmr_accumulate_semantic.py \
  --mav0_dir   /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --keyframes  /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --depth_dir  /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/depth_npy \
  --mask_dir   /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks \
  --stereo_meta /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/stereo_meta.json \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --out_dir    /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stmr_maps_dino_sam2_200_v3
'''
