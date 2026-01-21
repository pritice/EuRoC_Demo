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

def write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def quat_wxyz_to_R(q):
    qw,qx,qy,qz = [float(x) for x in q]
    n = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz)+1e-12
    qw,qx,qy,qz = qw/n,qx/n,qy/n,qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = np.array(t, dtype=np.float64)
    return T

def apply_T(T, pts):
    pts_h = np.concatenate([pts, np.ones((pts.shape[0],1), dtype=np.float64)], axis=1)
    out = (T @ pts_h.T).T
    return out[:, :3]

def load_cam0_KD(mav0_dir: str):
    if yaml is None:
        raise RuntimeError("pyyaml missing. Run: pip install pyyaml")
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
    assert wh == (W,H)
    map0x, map0y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (W,H), cv2.CV_32FC1)
    return map0x, map0y, st

def pca_2d_extents(xy: np.ndarray):
    """
    xy: (N,2), return (len, wid, yaw_rad)
    """
    if xy.shape[0] < 10:
        mn = xy.min(axis=0); mx = xy.max(axis=0)
        L = float(mx[0]-mn[0]); W = float(mx[1]-mn[1])
        return L, W, 0.0

    c = xy.mean(axis=0, keepdims=True)
    X = xy - c
    cov = (X.T @ X) / max(1, X.shape[0]-1)
    w, v = np.linalg.eigh(cov)  # ascending
    axis = v[:, 1]              # principal
    yaw = float(math.atan2(axis[1], axis[0]))
    proj = X @ v
    mn = proj.min(axis=0); mx = proj.max(axis=0)
    L = float(mx[1]-mn[1])  # along principal
    Wd = float(mx[0]-mn[0]) # along minor
    return L, Wd, yaw

def backproject_rect(Kr, xx, yy, Z):
    X = (xx - Kr[0,2]) / Kr[0,0] * Z
    Y = (yy - Kr[1,2]) / Kr[1,1] * Z
    return np.stack([X,Y,Z], axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--instances_rect_dir", required=True, help=".../instances_rect")
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--stereo_meta", required=True)
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--semantic_mask_dir", default="", help="optional: semantic label masks (original coords)")
    ap.add_argument("--mav0_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--pix_step", type=int, default=2)
    ap.add_argument("--min_depth", type=float, default=0.05)
    ap.add_argument("--max_depth", type=float, default=12.0)
    ap.add_argument("--min_pts", type=int, default=200)

    ap.add_argument("--floor_name", type=str, default="floor")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    perframe_dir = os.path.join(args.out_dir, "per_frame")
    os.makedirs(perframe_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "instances_3d.jsonl")
    if os.path.isfile(out_jsonl):
        os.remove(out_jsonl)

    map0x, map0y, st = build_rectify_map(args.mav0_dir, args.stereo_meta)

    W,H = st["image_wh"]
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    Kr = P0[:3,:3]
    R0 = np.array(st["rectify"]["R0"], dtype=np.float64)
    R_rect2orig = R0.T

    T_B_C0 = np.array(st["T_B_C0"], dtype=np.float64)
    T_C0_B = np.linalg.inv(T_B_C0)

    labels = read_json(args.labels_json)
    class_to_id = labels.get("class_to_id", {})
    floor_id = int(class_to_id.get(args.floor_name, -1))

    rows = read_jsonl(args.keyframes)

    for r in tqdm(rows, desc="[Step5b] 3D metrics"):
        t = int(r["t_ns"])
        depth_path = os.path.join(args.depth_dir, f"{t}.npy")
        inst_json = os.path.join(args.instances_rect_dir, str(t), "instances.json")
        if (not os.path.isfile(depth_path)) or (not os.path.isfile(inst_json)):
            continue

        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (H,W):
            continue

        inst_meta = read_json(inst_json)
        insts = inst_meta.get("instances", [])
        if len(insts) == 0:
            continue

        # pose
        T_W_B = make_T(quat_wxyz_to_R(r["q_WB_wxyz"]), r["p_WB"])
        pW = np.array(r["p_WB"], dtype=np.float64)

        # optional: estimate ground z from semantic floor (if provided)
        z_ground = None
        if args.semantic_mask_dir and floor_id > 0:
            sem_path = os.path.join(args.semantic_mask_dir, f"{t}.png")
            if os.path.isfile(sem_path):
                sem = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
                if sem is not None:
                    if sem.shape != (H,W):
                        sem = cv2.resize(sem, (W,H), interpolation=cv2.INTER_NEAREST)
                    # rectify semantic to match depth
                    sem_r = cv2.remap(sem, map0x, map0y, interpolation=cv2.INTER_NEAREST)
                    ys, xs = np.where(sem_r == floor_id)
                    if xs.size > 500:
                        # sample for speed
                        step = max(1, int(xs.size // 5000))
                        xs = xs[::step]; ys = ys[::step]
                        Z = depth[ys, xs].astype(np.float64)
                        ok = (Z > args.min_depth) & (Z < args.max_depth) & np.isfinite(Z)
                        xs = xs[ok].astype(np.float64)
                        ys = ys[ok].astype(np.float64)
                        Z = Z[ok]
                        if Z.size > 300:
                            P_rect = backproject_rect(Kr, xs, ys, Z)
                            P_orig = (R_rect2orig @ P_rect.T).T
                            P_B = apply_T(T_C0_B, P_orig)
                            P_W = apply_T(T_W_B, P_B)
                            z_ground = float(np.median(P_W[:,2]))

        frame_out = {"t_ns": t, "instances": [], "z_ground": z_ground}

        for ins in insts:
            mpath = ins["mask_rect_path"]
            if not os.path.isfile(mpath):
                continue
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != (H,W):
                m = cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)

            ys, xs = np.where(m > 0)
            if xs.size < 200:
                continue

            # subsample pixels
            step = max(1, int(args.pix_step))
            xs = xs[::step]; ys = ys[::step]
            Z = depth[ys, xs].astype(np.float64)
            ok = (Z > args.min_depth) & (Z < args.max_depth) & np.isfinite(Z)
            xs = xs[ok].astype(np.float64)
            ys = ys[ok].astype(np.float64)
            Z  = Z[ok]
            if Z.size < args.min_pts:
                continue

            P_rect = backproject_rect(Kr, xs, ys, Z)
            P_orig = (R_rect2orig @ P_rect.T).T
            P_B = apply_T(T_C0_B, P_orig)
            P_W = apply_T(T_W_B, P_B)

            mn = P_W.min(axis=0); mx = P_W.max(axis=0)
            size = mx - mn
            center = P_W.mean(axis=0)

            # oriented extents on XY
            L, Wd, yaw = pca_2d_extents(P_W[:, :2])

            # height relative to ground if available, else relative to instance min z
            if z_ground is not None:
                h_rel = float(mx[2] - z_ground)
            else:
                h_rel = float(mx[2] - mn[2])

            frame_out["instances"].append({
                "inst_id": int(ins["inst_id"]),
                "label_id": int(ins.get("label_id", 0)),
                "label": ins.get("label", ""),
                "score": float(ins.get("final_score", 0.0)),
                "n_pts": int(P_W.shape[0]),
                "center_W": [float(center[0]), float(center[1]), float(center[2])],
                "aabb_min_W": [float(mn[0]), float(mn[1]), float(mn[2])],
                "aabb_max_W": [float(mx[0]), float(mx[1]), float(mx[2])],
                "size_aabb_m": [float(size[0]), float(size[1]), float(size[2])],
                "len_width_yaw_m": [float(L), float(Wd), float(yaw)],
                "height_rel_ground_m": float(h_rel),
                "z_p50": float(np.median(P_W[:,2])),
                "z_p95": float(np.percentile(P_W[:,2], 95)),
            })

        if len(frame_out["instances"]) == 0:
            continue

        write_json(os.path.join(perframe_dir, f"{t}.json"), frame_out)
        append_jsonl(out_jsonl, frame_out)

    print("[Step5b] wrote:", out_jsonl)


if __name__ == "__main__":
    main()


'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step5b_instance_3d_metrics.py \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --instances_rect_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/instances_rect \
  --depth_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/depth_npy \
  --stereo_meta /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/stereo_meta.json \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --semantic_mask_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks \
  --mav0_dir /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/instance_metrics_200 \
  --pix_step 2 --min_pts 200
'''