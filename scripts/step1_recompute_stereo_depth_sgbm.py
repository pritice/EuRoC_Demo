import os, json, argparse
import numpy as np
import cv2
from tqdm import tqdm

try:
    import yaml
except Exception as e:
    raise RuntimeError("Missing pyyaml. Please: pip install pyyaml") from e

# Absolute paths (edit for your server)
DEFAULT_MAV0_DIR = "/root/autodl-tmp/sam-3d-objects/inputs/mav0"
DEFAULT_KEYFRAMES = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl"
DEFAULT_OUT_DIR = "/root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2"


def parse_mat_4x4(v):
    # EuRoC sensor.yaml: T_BS: {rows:4, cols:4, data:[...]}
    if isinstance(v, dict) and "data" in v:
        data = v["data"]
    elif isinstance(v, (list, tuple)):
        data = v
    else:
        raise TypeError(f"Unsupported matrix type: {type(v)}")
    arr = np.array(data, dtype=np.float64).reshape(4, 4)
    return arr


def load_cam_sensor_yaml(mav0_dir, cam_name):
    ypath = os.path.join(mav0_dir, cam_name, "sensor.yaml")
    with open(ypath, "r") as f:
        y = yaml.safe_load(f)

    intr = y["intrinsics"]  # [fx, fy, cx, cy]
    fx, fy, cx, cy = [float(x) for x in intr]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    D = np.array(y.get("distortion_coeffs", []), dtype=np.float64).reshape(-1, 1)

    # body -> camera
    T_B_C = parse_mat_4x4(y["T_BS"]).astype(np.float64)

    # image size
    res = y.get("resolution", None)
    if res is None:
        raise KeyError(f"{ypath} missing 'resolution'")
    W, H = int(res[0]), int(res[1])

    return {"K": K, "D": D, "T_B_C": T_B_C, "W": W, "H": H, "yaml": ypath}


def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mav0_dir", default=DEFAULT_MAV0_DIR)
    ap.add_argument("--keyframes", default=DEFAULT_KEYFRAMES)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=200)

    ap.add_argument("--num_disp", type=int, default=128)     # must be multiple of 16
    ap.add_argument("--block", type=int, default=7)          # odd
    ap.add_argument("--min_disp", type=int, default=0)
    ap.add_argument("--disp_thresh", type=float, default=1.0)  # <= this treated invalid
    ap.add_argument("--median_ksize", type=int, default=3, help="median filter size for disparity (odd, 0 to disable)")
    ap.add_argument("--lr_check", action="store_true", help="enable left-right consistency check (needs ximgproc)")
    ap.add_argument("--lr_max_diff", type=float, default=1.0, help="max LR disparity diff in pixels")
    ap.add_argument("--wls", action="store_true", help="enable WLS disparity filter (needs ximgproc)")
    ap.add_argument("--wls_lambda", type=float, default=8000.0)
    ap.add_argument("--wls_sigma", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    depth_dir = os.path.join(args.out_dir, "depth_npy")
    vis_dir   = os.path.join(args.out_dir, "depth_vis")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    cam0 = load_cam_sensor_yaml(args.mav0_dir, "cam0")
    cam1 = load_cam_sensor_yaml(args.mav0_dir, "cam1")
    assert cam0["W"] == cam1["W"] and cam0["H"] == cam1["H"], "cam0/cam1 resolution mismatch"
    W, H = cam0["W"], cam0["H"]

    # Relative transform cam0 -> cam1
    T_B_C0 = cam0["T_B_C"]
    T_B_C1 = cam1["T_B_C"]
    T_C0_B = np.linalg.inv(T_B_C0)
    T_C0_C1 = T_C0_B @ T_B_C1
    R = T_C0_C1[:3, :3].astype(np.float64)
    T = T_C0_C1[:3, 3].astype(np.float64).reshape(3, 1)

    # Rectify
    R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
        cam0["K"], cam0["D"], cam1["K"], cam1["D"],
        (W, H), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    map0x, map0y = cv2.initUndistortRectifyMap(cam0["K"], cam0["D"], R0, P0, (W, H), cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(cam1["K"], cam1["D"], R1, P1, (W, H), cv2.CV_32FC1)

    # baseline in meters from rectified P1: P1[0,3] = -fx*Tx
    fx_rect = float(P0[0, 0])
    baseline = abs(float(P1[0, 3]) / float(P1[0, 0]))  # = abs(-fx*Tx/fx)=abs(Tx)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=args.min_disp,
        numDisparities=int(args.num_disp),
        blockSize=int(args.block),
        P1=8 * 1 * int(args.block) * int(args.block),
        P2=32 * 1 * int(args.block) * int(args.block),
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    use_ximgproc = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createRightMatcher")
    if (args.lr_check or args.wls) and (not use_ximgproc):
        print("[WARN] ximgproc not available; disable --lr_check/--wls or install opencv-contrib-python")
    use_lr_check = bool(args.lr_check and use_ximgproc)
    use_wls = bool(args.wls and use_ximgproc)

    sgbmR = None
    wls_filter = None
    if use_lr_check or use_wls:
        sgbmR = cv2.ximgproc.createRightMatcher(sgbm)
    if use_wls:
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(sgbm)
        wls_filter.setLambda(float(args.wls_lambda))
        wls_filter.setSigmaColor(float(args.wls_sigma))

    rows = read_jsonl(args.keyframes)[: args.limit]
    stats = []

    for i, r in enumerate(tqdm(rows, desc="[Step2] stereo depth")):
        t = int(r["t_ns"])
        p0 = r["cam0_path"]
        p1 = r["cam1_path"]

        img0 = cv2.imread(p0, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        if img0 is None or img1 is None:
            stats.append({"t_ns": t, "error": "missing_image"})
            continue

        img0r = cv2.remap(img0, map0x, map0y, cv2.INTER_LINEAR)
        img1r = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)

        dispL_raw = sgbm.compute(img0r, img1r)
        dispR_raw = None
        if use_lr_check or use_wls:
            dispR_raw = sgbmR.compute(img1r, img0r)

        if use_wls and (dispR_raw is not None):
            dispW = wls_filter.filter(dispL_raw, img0r, None, dispR_raw)
            disp = dispW.astype(np.float32) / 16.0
        else:
            disp = dispL_raw.astype(np.float32) / 16.0

        if int(args.median_ksize) >= 3 and int(args.median_ksize) % 2 == 1:
            disp = cv2.medianBlur(disp, int(args.median_ksize))
        valid = disp > args.disp_thresh

        lr_valid_ratio = None
        if use_lr_check and (dispR_raw is not None):
            dispR = dispR_raw.astype(np.float32) / 16.0
            h, w = disp.shape
            xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
            xr = xs - disp
            xr_int = np.round(xr).astype(np.int32)
            inb = (xr_int >= 0) & (xr_int < w)
            rows_idx = np.repeat(np.arange(h, dtype=np.int32)[:, None], w, axis=1)
            dispR_at = np.zeros_like(disp, dtype=np.float32)
            dispR_at[inb] = dispR[rows_idx[inb], xr_int[inb]]
            lr_consistent = (np.abs(disp - dispR_at) <= float(args.lr_max_diff)) & inb
            valid = valid & lr_consistent
            lr_valid_ratio = float(lr_consistent.mean())

        depth = np.zeros((H, W), dtype=np.float32)
        depth[valid] = (fx_rect * baseline) / (disp[valid] + 1e-6)

        np.save(os.path.join(depth_dir, f"{t}.npy"), depth)

        # quick vis for first ~20 frames
        if i < 20:
            dv = depth.copy()
            v = dv[dv > 0]
            if v.size > 0:
                lo, hi = np.percentile(v, 5), np.percentile(v, 95)
                dv = np.clip((dv - lo) / (hi - lo + 1e-6), 0, 1)
            vis = (dv * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(vis_dir, f"{t}.png"), vis)

        inv = 1.0 - float(valid.mean())
        stats.append({
            "t_ns": t,
            "baseline_m": baseline,
            "fx_rect": fx_rect,
            "invalid_ratio": inv,
            "depth_pos_ratio": float((depth > 0).mean()),
            "depth_p50": float(np.median(depth[depth > 0])) if (depth > 0).any() else None,
            "use_lr_check": bool(use_lr_check),
            "use_wls": bool(use_wls),
            "lr_valid_ratio": lr_valid_ratio
        })

    # meta for Step4
    meta = {
        "image_wh": [W, H],
        "T_B_C0": T_B_C0.tolist(),
        "T_B_C1": T_B_C1.tolist(),
        "rectify": {
            "R0": R0.tolist(),
            "R1": R1.tolist(),
            "P0": P0.tolist(),
            "P1": P1.tolist(),
            "Q": Q.tolist(),
            "baseline_m": baseline,
            "fx_rect": fx_rect
        },
        "paths": {
            "cam0_yaml": cam0["yaml"],
            "cam1_yaml": cam1["yaml"]
        }
    }
    write_json(os.path.join(args.out_dir, "stereo_meta.json"), meta)
    write_json(os.path.join(args.out_dir, "depth_stats.json"), {
        "n": len(stats),
        "mean_invalid_ratio": float(np.mean([x["invalid_ratio"] for x in stats if "invalid_ratio" in x] or [1.0])),
        "samples": stats[:10]
    })
    print("[Step2] done. depth_dir =", depth_dir)

if __name__ == "__main__":
    main()

'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step1_recompute_stereo_depth_sgbm.py \
  --mav0_dir /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2 \
  --limit 200
'''
