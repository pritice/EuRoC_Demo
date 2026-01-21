import os, json, argparse
import numpy as np
import cv2
from tqdm import tqdm

def read_jsonl(path):
    out=[]
    with open(path,"r") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out

def read_json(path):
    with open(path,"r") as f:
        return json.load(f)

def backproject_rect(K, xs, ys, Z):
    X = (xs - K[0,2]) / K[0,0] * Z
    Y = (ys - K[1,2]) / K[1,1] * Z
    return np.stack([X,Y,Z], axis=1)

def fit_plane_ransac(P, iters=400, thresh=0.03, seed=0):
    if P.shape[0] < 300:
        return None
    rng = np.random.default_rng(seed)
    best = None
    best_inl = 0
    N = P.shape[0]
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        p0,p1,p2 = P[idx]
        n = np.cross(p1-p0, p2-p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -float(np.dot(n, p0))
        dist = np.abs(P @ n + d)
        inl = dist < thresh
        cnt = int(np.sum(inl))
        # prefer plane whose normal has large |Y| component (camera y points down)
        score = cnt * (abs(float(n[1])) + 0.2)
        if best is None or score > best[0]:
            best = (score, n, d, inl)
            best_inl = cnt
    if best is None:
        return None
    _, n, d, inl = best
    return n, d, inl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--stereo_meta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pix_step", type=int, default=6)
    ap.add_argument("--depth_trunc", type=float, default=8.0)
    ap.add_argument("--plane_iters", type=int, default=500)
    ap.add_argument("--plane_thresh", type=float, default=0.04)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mask_dir = os.path.join(args.out_dir, "masks")
    vis_dir = os.path.join(args.out_dir, "mask_vis")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    st = read_json(args.stereo_meta)
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    K0r = P0[:3,:3]
    W,H = st["image_wh"]

    rows = read_jsonl(args.keyframes)

    for r in tqdm(rows, desc="[Step3] cache masks"):
        t = int(r["t_ns"])
        out_png = os.path.join(mask_dir, f"{t}.png")
        out_jpg = os.path.join(vis_dir, f"{t}.jpg")
        if os.path.isfile(out_png) and os.path.isfile(out_jpg):
            continue

        dpath = os.path.join(args.depth_dir, f"{t}.npy")
        depth = np.load(dpath).astype(np.float32)

        # sample pixels (prefer bottom half for plane fit)
        ys = np.arange(H//2, H, args.pix_step, dtype=np.int32)
        xs = np.arange(0, W, args.pix_step, dtype=np.int32)
        xx, yy = np.meshgrid(xs, ys)
        xx = xx.reshape(-1); yy = yy.reshape(-1)
        Z = depth[yy, xx].astype(np.float64)
        valid = (Z > 0.2) & (Z < args.depth_trunc) & np.isfinite(Z)
        xx = xx[valid].astype(np.float64)
        yy = yy[valid].astype(np.float64)
        Z  = Z[valid]
        if Z.size < 300:
            # fallback: unknown
            mask = np.zeros((H,W), np.uint8)
            cv2.imwrite(out_png, mask)
            continue

        P = backproject_rect(K0r, xx, yy, Z)  # camera rect frame
        plane = fit_plane_ransac(P, iters=args.plane_iters, thresh=args.plane_thresh, seed=0)
        if plane is None:
            mask = np.zeros((H,W), np.uint8)
            cv2.imwrite(out_png, mask)
            continue

        n, d, _ = plane

        # label all valid depth pixels by distance to plane
        yy2 = np.arange(0, H, args.pix_step, dtype=np.int32)
        xx2 = np.arange(0, W, args.pix_step, dtype=np.int32)
        X2, Y2 = np.meshgrid(xx2, yy2)
        X2 = X2.reshape(-1); Y2 = Y2.reshape(-1)

        Z2 = depth[Y2, X2].astype(np.float64)
        v2 = (Z2 > 0.2) & (Z2 < args.depth_trunc) & np.isfinite(Z2)
        X2 = X2[v2].astype(np.float64)
        Y2 = Y2[v2].astype(np.float64)
        Z2 = Z2[v2]

        P2 = backproject_rect(K0r, X2, Y2, Z2)
        dist = np.abs(P2 @ n + d)

        mask = np.zeros((H,W), np.uint8)
        floor = dist < args.plane_thresh
        mask[Y2.astype(np.int32), X2.astype(np.int32)] = np.where(floor, 1, 2).astype(np.uint8)

        # small morphology to denoise
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        cv2.imwrite(out_png, mask)

        # visualization overlay
        img = cv2.imread(r["cam0_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (H,W):
            img = cv2.resize(img, (W,H), interpolation=cv2.INTER_AREA)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # floor green, nonfloor red
        vis[mask==1] = (0,255,0)
        vis[mask==2] = (0,0,255)
        cv2.imwrite(out_jpg, vis)

    meta = {"labels": {0:"unknown", 1:"floor", 2:"nonfloor"}}
    with open(os.path.join(args.out_dir, "mask_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[Step3] wrote masks to", args.out_dir)

if __name__ == "__main__":
    main()


'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step3_cache_masks_floor.py \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --depth_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200/depth_npy \
  --stereo_meta /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200/stereo_meta.json \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_floor_200
'''