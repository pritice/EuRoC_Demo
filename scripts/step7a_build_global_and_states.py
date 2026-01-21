# -*- coding: utf-8 -*-
import os, json, math, argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

try:
    import yaml
except Exception:
    yaml = None


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

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

def write_json(p: str, obj: Dict[str, Any]):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# Geometry
# -----------------------------
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

def yaw_from_q_WB(q_wxyz):
    R = quat_wxyz_to_R(q_wxyz)
    fwd = R[:,0]  # body x axis in world
    return float(math.atan2(fwd[1], fwd[0]))

def backproject_rect(Kr, xx, yy, Z):
    X = (xx - Kr[0,2]) / Kr[0,0] * Z
    Y = (yy - Kr[1,2]) / Kr[1,1] * Z
    return np.stack([X,Y,Z], axis=1)

def depth_autoscale(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    pos = d[np.isfinite(d) & (d > 0)]
    if pos.size == 0:
        return d
    p50 = float(np.percentile(pos, 50))
    # heuristic: if median depth looks like mm, convert to meters
    if p50 > 50.0:
        d = d * 0.001
    return d


# -----------------------------
# Rectify map (align raw mask -> rectified depth)
# -----------------------------
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
    if wh != (W,H):
        raise ValueError(f"resolution mismatch: sensor.yaml {wh} vs stereo_meta {(W,H)}")
    map0x, map0y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (W,H), cv2.CV_32FC1)
    return map0x, map0y, st


# -----------------------------
# Semantic / smoothing
# -----------------------------
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

def grid_majority_smooth(semN: np.ndarray,
                         unknown_id=0,
                         preserve_ids=None,
                         iters=6,
                         w_self=3,
                         allow_fill_unknown=False):
    """
    只做“块状一致性”，默认不把 unknown 填成别的类（保持真实性）。
    """
    if preserve_ids is None:
        preserve_ids = set()
    sem = semN.copy().astype(np.int32)
    H,W = sem.shape
    for _ in range(iters):
        pad = np.pad(sem, ((1,1),(1,1)), mode="edge")
        new = sem.copy()
        for y in range(H):
            for x in range(W):
                c0 = int(sem[y,x])
                if c0 in preserve_ids:
                    continue
                nb = pad[y:y+3, x:x+3].reshape(-1)
                if not allow_fill_unknown:
                    nb = nb[nb != unknown_id]
                if nb.size == 0:
                    continue
                bc = np.bincount(nb.astype(np.int64))
                if 0 <= c0 < bc.size:
                    bc[c0] += w_self
                c1 = int(bc.argmax())
                new[y,x] = c1
        sem = new
    return sem.astype(np.uint8)


def height_source_metrics(h_mask: np.ndarray) -> Dict[str, float]:
    total = int(h_mask.size)
    valid_cells = int(h_mask.sum())
    ratio = float(valid_cells / max(total, 1))
    return {"valid_cells": int(valid_cells), "ratio": float(ratio)}


def remove_small_components_sem(sem: np.ndarray, min_cells: int, preserve_ids=None) -> Tuple[np.ndarray, int]:
    if min_cells <= 1:
        return sem, 0
    if preserve_ids is None:
        preserve_ids = set()
    out = sem.copy().astype(np.uint8)
    removed = 0
    ids = [int(x) for x in np.unique(out) if int(x) > 0]
    for cid in ids:
        if cid in preserve_ids:
            continue
        m = (out == cid).astype(np.uint8)
        ncc, lab = cv2.connectedComponents(m, connectivity=4)
        for k in range(1, ncc):
            comp = (lab == k)
            area = int(comp.sum())
            if area < min_cells:
                out[comp] = 0
                removed += area
    return out, removed


def compute_align_angle(sem: np.ndarray, min_known_cells: int) -> Optional[float]:
    ys, xs = np.where(sem > 0)
    if xs.size < min_known_cells:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    edges = box[np.arange(4)] - box[(np.arange(4) + 1) % 4]
    lens = np.sum(edges * edges, axis=1)
    i = int(np.argmax(lens))
    dx, dy = edges[i]
    ang = float(np.degrees(np.arctan2(dy, dx)))
    return ang


def align_and_square_region(sem: np.ndarray,
                            min_known_cells: int,
                            pad_cells: int) -> Tuple[np.ndarray, np.ndarray, int, int, int, float]:
    H, W = sem.shape
    angle = compute_align_angle(sem, min_known_cells)
    if angle is None:
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return sem, M, 0, 0, W, 0.0
    center = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    sem_rot = cv2.warpAffine(sem, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    ys, xs = np.where(sem_rot > 0)
    if xs.size == 0:
        return sem_rot, M, 0, 0, W, angle
    x0 = int(xs.min()) - int(pad_cells)
    x1 = int(xs.max()) + 1 + int(pad_cells)
    y0 = int(ys.min()) - int(pad_cells)
    y1 = int(ys.max()) + 1 + int(pad_cells)
    side = int(max(x1 - x0, y1 - y0))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    x0 = int(np.floor(cx - side / 2.0))
    y0 = int(np.floor(cy - side / 2.0))
    return sem_rot, M, x0, y0, side, angle


def square_mask_from_xy(sem_shape: Tuple[int, int], x0: int, y0: int, side: int) -> np.ndarray:
    H, W = sem_shape
    mask = np.zeros((H, W), dtype=bool)
    x1 = x0 + side
    y1 = y0 + side
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(W, x1)
    sy1 = min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        mask[sy0:sy1, sx0:sx1] = True
    return mask


def resize_with_nan(h: np.ndarray, out_size: int) -> np.ndarray:
    h = h.astype(np.float32)
    valid = np.isfinite(h).astype(np.uint8)
    h2 = h.copy()
    h2[~np.isfinite(h2)] = 0.0
    h_res = cv2.resize(h2, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    m_res = cv2.resize(valid, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    h_res[m_res == 0] = np.nan
    return h_res


# -----------------------------
# RANSAC ground plane: z = a*x + b*y + c
# -----------------------------
def fit_plane_z_axbyc_from_3pts(p1, p2, p3, eps=1e-8):
    A = np.array([[p1[0], p1[1], 1.0],
                  [p2[0], p2[1], 1.0],
                  [p3[0], p3[1], 1.0]], dtype=np.float64)
    b = np.array([p1[2], p2[2], p3[2]], dtype=np.float64)
    det = np.linalg.det(A)
    if abs(det) < eps:
        return None
    x = np.linalg.solve(A, b)
    return x

def ransac_ground_plane(xy: np.ndarray, z: np.ndarray,
                        iters=250, thresh=0.10,
                        min_inliers=40, seed=0):
    rng = np.random.default_rng(seed)
    M = xy.shape[0]
    stats = {"ok": False, "reason": "", "inliers": 0, "M": int(M), "median_res": None,
             "thresh": float(thresh), "iters": int(iters), "min_inliers": int(min_inliers)}
    if M < 30:
        stats["reason"] = "too_few_points"
        return None, None, stats

    best = None
    best_inliers = -1
    best_med = 1e9
    best_mask = None

    pts = np.concatenate([xy, z.reshape(-1,1)], axis=1)
    for _ in range(iters):
        idx = rng.choice(M, size=3, replace=False)
        p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        abc = fit_plane_z_axbyc_from_3pts(p1, p2, p3)
        if abc is None:
            continue
        a, b, c = abc
        z_pred = a*xy[:,0] + b*xy[:,1] + c
        res = np.abs(z_pred - z)
        mask = res < thresh
        nin = int(mask.sum())
        if nin > best_inliers:
            med = float(np.median(res[mask])) if nin > 0 else 1e9
            best = (a,b,c)
            best_inliers = nin
            best_med = med
            best_mask = mask
        elif nin == best_inliers and nin > 0:
            med = float(np.median(res[mask]))
            if med < best_med:
                best = (a,b,c)
                best_med = med
                best_mask = mask

    if best is None or best_inliers < min_inliers:
        stats["reason"] = "no_good_model"
        stats["inliers"] = int(best_inliers if best_inliers > 0 else 0)
        return None, None, stats

    stats["ok"] = True
    stats["inliers"] = int(best_inliers)
    stats["median_res"] = float(best_med)
    return best, best_mask, stats

def clip_height_grid(h: np.ndarray, hmin=0.0, hmax=6.0):
    out = h.copy()
    m = np.isfinite(out)
    out[m] = np.clip(out[m], hmin, hmax)
    out[~m] = np.nan
    return out


# -----------------------------
# Objects (simple CC on semantic grid)
# -----------------------------
def extract_objects_from_grid(semN: np.ndarray, hrelN: np.ndarray,
                              id_to_name: Dict[int,str],
                              cell_m: float,
                              max_objects: int = 30,
                              min_area_cells: int = 2):
    objs = []
    ids = [int(x) for x in np.unique(semN) if int(x) > 0]
    for cid in ids:
        m = (semN == cid).astype(np.uint8)
        ncc, lab = cv2.connectedComponents(m, connectivity=4)
        for k in range(1, ncc):
            pts = np.argwhere(lab == k)  # (y,x)
            area = int(pts.shape[0])
            if area < min_area_cells:
                continue
            ys = pts[:,0]; xs = pts[:,1]
            y0,y1 = int(ys.min()), int(ys.max())
            x0,x1 = int(xs.min()), int(xs.max())
            w_m = float((x1-x0+1) * cell_m)
            l_m = float((y1-y0+1) * cell_m)
            cy = float(ys.mean()); cx = float(xs.mean())
            hh = hrelN[ys, xs]
            hh = hh[np.isfinite(hh)]
            h_m = float(np.percentile(hh, 95)) if hh.size > 0 else None
            objs.append({
                "class_id": int(cid),
                "class_name": id_to_name.get(cid, f"id{cid}"),
                "area_cells": area,
                "centroid_cell_xy": [float(cx), float(cy)],
                "bbox_cell_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "size_m_LW": [l_m, w_m],
                "height_m": h_m,
            })
    def sort_key(o):
        h = o.get("height_m", None)
        if isinstance(h, (int, float)) and np.isfinite(h):
            hv = float(h)
        else:
            hv = -1e9
        return (-int(o["area_cells"]), -hv)
    objs.sort(key=sort_key)
    return objs[:max_objects]

def default_cost_from_sem(semN: np.ndarray, id_to_name: Dict[int,str]) -> np.ndarray:
    # unknown is dangerous => high cost
    cost = np.ones_like(semN, dtype=np.float32) * 5.0
    cost[semN == 0] = 50.0
    for cid in np.unique(semN):
        cid = int(cid)
        name = id_to_name.get(cid, "")
        if name in ("floor", "road"):
            cost[semN == cid] = 1.0
        if name in ("wall", "ceiling"):
            cost[semN == cid] = 100.0
    return cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--stereo_meta", required=True)
    ap.add_argument("--mav0_dir", default="/root/autodl-tmp/sam-3d-objects/inputs/mav0")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--res", type=float, default=0.10)
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--pix_step", type=int, default=2)
    ap.add_argument("--min_depth", type=float, default=0.10)
    ap.add_argument("--max_depth", type=float, default=12.0)
    ap.add_argument("--zrel_min", type=float, default=-1.5, help="relative z min (m) w.r.t body z")
    ap.add_argument("--zrel_max", type=float, default=2.5, help="relative z max (m) w.r.t body z")
    ap.add_argument("--drop_unknown_depth", action="store_true", help="ignore depth points with label=0")

    # IMPORTANT: keep local_size_m fixed, increase N to get smaller cells
    ap.add_argument("--local_size_m", type=float, default=30.0)
    ap.add_argument("--N", type=int, default=60)

    ap.add_argument("--ego_align", action="store_true")
    ap.add_argument("--smooth_iters", type=int, default=6)
    ap.add_argument("--smooth_w_self", type=int, default=3)
    ap.add_argument("--sem_min_component_cells", type=int, default=4,
                    help="remove small semantic islands after smoothing")
    ap.add_argument("--max_objects", type=int, default=30)
    ap.add_argument("--min_area_cells", type=int, default=2)

    ap.add_argument("--ground_mode", type=str, default="auto",
                    choices=["floorref", "ransac", "auto"])
    ap.add_argument("--height_clip_max", type=float, default=6.0)
    ap.add_argument("--ransac_iters", type=int, default=250)
    ap.add_argument("--ransac_thresh", type=float, default=0.10)
    ap.add_argument("--ransac_min_inliers", type=int, default=40)
    ap.add_argument("--ransac_seed", type=int, default=0)
    ap.add_argument("--height_select_min_ratio", type=float, default=0.005)
    ap.add_argument("--height_select_min_cells", type=int, default=20)
    ap.add_argument("--export_square_states", action="store_true",
                    help="write square-aligned states for planning without overwriting originals")
    ap.add_argument("--square_states_dir", default="",
                    help="output dir for square states (default: out_dir/states_square)")
    ap.add_argument("--square_min_known_cells", type=int, default=60)
    ap.add_argument("--square_pad_cells", type=int, default=2)
    ap.add_argument("--no_square_fill_unknown_floor", action="store_true",
                    help="disable filling unknown inside square with floor label")
    ap.add_argument("--square_force_N", type=int, default=0,
                    help="force square state grid size (0 to keep original N)")

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    out_states = os.path.join(args.out_dir, "states")
    out_global = os.path.join(args.out_dir, "global_map")
    ensure_dir(out_states); ensure_dir(out_global)

    rows = read_jsonl(args.keyframes)
    labels = read_json(args.labels_json)
    class_to_id = labels.get("class_to_id", {})
    id_to_name = {int(v): k for k,v in class_to_id.items()}
    id_to_name[0] = "unknown"
    floor_id = int(class_to_id.get("floor", -1))

    preserve_names = ["floor", "road", "wall", "ceiling"]
    preserve_ids = set(int(class_to_id[n]) for n in preserve_names if n in class_to_id)

    map0x, map0y, st = build_rectify_map(args.mav0_dir, args.stereo_meta)
    W,H = st["image_wh"]
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    R0 = np.array(st["rectify"]["R0"], dtype=np.float64)
    Kr = P0[:3,:3]
    R_rect2orig = R0.T

    T_B_C0 = np.array(st["T_B_C0"], dtype=np.float64)
    T_C0_B = np.linalg.inv(T_B_C0)

    traj = np.array([r["p_WB"][:2] for r in rows], dtype=np.float64)
    xmin, ymin = traj.min(axis=0)
    xmax, ymax = traj.max(axis=0)
    xmin -= args.margin; ymin -= args.margin
    xmax += args.margin; ymax += args.margin

    res = float(args.res)
    GW = int(math.ceil((xmax - xmin) / res))
    GH = int(math.ceil((ymax - ymin) / res))
    GW = max(GW, 2); GH = max(GH, 2)

    g_ztop = np.full((GH, GW), -np.inf, dtype=np.float32)
    g_sem  = np.zeros((GH, GW), dtype=np.uint8)
    g_last = np.full((GH, GW), -1, dtype=np.int32)

    ys = np.arange(0, H, args.pix_step, dtype=np.int32)
    xs = np.arange(0, W, args.pix_step, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.reshape(-1); yy = yy.reshape(-1)

    # P is patch size in global grid cells (world res)
    P = int(round(args.local_size_m / res))
    P = max(P, args.N)
    if P % args.N != 0:
        P = int(math.ceil(P / args.N) * args.N)

    index_path = os.path.join(args.out_dir, "state_index.jsonl")
    idx_f = open(index_path, "w")
    out_states_square = ""
    idx_sq_f = None
    if args.export_square_states:
        out_states_square = args.square_states_dir or os.path.join(args.out_dir, "states_square")
        ensure_dir(out_states_square)
        idx_sq_f = open(os.path.join(out_states_square, "state_index.jsonl"), "w")

    processed = 0
    N = int(args.N)

    for fi, r in enumerate(rows):
        t = int(r["t_ns"])
        dp = os.path.join(args.depth_dir, f"{t}.npy")
        mp = os.path.join(args.mask_dir, f"{t}.png")
        if not (os.path.isfile(dp) and os.path.isfile(mp)):
            continue

        depth = np.load(dp).astype(np.float32)
        if depth.shape != (H,W):
            continue
        depth = depth_autoscale(depth)

        sem = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        if sem.shape != (H,W):
            sem = cv2.resize(sem, (W,H), interpolation=cv2.INTER_NEAREST)
        sem_r = cv2.remap(sem, map0x, map0y, interpolation=cv2.INTER_NEAREST)

        Z = depth[yy, xx].astype(np.float64)
        LB = sem_r[yy, xx].astype(np.int32)
        ok = np.isfinite(Z) & (Z > args.min_depth) & (Z < args.max_depth) & (LB >= 0)
        if not np.any(ok):
            continue

        xxo = xx[ok].astype(np.float64)
        yyo = yy[ok].astype(np.float64)
        Zo  = Z[ok]
        lbo = LB[ok].astype(np.uint8)

        P_rect = backproject_rect(Kr, xxo, yyo, Zo)
        P_c0   = (R_rect2orig @ P_rect.T).T
        P_B    = apply_T(T_C0_B, P_c0)
        T_W_B  = make_T(quat_wxyz_to_R(r["q_WB_wxyz"]), r["p_WB"])
        P_W    = apply_T(T_W_B, P_B)
        # z-gating to reduce ceiling/wall smear
        pz = float(r["p_WB"][2])
        zrel = P_W[:, 2] - pz
        keep = (zrel >= float(args.zrel_min)) & (zrel <= float(args.zrel_max))
        if np.any(keep):
            P_W = P_W[keep]
            lbo = lbo[keep]
        else:
            continue

        gx = np.floor((P_W[:,0] - xmin) / res).astype(np.int32)
        gy = np.floor((P_W[:,1] - ymin) / res).astype(np.int32)
        inb = (gx >= 0) & (gx < GW) & (gy >= 0) & (gy < GH)
        gx = gx[inb]; gy = gy[inb]
        zw = P_W[inb, 2].astype(np.float32)
        lb = lbo[inb]
        if args.drop_unknown_depth:
            kn = (lb > 0)
            if not np.any(kn):
                continue
            gx = gx[kn]; gy = gy[kn]; zw = zw[kn]; lb = lb[kn]

        if gx.size > 0:
            flat = gy.astype(np.int64) * GW + gx.astype(np.int64)
            order = np.lexsort((zw, flat))
            flat_s = flat[order]
            zw_s   = zw[order]
            lb_s   = lb[order]
            last = np.r_[flat_s[1:] != flat_s[:-1], True]
            flat_u = flat_s[last]
            zw_u   = zw_s[last]
            lb_u   = lb_s[last]

            gzt = g_ztop.reshape(-1)
            gsm = g_sem.reshape(-1)
            gls = g_last.reshape(-1)

            upd = zw_u > gzt[flat_u]
            if np.any(upd):
                fu = flat_u[upd]
                gzt[fu] = zw_u[upd]
                gsm[fu] = lb_u[upd]
                gls[fu] = fi

        # -------- local patch crop --------
        p0 = np.array(r["p_WB"][:2], dtype=np.float64)
        yaw = yaw_from_q_WB(r["q_WB_wxyz"])

        cxg = int(math.floor((p0[0] - xmin) / res))
        cyg = int(math.floor((p0[1] - ymin) / res))

        x0 = cxg - P//2; x1 = x0 + P
        y0 = cyg - P//2; y1 = y0 + P

        patch_sem = np.zeros((P, P), dtype=np.uint8)
        patch_zt  = np.full((P, P), np.nan, dtype=np.float32)

        xs0 = max(0, x0); xs1 = min(GW, x1)
        ys0 = max(0, y0); ys1 = min(GH, y1)

        if xs1 > xs0 and ys1 > ys0:
            px0 = xs0 - x0; px1 = px0 + (xs1 - xs0)
            py0 = ys0 - y0; py1 = py0 + (ys1 - ys0)
            sub_sem = g_sem[ys0:ys1, xs0:xs1]
            sub_zt  = g_ztop[ys0:ys1, xs0:xs1].copy()
            sub_zt[sub_zt <= -1e8] = np.nan
            patch_sem[py0:py1, px0:px1] = sub_sem
            patch_zt[py0:py1, px0:px1] = sub_zt

        if args.ego_align:
            angle_deg = -yaw * 180.0 / math.pi
            center = (P/2.0, P/2.0)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            patch_sem = cv2.warpAffine(patch_sem, M, (P, P),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_REPLICATE)
            patch_zt = cv2.warpAffine(patch_zt, M, (P, P),
                                      flags=cv2.INTER_NEAREST,
                                      borderValue=np.nan)

        # Pool P×P -> N×N (NO unknown fill here)
        semN = np.zeros((N,N), dtype=np.uint8)
        zN   = np.full((N,N), np.nan, dtype=np.float32)   # top proxy (p95)
        zL   = np.full((N,N), np.nan, dtype=np.float32)   # low proxy (p10)
        bins = np.linspace(0, P, N+1).astype(int)

        for gy2 in range(N):
            yb0,yb1 = bins[gy2], bins[gy2+1]
            yb1 = max(yb1, yb0+1)
            for gx2 in range(N):
                xb0,xb1 = bins[gx2], bins[gx2+1]
                xb1 = max(xb1, xb0+1)
                sb = patch_sem[yb0:yb1, xb0:xb1]
                zb = patch_zt[yb0:yb1, xb0:xb1]
                semN[gy2,gx2] = mode_ignore_zero(sb)
                zb2 = zb[np.isfinite(zb)]
                if zb2.size > 0:
                    zN[gy2,gx2] = float(np.percentile(zb2, 95))
                    zL[gy2,gx2] = float(np.percentile(zb2, 10))

        semS = grid_majority_smooth(
            semN,
            unknown_id=0,
            preserve_ids=preserve_ids,
            iters=int(args.smooth_iters),
            w_self=int(args.smooth_w_self),
            allow_fill_unknown=False,  # IMPORTANT: keep unknown as unknown
        )
        semS, sem_removed = remove_small_components_sem(
            semS,
            min_cells=int(args.sem_min_component_cells),
            preserve_ids=preserve_ids,
        )

        # -------- height estimation --------
        cell_m = float(args.local_size_m) / float(N)

        # floorref
        z_floor = compute_floor_ref(semS, zN, floor_id)
        h_floor = np.full_like(zN, np.nan, dtype=np.float32)
        m_top = np.isfinite(zN)
        h_floor[m_top] = (zN[m_top] - z_floor).astype(np.float32)
        h_floor = clip_height_grid(h_floor, 0.0, float(args.height_clip_max))
        h_floor_valid = np.isfinite(h_floor)

        # ransac (use bottom 25% of zL as candidates)
        half = float(args.local_size_m) / 2.0
        xs = (np.arange(N, dtype=np.float64) + 0.5) * cell_m - half
        ys = (np.arange(N, dtype=np.float64) + 0.5) * cell_m - half
        XX, YY = np.meshgrid(xs, ys)

        cand = np.isfinite(zL)
        zz_all = zL[cand].astype(np.float64)
        if zz_all.size >= 30:
            z_thr = np.percentile(zz_all, 25)
            cand = cand & (zL <= z_thr)

        xy = np.stack([XX[cand], YY[cand]], axis=1) if np.any(cand) else np.zeros((0,2), dtype=np.float64)
        zz = zL[cand].astype(np.float64) if np.any(cand) else np.zeros((0,), dtype=np.float64)

        abc, inl, rstats = ransac_ground_plane(
            xy, zz,
            iters=int(args.ransac_iters),
            thresh=float(args.ransac_thresh),
            min_inliers=int(args.ransac_min_inliers),
            seed=int(args.ransac_seed),
        )

        h_ransac = np.full_like(zN, np.nan, dtype=np.float32)
        if abc is not None:
            a,b,c = abc
            z_ground = (a*XX + b*YY + c).astype(np.float32)
            m2 = np.isfinite(zN)
            h_ransac[m2] = (zN[m2] - z_ground[m2]).astype(np.float32)
            h_ransac = clip_height_grid(h_ransac, 0.0, float(args.height_clip_max))
        h_ransac_valid = np.isfinite(h_ransac)

        # choose height
        floor_metrics = height_source_metrics(h_floor_valid)
        ransac_metrics = height_source_metrics(h_ransac_valid)
        ransac_metrics["model_ok"] = bool(abc is not None)

        candidates = []
        candidates.append(("floorref", h_floor, h_floor_valid, floor_metrics))
        if abc is not None:
            candidates.append(("ransac", h_ransac, h_ransac_valid, ransac_metrics))

        mode_used = "floorref"
        fallback_used = False
        if args.ground_mode == "floorref":
            h_use = h_floor; h_use_valid = h_floor_valid; mode_used = "floorref"
        elif args.ground_mode == "ransac":
            if abc is not None and ransac_metrics["valid_cells"] > 0:
                h_use = h_ransac; h_use_valid = h_ransac_valid; mode_used = "ransac"
            else:
                h_use = h_floor; h_use_valid = h_floor_valid; mode_used = "floorref_fallback"
                fallback_used = True
        else:
            min_ratio = float(args.height_select_min_ratio)
            min_cells = int(args.height_select_min_cells)
            usable = [c for c in candidates if (c[3]["ratio"] >= min_ratio and c[3]["valid_cells"] >= min_cells)]
            if usable:
                usable.sort(key=lambda x: (x[3]["ratio"], x[3]["valid_cells"]), reverse=True)
                mode_used, h_use, h_use_valid, _ = usable[0]
            else:
                candidates.sort(key=lambda x: (x[3]["ratio"], x[3]["valid_cells"]), reverse=True)
                mode_used, h_use, h_use_valid, _ = candidates[0]
                fallback_used = True

        # objects
        objs = extract_objects_from_grid(
            semS, h_use, id_to_name, cell_m,
            max_objects=int(args.max_objects),
            min_area_cells=int(args.min_area_cells),
        )

        cost = default_cost_from_sem(semS, id_to_name)

        # pack height (values + valid_mask). Important: no NaN->0 ambiguity anymore.
        def pack_height(h: np.ndarray):
            m = np.isfinite(h)
            v = np.where(m, h, 0.0).astype(np.float32)
            return v, m.astype(np.uint8)

        h_use_v, h_use_m = pack_height(h_use)
        h_floor_v, h_floor_m = pack_height(h_floor)
        h_ran_v,  h_ran_m  = pack_height(h_ransac)

        state = {
            "t_ns": t,
            "pose": {
                "p_WB": [float(x) for x in r["p_WB"]],
                "yaw_rad": float(yaw),
                "yaw_source": "body",
            },
            "map": {
                "N": int(N),
                "local_size_m": float(args.local_size_m),
                "cell_m": float(cell_m),
                "global_res_m": float(res),
                "ego_align": bool(args.ego_align),
            },
            "semantic_grid": semS.tolist(),
            "unknown_mask": (semS == 0).astype(np.uint8).tolist(),

            # heights
            "height_rel_grid": h_use_v.tolist(),
            "height_valid_mask": h_use_m.tolist(),
            "height_rel_grid_floorref": h_floor_v.tolist(),
            "height_valid_mask_floorref": h_floor_m.tolist(),
            "height_rel_grid_ransac": h_ran_v.tolist(),
            "height_valid_mask_ransac": h_ran_m.tolist(),

            "ground": {
                "mode_used": mode_used,
                "floor_ref_z_m": float(z_floor),
                "ransac_plane_z_axbyc": [float(x) for x in abc] if abc is not None else None,
                "ransac_stats": rstats,   # ALWAYS exists
            },
            "height_select": {
                "mode_used": mode_used,
                "fallback_used": bool(fallback_used),
                "min_ratio": float(args.height_select_min_ratio),
                "min_cells": int(args.height_select_min_cells),
                "candidates": {
                    "floorref": floor_metrics,
                    "ransac": ransac_metrics,
                },
            },

            "objects": objs,
            "cost_grid_default": cost.tolist(),
            "stats": {
                "unknown_ratio": float((semS == 0).mean()),
                "valid_height_ratio": float(h_use_valid.mean()),
                "valid_height_ratio_floorref": float(h_floor_valid.mean()),
                "valid_height_ratio_ransac": float(h_ransac_valid.mean()),
                "valid_height_cells": int(h_use_valid.sum()),
                "valid_height_cells_floorref": int(h_floor_valid.sum()),
                "valid_height_cells_ransac": int(h_ransac_valid.sum()),
                "sem_removed_small_cells": int(sem_removed),
            }
        }

        sp = os.path.join(out_states, f"state_{t}.json")
        write_json(sp, state)
        idx_f.write(json.dumps({"t_ns": t, "state_path": sp}, ensure_ascii=False) + "\n")

        if args.export_square_states and idx_sq_f is not None:
            sem_rot, M_sq, sx0, sy0, side, ang = align_and_square_region(
                semS,
                min_known_cells=int(args.square_min_known_cells),
                pad_cells=int(args.square_pad_cells),
            )
            square_mask = square_mask_from_xy(sem_rot.shape, sx0, sy0, side)
            sem_sq = np.zeros_like(sem_rot, dtype=np.uint8)
            sem_sq[square_mask] = sem_rot[square_mask]
            fill_unknown_floor = not bool(args.no_square_fill_unknown_floor)
            if fill_unknown_floor and floor_id > 0:
                sem_sq = sem_sq.copy()
                sem_sq[square_mask] = int(floor_id)
                keep = square_mask & (sem_rot > 0)
                sem_sq[keep] = sem_rot[keep]

            def warp_h(h, fill_value):
                hw = cv2.warpAffine(h, M_sq, (N, N), flags=cv2.INTER_NEAREST, borderValue=fill_value)
                out = hw.copy().astype(np.float32)
                if np.isfinite(fill_value):
                    out[~square_mask] = float(fill_value)
                else:
                    out[~square_mask] = np.nan
                return out

            def warp_m(m):
                mw = cv2.warpAffine(m.astype(np.uint8), M_sq, (N, N), flags=cv2.INTER_NEAREST, borderValue=0)
                out = mw.copy().astype(np.uint8)
                out[~square_mask] = 0
                return out

            h_use_sq = warp_h(h_use, float("nan"))
            h_floor_sq = warp_h(h_floor, float("nan"))
            h_ran_sq = warp_h(h_ransac, float("nan"))
            h_use_m_sq = warp_m(h_use_valid.astype(np.uint8))
            h_floor_m_sq = warp_m(h_floor_valid.astype(np.uint8))
            h_ran_m_sq = warp_m(h_ransac_valid.astype(np.uint8))

            N_sq = int(N)
            if int(args.square_force_N) > 0 and int(args.square_force_N) != int(N_sq):
                N_sq = int(args.square_force_N)
                sem_sq = cv2.resize(sem_sq, (N_sq, N_sq), interpolation=cv2.INTER_NEAREST)
                h_use_sq = resize_with_nan(h_use_sq, N_sq)
                h_floor_sq = resize_with_nan(h_floor_sq, N_sq)
                h_ran_sq = resize_with_nan(h_ran_sq, N_sq)
                h_use_m_sq = cv2.resize(h_use_m_sq, (N_sq, N_sq), interpolation=cv2.INTER_NEAREST)
                h_floor_m_sq = cv2.resize(h_floor_m_sq, (N_sq, N_sq), interpolation=cv2.INTER_NEAREST)
                h_ran_m_sq = cv2.resize(h_ran_m_sq, (N_sq, N_sq), interpolation=cv2.INTER_NEAREST)

            cell_m_sq = float(args.local_size_m) / float(N_sq)
            objs_sq = extract_objects_from_grid(
                sem_sq, h_use_sq, id_to_name, cell_m_sq,
                max_objects=int(args.max_objects),
                min_area_cells=int(args.min_area_cells),
            )
            cost_sq = default_cost_from_sem(sem_sq, id_to_name)

            def pack_height_sq(h: np.ndarray):
                m = np.isfinite(h)
                v = np.where(m, h, 0.0).astype(np.float32)
                return v, m.astype(np.uint8)

            h_use_v_sq, h_use_m_sq = pack_height_sq(h_use_sq)
            h_floor_v_sq, h_floor_m_sq = pack_height_sq(h_floor_sq)
            h_ran_v_sq, h_ran_m_sq = pack_height_sq(h_ran_sq)

            state_sq = {
                **state,
                "map": {
                    **state["map"],
                    "N": int(N_sq),
                    "cell_m": float(cell_m_sq),
                    "square_align": True,
                },
                "semantic_grid": sem_sq.tolist(),
                "unknown_mask": (sem_sq == 0).astype(np.uint8).tolist(),
                "height_rel_grid": h_use_v_sq.tolist(),
                "height_valid_mask": h_use_m_sq.tolist(),
                "height_rel_grid_floorref": h_floor_v_sq.tolist(),
                "height_valid_mask_floorref": h_floor_m_sq.tolist(),
                "height_rel_grid_ransac": h_ran_v_sq.tolist(),
                "height_valid_mask_ransac": h_ran_m_sq.tolist(),
                "objects": objs_sq,
                "cost_grid_default": cost_sq.tolist(),
                "stats": {
                    **state["stats"],
                    "unknown_ratio": float((sem_sq == 0).mean()),
                    "valid_height_ratio": float(np.isfinite(h_use_sq).mean()),
                    "valid_height_ratio_floorref": float(np.isfinite(h_floor_sq).mean()),
                    "valid_height_ratio_ransac": float(np.isfinite(h_ran_sq).mean()),
                    "valid_height_cells": int(np.isfinite(h_use_sq).sum()),
                    "valid_height_cells_floorref": int(np.isfinite(h_floor_sq).sum()),
                    "valid_height_cells_ransac": int(np.isfinite(h_ran_sq).sum()),
                },
                "square_meta": {
                    "angle_deg": float(ang),
                    "pad_cells": int(args.square_pad_cells),
                    "fill_unknown_floor": bool(fill_unknown_floor),
                    "force_N": int(N_sq),
                }
            }

            sp_sq = os.path.join(out_states_square, f"state_{t}.json")
            write_json(sp_sq, state_sq)
            idx_sq_f.write(json.dumps({"t_ns": t, "state_path": sp_sq}, ensure_ascii=False) + "\n")
        processed += 1

    idx_f.close()
    if idx_sq_f is not None:
        idx_sq_f.close()

    np.save(os.path.join(out_global, "global_semantic.npy"), g_sem)
    gzt_save = g_ztop.copy()
    gzt_save[gzt_save <= -1e8] = np.nan
    np.save(os.path.join(out_global, "global_ztop.npy"), gzt_save)
    np.save(os.path.join(out_global, "global_last.npy"), g_last)

    summary = {
        "processed_frames": processed,
        "global": {"GW": GW, "GH": GH, "res_m": res, "xmin": float(xmin), "ymin": float(ymin),
                   "xmax": float(xmax), "ymax": float(ymax)},
        "dirs": {
            "states": out_states,
            "global_map": out_global,
            "states_square": out_states_square if out_states_square else None
        },
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("[Step7a-v4] done. processed_frames =", processed)
    print("[Step7a-v4] outputs:", args.out_dir)


if __name__ == "__main__":
    main()



'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step7a_build_global_and_states.py \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --depth_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/depth_npy \
  --mask_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/masks_clean \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --stereo_meta /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/stereo_meta.json \
  --mav0_dir /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/step7_global_states_v4_N60 \
  --local_size_m 30 --N 60 --pix_step 1 --ego_align \
  --ground_mode auto

'''
