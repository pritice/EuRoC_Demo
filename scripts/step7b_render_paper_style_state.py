# -*- coding: utf-8 -*-
import os, json, argparse, glob
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# -------------------------
# I/O
# -------------------------
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def find_state_file(states_dir: str, t_ns: Optional[str]) -> str:
    files = sorted(glob.glob(os.path.join(states_dir, "state_*.json")))
    if not files:
        raise FileNotFoundError(f"no state_*.json found in {states_dir}")

    if t_ns is None or t_ns == "":
        return files[-1]

    t_ns = str(t_ns)
    # exact match by filename
    cand = os.path.join(states_dir, f"state_{t_ns}.json")
    if os.path.isfile(cand):
        return cand

    # fallback: scan for matching t_ns inside json (slow but safe)
    for p in files:
        try:
            s = read_json(p)
            if str(s.get("t_ns", "")) == t_ns:
                return p
        except Exception:
            pass
    raise FileNotFoundError(f"cannot find state for t_ns={t_ns} in {states_dir}")


# -------------------------
# Style consistent with step6d
# -------------------------
def color_for_id(i: int) -> Tuple[int, int, int]:
    # same simple deterministic palette (like 6d-style)
    return (int((i * 37) % 255), int((i * 17) % 255), int((i * 67) % 255))

def render_semantic_rgb(sem: np.ndarray) -> np.ndarray:
    H, W = sem.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid in np.unique(sem):
        cid = int(cid)
        if cid <= 0:
            continue
        rgb[sem == cid] = np.array(color_for_id(cid), dtype=np.uint8)
    return rgb

def build_legend_handles(sem: np.ndarray,
                         id_to_name: Dict[int, str],
                         max_items: int = 14):
    ids = [int(x) for x in np.unique(sem) if int(x) != 0]
    ids = sorted(ids)[:max_items]
    handles = []
    for cid in ids:
        name = id_to_name.get(cid, f"id{cid}")
        col = np.array(color_for_id(cid), dtype=np.float32) / 255.0
        handles.append(mpatches.Patch(color=col, label=name))
    return handles

def to_rgba(rgb_u8: np.ndarray, sem: np.ndarray, unknown_transparent: bool = True) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    if unknown_transparent:
        a = (sem != 0).astype(np.float32)
    else:
        a = np.ones_like(sem, dtype=np.float32)
    return np.dstack([rgb, a])

def render_25d_rgb(sem: np.ndarray,
                   base_rgb_u8: np.ndarray,
                   hgrid: Optional[np.ndarray],
                   hmask: Optional[np.ndarray],
                   hmax: float = 2.5) -> np.ndarray:
    """
    颜色=语义，亮度=高度（仅对有效高度调制）
    - known 但无高度：乘 0.80（避免过暗）
    - valid height：0.65~1.00
    """
    rgb = base_rgb_u8.astype(np.float32) / 255.0
    if hgrid is None or hmask is None:
        out = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        out[sem == 0] = 0
        return out

    h = hgrid.astype(np.float32)
    m = (hmask.astype(np.uint8) > 0) & (sem != 0) & np.isfinite(h)

    mult = np.full(sem.shape, 0.80, dtype=np.float32)
    if np.any(m):
        hh = np.clip(h[m], 0.0, float(hmax))
        hn = hh / float(hmax)
        mult[m] = 0.65 + 0.35 * hn

    rgb2 = rgb * mult[..., None]
    out = (rgb2 * 255.0).clip(0, 255).astype(np.uint8)
    out[sem == 0] = 0
    return out

def rot90_semantic(sem: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % 4
    if k == 0:
        return sem
    return np.rot90(sem, k=k)

def rot90_xy_pix(x: float, y: float, N: int, k: int) -> Tuple[float, float]:
    k = int(k) % 4
    if k == 0:
        return x, y
    if k == 1:
        return (N - 1 - y), x
    if k == 2:
        return (N - 1 - x), (N - 1 - y)
    return y, (N - 1 - x)

def apply_affine_xy(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2],
            M[1, 0] * x + M[1, 1] * y + M[1, 2])

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

def crop_square_2d(arr: np.ndarray, x0: int, y0: int, side: int, fill_value):
    out = np.full((side, side), fill_value, dtype=arr.dtype)
    x1 = x0 + side
    y1 = y0 + side
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(arr.shape[1], x1)
    sy1 = min(arr.shape[0], y1)
    dx0 = sx0 - x0
    dy0 = sy0 - y0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    return out

def align_and_crop_sem(sem: np.ndarray,
                       min_known_cells: int,
                       pad_cells: int,
                       force_square: bool):
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
    if force_square:
        side = int(max(x1 - x0, y1 - y0))
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        x0 = int(np.floor(cx - side / 2.0))
        y0 = int(np.floor(cy - side / 2.0))
    side = int(max(x1 - x0, y1 - y0))
    sem_crop = crop_square_2d(sem_rot, x0, y0, side, fill_value=0)
    return sem_crop, M, x0, y0, side, angle

def rot90_xy_meter(x: float, y: float, k: int) -> Tuple[float, float]:
    """
    对局部坐标 (x,y) 做 90deg*k 的逆时针旋转（用于显示对齐）
    """
    k = int(k) % 4
    if k == 0:
        return x, y
    if k == 1:
        return -y, x
    if k == 2:
        return -x, -y
    return y, -x


def resize_with_nan(h: np.ndarray, out_size: int) -> np.ndarray:
    h = h.astype(np.float32)
    valid = np.isfinite(h).astype(np.uint8)
    h2 = h.copy()
    h2[~np.isfinite(h2)] = 0.0
    h_res = cv2.resize(h2, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    m_res = cv2.resize(valid, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    h_res[m_res == 0] = np.nan
    return h_res


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


# -------------------------
# Height grid extraction from state
# -------------------------
def get_height_grid_from_state(state: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    尽可能从 state 里找高度网格与有效mask。
    兼容你前面提到的：state 里不一定有 ransac_stats / valid_height_ratio_ransac。
    """
    # 优先用 state 记录的选择结果
    select = state.get("height_select", {}) or {}
    mode = select.get("mode_used", "") or (state.get("ground", {}) or {}).get("mode_used", "")
    if isinstance(mode, str):
        mode = mode.replace("_fallback", "")
    mode_map = {
        "ransac": ("height_rel_grid_ransac", "height_valid_mask_ransac"),
        "floorref": ("height_rel_grid_floorref", "height_valid_mask_floorref"),
    }
    if mode in mode_map:
        hk, mk = mode_map[mode]
        if hk in state and mk in state:
            try:
                h = np.array(state[hk], dtype=np.float32)
                m = np.array(state[mk], dtype=np.uint8)
                return h, m
            except Exception:
                pass

    # 按有效率选择候选
    candidates = [
        ("height_rel_grid_ransac", "height_valid_mask_ransac", "valid_height_ratio_ransac"),
        ("height_rel_grid_floorref", "height_valid_mask_floorref", "valid_height_ratio_floorref"),
        ("height_rel_grid", "height_valid_mask", "valid_height_ratio"),
        ("height_grid_m", "height_valid_mask", "valid_height_ratio"),
        ("height_grid", "height_valid_mask", "valid_height_ratio"),
    ]
    stats = state.get("stats", {}) or {}
    scored = []
    for hk, mk, rk in candidates:
        if hk in state and mk in state:
            try:
                m = np.array(state[mk], dtype=np.uint8)
                ratio = float(stats.get(rk, float(m.mean())))
                scored.append((ratio, hk, mk))
            except Exception:
                pass
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        _, hk, mk = scored[0]
        try:
            h = np.array(state[hk], dtype=np.float32)
            m = np.array(state[mk], dtype=np.uint8)
            return h, m
        except Exception:
            pass

    # 如果只有高度没有mask：用 finite & sem!=0
    for hk in ["height_rel_grid", "height_grid_m", "height_grid"]:
        if hk in state:
            try:
                h = np.array(state[hk], dtype=np.float32)
                m = np.isfinite(h).astype(np.uint8)
                return h, m
            except Exception:
                pass

    return None, None


# -------------------------
# Object annotation from state["objects"]
# -------------------------
def collect_objects_for_annot(state: Dict[str, Any], max_n: int = 15) -> List[Dict[str, Any]]:
    objs = state.get("objects", []) or []
    # 尝试按面积/占比排序（没有就按原顺序）
    def score(o):
        # 兼容字段
        for k in ["area_cells", "area", "mask_area", "area_px"]:
            if k in o and isinstance(o[k], (int, float)):
                return float(o[k])
        return 0.0
    objs_sorted = sorted(objs, key=score, reverse=True)
    return objs_sorted[:max_n]


# -------------------------
# Paper-style drawing (step6d-like)
# -------------------------
def draw_paper_style_state(ax_map, ax_mat, state: Dict[str, Any], id_to_name: Dict[int, str],
                          *,
                          mode: str = "sem",  # "sem" or "sem25d"
                          hmax: float = 2.5,
                          rot90_k: int = 0,
                          show_grid: bool = True,
                          show_height_in_matrix: bool = False,
                          annotate_topk: int = 12,
                          unknown_opaque: bool = False,
                          align_known_square: bool = False,
                          min_known_cells: int = 60,
                          crop_pad_cells: int = 2,
                          fill_unknown_floor: bool = False,
                          floor_id: int = 0,
                          force_size: int = 0):
    sem = np.array(state["semantic_grid"], dtype=np.uint8)
    N = int(sem.shape[0])

    # map meta
    mp = state.get("map", {}) or {}
    cell_m = float(mp.get("cell_m", mp.get("cell_size_m", 1.0)))
    L = float(mp.get("local_size_m", mp.get("L", N * cell_m)))
    half = L / 2.0

    # rotate by 90deg*k for display (both sem & later annotation coords)
    sem_show = rot90_semantic(sem, rot90_k)

    square_mask = None
    if align_known_square:
        sem_rot, M_align, sx0, sy0, side, _ = align_and_square_region(
            sem_show,
            min_known_cells=int(min_known_cells),
            pad_cells=int(crop_pad_cells),
        )
        square_mask = square_mask_from_xy(sem_rot.shape, sx0, sy0, side)
        sem_show = np.zeros_like(sem_rot, dtype=np.uint8)
        sem_show[square_mask] = sem_rot[square_mask]
        crop_x0, crop_y0 = 0, 0
        N_show = int(sem_show.shape[0])
    else:
        M_align = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        crop_x0, crop_y0 = 0, 0
        N_show = int(sem_show.shape[0])

    if fill_unknown_floor and int(floor_id) > 0 and square_mask is not None:
        sem_show = sem_show.copy()
        sem_show[square_mask] = int(floor_id)
        keep = square_mask & (sem_rot > 0)
        sem_show[keep] = sem_rot[keep]

    left_m = (float(crop_x0) * cell_m) - half
    right_m = (float(crop_x0 + N_show) * cell_m) - half
    bottom_m = (float(crop_y0) * cell_m) - half
    top_m = (float(crop_y0 + N_show) * cell_m) - half

    h = hm = None
    if mode == "sem25d":
        h, hm = get_height_grid_from_state(state)
        if h is not None:
            h = rot90_semantic(h, rot90_k)
            if align_known_square:
                h = cv2.warpAffine(h, M_align, (N, N), flags=cv2.INTER_NEAREST, borderValue=np.nan)
                if square_mask is not None:
                    h2 = np.full_like(h, np.nan, dtype=np.float32)
                    h2[square_mask] = h[square_mask]
                    h = h2
        if hm is not None:
            hm = rot90_semantic(hm, rot90_k)
            if align_known_square:
                hm = cv2.warpAffine(hm, M_align, (N, N), flags=cv2.INTER_NEAREST, borderValue=0)
                if square_mask is not None:
                    hm2 = np.zeros_like(hm, dtype=np.uint8)
                    hm2[square_mask] = hm[square_mask]
                    hm = hm2

    if int(force_size) > 0 and int(force_size) != int(N_show):
        sem_show = cv2.resize(sem_show, (int(force_size), int(force_size)), interpolation=cv2.INTER_NEAREST)
        if h is not None:
            h = resize_with_nan(h, int(force_size))
        if hm is not None:
            hm = cv2.resize(hm.astype(np.uint8), (int(force_size), int(force_size)), interpolation=cv2.INTER_NEAREST)
        N_show = int(force_size)

    base_rgb = render_semantic_rgb(sem_show)

    if mode == "sem25d":
        rgb_show = render_25d_rgb(sem_show, base_rgb, h, hm, hmax=hmax)
        title = f"Local 2.5D map (semantic color + height brightness, hmax={hmax:.1f}m)"
    else:
        rgb_show = base_rgb
        title = "Local semantic map"

    cell_m_show = float((right_m - left_m) / max(N_show, 1))

    # left: top-down map (unknown transparent)
    rgba = to_rgba(rgb_show, sem_show, unknown_transparent=(not unknown_opaque))
    ax_map.set_facecolor("white")
    ax_map.imshow(rgba, origin="lower", extent=[left_m, right_m, bottom_m, top_m], interpolation="nearest")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlim(left_m, right_m)
    ax_map.set_ylim(bottom_m, top_m)
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.set_title(title)

    # grid lines
    if show_grid:
        for i in range(N_show + 1):
            vx = left_m + i * cell_m_show
            vy = bottom_m + i * cell_m_show
            ax_map.plot([vx, vx], [bottom_m, top_m], linewidth=0.3, color="black", alpha=0.15)
            ax_map.plot([left_m, right_m], [vy, vy], linewidth=0.3, color="black", alpha=0.15)

    # ego position
    if (left_m <= 0.0 <= right_m) and (bottom_m <= 0.0 <= top_m):
        ax_map.scatter([0], [0], marker="*", s=260, color="yellow",
                       edgecolor="black", linewidth=1.0, zorder=5)

    # legend bottom-left
    handles = build_legend_handles(sem_show, id_to_name, max_items=14)
    if handles:
        ax_map.legend(handles=handles, loc="lower left",
                      framealpha=0.85, fontsize=6,
                      borderpad=0.4, handlelength=1.2, labelspacing=0.3)

    # object annotations (use centroid_cell_xy from original grid, rotate coord for display)
    objs = collect_objects_for_annot(state, max_n=annotate_topk)
    for o in objs:
        name = str(o.get("class_name", o.get("name", "obj")))
        h = o.get("height_m", None)
        Lw = o.get("size_m_LW", o.get("size_m", [None, None]))
        try:
            l_m = float(Lw[0]) if Lw[0] is not None else None
            w_m = float(Lw[1]) if Lw[1] is not None else None
        except Exception:
            l_m, w_m = None, None

        cxcy = o.get("centroid_cell_xy", o.get("centroid_xy", None))
        if not cxcy or len(cxcy) < 2:
            continue
        cx, cy = float(cxcy[0]), float(cxcy[1])

        # original cell -> meters
        x = (cx + 0.5) * cell_m - half
        y = (cy + 0.5) * cell_m - half

        # rotate display
        x, y = rot90_xy_meter(x, y, rot90_k)
        if align_known_square:
            px = (x + half) / cell_m
            py = (y + half) / cell_m
            px, py = apply_affine_xy(M_align, px, py)
            x = left_m + (px + 0.5) * cell_m_show
            y = bottom_m + (py + 0.5) * cell_m_show

        # text
        if h is None or (isinstance(h, str) and h.strip() == "?"):
            hs = "?"
        else:
            try:
                hs = f"{float(h):.2f}"
            except Exception:
                hs = "?"
        if l_m is not None and w_m is not None:
            txt = f"{name}\nh={hs}m\n{l_m:.2f}x{w_m:.2f}m"
        else:
            txt = f"{name}\nh={hs}m"

        ax_map.text(x, y, txt, fontsize=6, color="black",
                    bbox=dict(facecolor=(1, 1, 1, 0.70),
                              edgecolor="none", boxstyle="round,pad=0.2"))

    # right: matrix view (like step6d)
    ax_mat.set_facecolor("white")
    # unknown -> white for readability
    rgb_mat = rgb_show.copy()
    rgb_mat[sem_show == 0] = 255
    ax_mat.imshow(rgb_mat, origin="lower", extent=[0, N_show, 0, N_show], interpolation="nearest")
    for i in range(N_show + 1):
        ax_mat.plot([i, i], [0, N_show], color="black", linewidth=0.3, alpha=0.6)
        ax_mat.plot([0, N_show], [i, i], color="black", linewidth=0.3, alpha=0.6)

    # optional text per cell
    if show_height_in_matrix and mode == "sem25d":
        h, hm = get_height_grid_from_state(state)
        if h is not None:
            h = rot90_semantic(h, rot90_k)
        for yy in range(N_show):
            for xx in range(N_show):
                cid = int(sem_show[yy, xx])
                if cid == 0:
                    continue
                if h is not None and np.isfinite(h[yy, xx]):
                    ax_mat.text(xx + 0.5, yy + 0.5, f"{cid}\n{h[yy,xx]:.1f}",
                                ha="center", va="center", fontsize=4, color="black")
                else:
                    ax_mat.text(xx + 0.5, yy + 0.5, f"{cid}",
                                ha="center", va="center", fontsize=4, color="black")
    else:
        for yy in range(N_show):
            for xx in range(N_show):
                cid = int(sem_show[yy, xx])
                if cid == 0:
                    continue
                ax_mat.text(xx + 0.5, yy + 0.5, f"{cid}",
                            ha="center", va="center", fontsize=4, color="black")

    ax_mat.scatter([N_show / 2], [N_show / 2], marker="*", s=260, color="yellow",
                   edgecolor="black", linewidth=1.0)
    ax_mat.set_xlim(0, N_show)
    ax_mat.set_ylim(0, N_show)
    ax_mat.set_xticks([])
    ax_mat.set_yticks([])
    ax_mat.set_title("Matrix (NxN)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", default="", help="directory containing state_*.json (default: states_square if exists)")
    ap.add_argument("--labels_json", required=True, help="labels.json with class_to_id mapping")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--t_ns", default="", help="which state to render; empty -> latest")

    ap.add_argument("--rot90", type=int, default=0, help="display rotate by 90deg*k (0/1/2/3)")
    ap.add_argument("--annotate_topk", type=int, default=12)
    ap.add_argument("--show_grid", action="store_true")

    ap.add_argument("--save_25d", action="store_true")
    ap.add_argument("--hmax_25d", type=float, default=2.5)
    ap.add_argument("--show_height_in_matrix", action="store_true")
    ap.add_argument("--unknown_opaque", action="store_true",
                    help="render unknown as opaque (avoid tilted visible area)")
    ap.add_argument("--no_align_known_square", action="store_true",
                    help="disable alignment and square masking")
    ap.add_argument("--min_known_cells", type=int, default=60)
    ap.add_argument("--crop_pad_cells", type=int, default=2)
    ap.add_argument("--no_fill_unknown_floor", action="store_true",
                    help="do not convert unknown to floor in render")
    ap.add_argument("--force_size", type=int, default=0,
                    help="force rendered grid size (0 to disable)")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    labels = read_json(args.labels_json)
    class_to_id = labels.get("class_to_id", {})
    id_to_name = {int(v): k for k, v in class_to_id.items()}
    id_to_name[0] = "unknown"
    floor_id = int(class_to_id.get("floor", 0))

    states_dir = args.states_dir
    if not states_dir:
        cand = os.path.join(os.path.dirname(args.out_dir), "states_square")
        if os.path.isdir(cand):
            states_dir = cand
    if not states_dir:
        raise ValueError("states_dir is required if states_square does not exist")

    state_path = find_state_file(states_dir, args.t_ns)
    state = read_json(state_path)
    t = state.get("t_ns", "")
    print("[load]", state_path)
    align_known_square = (not bool(args.no_align_known_square))
    fill_unknown_floor = (not bool(args.no_fill_unknown_floor))
    force_size = int(args.force_size)

    # semantic figure
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    draw_paper_style_state(
        ax1, ax2, state, id_to_name,
        mode="sem",
        rot90_k=args.rot90,
        show_grid=args.show_grid,
        show_height_in_matrix=args.show_height_in_matrix,
        annotate_topk=args.annotate_topk,
        unknown_opaque=bool(args.unknown_opaque),
        align_known_square=align_known_square,
        min_known_cells=int(args.min_known_cells),
        crop_pad_cells=int(args.crop_pad_cells),
        fill_unknown_floor=fill_unknown_floor,
        floor_id=int(floor_id),
        force_size=force_size,
    )
    plt.tight_layout()
    out1 = os.path.join(args.out_dir, f"paper_style_state_t{t}.png")
    plt.savefig(out1, dpi=220)
    plt.close(fig)
    print("[saved]", out1)

    # 2.5D figure
    if args.save_25d:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        draw_paper_style_state(
            ax1, ax2, state, id_to_name,
            mode="sem25d",
            hmax=float(args.hmax_25d),
            rot90_k=args.rot90,
            show_grid=args.show_grid,
            show_height_in_matrix=True,
            annotate_topk=args.annotate_topk,
            unknown_opaque=bool(args.unknown_opaque),
            align_known_square=align_known_square,
            min_known_cells=int(args.min_known_cells),
            crop_pad_cells=int(args.crop_pad_cells),
            fill_unknown_floor=fill_unknown_floor,
            floor_id=int(floor_id),
            force_size=force_size,
        )
        plt.tight_layout()
        out2 = os.path.join(args.out_dir, f"paper_style_state25d_t{t}.png")
        plt.savefig(out2, dpi=220)
        plt.close(fig)
        print("[saved]", out2)


if __name__ == "__main__":
    main()

'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step7b_render_paper_style_state.py \
  --states_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/step7_global_states_v4_N60/states \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/step7b_paper_viz \
  --save_25d --hmax_25d 2.5 \
  --show_grid \
  --rot90 0

'''
#   --height_source floorref
