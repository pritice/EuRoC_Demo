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


# -------------------------
# Height grid extraction from state
# -------------------------
def get_height_grid_from_state(state: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    尽可能从 state 里找高度网格与有效mask。
    兼容你前面提到的：state 里不一定有 ransac_stats / valid_height_ratio_ransac。
    """
    # 常见候选键（按优先级）
    candidates = [
        ("height_rel_grid_ransac", "height_valid_mask_ransac"),
        ("height_rel_grid_floorref", "height_valid_mask_floorref"),
        ("height_rel_grid", "height_valid_mask"),
        ("height_grid_m", "height_valid_mask"),
        ("height_grid", "height_valid_mask"),
    ]
    for hk, mk in candidates:
        if hk in state and mk in state:
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
                          annotate_topk: int = 12):
    sem = np.array(state["semantic_grid"], dtype=np.uint8)
    N = int(sem.shape[0])

    # map meta
    mp = state.get("map", {}) or {}
    cell_m = float(mp.get("cell_m", mp.get("cell_size_m", 1.0)))
    L = float(mp.get("local_size_m", mp.get("L", N * cell_m)))
    half = L / 2.0

    # rotate by 90deg*k for display (both sem & later annotation coords)
    sem_show = rot90_semantic(sem, rot90_k)

    base_rgb = render_semantic_rgb(sem_show)

    if mode == "sem25d":
        h, hm = get_height_grid_from_state(state)
        if h is not None:
            h = rot90_semantic(h, rot90_k)
        if hm is not None:
            hm = rot90_semantic(hm, rot90_k)
        rgb_show = render_25d_rgb(sem_show, base_rgb, h, hm, hmax=hmax)
        title = f"Local 2.5D map (semantic color + height brightness, hmax={hmax:.1f}m)"
    else:
        rgb_show = base_rgb
        title = "Local semantic map"

    # left: top-down map (unknown transparent)
    rgba = to_rgba(rgb_show, sem_show, unknown_transparent=True)
    ax_map.set_facecolor("white")
    ax_map.imshow(rgba, origin="lower", extent=[-half, half, -half, half], interpolation="nearest")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlim(-half, half)
    ax_map.set_ylim(-half, half)
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.set_title(title)

    # grid lines
    if show_grid:
        for i in range(N + 1):
            v = -half + i * cell_m
            ax_map.plot([v, v], [-half, half], linewidth=0.3, color="black", alpha=0.15)
            ax_map.plot([-half, half], [v, v], linewidth=0.3, color="black", alpha=0.15)

    # ego position
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
    ax_mat.imshow(rgb_mat, origin="lower", extent=[0, N, 0, N], interpolation="nearest")
    for i in range(N + 1):
        ax_mat.plot([i, i], [0, N], color="black", linewidth=0.3, alpha=0.6)
        ax_mat.plot([0, N], [i, i], color="black", linewidth=0.3, alpha=0.6)

    # optional text per cell
    if show_height_in_matrix and mode == "sem25d":
        h, hm = get_height_grid_from_state(state)
        if h is not None:
            h = rot90_semantic(h, rot90_k)
        for yy in range(N):
            for xx in range(N):
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
        for yy in range(N):
            for xx in range(N):
                cid = int(sem_show[yy, xx])
                if cid == 0:
                    continue
                ax_mat.text(xx + 0.5, yy + 0.5, f"{cid}",
                            ha="center", va="center", fontsize=4, color="black")

    ax_mat.scatter([N / 2], [N / 2], marker="*", s=260, color="yellow",
                   edgecolor="black", linewidth=1.0)
    ax_mat.set_xlim(0, N)
    ax_mat.set_ylim(0, N)
    ax_mat.set_xticks([])
    ax_mat.set_yticks([])
    ax_mat.set_title("Matrix (NxN)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True, help="directory containing state_*.json")
    ap.add_argument("--labels_json", required=True, help="labels.json with class_to_id mapping")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--t_ns", default="", help="which state to render; empty -> latest")

    ap.add_argument("--rot90", type=int, default=0, help="display rotate by 90deg*k (0/1/2/3)")
    ap.add_argument("--annotate_topk", type=int, default=12)
    ap.add_argument("--show_grid", action="store_true")

    ap.add_argument("--save_25d", action="store_true")
    ap.add_argument("--hmax_25d", type=float, default=2.5)
    ap.add_argument("--show_height_in_matrix", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    labels = read_json(args.labels_json)
    class_to_id = labels.get("class_to_id", {})
    id_to_name = {int(v): k for k, v in class_to_id.items()}
    id_to_name[0] = "unknown"

    state_path = find_state_file(args.states_dir, args.t_ns)
    state = read_json(state_path)
    t = state.get("t_ns", "")
    print("[load]", state_path)

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