#clean 后每个实体尽量是整块整块、减少交叉混杂
# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def color_for_id(i: int):
    return (int((i*37) % 255), int((i*17) % 255), int((i*67) % 255))

def render_sem_grid_rgb(semN: np.ndarray) -> np.ndarray:
    H,W = semN.shape
    rgb = np.zeros((H,W,3), dtype=np.uint8)
    for cid in np.unique(semN):
        cid = int(cid)
        if cid <= 0: 
            continue
        rgb[semN == cid] = np.array(color_for_id(cid), dtype=np.uint8)
    return rgb

def isolated_cell_ratio(semN: np.ndarray) -> float:
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
    total = 0
    ids = [int(x) for x in np.unique(semN) if int(x) > 0]
    for cid in ids:
        m = (semN == cid).astype(np.uint8)
        ncc, _ = cv2.connectedComponents(m, connectivity=4)
        total += (ncc - 1)
    return int(total)

def grid_majority_smooth(semN: np.ndarray,
                         unknown_id=0,
                         preserve_ids=None,
                         iters=6,
                         w_self=2,
                         allow_fill_unknown=False):
    """
    Iterative majority smoothing on NxN grid.
    - preserve_ids: set of class ids that should be very stable (e.g., floor/road/wall).
    - allow_fill_unknown: if True, unknown cells can be filled by neighbors; otherwise keep 0.
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

                # keep unknown if not allowed to fill
                if c0 == unknown_id and (not allow_fill_unknown):
                    continue

                # preserve strong classes
                if c0 in preserve_ids:
                    continue

                nb = pad[y:y+3, x:x+3].reshape(-1)
                if not allow_fill_unknown:
                    nb = nb[nb != unknown_id]
                if nb.size == 0:
                    continue

                # voting
                bc = np.bincount(nb.astype(np.int64))
                # add self weight
                if c0 >= 0 and c0 < bc.size:
                    bc[c0] += w_self
                c1 = int(bc.argmax())

                # if winner is unknown and we don't fill unknown, skip
                if (c1 == unknown_id) and (not allow_fill_unknown):
                    continue

                new[y,x] = c1
        sem = new

    return sem.astype(np.uint8)

def draw_matrix(ax, semN, title="", with_text=False):
    N = semN.shape[0]
    rgb = render_sem_grid_rgb(semN)
    ax.imshow(rgb, origin="lower", extent=[0,N,0,N], interpolation="nearest")
    for i in range(N+1):
        ax.plot([i,i],[0,N], color="black", linewidth=0.3, alpha=0.6)
        ax.plot([0,N],[i,i], color="black", linewidth=0.3, alpha=0.6)
    if with_text:
        for y in range(N):
            for x in range(N):
                ax.text(x+0.5, y+0.5, str(int(semN[y,x])),
                        ha="center", va="center", fontsize=6, color="black")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0,N); ax.set_ylim(0,N)
    ax.set_title(title)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--sem_raw_npy", required=True)
    ap.add_argument("--sem_clean_npy", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--w_self", type=int, default=2)
    ap.add_argument("--allow_fill_unknown", action="store_true")
    ap.add_argument("--with_text", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    labels = json.load(open(args.labels_json, "r"))
    class_to_id = labels.get("class_to_id", {})
    # preserve common indoor stable classes if exist
    preserve_names = ["floor", "road", "wall", "ceiling"]
    preserve_ids = set(int(class_to_id[n]) for n in preserve_names if n in class_to_id)

    sem_raw = np.load(args.sem_raw_npy).astype(np.uint8)
    sem_cln = np.load(args.sem_clean_npy).astype(np.uint8)

    sem_smooth = grid_majority_smooth(
        sem_cln,
        unknown_id=0,
        preserve_ids=preserve_ids,
        iters=args.iters,
        w_self=args.w_self,
        allow_fill_unknown=bool(args.allow_fill_unknown),
    )

    stats = {
        "raw": {
            "isolated_cell_ratio": isolated_cell_ratio(sem_raw),
            "component_count_total": component_count_total(sem_raw),
        },
        "pixel_clean": {
            "isolated_cell_ratio": isolated_cell_ratio(sem_cln),
            "component_count_total": component_count_total(sem_cln),
        },
        "grid_smooth": {
            "isolated_cell_ratio": isolated_cell_ratio(sem_smooth),
            "component_count_total": component_count_total(sem_smooth),
        },
        "params": {
            "iters": int(args.iters),
            "w_self": int(args.w_self),
            "allow_fill_unknown": bool(args.allow_fill_unknown),
            "preserve_ids": sorted(list(preserve_ids)),
        }
    }
    with open(os.path.join(args.out_dir, "grid_smooth_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1,3,1)
    draw_matrix(ax1, sem_raw,
                title=f"RAW\niso={stats['raw']['isolated_cell_ratio']:.3f}, cc={stats['raw']['component_count_total']}",
                with_text=args.with_text)
    ax2 = fig.add_subplot(1,3,2)
    draw_matrix(ax2, sem_cln,
                title=f"Pixel-clean\niso={stats['pixel_clean']['isolated_cell_ratio']:.3f}, cc={stats['pixel_clean']['component_count_total']}",
                with_text=args.with_text)
    ax3 = fig.add_subplot(1,3,3)
    draw_matrix(ax3, sem_smooth,
                title=f"Grid-smooth (iters={args.iters})\niso={stats['grid_smooth']['isolated_cell_ratio']:.3f}, cc={stats['grid_smooth']['component_count_total']}",
                with_text=args.with_text)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "grid_smooth_comparison.png")
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    np.save(os.path.join(args.out_dir, "sem_grid_smooth.npy"), sem_smooth)

    print("[Step6f] saved:", out_png)
    print("[Step6f] stats:", stats)

if __name__ == "__main__":
    main()

'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step6f_grid_smooth_and_compare.py \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --sem_raw_npy /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/final_compare/sem_raw_t1413393925705760512.npy \
  --sem_clean_npy /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/final_compare/sem_clean_t1413393925705760512.npy \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/final_compare_grid_smooth \
  --iters 8 --w_self 2
'''
'''
grid_smooth_comparison.png：三列对比（Raw / Pixel-clean / Grid-smooth）

grid_smooth_stats.json：指标对比（会非常直观地下降）

sem_grid_smooth.npy：可直接给 Step7 用的最终语义矩阵'''