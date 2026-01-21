# -*- coding: utf-8 -*-
"""
Step3 (STMR Demo): GroundingDINO + SAM2 生成并缓存语义/实例 mask（仅关键帧 ~200）
- 输入：keyframes_200.jsonl（每行包含 t_ns、cam0_path 等）
- 输出：
  out_dir/
    masks/{t}.png           # uint8 label mask，0=unknown，1..K=classes
    mask_vis/{t}.jpg        # 叠加可视化（便于人工检查）
    frame_meta/{t}.json     # 每帧检测与分割元信息（可写论文）
    labels.json             # 类别映射与超参

关键修复：
- SAM2 build_sam2 必须传 config_name（如 configs/sam2.1/sam2.1_hiera_l.yaml），不能传绝对路径。
  该坑你在旧 Step1/Step2 里已经总结过。参考：step1_groundingdino_detect.py / step2_sam2_instance_masks.py
"""

import os
import sys
import json
import math
import argparse
import inspect
from typing import List, Dict, Tuple

import numpy as np
import cv2
from tqdm import tqdm


# -------------------------
# IO
# -------------------------
def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def ensure_exists(path: str, kind: str):
    ok = os.path.isfile(path) if kind == "file" else os.path.isdir(path)
    if not ok:
        raise FileNotFoundError(f"[PATH NOT FOUND:{kind}] {path}")


# -------------------------
# Geometry utils
# -------------------------
def cxcywh_norm_to_xyxy_px(box: np.ndarray, W: int, H: int) -> Tuple[float, float, float, float]:
    cx, cy, w, h = [float(x) for x in box.tolist()]
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    x1 = float(np.clip(x1, 0, W - 1))
    y1 = float(np.clip(y1, 0, H - 1))
    x2 = float(np.clip(x2, 0, W - 1))
    y2 = float(np.clip(y2, 0, H - 1))
    return x1, y1, x2, y2

def xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-12))

def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> List[int]:
    idxs = scores.argsort()[::-1].tolist()
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        rest = []
        for j in idxs:
            if xyxy_iou(boxes[i], boxes[j]) < iou_th:
                rest.append(j)
        idxs = rest
    return keep

def clamp_box_xyxy(box: List[float], W: int, H: int) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, W - 1))
    y1 = float(np.clip(y1, 0, H - 1))
    x2 = float(np.clip(x2, 0, W - 1))
    y2 = float(np.clip(y2, 0, H - 1))
    if x2 <= x1 + 1:
        x2 = min(float(W - 1), x1 + 2.0)
    if y2 <= y1 + 1:
        y2 = min(float(H - 1), y1 + 2.0)
    return [x1, y1, x2, y2]


# -------------------------
# Text utils
# -------------------------
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace(",", " ").replace(";", " ").replace(":", " ")
    s = " ".join(s.split())
    return s

def build_caption(classes: List[str]) -> str:
    # DINO 推荐用 "a . b . c ." 格式
    return " . ".join(classes) + " ."

def map_phrase_to_class_id(phrase: str, class_to_id: Dict[str, int]) -> int:
    p = normalize_text(phrase)
    # 优先精确匹配，再做包含匹配
    if p in class_to_id:
        return int(class_to_id[p])
    for k in sorted(class_to_id.keys(), key=len, reverse=True):
        if k in p:
            return int(class_to_id[k])
    return 0


# -------------------------
# GroundingDINO (full image) - 参考你 Step1 的 transform 写法
# -------------------------
def load_groundingdino(dino_repo_dir: str, dino_config: str, dino_weights: str):
    if dino_repo_dir not in sys.path:
        sys.path.insert(0, dino_repo_dir)
    from groundingdino.util.inference import load_model  # noqa
    import torch  # noqa

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(dino_config, dino_weights).to(device)
    model.eval()
    return model, device

def dino_predict_full(model, device: str, image_bgr: np.ndarray, caption: str,
                      box_th: float, text_th: float):
    """
    返回：
      boxes_xyxy_px: (N,4) float
      scores: (N,) float
      phrases: list[str]
    """
    if "groundingdino" not in sys.modules:
        # transforms 来自 groundingdino.datasets.transforms
        pass

    from groundingdino.util.inference import predict  # noqa
    from groundingdino.datasets.transforms import Compose, RandomResize, ToTensor, Normalize  # noqa
    from PIL import Image  # noqa
    import torch  # noqa

    transform = Compose([
        RandomResize([800], max_size=1333),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    img_t, _ = transform(pil, None)
    img_t = img_t.to(device)

    boxes, logits, phrases = predict(
        model=model,
        image=img_t,
        caption=caption,
        box_threshold=box_th,
        text_threshold=text_th,
    )

    # boxes: cxcywh normalized
    boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)

    boxes_xyxy = []
    for b in boxes_np:
        x1, y1, x2, y2 = cxcywh_norm_to_xyxy_px(np.array(b, dtype=float), W=W, H=H)
        boxes_xyxy.append([x1, y1, x2, y2])
    boxes_xyxy = np.array(boxes_xyxy, dtype=np.float32)
    scores = logits_np.astype(np.float32).reshape(-1)

    return boxes_xyxy, scores, [str(p) for p in phrases]


# -------------------------
# SAM2 - 参考你 Step1/Step2 的 build 方式（config_name + ckpt_abs）
# -------------------------
def build_sam2_image_predictor(sam2_repo_dir: str, sam2_cfg_name: str, sam2_cfg_abs: str, sam2_ckpt_abs: str):
    import torch

    if sam2_repo_dir not in sys.path:
        sys.path.insert(0, sam2_repo_dir)

    ensure_exists(sam2_cfg_abs, "file")
    ensure_exists(sam2_ckpt_abs, "file")

    from sam2.build_sam import build_sam2  # noqa
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sig = inspect.signature(build_sam2)
    kwargs = {}
    if "device" in sig.parameters:
        kwargs["device"] = device

    # 关键：传 config_name（相对 pkg://sam2），不要传绝对路径
    if "config_file" in sig.parameters:
        model = build_sam2(config_file=sam2_cfg_name, ckpt_path=sam2_ckpt_abs, **kwargs)
    else:
        model = build_sam2(sam2_cfg_name, sam2_ckpt_abs, **kwargs)

    predictor = SAM2ImagePredictor(model)
    predictor_meta = {
        "config_name": sam2_cfg_name,
        "config_abs": sam2_cfg_abs,
        "ckpt_abs": sam2_ckpt_abs,
        "device": device,
    }
    print("[INFO] SAM2 config_name =", sam2_cfg_name)
    print("[INFO] SAM2 cfg(abs)    =", sam2_cfg_abs)
    print("[INFO] SAM2 ckpt(abs)   =", sam2_ckpt_abs)
    print("[INFO] SAM2 device      =", device)
    return predictor, predictor_meta

def sam2_box_to_mask(predictor, image_rgb: np.ndarray, box_xyxy: List[float], multimask_output: bool = False):
    """
    返回：mask_bool(H,W), best_score(float)
    """
    predictor.set_image(image_rgb)
    box = np.array(box_xyxy, dtype=np.float32)

    try:
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=multimask_output)
    except TypeError:
        masks, scores, _ = predictor.predict(box=box, multimask_output=multimask_output)

    if masks is None:
        return None, 0.0

    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)

    # 兼容 masks 形状
    if masks.ndim == 4:
        masks = masks[0]  # (M,H,W)
    if masks.ndim != 3:
        return None, 0.0

    best = int(np.argmax(scores)) if scores.size > 0 else 0
    m = masks[best].astype(bool)
    sc = float(scores[best]) if scores.size > 0 else 0.0
    return m, sc


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--out_dir", required=True)

    # DINO
    ap.add_argument("--dino_repo_dir", required=True)
    ap.add_argument("--dino_config", required=True)
    ap.add_argument("--dino_weights", required=True)
    ap.add_argument("--box_thresh", type=float, default=0.30)
    ap.add_argument("--text_thresh", type=float, default=0.25)
    ap.add_argument("--nms_iou", type=float, default=0.55)
    ap.add_argument("--max_dets", type=int, default=40)

    # SAM2
    ap.add_argument("--sam2_repo_dir", required=True)
    ap.add_argument("--sam2_cfg_name", required=True, help="e.g. configs/sam2.1/sam2.1_hiera_l.yaml (NOT abs path)")
    ap.add_argument("--sam2_cfg_abs", required=True, help="abs path used only for existence check")
    ap.add_argument("--sam2_ckpt_abs", required=True)

    # classes
    ap.add_argument("--classes", type=str, default="wall,door,window,table,chair,sofa,bed,cabinet,box,person,floor")
    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    ensure_exists(args.keyframes, "file")
    ensure_exists(args.dino_repo_dir, "dir")
    ensure_exists(args.dino_config, "file")
    ensure_exists(args.dino_weights, "file")
    ensure_exists(args.sam2_repo_dir, "dir")
    ensure_exists(args.sam2_cfg_abs, "file")
    ensure_exists(args.sam2_ckpt_abs, "file")

    os.makedirs(args.out_dir, exist_ok=True)
    mask_dir = os.path.join(args.out_dir, "masks")
    vis_dir = os.path.join(args.out_dir, "mask_vis")
    meta_dir = os.path.join(args.out_dir, "frame_meta")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    classes = [normalize_text(x) for x in args.classes.split(",") if normalize_text(x)]
    class_to_id = {c: (i + 1) for i, c in enumerate(classes)}  # 0 reserved
    id_to_class = {v: k for k, v in class_to_id.items()}
    caption = build_caption(classes)

    write_json(os.path.join(args.out_dir, "labels.json"), {
        "classes": classes,
        "class_to_id": class_to_id,
        "dino": {
            "box_thresh": args.box_thresh,
            "text_thresh": args.text_thresh,
            "nms_iou": args.nms_iou,
            "max_dets": args.max_dets,
            "caption": caption,
        },
        "sam2": {
            "cfg_name": args.sam2_cfg_name,
            "cfg_abs": args.sam2_cfg_abs,
            "ckpt_abs": args.sam2_ckpt_abs,
        }
    })

    print("[INFO] caption:", caption)

    dino_model, dino_device = load_groundingdino(args.dino_repo_dir, args.dino_config, args.dino_weights)
    sam2_pred, sam2_meta = build_sam2_image_predictor(
        args.sam2_repo_dir, args.sam2_cfg_name, args.sam2_cfg_abs, args.sam2_ckpt_abs
    )

    rows = read_jsonl(args.keyframes)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    for r in tqdm(rows, desc="[Step3] cache masks"):
        t = int(r["t_ns"])
        img_path = r["cam0_path"]

        out_png = os.path.join(mask_dir, f"{t}.png")
        out_vis = os.path.join(vis_dir, f"{t}.jpg")
        out_meta = os.path.join(meta_dir, f"{t}.json")
        if os.path.isfile(out_png) and os.path.isfile(out_vis) and os.path.isfile(out_meta):
            continue

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1) DINO full image boxes
        boxes, scores, phrases = dino_predict_full(
            dino_model, dino_device, img_bgr, caption,
            box_th=args.box_thresh, text_th=args.text_thresh
        )

        dets = []
        for b, sc, ph in zip(boxes, scores, phrases):
            cid = map_phrase_to_class_id(ph, class_to_id)
            if cid == 0:
                continue
            box = clamp_box_xyxy([float(b[0]), float(b[1]), float(b[2]), float(b[3])], W, H)
            dets.append({
                "label_id": int(cid),
                "label": id_to_class[int(cid)],
                "phrase_raw": str(ph),
                "score": float(sc),
                "box_xyxy_px": box,
            })

        if len(dets) == 0:
            sem = np.zeros((H, W), dtype=np.uint8)
            cv2.imwrite(out_png, sem)
            cv2.imwrite(out_vis, img_bgr)
            write_json(out_meta, {"t_ns": t, "note": "no dino dets after mapping"})
            continue

        # 2) class-agnostic NMS + topK
        boxes_np = np.array([d["box_xyxy_px"] for d in dets], dtype=np.float32)
        scores_np = np.array([d["score"] for d in dets], dtype=np.float32)
        keep = nms_xyxy(boxes_np, scores_np, args.nms_iou)
        dets = [dets[i] for i in keep]
        dets.sort(key=lambda x: float(x["score"]), reverse=True)
        dets = dets[:args.max_dets]

        # 3) SAM2 box-prompt segmentation → 合成语义 label mask（按 final_score 覆盖）
        sem = np.zeros((H, W), dtype=np.uint8)
        conf = np.zeros((H, W), dtype=np.float32)
        used = []

        for d in dets:
            box = d["box_xyxy_px"]
            m, sam_sc = sam2_box_to_mask(sam2_pred, img_rgb, box, multimask_output=False)
            if m is None:
                continue
            area = int(m.sum())
            if area < 50:
                continue
            final_sc = float(d["score"]) * float(sam_sc)
            upd = m & (final_sc > conf)
            sem[upd] = np.uint8(d["label_id"])
            conf[upd] = np.float32(final_sc)

            used.append({
                **d,
                "sam_score": float(sam_sc),
                "final_score": float(final_sc),
                "mask_area": int(area),
            })

        cv2.imwrite(out_png, sem)

        # 4) 可视化 overlay
        vis = img_bgr.copy()
        for cid, name in id_to_class.items():
            color = (int((cid * 37) % 255), int((cid * 17) % 255), int((cid * 67) % 255))
            vis[sem == cid] = (0.6 * vis[sem == cid] + 0.4 * np.array(color, dtype=np.float32)).astype(np.uint8)

        top_draw = sorted(used, key=lambda x: x["final_score"], reverse=True)[:10]
        for d in top_draw:
            x1, y1, x2, y2 = [int(round(v)) for v in d["box_xyxy_px"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
            txt = f'{d["label"]}:{d["final_score"]:.2f}'
            cv2.putText(vis, txt, (x1, max(15, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(out_vis, vis)

        write_json(out_meta, {
            "t_ns": t,
            "cam0_path": img_path,
            "caption": caption,
            "classes": classes,
            "sam2": sam2_meta,
            "dets_used": used,
            "num_dets_raw": int(len(boxes)),
            "num_dets_mapped": int(len(dets)),
            "num_masks_used": int(len(used)),
        })

    print("[OK] Step3 done:", args.out_dir)


if __name__ == "__main__":
    main()


'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step3_cache_masks_dino_sam2.py \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --out_dir   /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200 \
  --dino_repo_dir "/root/autodl-tmp/sam-3d-objects/third_party/GroundingDINO" \
  --dino_config   "/root/autodl-tmp/sam-3d-objects/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --dino_weights  "/root/autodl-tmp/sam-3d-objects/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth" \
  --sam2_repo_dir "/root/autodl-tmp/sam-3d-objects/third_party/sam2" \
  --sam2_cfg_name "configs/sam2.1/sam2.1_hiera_l.yaml" \
  --sam2_cfg_abs  "/root/autodl-tmp/sam-3d-objects/third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" \
  --sam2_ckpt_abs "/root/autodl-tmp/sam-3d-objects/third_party/sam2/checkpoints/sam2.1_hiera_large.pt" \
  --box_thresh 0.30 --text_thresh 0.25 --nms_iou 0.55 --max_dets 40 \
  --classes "wall,door,window,table,chair,sofa,bed,cabinet,box,person,floor"
'''