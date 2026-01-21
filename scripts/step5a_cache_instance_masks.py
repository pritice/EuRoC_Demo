# -*- coding: utf-8 -*-
import os, sys, json, argparse
from typing import Dict, Any, List, Tuple, Optional

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

def ensure_file(p: str):
    if not os.path.isfile(p):
        raise FileNotFoundError(p)

def load_cam0_KD(mav0_dir: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int]]:
    """
    EuRoC cam0/sensor.yaml:
      intrinsics: [fx, fy, cx, cy]
      distortion_coeffs: [...]
      resolution: [W, H]
    """
    if yaml is None:
        raise RuntimeError("pyyaml missing. Run: pip install pyyaml")
    ypath = os.path.join(mav0_dir, "cam0", "sensor.yaml")
    with open(ypath, "r") as f:
        y = yaml.safe_load(f)
    fx, fy, cx, cy = [float(x) for x in y["intrinsics"]]
    K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]], dtype=np.float64)
    D = np.array(y.get("distortion_coeffs", []), dtype=np.float64).reshape(-1,1)
    W, H = int(y["resolution"][0]), int(y["resolution"][1])
    return K, D, (W, H)

def build_rectify_map(mav0_dir: str, stereo_meta_path: str):
    st = read_json(stereo_meta_path)
    W, H = st["image_wh"]
    P0 = np.array(st["rectify"]["P0"], dtype=np.float64)
    R0 = np.array(st["rectify"]["R0"], dtype=np.float64)
    K0, D0, wh = load_cam0_KD(mav0_dir)
    assert wh == (W, H), f"resolution mismatch: sensor.yaml {wh} vs stereo_meta {(W,H)}"
    map0x, map0y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (W, H), cv2.CV_32FC1)
    return map0x, map0y, (W, H)

def load_sam2_predictor(sam2_repo_dir: str, sam2_cfg_name: str, sam2_ckpt_abs: str, device: str = "cuda"):
    """
    关键点：sam2_cfg_name 必须是 configs/... 的相对路径（不要传绝对路径）。
    """
    ensure_file(sam2_ckpt_abs)
    if sam2_repo_dir not in sys.path:
        sys.path.insert(0, sam2_repo_dir)

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    dev = device
    if dev == "cuda" and (not torch.cuda.is_available()):
        dev = "cpu"

    # build_sam2 在不同版本参数名略不同，这里做兼容
    try:
        model = build_sam2(config_file=sam2_cfg_name, ckpt_path=sam2_ckpt_abs, device=dev)
    except TypeError:
        model = build_sam2(sam2_cfg_name, sam2_ckpt_abs, device=dev)

    predictor = SAM2ImagePredictor(model)
    return predictor, dev

def sam2_predict_union(predictor, box_xyxy: List[float], multimask: bool = True, topk_union: int = 1):
    box_np = np.array(box_xyxy, dtype=np.float32)
    kwargs = {"multimask_output": bool(multimask)}
    try:
        masks, scores, _ = predictor.predict(box=box_np[None, :], **kwargs)
    except TypeError:
        masks, scores, _ = predictor.predict(box=box_np, **kwargs)

    if masks is None:
        return None, 0.0
    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)

    if masks.ndim == 4:
        masks = masks[0]
    if masks.ndim != 3:
        return None, 0.0

    order = scores.argsort()[::-1]
    topk = order[: max(1, min(int(topk_union), len(order)))]
    m = np.zeros_like(masks[0], dtype=bool)
    for i in topk:
        m |= masks[int(i)].astype(bool)
    return m, float(scores[int(order[0])]) if len(order) else 0.0

def clamp_box_xyxy(box, W, H):
    x1,y1,x2,y2 = [float(x) for x in box]
    x1 = float(np.clip(x1, 0, W-1))
    y1 = float(np.clip(y1, 0, H-1))
    x2 = float(np.clip(x2, 0, W-1))
    y2 = float(np.clip(y2, 0, H-1))
    if x2 <= x1 + 1: x2 = min(W-1, x1 + 2)
    if y2 <= y1 + 1: y2 = min(H-1, y1 + 2)
    return [x1,y1,x2,y2]

def mask_bbox_centroid(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return (0,0,0,0), (0.0,0.0), 0
    x1,x2 = int(xs.min()), int(xs.max())
    y1,y2 = int(ys.min()), int(ys.max())
    cx, cy = float(xs.mean()), float(ys.mean())
    return (x1,y1,x2,y2), (cx,cy), int(xs.size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True, help="keyframes_200.jsonl")
    ap.add_argument("--frame_meta_dir", required=True, help=".../frame_meta")
    ap.add_argument("--out_dir", required=True, help="output dir for instances")
    ap.add_argument("--mav0_dir", required=True, help="/root/.../inputs/mav0 (for rect map)")
    ap.add_argument("--stereo_meta", required=True, help="stereo_meta.json (contains R0,P0,image_wh)")

    ap.add_argument("--sam2_repo_dir", required=True)
    ap.add_argument("--sam2_cfg_name", required=True, help="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--sam2_ckpt_abs", required=True)

    ap.add_argument("--score_thr", type=float, default=0.30, help="filter dets by final_score")
    ap.add_argument("--max_inst", type=int, default=12)
    ap.add_argument("--multimask", action="store_true")
    ap.add_argument("--topk_union", type=int, default=1)

    ap.add_argument("--save_raw_masks", action="store_true", help="also save raw (unrectified) masks")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rect_dir = os.path.join(args.out_dir, "instances_rect")
    raw_dir  = os.path.join(args.out_dir, "instances_raw")
    vis_dir  = os.path.join(args.out_dir, "instances_vis")
    os.makedirs(rect_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    if args.save_raw_masks:
        os.makedirs(raw_dir, exist_ok=True)

    map0x, map0y, (W,H) = build_rectify_map(args.mav0_dir, args.stereo_meta)

    predictor, dev = load_sam2_predictor(args.sam2_repo_dir, args.sam2_cfg_name, args.sam2_ckpt_abs)
    print("[Step5a] SAM2 device:", dev)

    rows = read_jsonl(args.keyframes)

    summary = {"processed_frames": 0, "total_instances": 0, "rectified_masks": True}
    for r in tqdm(rows, desc="[Step5a] instance masks"):
        t = int(r["t_ns"])
        meta_path = os.path.join(args.frame_meta_dir, f"{t}.json")
        if not os.path.isfile(meta_path):
            continue

        fm = read_json(meta_path)
        img_path = fm.get("cam0_path") or r.get("cam0_path")
        if not img_path or (not os.path.isfile(img_path)):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[0] != H or img_rgb.shape[1] != W:
            img_rgb = cv2.resize(img_rgb, (W,H), interpolation=cv2.INTER_AREA)

        predictor.set_image(img_rgb)

        dets = fm.get("dets_used", [])
        dets = [d for d in dets if float(d.get("final_score", d.get("score", 0.0))) >= args.score_thr]
        dets.sort(key=lambda d: float(d.get("final_score", d.get("score", 0.0))), reverse=True)
        dets = dets[: max(0, int(args.max_inst))]

        frame_out_rect = os.path.join(rect_dir, str(t))
        os.makedirs(frame_out_rect, exist_ok=True)
        frame_out_raw = os.path.join(raw_dir, str(t)) if args.save_raw_masks else None
        if frame_out_raw:
            os.makedirs(frame_out_raw, exist_ok=True)

        inst_list = []
        overlay = img.copy()
        for i, d in enumerate(dets):
            box = clamp_box_xyxy(d["box_xyxy_px"], W, H)
            m_bool, sam_score = sam2_predict_union(
                predictor, box_xyxy=box, multimask=args.multimask, topk_union=args.topk_union
            )
            if m_bool is None or (m_bool.sum() < 50):
                continue

            # rectified mask aligned with rectified depth
            m_u8 = (m_bool.astype(np.uint8) * 255)
            m_rect = cv2.remap(m_u8, map0x, map0y, interpolation=cv2.INTER_NEAREST)

            bbox, cen, area = mask_bbox_centroid(m_bool)

            inst_id = len(inst_list)
            rect_path = os.path.join(frame_out_rect, f"inst_{inst_id:03d}.png")
            cv2.imwrite(rect_path, m_rect)

            raw_path = ""
            if frame_out_raw:
                raw_path = os.path.join(frame_out_raw, f"inst_{inst_id:03d}.png")
                cv2.imwrite(raw_path, m_u8)

            # vis on original image
            x1,y1,x2,y2 = bbox
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.putText(overlay, f"{inst_id}:{d.get('label','')}", (x1, max(0,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)

            inst_list.append({
                "inst_id": inst_id,
                "label_id": int(d.get("label_id", 0)),
                "label": d.get("label", ""),
                "final_score": float(d.get("final_score", d.get("score", 0.0))),
                "sam_score": float(d.get("sam_score", sam_score)),
                "box_xyxy_px": [float(x) for x in d["box_xyxy_px"]],
                "mask_rect_path": rect_path,
                "mask_raw_path": raw_path,
                "mask_area_px": int(area),
                "centroid_uv": [float(cen[0]), float(cen[1])]
            })

        if len(inst_list) == 0:
            continue

        # write per-frame instances.json
        write_json(os.path.join(frame_out_rect, "instances.json"), {
            "t_ns": t,
            "cam0_path": img_path,
            "num_instances": len(inst_list),
            "instances": inst_list
        })

        # write vis
        vis_path = os.path.join(vis_dir, f"{t}.jpg")
        cv2.imwrite(vis_path, overlay)

        summary["processed_frames"] += 1
        summary["total_instances"] += len(inst_list)

    write_json(os.path.join(args.out_dir, "summary_instances.json"), summary)
    print("[Step5a] done:", summary)


if __name__ == "__main__":
    main()

'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step5a_cache_instance_masks.py \
  --keyframes /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --frame_meta_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/frame_meta \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200 \
  --mav0_dir /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --stereo_meta /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/stereo_depth_200_v2/stereo_meta.json \
  --sam2_repo_dir /root/autodl-tmp/sam-3d-objects/third_party/sam2 \
  --sam2_cfg_name configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2_ckpt_abs /root/autodl-tmp/sam-3d-objects/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --score_thr 0.30 --max_inst 12 --multimask --topk_union 1
'''
