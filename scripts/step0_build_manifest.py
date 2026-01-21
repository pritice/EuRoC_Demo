import os, csv, json, argparse
import numpy as np

def list_timestamps(img_dir):
    ts = []
    for fn in os.listdir(img_dir):
        if fn.endswith(".png"):
            try:
                ts.append(int(os.path.splitext(fn)[0]))
            except:
                pass
    ts.sort()
    return ts

def load_gt_csv(csv_path):
    # EuRoC state_groundtruth_estimate0/data.csv
    # columns often: timestamp, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z, ...
    rows = []
    with open(csv_path, "r") as f:
        r = csv.reader(f)
        header = next(r)
        for line in r:
            if not line:
                continue
            t = int(line[0])
            px, py, pz = float(line[1]), float(line[2]), float(line[3])
            qw, qx, qy, qz = float(line[4]), float(line[5]), float(line[6]), float(line[7])
            rows.append((t, np.array([px,py,pz], np.float64), np.array([qw,qx,qy,qz], np.float64)))
    rows.sort(key=lambda x: x[0])
    ts = np.array([x[0] for x in rows], dtype=np.int64)
    return rows, ts

def nearest_pose(t, gt_rows, gt_ts):
    idx = int(np.searchsorted(gt_ts, t))
    cand = []
    if idx < len(gt_ts):
        cand.append(idx)
    if idx-1 >= 0:
        cand.append(idx-1)
    best = None
    best_dt = None
    for j in cand:
        dt = abs(int(gt_rows[j][0]) - int(t))
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = gt_rows[j]
    return best, best_dt

def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mav0", required=True, help=".../inputs/mav0")
    ap.add_argument("--out", required=True, help="manifest.jsonl output path")
    ap.add_argument("--max_dt_ns", type=int, default=2_000_000, help="pose匹配允许的最大时间差(ns)")
    args = ap.parse_args()

    cam0_dir = os.path.join(args.mav0, "cam0", "data")
    cam1_dir = os.path.join(args.mav0, "cam1", "data")
    gt_csv = os.path.join(args.mav0, "state_groundtruth_estimate0", "data.csv")

    assert os.path.isdir(cam0_dir), cam0_dir
    assert os.path.isdir(cam1_dir), cam1_dir
    assert os.path.isfile(gt_csv), gt_csv

    cam0_ts = set(list_timestamps(cam0_dir))
    cam1_ts = set(list_timestamps(cam1_dir))
    common = sorted(list(cam0_ts.intersection(cam1_ts)))

    gt_rows, gt_ts = load_gt_csv(gt_csv)

    items = []
    dropped = 0
    for t in common:
        pose, dt = nearest_pose(t, gt_rows, gt_ts)
        if dt is None or dt > args.max_dt_ns:
            dropped += 1
            continue
        _, p, q = pose
        items.append({
            "t_ns": int(t),
            "cam0_path": os.path.join(cam0_dir, f"{t}.png"),
            "cam1_path": os.path.join(cam1_dir, f"{t}.png"),
            "gt_dt_ns": int(dt),
            "p_WB": [float(p[0]), float(p[1]), float(p[2])],
            "q_WB_wxyz": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
        })

    write_jsonl(args.out, items)
    meta = {
        "mav0": args.mav0,
        "cam0_frames": len(cam0_ts),
        "cam1_frames": len(cam1_ts),
        "common": len(common),
        "kept": len(items),
        "dropped_by_dt": dropped,
        "max_dt_ns": args.max_dt_ns,
    }
    meta_path = os.path.join(os.path.dirname(args.out), "dataset_meta_step0.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[Step0] wrote", args.out)
    print("[Step0] meta ", meta_path)

if __name__ == "__main__":
    main()


'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step0_build_manifest.py \
  --mav0 /root/autodl-tmp/sam-3d-objects/inputs/mav0 \
  --out  /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/manifest.jsonl
'''