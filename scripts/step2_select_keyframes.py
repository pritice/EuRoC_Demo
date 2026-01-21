import os, json, math, argparse
import numpy as np

def read_jsonl(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def quat_wxyz_to_R(q):
    qw, qx, qy, qz = [float(x) for x in q]
    n = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz) + 1e-12
    qw,qx,qy,qz = qw/n,qx/n,qy/n,qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)

def yaw_from_R(R):
    # yaw about world z (approx)
    return float(math.atan2(R[1,0], R[0,0]))

def ang_diff(a,b):
    d = a-b
    while d > math.pi: d -= 2*math.pi
    while d < -math.pi: d += 2*math.pi
    return abs(d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_n", type=int, default=200)
    ap.add_argument("--min_trans_m", type=float, default=0.20)
    ap.add_argument("--min_yaw_deg", type=float, default=8.0)
    ap.add_argument("--fallback_stride", type=int, default=5, help="若触发过少，用stride补齐")
    args = ap.parse_args()

    rows = read_jsonl(args.manifest)
    if not rows:
        raise RuntimeError("empty manifest")

    min_yaw = math.radians(args.min_yaw_deg)

    picked = []
    last_p = None
    last_yaw = None

    for r in rows:
        p = np.array(r["p_WB"], dtype=np.float64)
        R = quat_wxyz_to_R(r["q_WB_wxyz"])
        y = yaw_from_R(R)
        if last_p is None:
            picked.append(r)
            last_p = p
            last_yaw = y
            continue

        dp = float(np.linalg.norm(p - last_p))
        dy = ang_diff(y, last_yaw)
        if (dp >= args.min_trans_m) or (dy >= min_yaw):
            picked.append(r)
            last_p = p
            last_yaw = y
        if len(picked) >= args.target_n:
            break

    # 若触发不足，用 stride 补齐到 target_n
    if len(picked) < args.target_n:
        extra = []
        stride = max(1, args.fallback_stride)
        for i in range(0, len(rows), stride):
            extra.append(rows[i])
            if len(extra) >= args.target_n:
                break
        # merge by timestamp
        seen = set()
        merged = []
        for r in picked + extra:
            t = int(r["t_ns"])
            if t not in seen:
                seen.add(t)
                merged.append(r)
            if len(merged) >= args.target_n:
                break
        picked = merged

    write_jsonl(args.out, picked)
    meta = {
        "target_n": args.target_n,
        "picked": len(picked),
        "min_trans_m": args.min_trans_m,
        "min_yaw_deg": args.min_yaw_deg,
        "fallback_stride": args.fallback_stride
    }
    with open(os.path.join(os.path.dirname(args.out), "keyframes_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[Step2] wrote", args.out, "picked=", len(picked))

if __name__ == "__main__":
    main()


'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step2_select_keyframes.py \
  --manifest /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/manifest.jsonl \
  --out      /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/keyframes_200.jsonl \
  --target_n 200
'''