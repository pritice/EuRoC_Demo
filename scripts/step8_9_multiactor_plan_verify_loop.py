# -*- coding: utf-8 -*-
import os, json, argparse, glob, math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ----------------------------
# Utils
# ----------------------------
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

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def find_state(states_dir: str, t_ns: Optional[str], index_jsonl: Optional[str]) -> str:
    # Prefer index_jsonl if provided (stable ordering / metadata)
    if index_jsonl and os.path.isfile(index_jsonl):
        rows = read_jsonl(index_jsonl)
        if not rows:
            raise FileNotFoundError("empty state_index.jsonl")
        if not t_ns:
            return rows[-1]["state_path"] if "state_path" in rows[-1] else os.path.join(states_dir, rows[-1]["state_file"])
        # match by t_ns
        for r in rows:
            if str(r.get("t_ns", "")) == str(t_ns):
                if "state_path" in r:
                    return r["state_path"]
                if "state_file" in r:
                    return os.path.join(states_dir, r["state_file"])
        # fallback below

    # Fallback: scan states_dir
    files = sorted(glob.glob(os.path.join(states_dir, "state_*.json")))
    if not files:
        raise FileNotFoundError(f"no state_*.json in {states_dir}")
    if not t_ns:
        return files[-1]
    cand = os.path.join(states_dir, f"state_{t_ns}.json")
    if os.path.isfile(cand):
        return cand
    # slow fallback: read json
    for p in files:
        try:
            s = read_json(p)
            if str(s.get("t_ns", "")) == str(t_ns):
                return p
        except Exception:
            pass
    raise FileNotFoundError(f"cannot find t_ns={t_ns} in {states_dir}")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ----------------------------
# Palette (paper-style)
# ----------------------------
def color_for_id(i: int) -> Tuple[int, int, int]:
    return (int((i * 37) % 255), int((i * 17) % 255), int((i * 67) % 255))

def render_sem_rgb(sem: np.ndarray) -> np.ndarray:
    H, W = sem.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid in np.unique(sem):
        cid = int(cid)
        if cid <= 0:
            continue
        rgb[sem == cid] = np.array(color_for_id(cid), dtype=np.uint8)
    return rgb

def to_rgba(rgb_u8: np.ndarray, sem: np.ndarray, unknown_transparent: bool = True) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    if unknown_transparent:
        a = (sem != 0).astype(np.float32)
    else:
        a = np.ones_like(sem, dtype=np.float32)
    return np.dstack([rgb, a])

def build_legend_handles(sem: np.ndarray, id_to_name: Dict[int, str], max_items: int = 14):
    ids = [int(x) for x in np.unique(sem) if int(x) != 0]
    ids = sorted(ids)[:max_items]
    handles = []
    for cid in ids:
        name = id_to_name.get(cid, f"id{cid}")
        col = np.array(color_for_id(cid), dtype=np.float32) / 255.0
        handles.append(mpatches.Patch(color=col, label=name))
    return handles


# ----------------------------
# Intent / Policy (LLM hook)
# ----------------------------
def parse_instruction_stub(text: str) -> Dict[str, Any]:
    """
    LLM 应替换这里：把用户自然语言 -> 规范化 intent JSON。
    这里先给一个默认结构，保证 demo 可跑。
    """
    return {
        "goal": {"type": "go_to", "target": "frontier"},
        "constraints": {
            "avoid_classes": [],         # e.g. ["water", "person"]
            "keep_distance": [],         # e.g. [{"class":"person","meters":2.0}]
        },
        "preferences": {
            "prefer_classes": [],        # e.g. ["floor"]
        }
    }

def llm_revision_suggester_stub(reports: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 仲裁器/修订器应替换这里：读取 verifier 报告，输出修订建议。
    这里先做规则化修订，模拟多 agent 闭环。
    """
    action = {"type": "accept"}
    if not reports["collision"]["pass"]:
        action = {"type": "increase_inflation", "delta_m": 0.10}
    elif not reports["constraints"]["pass"]:
        # 提高违反类的代价
        bad = reports["constraints"].get("bad_classes", [])
        action = {"type": "raise_cost", "classes": bad, "delta": 2.0}
    elif not reports["completion"]["pass"]:
        action = {"type": "relax_goal", "radius_cells_add": 2}
    elif not reports["efficiency"]["pass"]:
        action = {"type": "replan_baseline"}  # 直接回退到基线最短代价
    return action


# ----------------------------
# Cost map
# ----------------------------
def default_cost_table(class_to_id: Dict[str, int]) -> Dict[int, float]:
    """
    默认代价：可走区域低代价，障碍高代价。
    你后续会用 LLM 动态调整，这里先保证 demo 可用且合理。
    """
    # 默认：unknown 不禁行但较贵（避免盲走）
    base_unknown = 6.0
    costs = {0: base_unknown}

    def set_if(name: str, v: float):
        if name in class_to_id:
            costs[int(class_to_id[name])] = float(v)

    # 常见室内
    set_if("floor", 1.0)
    set_if("road", 1.0)
    set_if("corridor", 1.2)

    set_if("wall", 1e9)
    set_if("door", 2.0)      # 门可通行但略高（你也可设更低）
    set_if("stairs", 8.0)

    set_if("chair", 30.0)
    set_if("sofa", 30.0)
    set_if("table", 40.0)
    set_if("bed", 40.0)
    set_if("cabinet", 40.0)

    set_if("person", 80.0)
    set_if("water", 1e9)
    return costs

def build_cost_grid(sem: np.ndarray, id_cost: Dict[int, float], default_cost: float = 8.0) -> np.ndarray:
    cost = np.full_like(sem, float(default_cost), dtype=np.float32)
    for cid in np.unique(sem):
        cid = int(cid)
        if cid in id_cost:
            cost[sem == cid] = float(id_cost[cid])
    return cost

def inflate_obstacles(cost: np.ndarray, obstacle_thr: float, inflate_cells: int) -> np.ndarray:
    """
    将 cost>=obstacle_thr 的格子视为障碍，进行膨胀，膨胀区域也视为高代价/不可通行。
    """
    obs = (cost >= obstacle_thr).astype(np.uint8) * 255
    if inflate_cells <= 0:
        return cost
    k = 2 * inflate_cells + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    obs2 = cv2.dilate(obs, kernel, iterations=1)
    out = cost.copy()
    out[obs2 > 0] = max(float(obstacle_thr), 1e9)  # inflated as hard obstacle
    return out


# ----------------------------
# A* planner
# ----------------------------
def astar(cost: np.ndarray,
          start: Tuple[int, int],
          goal: Tuple[int, int],
          allow_diag: bool = True) -> Optional[List[Tuple[int, int]]]:
    """
    cost: N×N, large cost treated as blocked (>=1e8)
    start/goal: (y,x)
    """
    N, M = cost.shape
    sy, sx = start
    gy, gx = goal

    def inb(y, x): return 0 <= y < N and 0 <= x < M
    if not inb(sy, sx) or not inb(gy, gx):
        return None
    if cost[sy, sx] >= 1e8 or cost[gy, gx] >= 1e8:
        return None

    if allow_diag:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        step = [1,1,1,1,math.sqrt(2),math.sqrt(2),math.sqrt(2),math.sqrt(2)]
    else:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1)]
        step = [1,1,1,1]

    def h(y,x):
        return math.hypot(y-gy, x-gx)

    # open set
    import heapq
    pq = []
    heapq.heappush(pq, (h(sy,sx), 0.0, (sy,sx)))
    came = { (sy,sx): None }
    gscore = { (sy,sx): 0.0 }

    while pq:
        f, g, (y,x) = heapq.heappop(pq)
        if (y,x) == (gy,gx):
            # reconstruct
            path = []
            cur = (y,x)
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path

        for (d,(dy,dx)) in enumerate(nbrs):
            ny, nx = y+dy, x+dx
            if not inb(ny,nx): 
                continue
            if cost[ny,nx] >= 1e8:
                continue
            ng = g + step[d] + float(cost[ny,nx])
            if (ny,nx) not in gscore or ng < gscore[(ny,nx)]:
                gscore[(ny,nx)] = ng
                came[(ny,nx)] = (y,x)
                heapq.heappush(pq, (ng + h(ny,nx), ng, (ny,nx)))
    return None

def path_cost(cost: np.ndarray, path: List[Tuple[int,int]]) -> float:
    if not path:
        return float("inf")
    s = 0.0
    for (y,x) in path:
        s += float(cost[y,x])
    return float(s)


# ----------------------------
# Verifiers (deterministic)
# ----------------------------
def collision_verifier(path: Optional[List[Tuple[int,int]]], inflated_cost: np.ndarray) -> Dict[str, Any]:
    if not path:
        return {"pass": False, "reason": "no_path"}
    bad = [(i, y, x) for i,(y,x) in enumerate(path) if inflated_cost[y,x] >= 1e8]
    if bad:
        return {"pass": False, "reason": "hit_inflated_obstacle", "first_bad": bad[0], "bad_count": len(bad)}
    return {"pass": True}

def constraint_verifier(path: Optional[List[Tuple[int,int]]], sem: np.ndarray, avoid_ids: List[int]) -> Dict[str, Any]:
    if not path:
        return {"pass": False, "reason": "no_path"}
    if not avoid_ids:
        return {"pass": True}
    hits = []
    aset = set(int(x) for x in avoid_ids)
    for i,(y,x) in enumerate(path):
        if int(sem[y,x]) in aset:
            hits.append((i,y,x,int(sem[y,x])))
    if hits:
        bad_ids = sorted(list({h[3] for h in hits}))
        return {"pass": False, "reason": "avoid_violation", "first_hit": hits[0], "bad_ids": bad_ids}
    return {"pass": True}

def completion_verifier(path: Optional[List[Tuple[int,int]]], goal: Tuple[int,int], radius_cells: int = 0) -> Dict[str, Any]:
    if not path:
        return {"pass": False, "reason": "no_path"}
    ey, ex = path[-1]
    gy, gx = goal
    d = max(abs(ey-gy), abs(ex-gx))
    ok = d <= int(radius_cells)
    return {"pass": ok, "end": [ey,ex], "goal": [gy,gx], "chebyshev_dist": int(d), "radius_cells": int(radius_cells)}

def efficiency_verifier(candidate_cost: float, baseline_cost: float, ratio_thr: float = 1.20) -> Dict[str, Any]:
    if not np.isfinite(baseline_cost) or baseline_cost <= 0:
        return {"pass": True, "note": "no_baseline"}
    ratio = float(candidate_cost / baseline_cost)
    return {"pass": ratio <= ratio_thr, "ratio": ratio, "thr": float(ratio_thr), "cand": float(candidate_cost), "base": float(baseline_cost)}


# ----------------------------
# Visualization (paper-style, step6d-like)
# ----------------------------
def draw_paper_style_with_path(out_png: str,
                               sem: np.ndarray,
                               id_to_name: Dict[int,str],
                               cell_m: float,
                               local_size_m: float,
                               start: Tuple[int,int],
                               goal: Tuple[int,int],
                               path: Optional[List[Tuple[int,int]]],
                               title: str,
                               annotate_topk: int = 12,
                               fs_matrix: float = 4.0,
                               fs_annot: float = 6.5,
                               fs_legend: float = 8.0):
    N = sem.shape[0]
    half = local_size_m/2.0

    rgb = render_sem_rgb(sem)
    rgba = to_rgba(rgb, sem, unknown_transparent=True)

    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # left map
    ax1.set_facecolor("white")
    ax1.imshow(rgba, origin="lower", extent=[-half,half,-half,half], interpolation="nearest")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title(title, fontsize=12)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    # grid lines (light)
    for i in range(N+1):
        v = -half + i*cell_m
        ax1.plot([v,v],[-half,half],linewidth=0.3,color="black",alpha=0.12)
        ax1.plot([-half,half],[v,v],linewidth=0.3,color="black",alpha=0.12)

    # markers
    sy,sx = start
    gy,gx = goal
    sxm = (sx+0.5)*cell_m - half
    sym = (sy+0.5)*cell_m - half
    gxm = (gx+0.5)*cell_m - half
    gym = (gy+0.5)*cell_m - half

    ax1.scatter([sxm],[sym], s=80, marker="o", color="lime", edgecolor="black", linewidth=1.0, zorder=5)
    ax1.scatter([gxm],[gym], s=120, marker="X", color="red", edgecolor="black", linewidth=1.0, zorder=5)

    # path overlay
    if path:
        xs = [(x+0.5)*cell_m - half for (y,x) in path]
        ys = [(y+0.5)*cell_m - half for (y,x) in path]
        ax1.plot(xs, ys, linewidth=2.2, color="white", alpha=0.9)
        ax1.plot(xs, ys, linewidth=1.4, color="black", alpha=0.7)

    # legend
    handles = build_legend_handles(sem, id_to_name, max_items=14)
    if handles:
        ax1.legend(handles=handles, loc="lower left", framealpha=0.85, fontsize=fs_legend)

    # right matrix
    rgbM = rgb.copy()
    rgbM[sem==0] = 255
    ax2.set_facecolor("white")
    ax2.imshow(rgbM, origin="lower", extent=[0,N,0,N], interpolation="nearest")
    for i in range(N+1):
        ax2.plot([i,i],[0,N],color="black",linewidth=0.3,alpha=0.6)
        ax2.plot([0,N],[i,i],color="black",linewidth=0.3,alpha=0.6)

    # text smaller (你要调小右边数字，就是这里)
    for y in range(N):
        for x in range(N):
            cid = int(sem[y,x])
            if cid == 0:
                continue
            ax2.text(x+0.5, y+0.5, f"{cid}", ha="center", va="center",
                     fontsize=fs_matrix, color="black")

    # path on matrix
    if path:
        px = [x+0.5 for (y,x) in path]
        py = [y+0.5 for (y,x) in path]
        ax2.plot(px, py, linewidth=2.0, color="white", alpha=0.9)
        ax2.plot(px, py, linewidth=1.0, color="black", alpha=0.7)

    ax2.scatter([sx+0.5],[sy+0.5], s=80, marker="o", color="lime", edgecolor="black", linewidth=1.0)
    ax2.scatter([gx+0.5],[gy+0.5], s=120, marker="X", color="red", edgecolor="black", linewidth=1.0)
    ax2.set_xlim(0,N); ax2.set_ylim(0,N)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Matrix (NxN)", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


# ----------------------------
# Main loop
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True)
    ap.add_argument("--state_index", default="")
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--t_ns", default="")

    # goal specification (demo)
    ap.add_argument("--start_cell", default="", help="format: y,x ; empty->center or state-provided")
    ap.add_argument("--goal_cell", default="", help="format: y,x ; empty->choose a reachable frontier")

    # map/planning params
    ap.add_argument("--unknown_block", action="store_true", help="treat unknown as hard obstacle")
    ap.add_argument("--inflate_m", type=float, default=0.25, help="inflation radius in meters")
    ap.add_argument("--obstacle_thr", type=float, default=80.0, help="cells with cost>=thr treated as obstacle")
    ap.add_argument("--iters", type=int, default=4, help="max verify-revise iterations")
    ap.add_argument("--eff_ratio_thr", type=float, default=1.20)

    # instruction (LLM hook)
    ap.add_argument("--instruction", default="", help="user natural language instruction (demo)")
    ap.add_argument("--fs_matrix", type=float, default=3.6, help="matrix numbers fontsize (right panel)")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    labels = read_json(args.labels_json)
    class_to_id = labels.get("class_to_id", {})
    id_to_name = {int(v): k for k,v in class_to_id.items()}
    id_to_name[0] = "unknown"

    state_path = find_state(args.states_dir, args.t_ns, args.state_index if args.state_index else None)
    state = read_json(state_path)

    sem = np.array(state["semantic_grid"], dtype=np.uint8)
    N = int(sem.shape[0])
    mp = state.get("map", {}) or {}
    cell_m = float(mp.get("cell_m", mp.get("cell_size_m", 1.0)))
    local_size_m = float(mp.get("local_size_m", mp.get("L", N*cell_m)))

    # parse start/goal
    def parse_yx(s: str) -> Optional[Tuple[int,int]]:
        if not s:
            return None
        a = s.split(",")
        if len(a) != 2:
            return None
        return (int(a[0]), int(a[1]))

    start = parse_yx(args.start_cell)
    if start is None:
        # try state provided
        for k in ["agent_cell_yx", "agent_cell", "pose_cell_yx", "uav_cell_yx"]:
            if k in state and isinstance(state[k], (list,tuple)) and len(state[k]) >= 2:
                start = (int(state[k][0]), int(state[k][1]))
                break
    if start is None:
        start = (N//2, N//2)

    goal = parse_yx(args.goal_cell)

    # LLM parse instruction (stub now)
    intent = parse_instruction_stub(args.instruction) if args.instruction else parse_instruction_stub("")
    avoid_names = intent.get("constraints", {}).get("avoid_classes", [])
    avoid_ids = [int(class_to_id[n]) for n in avoid_names if n in class_to_id]

    # choose a demo goal if not specified: pick farthest reachable cell along +x direction
    if goal is None:
        # heuristic: choose a cell on the rightmost column that is not blocked
        # if none, choose farthest non-block cell
        goal = (start[0], N-2)

    # initial policy
    id_cost = default_cost_table(class_to_id)
    if args.unknown_block:
        id_cost[0] = 1e9

    # baseline (for efficiency verifier): plan on base cost without avoid escalation
    base_cost_grid = build_cost_grid(sem, id_cost, default_cost=8.0)

    # inflation params
    inflate_cells = int(round(float(args.inflate_m) / max(cell_m, 1e-6)))

    # revision trace
    trace_path = os.path.join(args.out_dir, "revision_trace.jsonl")
    if os.path.isfile(trace_path):
        os.remove(trace_path)

    best = {"ok": False, "score": float("inf"), "iter": -1, "path": None, "reports": None}

    for it in range(int(args.iters)):
        # build cost
        cost_grid = build_cost_grid(sem, id_cost, default_cost=8.0)

        # apply avoid constraints as hard-ish penalty
        for cid in avoid_ids:
            cost_grid[sem == cid] = max(cost_grid[sem == cid], 1e9)

        inflated = inflate_obstacles(cost_grid, obstacle_thr=float(args.obstacle_thr), inflate_cells=inflate_cells)

        # plan candidate
        cand = astar(inflated, start=start, goal=goal, allow_diag=True)

        cand_cost = path_cost(cost_grid, cand) if cand else float("inf")

        # baseline for efficiency (no avoid hard block; use base_cost_grid + same inflation)
        base_inflated = inflate_obstacles(base_cost_grid, obstacle_thr=float(args.obstacle_thr), inflate_cells=inflate_cells)
        base_path = astar(base_inflated, start=start, goal=goal, allow_diag=True)
        base_cost = path_cost(base_cost_grid, base_path) if base_path else float("inf")

        # verifiers
        reports = {
            "collision": collision_verifier(cand, inflated),
            "constraints": constraint_verifier(cand, sem, avoid_ids),
            "completion": completion_verifier(cand, goal, radius_cells=0),
            "efficiency": efficiency_verifier(cand_cost, base_cost, ratio_thr=float(args.eff_ratio_thr)),
        }
        ok = reports["collision"]["pass"] and reports["constraints"]["pass"] and reports["completion"]["pass"]

        # score: lower is better
        penalty = 0.0
        if not reports["collision"]["pass"]: penalty += 1e6
        if not reports["constraints"]["pass"]: penalty += 1e5
        if not reports["completion"]["pass"]: penalty += 1e4
        if not reports["efficiency"]["pass"]: penalty += 1e3
        score = cand_cost + penalty

        # save artifacts for this iter
        np.save(os.path.join(args.out_dir, f"cost_grid_iter{it}.npy"), cost_grid)
        np.save(os.path.join(args.out_dir, f"inflated_cost_iter{it}.npy"), inflated)

        path_json = {
            "iter": it,
            "t_ns": state.get("t_ns", ""),
            "start": list(start),
            "goal": list(goal),
            "path_yx": [list(p) for p in cand] if cand else [],
            "cand_cost": float(cand_cost),
            "base_cost": float(base_cost),
            "inflate_cells": int(inflate_cells),
            "avoid_names": avoid_names,
        }
        with open(os.path.join(args.out_dir, f"path_iter{it}.json"), "w") as f:
            json.dump(path_json, f, indent=2, ensure_ascii=False)

        with open(os.path.join(args.out_dir, f"verifier_iter{it}.json"), "w") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)

        title = (f"Plan+Verify loop iter={it}  ok={ok}\n"
                 f"cand_cost={cand_cost:.1f}  base_cost={base_cost:.1f}  inflate={inflate_cells} cells")
        out_png = os.path.join(args.out_dir, f"viz_iter{it}.png")
        draw_paper_style_with_path(out_png, sem, id_to_name, cell_m, local_size_m,
                                   start=start, goal=goal, path=cand, title=title,
                                   fs_matrix=float(args.fs_matrix))

        # append trace
        rec = {
            "iter": it,
            "ok": bool(ok),
            "cand_cost": float(cand_cost),
            "base_cost": float(base_cost),
            "inflate_cells": int(inflate_cells),
            "avoid_names": avoid_names,
            "reports": reports,
        }
        with open(trace_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if score < best["score"]:
            best = {"ok": bool(ok), "score": float(score), "iter": it, "path": cand, "reports": reports}

        if ok:
            break

        # multi-agent revision (LLM hook)
        action = llm_revision_suggester_stub(reports)

        # apply revision
        if action["type"] == "increase_inflation":
            inflate_cells = int(round((inflate_cells*cell_m + float(action.get("delta_m", 0.1))) / cell_m))
        elif action["type"] == "raise_cost":
            delta = float(action.get("delta", 2.0))
            for name in action.get("classes", []):
                if name in class_to_id:
                    cid = int(class_to_id[name])
                    id_cost[cid] = float(id_cost.get(cid, 8.0) + delta)
        elif action["type"] == "relax_goal":
            # 放宽 completion radius（这里只做记录，不改变 goal）
            pass
        elif action["type"] == "replan_baseline":
            avoid_ids = []  # 直接取消约束回退
        # else accept (do nothing)

    # summary
    summ = {
        "state_path": state_path,
        "t_ns": state.get("t_ns", ""),
        "best_iter": int(best["iter"]),
        "best_ok": bool(best["ok"]),
        "best_score": float(best["score"]),
        "note": "LLM modules are stubbed; replace parse_instruction_stub and llm_revision_suggester_stub with real LLM calls."
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)

    print("[DONE] out_dir =", args.out_dir)
    print("[BEST] iter =", best["iter"], " ok =", best["ok"])


if __name__ == "__main__":
    main()

'''
python /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/scripts/step8_9_multiactor_plan_verify_loop.py \
  --states_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/step7_global_states_v4_N60/states \
  --state_index /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/step7_global_states_v4_N60/state_index.jsonl \
  --labels_json /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/masks_dino_sam2_200/labels.json \
  --out_dir /root/autodl-tmp/sam-3d-objects/EuRoC/stmr_demo/outputs/step8_9_plan_verify_demo \
  --iters 4 \
  --inflate_m 0.25 \
  --fs_matrix 3.0
'''