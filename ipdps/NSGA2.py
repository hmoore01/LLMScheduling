# NSGA2.py — Serverless NSGA-II baseline (hardened stats coercion)

from typing import Dict, List, Any, Tuple
import math, random
from copy import deepcopy
from simulation import LLM_Simulator

# ---------- utils ----------
def _normalize_node_props(node_properties) -> Dict[str, dict]:
    norm: Dict[str, dict] = {}
    if isinstance(node_properties, dict):
        for nid, props in node_properties.items():
            p = dict(props) if isinstance(props, dict) else {}
            p.setdefault("node_id", str(nid))
            norm[str(nid)] = p
        return norm
    if isinstance(node_properties, (list, tuple)):
        for i, item in enumerate(node_properties):
            p = dict(item) if isinstance(item, dict) else {}
            nid = str(p.get("node_id") or p.get("id") or p.get("name") or f"node_{i}")
            p["node_id"] = nid
            norm[nid] = p
        return norm
    for nid, props in dict(node_properties).items():
        p = dict(props) if isinstance(props, dict) else {}
        p.setdefault("node_id", str(nid))
        norm[str(nid)] = p
    return norm

def _make_dc_index_maps(node_properties, epoch_summary):
    dc_list = epoch_summary.get("datacenters")
    if isinstance(dc_list, list) and dc_list:
        dc_to_idx = {str(dc): i for i, dc in enumerate(dc_list)}
        idx_to_dc = [str(dc) for dc in dc_list]
        return dc_to_idx, idx_to_dc
    props = _normalize_node_props(node_properties)
    uniq = []
    for _, p in props.items():
        r = str(p.get("region", "0"))
        if r not in uniq: uniq.append(r)
    try:
        _ = [int(r) for r in uniq]
        dc_to_idx = {str(r): int(r) for r in uniq}
        idx_to_dc = [str(i) for i in sorted({int(r) for r in uniq})]
        return dc_to_idx, idx_to_dc
    except Exception:
        dc_to_idx = {r: i for i, r in enumerate(uniq)}
        idx_to_dc = uniq
        return dc_to_idx, idx_to_dc

def _coerce_epoch_rows(epoch_data, avg_in: int, avg_out: int) -> List[dict]:
    rows: List[dict] = []
    if hasattr(epoch_data, "iterrows") and hasattr(epoch_data, "columns"):
        for _, row in epoch_data.iterrows():
            rows.append({
                "source_dc_id": int(row["source_dc_id"]),
                "model_type": row["model_type"],
                "num_tokens": int(row.get("num_tokens", row.get("prompt_tokens", avg_in))),
                "output_tokens": int(row.get("output_tokens", avg_out)),
                "batch_size": int(row.get("batch_size", 1)),
                "time_index": int(row.get("time_index", 0)),
            })
        return rows
    if isinstance(epoch_data, list):
        for r in epoch_data:
            rows.append({
                "source_dc_id": int(r.get("source_dc_id", 0)),
                "model_type": r.get("model_type", "Llama-7b"),
                "num_tokens": int(r.get("num_tokens", r.get("prompt_tokens", avg_in))),
                "output_tokens": int(r.get("output_tokens", avg_out)),
                "batch_size": int(r.get("batch_size", 1)),
                "time_index": int(r.get("time_index", 0)),
            })
        return rows
    return rows

def _build_dc_caps(node_properties, alpha: float, beta: float):
    props = _normalize_node_props(node_properties)
    dc_to_idx, _ = _make_dc_index_maps(node_properties, {})
    prompt_cap: Dict[int, float] = {}
    token_cap_tps: Dict[int, float] = {}
    token_cap_mbt: Dict[int, float] = {}
    for _, p in props.items():
        dc_key = str(p.get("region", "0"))
        dc_id = dc_to_idx.get(dc_key, int(dc_key) if dc_key.isdigit() else 0)
        tps = float(p.get("tp_tokens_per_s", 4000.0))
        mbt = float(p.get("max_batch_tokens", 8192))
        prompt_cap[dc_id]     = prompt_cap.get(dc_id, 0.0) + tps
        token_cap_tps[dc_id]  = token_cap_tps.get(dc_id, 0.0) + tps
        token_cap_mbt[dc_id]  = token_cap_mbt.get(dc_id, 0.0) + mbt
    token_cap: Dict[int, float] = {}
    for dc in set(list(prompt_cap.keys()) + list(token_cap_tps.keys())):
        token_cap[dc]  = max(1e-6, alpha * token_cap_tps.get(dc, 0.0) + beta * token_cap_mbt.get(dc, 0.0))
        prompt_cap[dc] = max(1e-6, prompt_cap.get(dc, 0.0))
    return prompt_cap, token_cap

def _build_power_plan(routed_token_share_by_dc: Dict[int, float], epoch_summary) -> Dict[int, Dict[int, str]]:
    node_types = list(epoch_summary.get("node_types", [0,1,2,3,4,5]))
    min_idle = int(epoch_summary.get("min_idle_types", 1))
    max_idle = int(epoch_summary.get("max_idle_types", len(node_types)))
    max_idle = max(1, min(max_idle, len(node_types)))
    total = sum(max(0.0, v) for v in routed_token_share_by_dc.values()) or 1.0
    shares = {dc: max(0.0, v) / total for dc, v in routed_token_share_by_dc.items()}
    plan: Dict[int, Dict[int, str]] = {}
    for dc, s in shares.items():
        if s <= 0.0:
            plan[dc] = {nt: "Off" for nt in node_types}
            continue
        k = min_idle + int(round((max_idle - min_idle) * s))
        k = max(min_idle, min(max_idle, k))
        plan[dc] = {nt: ("Idle" if i < k else "Off") for i, nt in enumerate(node_types)}
    return plan

# ---------- NSGA-II primitives ----------
def _repair_theta(theta):
    t = list(theta)
    t[0] = max(0.5, min(3.0, float(t[0])))     # w_token
    t[1] = max(0.0, min(0.01, float(t[1])))    # w_lat
    t[2] = int(max(512, min(4096, float(t[2])))) # prompt cap
    t[3] = max(0.3, min(0.95, float(t[3])))    # alpha
    t[4] = max(0.05, min(0.7,  float(t[4])))   # beta
    s = t[3] + t[4]
    if s <= 1e-9: t[3], t[4] = 0.7, 0.3
    else:         t[3], t[4] = t[3]/s, t[4]/s
    return t

def _sbx_crossover(a, b, eta_c=15.0, p_c=0.9):
    if random.random() > p_c: return deepcopy(a), deepcopy(b)
    c1, c2 = [], []
    for x, y in zip(a, b):
        if random.random() < 0.5 and abs(x-y) > 1e-14:
            x1, x2 = min(x,y), max(x,y); u = random.random()
            beta = 1.0 + (2.0*(x1-0.0)/(x2-x1+1e-12)); alpha = 2.0 - beta**(-(eta_c+1.0))
            betaq = (u*alpha)**(1.0/(eta_c+1.0)) if u <= 1.0/alpha else (1.0/(2.0-u*alpha))**(1.0/(eta_c+1.0))
            child1 = 0.5*((x1+x2) - betaq*(x2-x1))
            beta = 1.0 + (2.0*(1.0-x2)/(x2-x1+1e-12)); alpha = 2.0 - beta**(-(eta_c+1.0))
            betaq = (u*alpha)**(1.0/(eta_c+1.0)) if u <= 1.0/alpha else (1.0/(2.0-u*alpha))**(1.0/(eta_c+1.0))
            child2 = 0.5*((x1+x2) + betaq*(x2-x1))
            c1.append(child1); c2.append(child2)
        else:
            c1.append(x); c2.append(y)
    return _repair_theta(c1), _repair_theta(c2)

def _poly_mutation(x, eta_m=20.0, p_m=0.2):
    y = list(x)
    for i in range(len(y)):
        if random.random() < p_m:
            u = random.random()
            delta = (2*u)**(1.0/(eta_m+1))-1.0 if u<0.5 else 1.0-(2*(1-u))**(1.0/(eta_m+1))
            y[i] = y[i] + delta * 0.1 * (1.0 if i != 2 else 512.0)
    return _repair_theta(y)

def _fast_nondominated_sort(F):
    S = [[] for _ in F]; n = [0]*len(F); fronts=[[]]; rank=[0]*len(F)
    for p in range(len(F)):
        Sp=[]; np=0
        for q in range(len(F)):
            if p==q: continue
            pdom = all(F[p][k]<=F[q][k] for k in range(len(F[p]))) and any(F[p][k]<F[q][k] for k in range(len(F[p])))
            qdom = all(F[q][k]<=F[p][k] for k in range(len(F[p]))) and any(F[q][k]<F[p][k] for k in range(len(F[p])))
            if pdom: Sp.append(q)
            elif qdom: np += 1
        S[p]=Sp; n[p]=np
        if np==0: rank[p]=1; fronts[0].append(p)
    i=0
    while fronts[i]:
        Q=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q]-=1
                if n[q]==0: rank[q]=i+2; Q.append(q)
        i+=1; fronts.append(Q)
    if not fronts[-1]: fronts.pop()
    return fronts, rank

def _crowding_distance(front_indices, F):
    if not front_indices: return {}
    m=len(F[0]); dist={i:0.0 for i in front_indices}
    for k in range(m):
        idx=sorted(front_indices, key=lambda i:F[i][k]); fmin=F[idx[0]][k]; fmax=F[idx[-1]][k]
        dist[idx[0]] = float("inf"); dist[idx[-1]] = float("inf")
        if fmax - fmin < 1e-12: continue
        for j in range(1,len(idx)-1):
            i_prev,i_next=idx[j-1],idx[j+1]
            dist[idx[j]] += (F[i_next][k]-F[i_prev][k])/(fmax-fmin)
    return dist

def _crowded_tournament(a_idx,b_idx,rank,dist):
    if rank[a_idx] < rank[b_idx]: return a_idx
    if rank[b_idx] < rank[a_idx]: return b_idx
    return a_idx if dist.get(a_idx,0.0) > dist.get(b_idx,0.0) else b_idx

# ---------- schedule from policy ----------
def _make_schedule_from_policy(theta, epoch_data, node_properties, epoch_summary):
    w_token, w_lat, prompt_cap, alpha, beta = theta
    avg_in  = int(epoch_summary.get("avg_input_tokens", 700))
    avg_out = int(epoch_summary.get("avg_output_tokens", 250))

    rows = _coerce_epoch_rows(epoch_data, avg_in, avg_out)
    dc_to_idx, _ = _make_dc_index_maps(node_properties, epoch_summary)
    prompt_cap_dc, token_cap_dc = _build_dc_caps(node_properties, alpha, beta)

    pend_p = {dc:0.0 for dc in prompt_cap_dc}
    pend_t = {dc:0.0 for dc in token_cap_dc}
    routed_token_by_dc = {dc:0.0 for dc in token_cap_dc}

    schedule_plan: List[dict] = []

    for r in rows:
        p_tokens = r["num_tokens"]; o_tokens = r["output_tokens"]

        best_dc, best_cost = None, None
        for dc in prompt_cap_dc.keys():
            p_term = (pend_p[dc] + min(p_tokens, prompt_cap)) / prompt_cap_dc[dc]
            t_term = (pend_t[dc] + o_tokens) / token_cap_dc[dc]
            lat_ms = 0.0  # add latency table if you have one
            cost = p_term + w_token*t_term + w_lat*lat_ms
            if best_cost is None or cost < best_cost:
                best_cost, best_dc = cost, dc

        target_dc_id = int(best_dc)
        pend_p[target_dc_id] += min(p_tokens, prompt_cap)
        pend_t[target_dc_id] += o_tokens
        routed_token_by_dc[target_dc_id] += o_tokens
        pend_p[target_dc_id] = max(0.0, pend_p[target_dc_id]-prompt_cap_dc[target_dc_id])
        pend_t[target_dc_id] = max(0.0, pend_t[target_dc_id]-token_cap_dc[target_dc_id])

        schedule_plan.append({
            "target_dc_id": target_dc_id,
            "model_type": r["model_type"],
            "num_tokens": p_tokens,
            "batch_size": r["batch_size"],
            "source_dc_id": r["source_dc_id"],
            "time_index": r["time_index"],
        })

    power_plan = _build_power_plan(routed_token_by_dc, epoch_summary)
    return schedule_plan, power_plan

# ---------- robust coercion ----------
def _force_stats_dict(stats_any) -> dict:
    if isinstance(stats_any, dict):
        return dict(stats_any)
    if isinstance(stats_any, list):
        if stats_any and isinstance(stats_any[0], dict):
            return dict(stats_any[0])
        if all(isinstance(x, (list, tuple)) and len(x)==2 for x in stats_any):
            return {k:v for (k,v) in stats_any}
    return {}

def _coerce_sim_output(out):
    stats, results, leftovers = {}, [], []
    if isinstance(out, tuple):
        if len(out) >= 1: stats = out[0]
        if len(out) >= 2: results = out[1] if out[1] is not None else []
        if len(out) >= 3: leftovers = out[2]
    elif isinstance(out, dict):
        stats = out.get("metrics", {})
        results = out.get("results", [])
        leftovers = out.get("leftover_requests", [])
    stats = _force_stats_dict(stats)
    if not isinstance(results, list): results = list(results) if results is not None else []
    if leftovers is None: leftovers = []
    return stats, results, leftovers

# ---------- public API ----------
class NSGA2:
    @staticmethod
    def milp_optimizer(epoch_data, epoch_idx: int, node_properties, epoch_summary: Dict[str, Any]):

        pop_size    = int(epoch_summary.get("nsga2_pop_size", 20))
        generations = int(epoch_summary.get("nsga2_generations", 5))
        p_c         = float(epoch_summary.get("nsga2_crossover_prob", 0.9))
        p_m         = float(epoch_summary.get("nsga2_mutation_prob", 0.2))
        eta_c       = float(epoch_summary.get("nsga2_eta_c", 15.0))
        eta_m       = float(epoch_summary.get("nsga2_eta_m", 20.0))

        base_theta = _repair_theta([
            epoch_summary.get("splitwise_token_weight", 1.5),
            epoch_summary.get("latency_weight", 0.001),
            epoch_summary.get("prompt_batch_cap_tokens", 2048),
            epoch_summary.get("token_capacity_alpha", 0.7),
            epoch_summary.get("token_capacity_beta",  0.3),
        ])

        pop: List[List[float]] = [base_theta]
        while len(pop) < pop_size:
            jitter = [
                base_theta[0] * random.uniform(0.7, 1.3),
                base_theta[1] * random.uniform(0.5, 1.5),
                base_theta[2] * random.uniform(0.5, 1.5),
                base_theta[3] * random.uniform(0.7, 1.3),
                base_theta[4] * random.uniform(0.7, 1.3),
            ]
            pop.append(_repair_theta(jitter))

        cache: Dict[Tuple, Tuple] = {}
        def eval_cached(theta):
            key = tuple(_repair_theta(theta))
            if key in cache: return cache[key]
            schedule_plan, power_plan = _make_schedule_from_policy(list(key), epoch_data, node_properties, epoch_summary)
            out = LLM_Simulator(epoch_idx, epoch_data, schedule_plan, power_plan)
            stats, results, leftovers = _coerce_sim_output(out)
            f1 = float(stats.get("avg_ttft", 0.0))
            f2 = float(stats.get("energy_cost", 0.0))
            f3 = float(stats.get("carbon_emissions", 0.0))
            cache[key] = (f1, f2, f3, stats, results, leftovers)
            return cache[key]

        objs, payloads = [], []
        for th in pop:
            f1, f2, f3, s, r, l = eval_cached(th)
            objs.append((f1, f2, f3)); payloads.append((s, r, l))

        for _ in range(generations):
            fronts, rank = _fast_nondominated_sort(objs)
            dist={}
            for fr in fronts: dist.update(_crowding_distance(fr, objs))

            mating: List[List[float]] = []
            while len(mating) < pop_size:
                i, j = random.randrange(len(pop)), random.randrange(len(pop))
                winner = _crowded_tournament(i, j, rank, dist)
                mating.append(deepcopy(pop[winner]))

            offspring: List[List[float]] = []
            for i in range(0, pop_size, 2):
                a = mating[i]; b = mating[(i+1) % pop_size]
                c1, c2 = _sbx_crossover(a, b, eta_c=eta_c, p_c=p_c)
                c1 = _poly_mutation(c1, eta_m=eta_m, p_m=p_m)
                c2 = _poly_mutation(c2, eta_m=eta_m, p_m=p_m)
                offspring.extend([c1, c2])
            offspring = offspring[:pop_size]

            off_objs, off_payloads = [], []
            for th in offspring:
                f1, f2, f3, s, r, l = eval_cached(th)
                off_objs.append((f1, f2, f3)); off_payloads.append((s, r, l))

            combined       = pop + offspring
            combined_objs  = objs + off_objs
            combined_payld = payloads + off_payloads

            fronts, rank = _fast_nondominated_sort(combined_objs)
            new_pop, new_objs, new_payld = [], [], []
            for fr in fronts:
                if len(new_pop) + len(fr) <= pop_size:
                    for idx in fr:
                        new_pop.append(combined[idx]); new_objs.append(combined_objs[idx]); new_payld.append(combined_payld[idx])
                else:
                    d = _crowding_distance(fr, combined_objs)
                    fr_sorted = sorted(fr, key=lambda i: d[i], reverse=True)
                    for idx in fr_sorted[:pop_size - len(new_pop)]:
                        new_pop.append(combined[idx]); new_objs.append(combined_objs[idx]); new_payld.append(combined_payld[idx])
                    break
            pop, objs, payloads = new_pop, new_objs, new_payld

        mins = [min(o[k] for o in objs) for k in range(3)]
        maxs = [max(o[k] for o in objs) for k in range(3)]
        def _score(o):
            s=0.0
            for k in range(3):
                rng = max(1e-9, maxs[k]-mins[k])
                s += (o[k]-mins[k]) / rng
            return s
        fronts, _ = _fast_nondominated_sort(objs)
        best_idx = min(fronts[0], key=lambda i: _score(objs[i]))
        best_theta = pop[best_idx]
        best_stats, best_results, best_leftovers = payloads[best_idx]

        # FINAL guard: force dict before returning
        best_stats = _force_stats_dict(best_stats)
        best_stats.setdefault("avg_ttft", best_stats.get("avg_ttft", 0.0))
        best_stats.setdefault("energy_cost", best_stats.get("energy_cost", 0.0))
        best_stats.setdefault("carbon_emissions", best_stats.get("carbon_emissions", 0.0))
        best_stats.setdefault("water_usage", best_stats.get("water_usage", 0.0))
        best_stats["nsga2_meta"] = {
            "best_theta": list(map(float, best_theta)),
            "population_size": pop_size,
            "generations": generations,
            "pareto_size": len(fronts[0]),
        }

        return best_stats, best_results, best_leftovers


