# PerLLM.py — Serverless CS-UCB (PerLLM) comparison work
# API: stats, results, leftovers = PerLLM.milp_optimizer(epoch_data, epoch_idx, node_properties, epoch_summary)

from typing import Dict, List, Any, Tuple
import math
from simulation import LLM_Simulator

# ---------- helpers ----------
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
            nid = (
                str(p.get("node_id"))
                if p.get("node_id") is not None else
                str(p.get("id")) if p.get("id") is not None else
                str(p.get("name")) if p.get("name") is not None else
                f"node_{i}"
            )
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
    regions = []
    for _, p in props.items():
        r = str(p.get("region", "0"))
        if r not in regions:
            regions.append(r)
    # Numeric IDs? preserve
    try:
        _ = [int(r) for r in regions]
        dc_to_idx = {str(r): int(r) for r in regions}
        idx_to_dc = [str(i) for i in sorted({int(r) for r in regions})]
        return dc_to_idx, idx_to_dc
    except Exception:
        dc_to_idx = {r: i for i, r in enumerate(regions)}
        idx_to_dc = regions
        return dc_to_idx, idx_to_dc

def _coerce_rows(epoch_data, avg_in: int, avg_out: int) -> List[dict]:
    rows: List[dict] = []
    if hasattr(epoch_data, "iterrows") and hasattr(epoch_data, "columns"):
        for _, row in epoch_data.iterrows():
            rows.append({
                "source_dc_id": int(row["source_dc_id"]),
                "model_type":   row["model_type"],
                "num_tokens":   int(row.get("num_tokens", row.get("prompt_tokens", avg_in))),
                "output_tokens":int(row.get("output_tokens", avg_out)),
                "batch_size":   int(row.get("batch_size", 1)),
                "time_index":   int(row.get("time_index", 0)),
                # optional SLA per-request; else global
                "time_requirement": float(row.get("time_requirement", float("nan")))
            })
        return rows
    if isinstance(epoch_data, list):
        for r in epoch_data:
            rows.append({
                "source_dc_id": int(r.get("source_dc_id", 0)),
                "model_type":   r.get("model_type", "Llama-7b"),
                "num_tokens":   int(r.get("num_tokens", r.get("prompt_tokens", avg_in))),
                "output_tokens":int(r.get("output_tokens", avg_out)),
                "batch_size":   int(r.get("batch_size", 1)),
                "time_index":   int(r.get("time_index", 0)),
                "time_requirement": float(r.get("time_requirement", float("nan")))
            })
        return rows
    return rows

def _build_dc_caps(node_properties, epoch_summary):
    """
    Returns per-DC:
      - compute_tps[dc]: sum tp_tokens_per_s
      - bw_Bps[dc]: uplink bandwidth (sum of 'uplink_Bps' in node props or from epoch_summary['dc_bandwidth_Bps'])
      - e_infer_Jptok[dc], e_tran_JpB[dc]: energy per token / per byte (optional)
    """
    props = _normalize_node_props(node_properties)
    dc_to_idx, _ = _make_dc_index_maps(node_properties, epoch_summary)

    # defaults (can be overridden by epoch_summary)
    default_tps  = float(epoch_summary.get("default_tp_tokens_per_s", 4000.0))
    # bandwidth defaults: allow an explicit map; else 1 GByte/s fallback
    dc_bw_map = epoch_summary.get("dc_bandwidth_Bps", {})
    default_bw = float(epoch_summary.get("default_uplink_Bps", 1e9))

    # energy model knobs
    default_e_infer = float(epoch_summary.get("energy_per_token_joule", 0.0))
    default_e_tran  = float(epoch_summary.get("energy_per_byte_joule", 0.0))

    compute_tps: Dict[int, float] = {}
    bw_Bps: Dict[int, float] = {}
    e_infer_Jptok: Dict[int, float] = {}
    e_tran_JpB: Dict[int, float] = {}

    # sum capacities from nodes
    for _, p in props.items():
        dc_key = str(p.get("region", "0"))
        dc = dc_to_idx.get(dc_key, int(dc_key) if dc_key.isdigit() else 0)
        tps = float(p.get("tp_tokens_per_s", default_tps))
        bw  = float(p.get("uplink_Bps", dc_bw_map.get(dc_key, default_bw)))
        compute_tps[dc] = compute_tps.get(dc, 0.0) + tps
        bw_Bps[dc]      = bw_Bps.get(dc,      0.0) + bw  # aggregated uplink
        e_infer_Jptok.setdefault(dc, float(p.get("energy_per_token_joule", default_e_infer)))
        e_tran_JpB.setdefault(dc,  float(p.get("energy_per_byte_joule",  default_e_tran)))

    # if nothing discovered (unlikely), ensure at least one DC 0
    if not compute_tps:
        compute_tps[0] = default_tps
        bw_Bps[0] = default_bw
        e_infer_Jptok[0] = default_e_infer
        e_tran_JpB[0] = default_e_tran

    # avoid zero
    for dc in list(compute_tps.keys()):
        compute_tps[dc] = max(1e-6, compute_tps[dc])
        bw_Bps[dc]      = max(1e-6, bw_Bps.get(dc, default_bw))
    return compute_tps, bw_Bps, e_infer_Jptok, e_tran_JpB

def _estimate_bytes_for_request(num_tokens: int, epoch_summary: dict) -> int:
    # crude: bytes ~= tokens * token_bytes (default 4), plus overhead for prompt payloads
    tok_bytes = int(epoch_summary.get("token_bytes", 4))
    prompt_overhead = int(epoch_summary.get("prompt_overhead_bytes", 1024))
    return prompt_overhead + num_tokens * tok_bytes

def _build_power_plan(routed_token_by_dc: Dict[int, float], epoch_summary) -> Dict[int, Dict[int, str]]:
    node_types = list(epoch_summary.get("node_types", [0,1,2,3,4,5]))
    min_idle = int(epoch_summary.get("min_idle_types", 1))
    max_idle = int(epoch_summary.get("max_idle_types", len(node_types)))
    max_idle = max(1, min(max_idle, len(node_types)))
    total = sum(max(0.0, v) for v in routed_token_by_dc.values()) or 1.0
    shares = {dc: max(0.0, v) / total for dc, v in routed_token_by_dc.items()}
    plan: Dict[int, Dict[int, str]] = {}
    for dc, s in shares.items():
        if s <= 0.0:
            plan[dc] = {nt: "Off" for nt in node_types}
            continue
        k = min_idle + int(round((max_idle - min_idle) * s))
        k = max(min_idle, min(max_idle, k))
        plan[dc] = {nt: ("Idle" if i < k else "Off") for i, nt in enumerate(node_types)}
    return plan

# ---------- PerLLM (CS-UCB over DCs) ----------
class PerLLM:
    @staticmethod
    def milp_optimizer(epoch_data, epoch_idx: int, node_properties, epoch_summary: Dict[str, Any]):
        """
        PerLLM as CS-UCB over DC choices (co-located phases):
          - For each request, choose DC with highest UCB among *feasible* actions (constraints),
            otherwise include a penalty for infeasible ones.
          - Constraints mirror paper: processing time (Di <= D_req), bandwidth and compute caps (queuing-aware).
          - Schedule plan: simulator's list[dict].
          - Power plan: scales Idle node types by routed token share per DC.
        """
        # ---- knobs from paper / environment ----
        # Objective weights for a negative-energy reward proxy (only used for UCB reward estimate)
        w_tran  = float(epoch_summary.get("omega_tran", 1.0))
        w_infer = float(epoch_summary.get("omega_infer", 1.0))
        w_idle  = float(epoch_summary.get("omega_idle",  0.0))  # usually 0 unless you model it

        # CS-UCB params (lambda, delta, theta in the paper; using similar symbols)
        lam    = float(epoch_summary.get("perllm_lambda", 1.0))     # constraint satisfaction coef
        delta  = float(epoch_summary.get("perllm_delta",  1.0))     # exploration scaling
        thetaP = float(epoch_summary.get("perllm_thetaP", 1.0))     # penalty scaling in UCB
        # SLA / constraints
        default_req_s = float(epoch_summary.get("default_time_requirement_s", 6.0))

        # Workload defaults
        avg_in  = int(epoch_summary.get("avg_input_tokens", 700))
        avg_out = int(epoch_summary.get("avg_output_tokens", 250))

        # Capacities / energy
        compute_tps, bw_Bps, e_infer_Jptok, e_tran_JpB = _build_dc_caps(node_properties, epoch_summary)
        dc_ids = sorted(compute_tps.keys())

        # Queues (simple token/byte backlogs to approximate waiting time)
        pend_tokens: Dict[int, float] = {dc: 0.0 for dc in dc_ids}
        pend_bytes:  Dict[int, float] = {dc: 0.0 for dc in dc_ids}

        # UCB statistics per DC
        picks: Dict[int, int]     = {dc: 0 for dc in dc_ids}
        meanR: Dict[int, float]   = {dc: 0.0 for dc in dc_ids}  # running mean reward

        # Per-DC token accounting for power plan
        routed_token_by_dc: Dict[int, float] = {dc: 0.0 for dc in dc_ids}

        # Build schedule
        schedule_plan: List[dict] = []
        rows = _coerce_rows(epoch_data, avg_in, avg_out)

        # --- helper: compute feasibility + reward proxy for (request r, DC j) ---
        def eval_dc_for_request(r: dict, dc: int):
            # size, compute demand
            p_tok = int(r["num_tokens"])
            o_tok = int(r["output_tokens"])
            tot_tok = p_tok + o_tok  # simple proxy for total compute
            bytes_in = _estimate_bytes_for_request(p_tok, epoch_summary)

            # waiting + service time estimates (queueing by backlogs)
            wait_tran = pend_bytes[dc] / bw_Bps[dc]
            svc_tran  = bytes_in   / bw_Bps[dc]
            wait_inf  = pend_tokens[dc] / compute_tps[dc]
            svc_inf   = tot_tok    / compute_tps[dc]
            Di = wait_tran + svc_tran + wait_inf + svc_inf

            # constraint: processing time must be within requirement
            Dreq = r["time_requirement"]
            if math.isnan(Dreq):  # use default if not provided
                Dreq = default_req_s

            # Feasibility score f(y) >= 0 → satisfied (paper eq. 3 idea, simplified to time constraint here)
            f_y = (Dreq - Di) / max(1e-6, Dreq)  # >=0 if feasible; negative if violation

            # (Optional) also guard instantaneous caps: if backlog would exceed a soft window, make f_y more negative
            # Soft cap windows (seconds worth of work)
            soft_win_s = float(epoch_summary.get("perllm_soft_window_s", 1.0))
            cap_tokens_soft = compute_tps[dc] * soft_win_s
            cap_bytes_soft  = bw_Bps[dc]      * soft_win_s
            if pend_tokens[dc] + tot_tok > cap_tokens_soft or pend_bytes[dc] + bytes_in > cap_bytes_soft:
                f_y -= 0.2  # nudge penalty

            # Energy proxy for reward (lower is better; we negate)
            Einfer = e_infer_Jptok[dc] * tot_tok
            Etran  = e_tran_JpB[dc]    * bytes_in
            Eidle  = 0.0  # optionally, add idle energy if you track it externally
            reward = -(w_tran * Etran + w_infer * Einfer + w_idle * Eidle)

            # penalty term P(t): zero if feasible, |f_y| otherwise
            P_t = 0.0 if f_y >= 0.0 else -abs(f_y)
            return Di, f_y, reward, P_t, bytes_in, tot_tok

        # --- scheduling loop with CS-UCB selection ---
        t = 0
        for r in rows:
            t += 1
            # compute UCB for each DC
            best_dc, best_ucb = None, None
            best_eval = None
            for dc in dc_ids:
                Di, f_y, reward, P_t, bytes_in, tot_tok = eval_dc_for_request(r, dc)
                n = max(1, picks[dc])  # to avoid div by zero in first rounds
                ucb = meanR[dc] + delta * math.sqrt(max(0.0, math.log(max(1.0, t)) / n)) + thetaP * P_t
                # If feasible, we can bias toward higher reward by blending some of it into UCB
                if f_y >= 0.0:
                    ucb += lam * (reward)
                # choose max UCB
                if (best_ucb is None) or (ucb > best_ucb):
                    best_ucb, best_dc, best_eval = ucb, dc, (Di, f_y, reward, P_t, bytes_in, tot_tok)

            # assign to best_dc
            target_dc_id = int(best_dc)
            Di, f_y, reward, P_t, bytes_in, tot_tok = best_eval

            # update per-DC UCB stats (incremental mean)
            picks[target_dc_id] += 1
            n = picks[target_dc_id]
            meanR[target_dc_id] += (reward - meanR[target_dc_id]) / float(n)

            # update queues (backlogs) — “commit”
            pend_bytes[target_dc_id]  += bytes_in
            pend_tokens[target_dc_id] += tot_tok
            # drain a soft window so queues don’t explode (proxy for time passing within the epoch)
            soft_win_s = float(epoch_summary.get("perllm_soft_window_s", 1.0))
            pend_bytes[target_dc_id]  = max(0.0, pend_bytes[target_dc_id]  - bw_Bps[target_dc_id]     * soft_win_s)
            pend_tokens[target_dc_id] = max(0.0, pend_tokens[target_dc_id] - compute_tps[target_dc_id] * soft_win_s)

            routed_token_by_dc[target_dc_id] += tot_tok

            schedule_plan.append({
                "target_dc_id": target_dc_id,
                "model_type":   r["model_type"],
                "num_tokens":   r["num_tokens"],
                "batch_size":   r["batch_size"],
                "source_dc_id": r["source_dc_id"],
                "time_index":   r["time_index"],
            })

        # ---- power plan from routed load share ----
        power_plan = _build_power_plan(routed_token_by_dc, epoch_summary)

        # ---- run simulator ----
        sim_out = LLM_Simulator(epoch_idx, epoch_data, schedule_plan, power_plan)

        # ---- unpack standard tuple/dict ----
        stats, results, leftovers = {}, [], []
        if isinstance(sim_out, tuple) and len(sim_out) >= 2:
            stats = dict(sim_out[0]) if sim_out[0] else {}
            results = list(sim_out[1]) if sim_out[1] else []
            if len(sim_out) >= 3 and sim_out[2] is not None:
                leftovers = sim_out[2]
        elif isinstance(sim_out, dict):
            stats = dict(sim_out.get("metrics", {}))
            results = list(sim_out.get("results", []))
            leftovers = sim_out.get("leftover_requests", [])

        # ensure keys
        stats.setdefault("avg_ttft", 0.0)
        stats.setdefault("energy_cost", 0.0)
        stats.setdefault("carbon_emissions", 0.0)
        stats.setdefault("water_usage", 0.0)
        # telemetry
        stats.setdefault("network_load", {"dc_token_totals": {int(k): float(v) for k, v in routed_token_by_dc.items()}})

        return stats, results, leftovers
