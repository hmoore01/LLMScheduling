# Helix.py — Serverless/DC-level Helix comparison work
# API: stats, results, leftovers = Helix.milp_optimizer(epoch_data, epoch_idx, node_properties, epoch_summary)
# Generates the simulator-ready schedule list (no adapters) and an intelligent power plan.

from typing import Dict, List, Tuple, Any
from simulation import LLM_Simulator
from dataclasses import dataclass


# --------- Helpers: normalization & data coercion ----------
def _normalize_node_props(node_properties) -> Dict[str, dict]:
    """Accept dict or list[dict]; return {node_id: props} with node_id present."""
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
    # best effort
    for nid, props in dict(node_properties).items():
        p = dict(props) if isinstance(props, dict) else {}
        p.setdefault("node_id", str(nid))
        norm[str(nid)] = p
    return norm

def _coerce_epoch_data(epoch_data, avg_in: int, avg_out: int):
    """Return (req_rows) where each row is a simple dict with required fields present."""
    rows: List[dict] = []
    # pandas?
    if hasattr(epoch_data, "iterrows") and hasattr(epoch_data, "columns"):
        for _, row in epoch_data.iterrows():
            rows.append({
                "source_dc_id": int(row["source_dc_id"]),
                "model_type": row["model_type"],
                "num_tokens": int(row.get("num_tokens", row.get("prompt_tokens", avg_in))),
                "batch_size": int(row.get("batch_size", 1)),
                "time_index": int(row.get("time_index", 0))
            })
        return rows
    # list of dicts?
    if isinstance(epoch_data, list):
        for r in epoch_data:
            rows.append({
                "source_dc_id": int(r.get("source_dc_id", 0)),
                "model_type": r.get("model_type", "Llama-7b"),
                "num_tokens": int(r.get("num_tokens", r.get("prompt_tokens", avg_in))),
                "batch_size": int(r.get("batch_size", 1)),
                "time_index": int(r.get("time_index", 0))
            })
        return rows
    # fallback (empty)
    return rows

def _make_dc_index_maps(node_properties, epoch_summary):
    """
    Build DC <-> index mapping. If epoch_summary['datacenters'] provided, use it.
    Else infer from node_properties['region'] values; if numeric, preserve numbers.
    Returns: (dc_to_idx: dict[str,int], idx_to_dc: list[str])
    """
    # explicit list wins
    dc_list = epoch_summary.get("datacenters")
    if isinstance(dc_list, list) and dc_list:
        dc_to_idx = {str(dc): i for i, dc in enumerate(dc_list)}
        idx_to_dc = [str(dc) for dc in dc_list]
        return dc_to_idx, idx_to_dc

    # infer from node regions
    props = _normalize_node_props(node_properties)
    regions = []
    for _, p in props.items():
        r = str(p.get("region", "0"))
        if r not in regions:
            regions.append(r)
    # numeric regions?
    try:
        ints = [int(r) for r in regions]
        dc_to_idx = {str(r): int(r) for r in regions}
        idx_to_dc = [str(i) for i in sorted(set(ints))]
        return dc_to_idx, idx_to_dc
    except Exception:
        dc_to_idx = {r: i for i, r in enumerate(regions)}
        idx_to_dc = regions
        return dc_to_idx, idx_to_dc

def _latency_ms(src_dc: str, dst_dc: str, dc_latency_ms: Dict[str, Dict[str, float]]) -> float:
    """Lookup latency if provided; otherwise 0."""
    try:
        return float(dc_latency_ms.get(str(src_dc), {}).get(str(dst_dc), 0.0))
    except Exception:
        return 0.0


# --------- Power plan logic (Idle/Off only; On is automatic in your sim) ----------
def _build_power_plan(routed_token_share_by_dc: Dict[int, float], epoch_summary) -> Dict[int, Dict[int, str]]:
    """
    Heuristic:
      - No load -> all Off
      - With load -> enable k Idle types where k scales with share (1..6)
    Tunables:
      - node_types (default [0..5])
      - min_idle_types (default 1), max_idle_types (default 6)
    """
    node_types = list(epoch_summary.get("node_types", [0, 1, 2, 3, 4, 5]))
    min_idle = int(epoch_summary.get("min_idle_types", 1))
    max_idle = int(epoch_summary.get("max_idle_types", len(node_types)))
    max_idle = max(1, min(max_idle, len(node_types)))

    # normalize shares
    total_share = sum(max(0.0, v) for v in routed_token_share_by_dc.values()) or 1.0
    shares = {dc: max(0.0, v) / total_share for dc, v in routed_token_share_by_dc.items()}

    power_plan: Dict[int, Dict[int, str]] = {}
    for dc_id, share in shares.items():
        if share <= 0.0:
            power_plan[dc_id] = {nt: "Off" for nt in node_types}
            continue
        k = min_idle + int(round( (max_idle - min_idle) * share ))
        k = max(min_idle, min(max_idle, k))
        plan = {}
        # enable first k as Idle, rest Off (order is just index order)
        for idx, nt in enumerate(node_types):
            plan[nt] = "Idle" if idx < k else "Off"
        power_plan[dc_id] = plan
    return power_plan


# --------- Public API ----------
class Helix:
    @staticmethod
    def milp_optimizer(epoch_data, epoch_idx: int, node_properties, epoch_summary: Dict[str, Any]):
        """
        Serverless/DC-level Helix:
          - Aggregate DC capacity from node_properties
          - Choose a single target DC per request using capacity-aware + latency-aware cost
          - Emit schedule (list of dict) matching simulator schema
          - Build intelligent power plan (Idle/Off) based on routed load share
          - Run LLM_Simulator and return (stats, results, leftovers)
        """

        # ---- Tunables / defaults ----
        avg_in  = int(epoch_summary.get("avg_input_tokens", 700))
        avg_out = int(epoch_summary.get("avg_output_tokens", 250))  # not directly used here
        latency_weight = float(epoch_summary.get("latency_weight", 0.001))  # converts ms into cost units
        dc_latency_ms = epoch_summary.get("dc_latency_ms", {})  # dict[src][dst] = ms

        # ---- DC capacity aggregation ----
        props = _normalize_node_props(node_properties)
        dc_to_idx, idx_to_dc = _make_dc_index_maps(node_properties, epoch_summary)

        # capacity per dc: sum of tp_tokens_per_s (tokens/s)
        cap_tps: Dict[int, float] = {}
        for _, p in props.items():
            dc_key = str(p.get("region", "0"))
            dc_id  = dc_to_idx.get(dc_key, int(dc_key) if dc_key.isdigit() else 0)
            cap_tps[dc_id] = cap_tps.get(dc_id, 0.0) + float(p.get("tp_tokens_per_s", 4000.0))
        # avoid zero
        for dc in list(cap_tps.keys()):
            cap_tps[dc] = max(1e-6, cap_tps[dc])

        # if no DCs discovered, short-circuit
        if not cap_tps:
            sim_out = LLM_Simulator(epoch_idx, epoch_data, [], epoch_summary.get("power_plan"), epoch_summary)
            if isinstance(sim_out, tuple):
                return sim_out[0] if len(sim_out) > 0 else {}, sim_out[1] if len(sim_out) > 1 else [], sim_out[2] if len(sim_out) > 2 else []
            return {}, [], []

        # ---- Coerce workload rows ----
        rows = _coerce_epoch_data(epoch_data, avg_in, avg_out)

        # ---- Routing state ----
        pending_tokens: Dict[int, float] = {dc: 0.0 for dc in cap_tps.keys()}
        routed_tokens_by_dc: Dict[int, float] = {dc: 0.0 for dc in cap_tps.keys()}

        # ---- Build schedule list (simulator schema) ----
        schedule_plan: List[dict] = []

        # We need a mapping from numeric source_dc_id in rows to DC key strings for latency lookup.
        # Build reverse map idx_to_dc_str:
        idx_to_dc_str = {idx: key for key, idx in dc_to_idx.items()}

        for row in rows:
            src_id = int(row["source_dc_id"])
            src_key = idx_to_dc_str.get(src_id, str(src_id))

            # pick best dc by cost = load_ratio + latency_weight * latency_ms
            best_dc = None
            best_cost = None
            for dc_id, cap in cap_tps.items():
                load_ratio = pending_tokens[dc_id] / cap
                lat_ms = _latency_ms(src_key, idx_to_dc_str.get(dc_id, str(dc_id)), dc_latency_ms)
                cost = load_ratio + latency_weight * lat_ms
                if (best_cost is None) or (cost < best_cost):
                    best_cost = cost
                    best_dc = dc_id

            target_dc_id = int(best_dc)
            # enqueue its tokens (simple proxy; simulator does the real timing)
            pending_tokens[target_dc_id] += row["num_tokens"]
            routed_tokens_by_dc[target_dc_id] += row["num_tokens"]

            schedule_plan.append({
                "target_dc_id": target_dc_id,
                "model_type": row["model_type"],
                "num_tokens": row["num_tokens"],
                "batch_size": row["batch_size"],
                "source_dc_id": row["source_dc_id"],
                "time_index": row["time_index"],
            })

            # simple smoothing so one big req doesn't block the queue forever
            pending_tokens[target_dc_id] = max(0.0, pending_tokens[target_dc_id] - cap_tps[target_dc_id])

        # ---- Intelligent power plan (Idle/Off) ----
        # Scale Idle node-types with each DC's share of routed tokens
        total_tokens = sum(routed_tokens_by_dc.values()) or 1.0
        share = {dc: (routed_tokens_by_dc[dc] / total_tokens) for dc in routed_tokens_by_dc}
        power_plan = _build_power_plan(share, epoch_summary)

        print(schedule_plan)

        # ---- Run simulator ----
        from simulation import LLM_Simulator

        def _coerce_dc_dict(obj):
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, (list, tuple)):
                return {i: v for i, v in enumerate(obj)}
            return {}

        schedule_plan = _coerce_dc_dict(schedule_plan)
        power_plan = _coerce_dc_dict(power_plan)



        sim_out = LLM_Simulator(epoch_idx, epoch_data, schedule_plan, power_plan, epoch_summary)

        stats = {}
        results = []
        leftovers = []
        if isinstance(sim_out, tuple) and len(sim_out) >= 2:
            stats = dict(sim_out[0]) if sim_out[0] is not None else {}
            results = list(sim_out[1]) if sim_out[1] is not None else []
            if len(sim_out) >= 3 and sim_out[2] is not None:
                leftovers = sim_out[2]
        elif isinstance(sim_out, dict):
            stats = dict(sim_out.get("metrics", {}))
            results = list(sim_out.get("results", []))
            leftovers = sim_out.get("leftover_requests", [])
        else:
            stats, results, leftovers = {}, [], []

        # Ensure common keys exist
        stats.setdefault("avg_ttft", stats.get("avg_ttft", 0.0))
        stats.setdefault("energy_cost", stats.get("energy_cost", 0.0))
        stats.setdefault("carbon_emissions", stats.get("carbon_emissions", 0.0))
        stats.setdefault("water_usage", stats.get("water_usage", 0.0))

        # Optional: attach simple “network_load” DC→DC KV proxy (here we have no prompt/token split,
        # so we just report routed token totals per DC)
        stats.setdefault("network_load", {"dc_token_totals": {int(k): float(v) for k, v in routed_tokens_by_dc.items()}})

        return stats, results, leftovers

