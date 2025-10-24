# Splitwise.py — Co-located Splitwise comparison work
# API: stats, results, leftovers = Splitwise.milp_optimizer(epoch_data, epoch_idx, node_properties, epoch_summary)

from typing import Dict, List, Any
from simulation import LLM_Simulator   # <-- unified import

# ---------------- Helpers ----------------
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
    try:
        _ = [int(r) for r in regions]
        dc_to_idx = {str(r): int(r) for r in regions}
        idx_to_dc = [str(i) for i in sorted({int(r) for r in regions})]
        return dc_to_idx, idx_to_dc
    except Exception:
        dc_to_idx = {r: i for i, r in enumerate(regions)}
        idx_to_dc = regions
        return dc_to_idx, idx_to_dc


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

# ---------------- Splitwise (co-located) ----------------
class Splitwise:
    @staticmethod
    def milp_optimizer(epoch_data, epoch_idx: int, node_properties, epoch_summary: Dict[str, Any]):
        """
        Co-located Splitwise:
          - Both prompt + token handled in the same DC.
          - Requests routed based on prompt capacity + token capacity + latency.
          - Builds schedule_plan and a token-share-based power_plan.
        """
        # Extract maps
        props = _normalize_node_props(node_properties)
        dc_to_idx, idx_to_dc = _make_dc_index_maps(node_properties, epoch_summary)
        idx_to_dc_str = {idx: key for key, idx in dc_to_idx.items()}

        # Basic workload parameters
        avg_in  = int(epoch_summary.get("avg_input_tokens", 700))
        avg_out = int(epoch_summary.get("avg_output_tokens", 250))
        w_token = float(epoch_summary.get("splitwise_token_weight", 1.5))
        w_lat   = float(epoch_summary.get("latency_weight", 0.001))

        prompt_cap = {dc: 4000.0 for dc in dc_to_idx.values()}
        token_cap  = {dc: 8000.0 for dc in dc_to_idx.values()}

        schedule_plan: List[dict] = []
        routed_token_by_dc: Dict[int, float] = {dc: 0.0 for dc in token_cap}

        # Coerce rows
        if hasattr(epoch_data, "iterrows"):
            rows = [
                {
                    "source_dc_id": int(row["source_dc_id"]),
                    "model_type": row["model_type"],
                    "num_tokens": int(row.get("num_tokens", avg_in)),
                    "output_tokens": int(row.get("output_tokens", avg_out)),
                    "batch_size": int(row.get("batch_size", 1)),
                    "time_index": int(row.get("time_index", 0)),
                }
                for _, row in epoch_data.iterrows()
            ]
        else:
            rows = [
                {
                    "source_dc_id": int(r.get("source_dc_id", 0)),
                    "model_type": r.get("model_type", "Llama-7b"),
                    "num_tokens": int(r.get("num_tokens", avg_in)),
                    "output_tokens": int(r.get("output_tokens", avg_out)),
                    "batch_size": int(r.get("batch_size", 1)),
                    "time_index": int(r.get("time_index", 0)),
                }
                for r in epoch_data
            ]

        for row in rows:
            p_tokens = row["num_tokens"]
            o_tokens = row["output_tokens"]
            src_id   = row["source_dc_id"]

            best_dc, best_cost = None, None
            for dc in prompt_cap.keys():
                cost = (p_tokens / prompt_cap[dc]) + w_token * (o_tokens / token_cap[dc])
                if best_cost is None or cost < best_cost:
                    best_cost, best_dc = cost, dc

            target_dc_id = int(best_dc)
            routed_token_by_dc[target_dc_id] += o_tokens

            schedule_plan.append({
                "target_dc_id": target_dc_id,
                "model_type": row["model_type"],
                "num_tokens": p_tokens,
                "batch_size": row["batch_size"],
                "source_dc_id": src_id,
                "time_index": row["time_index"],
            })

        power_plan = _build_power_plan(routed_token_by_dc, epoch_summary)

        # Run simulator
        sim_out = LLM_Simulator(epoch_idx, epoch_data, schedule_plan, power_plan)

        # Unpack results
        stats, results, leftovers = {}, [], []
        if isinstance(sim_out, tuple) and len(sim_out) >= 2:
            stats = dict(sim_out[0]) if sim_out[0] else {}
            results = list(sim_out[1]) if sim_out[1] else []
            if len(sim_out) >= 3 and sim_out[2] is not None:
                leftovers = sim_out[2]

        stats.setdefault("avg_ttft", 0.0)
        stats.setdefault("energy_cost", 0.0)
        stats.setdefault("carbon_emissions", 0.0)
        stats.setdefault("water_usage", 0.0)
        stats.setdefault("network_load", {"dc_token_totals": {int(k): float(v) for k, v in routed_token_by_dc.items()}})

        return stats, results, leftovers
