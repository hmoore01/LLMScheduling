# Helix.py
from typing import Dict, List, Tuple, Any

# --- Import the simulator with fallbacks ---
try:
    from Rate_Flow_Sim import LLM_Simulator  # your file Rate_Flow_Sim.py
except Exception:
    # last-ditch: some repos name it simulation.py
    from simulation import LLM_Simulator  # type: ignore


# ----------------- Helpers -----------------
def _coerce_epoch_data(epoch_data, default_tokens_in: int = 700) -> List[dict]:
    """
    Return list of dict rows with required fields:
      source_dc_id, model_type, num_tokens, batch_size, time_index
    """
    rows: List[dict] = []
    if hasattr(epoch_data, "iterrows") and hasattr(epoch_data, "columns"):
        for _, r in epoch_data.iterrows():
            rows.append({
                "source_dc_id": int(r.get("source_dc_id", 0)),
                "model_type": str(r.get("model_type", "Llama7b")),
                "num_tokens": int(r.get("num_tokens", r.get("prompt_tokens", default_tokens_in))),
                "batch_size": int(r.get("batch_size", 1)),
                "time_index": int(r.get("time_index", 0)),
            })
        return rows
    if isinstance(epoch_data, list):
        for r in epoch_data:
            rows.append({
                "source_dc_id": int(r.get("source_dc_id", 0)),
                "model_type": str(r.get("model_type", "Llama7b")),
                "num_tokens": int(r.get("num_tokens", r.get("prompt_tokens", default_tokens_in))),
                "batch_size": int(r.get("batch_size", 1)),
                "time_index": int(r.get("time_index", 0)),
            })
        return rows
    return rows


def _discover_dcs_from_node_props(node_properties) -> List[int]:
    """Pull DC ids from node_properties (expects 'datacenter_id' per entry)."""
    dcs: List[int] = []
    if isinstance(node_properties, dict):
        iterable = node_properties.values()
    elif isinstance(node_properties, (list, tuple)):
        iterable = node_properties
    else:
        try:
            iterable = dict(node_properties).values()
        except Exception:
            iterable = []

    for p in iterable:
        try:
            dc = int(p.get("datacenter_id"))
            if dc not in dcs:
                dcs.append(dc)
        except Exception:
            continue
    dcs.sort()
    return dcs


def _discover_dcs_from_epoch(epoch_data) -> List[int]:
    """Fallback discovery from workload rows."""
    dcs: List[int] = []
    if hasattr(epoch_data, "iterrows") and hasattr(epoch_data, "columns"):
        if "source_dc_id" in epoch_data.columns:
            for v in epoch_data["source_dc_id"].unique().tolist():
                try:
                    iv = int(v)
                    if iv not in dcs:
                        dcs.append(iv)
                except Exception:
                    pass
    elif isinstance(epoch_data, list):
        for r in epoch_data:
            try:
                iv = int(r.get("source_dc_id", -1))
                if iv >= 0 and iv not in dcs:
                    dcs.append(iv)
            except Exception:
                pass
    dcs.sort()
    return dcs


def _capacity_per_dc(node_properties, dcs: List[int]) -> Dict[int, float]:
    """
    Simple capacity proxy:
      capacity ∝ number of nodes listed in node_properties per DC.
      If node_properties is empty or has no datacenter_id, fall back to equal capacities.
    Units are tokens/second but only relative magnitudes matter for this router.
    """
    caps = {dc: 0.0 for dc in dcs}
    count_any = False

    if isinstance(node_properties, dict):
        iterable = node_properties.values()
    elif isinstance(node_properties, (list, tuple)):
        iterable = node_properties
    else:
        try:
            iterable = dict(node_properties).values()
        except Exception:
            iterable = []

    for p in iterable:
        try:
            dc = int(p.get("datacenter_id"))
            if dc in caps:
                caps[dc] += 1.0
                count_any = True
        except Exception:
            continue

    if not count_any:
        # Equal capacity over discovered DCs
        for dc in caps:
            caps[dc] = 1.0

    # Avoid zeros
    for dc in caps:
        caps[dc] = max(1e-6, caps[dc])
    return caps


def _build_power_plan(routed_token_share_by_dc: Dict[int, float], epoch_summary) -> Dict[int, Dict[int, str]]:
    """
    Heuristic:
      - No load -> all Off
      - With load -> enable k Idle types where k scales with share (1..6)
    Tunables via epoch_summary:
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
        k = min_idle + int(round((max_idle - min_idle) * share))
        k = max(min_idle, min(max_idle, k))
        plan = {}
        # enable first k as Idle, rest Off (index order is arbitrary but deterministic)
        for idx, nt in enumerate(node_types):
            plan[nt] = "Idle" if idx < k else "Off"
        power_plan[dc_id] = plan
    return power_plan


def _normalize_sim_output(sim_out):
    """Return (stats, results, leftovers) with required keys present."""
    stats = {}
    results = []
    leftovers = []
    if isinstance(sim_out, tuple):
        if len(sim_out) > 0 and sim_out[0] is not None:
            stats = dict(sim_out[0])
        if len(sim_out) > 1 and sim_out[1] is not None:
            results = list(sim_out[1])
        if len(sim_out) > 2 and sim_out[2] is not None:
            leftovers = sim_out[2]
    elif isinstance(sim_out, dict):
        stats = dict(sim_out.get("metrics", {}))
        results = list(sim_out.get("results", []))
        leftovers = sim_out.get("leftover_requests", [])
    # Ensure common keys exist
    stats.setdefault("avg_ttft", stats.get("avg_ttft", 0.0))
    stats.setdefault("energy_cost", stats.get("energy_cost", 0.0))
    stats.setdefault("carbon_emissions", stats.get("carbon_emissions", 0.0))
    stats.setdefault("water_usage", stats.get("water_usage", 0.0))
    return stats, results, leftovers


# ----------------- Public API -----------------
class Helix:
    @staticmethod
    def milp_optimizer(epoch_data, epoch_idx: int, node_properties, epoch_summary: Any):
        """
        Build a request-level schedule + power plan, run LLM_Simulator, and return:
          (stats, results, leftovers)
        """
        # 1) Coerce workload rows
        rows = _coerce_epoch_data(epoch_data, default_tokens_in=int(epoch_summary.get("avg_input_tokens", 700)))

        # 2) Discover DC universe
        dcs = _discover_dcs_from_node_props(node_properties)
        if not dcs:
            dcs = _discover_dcs_from_epoch(epoch_data)
        if not dcs:
            dcs = [0]  # final fallback

        # 3) Capacity model (relative)
        cap_tps = _capacity_per_dc(node_properties, dcs)

        # 4) Greedy routing: pick DC with minimal (pending / capacity)
        pending_tokens = {dc: 0.0 for dc in dcs}
        routed_tokens_by_dc = {dc: 0.0 for dc in dcs}

        schedule_plan: List[dict] = []
        for r in rows:
            best_dc = None
            best_score = None
            for dc in dcs:
                score = pending_tokens[dc] / cap_tps[dc]
                if best_score is None or score < best_score:
                    best_score = score
                    best_dc = dc

            target_dc = int(best_dc)
            pending_tokens[target_dc] += float(r["num_tokens"])
            routed_tokens_by_dc[target_dc] += float(r["num_tokens"])

            schedule_plan.append({
                "target_dc_id": target_dc,
                "model_type": str(r["model_type"]),
                "num_tokens": int(r["num_tokens"]),
                "batch_size": int(r["batch_size"]),
                "source_dc_id": int(r["source_dc_id"]),
                "time_index": int(r["time_index"]),
            })

            # small smoothing step
            pending_tokens[target_dc] = max(0.0, pending_tokens[target_dc] - cap_tps[target_dc])

        # 5) Power plan based on routed share
        total_tokens = sum(routed_tokens_by_dc.values()) or 1.0
        share = {dc: (routed_tokens_by_dc[dc] / total_tokens) for dc in routed_tokens_by_dc}
        power_plan = _build_power_plan(share, epoch_summary)

        # 6) Run simulator (frameworks are responsible for this)
        sim_out = LLM_Simulator(epoch_idx, epoch_data, schedule_plan, power_plan)

        # 7) Normalize and return
        stats, results, leftovers = _normalize_sim_output(sim_out)

        # Attach a tiny network-load proxy to stats (optional)
        stats.setdefault("network_load", {"dc_token_totals": {int(k): float(v) for k, v in routed_tokens_by_dc.items()}})

        return stats, results, leftovers

