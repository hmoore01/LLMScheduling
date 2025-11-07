# Helix.py
from typing import Dict, List, Tuple, Any
import math

# --- Simulator imports ---
import pandas as pd

try:
    import Rate_Flow_Sim as rfs  # your file Rate_Flow_Sim.py
    from Rate_Flow_Sim import LLM_Simulator
except Exception:  # fallback
    from simulation import LLM_Simulator  # type: ignore
    rfs = None

# ---- Default simulator spec paths (can be overridden later if you add args) ----
DEFAULT_NODE_SPECS = "sim_specs/Node_Specs.csv"
DEFAULT_DC_SPECS   = "sim_specs/Datacenter_specs.csv"
DEFAULT_A100       = "sim_specs/A100_GPU.csv"
DEFAULT_H100       = "sim_specs/H100_GPU.csv"
DEFAULT_LATENCIES  = "sim_specs/Geo_Latencies.csv"
DEFAULT_EPOCH_LEN  = 900


# ----------------- Helpers: align to your epoch data -----------------
def _ensure_epoch_columns(df: pd.DataFrame, epoch_length: int = DEFAULT_EPOCH_LEN,
                          default_tokens: int = 400) -> pd.DataFrame:
    """
    Ensure required columns for the rate path exist:
      source_dc_id (int), model_type (str), num_tokens (int), time_index (int; unused in rate)
    """
    out = df.copy()

    if "source_dc_id" not in out.columns:
        out["source_dc_id"] = 0
    out["source_dc_id"] = pd.to_numeric(out["source_dc_id"], errors="coerce").fillna(0).astype(int)

    if "model_type" not in out.columns:
        out["model_type"] = "Llama7b"
    out["model_type"] = out["model_type"].astype(str).fillna("Llama7b")

    if "num_tokens" not in out.columns:
        if "tokens" in out.columns:
            out["num_tokens"] = out["tokens"]
        elif {"prompt_tokens", "gen_tokens"}.issubset(out.columns):
            out["num_tokens"] = out["prompt_tokens"].fillna(0) + out["gen_tokens"].fillna(0)
        elif "prompt_tokens" in out.columns:
            out["num_tokens"] = out["prompt_tokens"]
        else:
            out["num_tokens"] = default_tokens
    out["num_tokens"] = pd.to_numeric(out["num_tokens"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    if "time_index" not in out.columns:
        out["time_index"] = 0
    out["time_index"] = (
        pd.to_numeric(out["time_index"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=max(0, epoch_length - 1))
        .astype(int)
    )
    return out


def _discover_dcs_from_node_props(node_properties) -> List[int]:
    """Pull DC ids from node_properties (expects 'datacenter_id')."""
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
            pass
    dcs.sort()
    return dcs


def _discover_dcs_from_epoch(epoch_df: pd.DataFrame) -> List[int]:
    """Fallback discovery from workload rows."""
    dcs: List[int] = []
    if "source_dc_id" in epoch_df.columns:
        for v in epoch_df["source_dc_id"].unique().tolist():
            try:
                iv = int(v)
                if iv not in dcs:
                    dcs.append(iv)
            except Exception:
                pass
    dcs.sort()
    return dcs


def _capacity_per_dc(node_properties, dcs: List[int]) -> Dict[int, float]:
    """
    Relative capacity proxy:
      capacity ∝ number of nodes in node_properties per DC.
      If unavailable, use equal capacities across discovered DCs.
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
            pass

    if not count_any:
        for dc in caps:
            caps[dc] = 1.0

    # avoid zeros
    for dc in caps:
        caps[dc] = max(1e-6, caps[dc])
    return caps


def _build_power_plan(routed_token_share_by_dc: Dict[int, float], epoch_summary: Any) -> Dict[int, Dict[int, str]]:
    """
    Heuristic Idle/Off power plan scaled by per-DC share.
    epoch_summary may contain:
      - node_types (default [0..5])
      - min_idle_types (default 1), max_idle_types (default len(node_types))
    """
    node_types = list(epoch_summary.get("node_types", [0, 1, 2, 3, 4, 5])) if isinstance(epoch_summary, dict) else [0,1,2,3,4,5]
    min_idle = int(epoch_summary.get("min_idle_types", 1)) if isinstance(epoch_summary, dict) else 1
    max_idle = int(epoch_summary.get("max_idle_types", len(node_types))) if isinstance(epoch_summary, dict) else len(node_types)
    max_idle = max(1, min(max_idle, len(node_types)))

    total = sum(max(0.0, v) for v in routed_token_share_by_dc.values()) or 1.0
    shares = {dc: max(0.0, v) / total for dc, v in routed_token_share_by_dc.items()}

    power_plan: Dict[int, Dict[int, str]] = {}
    for dc_id, share in shares.items():
        if share <= 0.0:
            power_plan[dc_id] = {nt: "Off" for nt in node_types}
            continue
        k = min_idle + int(round((max_idle - min_idle) * share))
        k = max(min_idle, min(max_idle, k))
        plan = {nt: ("Idle" if idx < k else "Off") for idx, nt in enumerate(node_types)}
        power_plan[dc_id] = plan
    return power_plan


def _load_latencies_csv(path: str) -> List[List[float]]:
    try:
        import csv
        rows: List[List[str]] = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Try to skip a header row
        def is_float(s: str) -> bool:
            try: float(s); return True
            except Exception: return False
        if rows and rows[0] and not is_float(rows[0][0]):
            data = []
            for r in rows[1:]:
                vals = r[1:] if (r and not is_float(r[0])) else r
                data.append([float(x) for x in vals])
            return data
        return [[float(x) for x in r] for r in rows]
    except Exception:
        return []  # treated as zeros by the simulator


def _normalize_sim_output(sim_out):
    """Return (stats, results, leftovers) with required keys present."""
    stats, results, leftovers = {}, [], []
    if isinstance(sim_out, tuple):
        if len(sim_out) > 0 and sim_out[0] is not None: stats = dict(sim_out[0])
        if len(sim_out) > 1 and sim_out[1] is not None: results = list(sim_out[1])
        if len(sim_out) > 2 and sim_out[2] is not None: leftovers = sim_out[2]
    elif isinstance(sim_out, dict):
        stats = dict(sim_out.get("metrics", {}))
        results = list(sim_out.get("results", []))
        leftovers = sim_out.get("leftover_requests", [])
    # Common keys
    stats.setdefault("avg_ttft", stats.get("avg_ttft", stats.get("avg_ttft_sec", 0.0)))
    stats.setdefault("energy_cost", stats.get("energy_cost", 0.0))
    stats.setdefault("carbon_emissions", stats.get("carbon_emissions", 0.0))
    stats.setdefault("water_usage", stats.get("water_usage", 0.0))
    return stats, results, leftovers


# ----------------- Public API -----------------
class Helix:
    @staticmethod
    def milp_optimizer(epoch_data, epoch_idx: int, node_properties, epoch_summary: Any):
        """
        Build a **rate-mode** schedule + power plan, run Rate_Flow_Sim.LLM_Simulator, and return:
            (stats, results, leftovers)

        Inputs:
          - epoch_data: DataFrame with at least ['source_dc_id','model_type','num_tokens'] (we normalize if missing)
          - epoch_summary: can be a DataFrame or dict; only used for small power plan knobs if dict
          - node_properties: used to estimate relative DC capacities (counts per 'datacenter_id')
        """
        # 0) Normalize the epoch rows to ensure required columns exist
        if hasattr(epoch_data, "iterrows") and hasattr(epoch_data, "columns"):
            df = _ensure_epoch_columns(epoch_data, DEFAULT_EPOCH_LEN)
        else:
            # best-effort conversion from list[dict]
            df = pd.DataFrame(epoch_data)
            df = _ensure_epoch_columns(df, DEFAULT_EPOCH_LEN)

        # 1) Summarize to the rate format that the simulator's rate path expects
        #    work_df columns: ['src_dc','model_type','total_tokens']
        work_df = (df.groupby(["source_dc_id", "model_type"], as_index=False)["num_tokens"]
                     .sum()
                     .rename(columns={"source_dc_id": "src_dc", "num_tokens": "total_tokens"}))
        # If the epoch has no tokens, short-circuit with empty results
        if work_df["total_tokens"].sum() <= 0:
            empty_stats = {
                "processed_tokens": 0.0, "avg_ttft_sec": 0.0, "energy_kwh": 0.0,
                "carbon_emissions": 0.0, "water_usage": 0.0
            }
            return empty_stats, [], []

        # 2) Discover DC set and relative capacities
        dcs = _discover_dcs_from_node_props(node_properties)
        if not dcs:
            dcs = _discover_dcs_from_epoch(df)
        if not dcs:
            dcs = sorted(work_df["src_dc"].unique().astype(int).tolist() or [0])
        cap_tps = _capacity_per_dc(node_properties, dcs)

        # 3) Build a routing FRACTION plan per (src_dc, model) → {tgt_dc: frac}
        #    Greedy: send each (src,model) bucket to the least-loaded DC by (pending/capacity).
        pending: Dict[int, float] = {dc: 0.0 for dc in dcs}
        routed_tokens_by_dc: Dict[int, float] = {dc: 0.0 for dc in dcs}
        schedule_plan: Dict[Tuple[int, str], Dict[int, float]] = {}

        for row in work_df.itertuples(index=False):
            src_dc  = int(getattr(row, "src_dc"))
            model   = str(getattr(row, "model_type"))
            tokens  = float(getattr(row, "total_tokens"))

            # choose DC with minimal load ratio
            best_dc, best_score = None, None
            for dc in dcs:
                score = pending[dc] / cap_tps[dc]
                if (best_score is None) or (score < best_score):
                    best_score, best_dc = score, dc

            tgt = int(best_dc)
            pending[tgt] += tokens
            routed_tokens_by_dc[tgt] += tokens

            key = (src_dc, model)
            d = schedule_plan.setdefault(key, {})
            d[tgt] = d.get(tgt, 0.0) + 1.0  # accumulate “weight” for this target

            # Smooth a bit so one huge bucket doesn’t dominate the next pick
            pending[tgt] = max(0.0, pending[tgt] - cap_tps[tgt])

        # Normalize the accumulated weights into fractions per (src,model)
        for key, dist in schedule_plan.items():
            s = sum(dist.values())
            if s > 0:
                for dc in list(dist.keys()):
                    dist[dc] = dist[dc] / s

        # 4) Simple power plan heuristic scaled by routed share
        total_tokens = sum(routed_tokens_by_dc.values()) or 1.0
        shares = {dc: routed_tokens_by_dc[dc] / total_tokens for dc in routed_tokens_by_dc}
        power_plan = _build_power_plan(shares, epoch_summary if isinstance(epoch_summary, dict) else {})

        # 5) Build or reuse simulator infra and run in RATE mode
        #    Since frameworks own the simulator run now, we construct DCs here with defaults.
        datacenters = None
        lat = []
        try:
            if rfs is not None:
                datacenters = rfs.build_datacenters_from_csv(
                    node_specs_csv=DEFAULT_NODE_SPECS,
                    gpu_perf_csvs={"A100": DEFAULT_A100, "H100": DEFAULT_H100},
                    dc_specs_csv=DEFAULT_DC_SPECS,
                    epoch_length=DEFAULT_EPOCH_LEN,
                    debug=False,
                )
                lat = _load_latencies_csv(DEFAULT_LATENCIES)
        except Exception:
            # If DC build fails for any reason, let the simulator fall back if it supports it
            datacenters = datacenters or []

        sim_out = LLM_Simulator(
            epoch_idx=epoch_idx,
            epoch_work_df=work_df,          # summarized work
            schedule_plan=schedule_plan,    # (src,model) -> {tgt: frac}
            power_plan=power_plan,          # node-type Idle/Off mapping per DC
            node_properties={},             # not needed in rate path
            epoch_length=DEFAULT_EPOCH_LEN,
            dc_latency_ms=lat,
            datacenters=datacenters,
            mode="rate",
            leftover_carry_in=None,
            epoch_hour=(epoch_idx % 24),
        )

        # 6) Normalize & attach a tiny “network_load” proxy
        stats, results, leftovers = _normalize_sim_output(sim_out)
        stats.setdefault("network_load", {"dc_token_totals": {int(k): float(v) for k, v in routed_tokens_by_dc.items()}})
        return stats, results, leftovers

