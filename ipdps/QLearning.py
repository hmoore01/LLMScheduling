import pandas as pd
import numpy as np
import Rate_Flow_Sim_v2 as Rate_Flow_Sim

GAMMA = 0.9
THETA_DEFAULT = 0.87
FIXED_VARIANT = "_FP16 (Base)_B16"
DEFAULT_NODE_TYPES = [0, 1, 2, 3, 4, 5]

class QLearningAgent:
    def __init__(self, num_dcs, theta=THETA_DEFAULT):
        self.num_dcs = num_dcs
        self.theta = theta
        self.q_table = np.zeros((num_dcs + 1, num_dcs))

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_dcs)
        return np.argmax(self.q_table[state])

    def update(self, s, a, r, s_next, alpha):
        max_future_q = np.max(self.q_table[s_next])
        self.q_table[s, a] = (1 - alpha) * self.q_table[s, a] + alpha * (r + GAMMA * max_future_q)

_AGENT = None
_last_state = None

def _prepare_sim_data(epoch_data: pd.DataFrame) -> pd.DataFrame:
    df = epoch_data.copy() if isinstance(epoch_data, pd.DataFrame) else pd.DataFrame(epoch_data)
    df = df.rename(columns={"source_dc_id": "source_dc", "src_dc": "source_dc", "model_type": "model", "num_tokens": "tokens", "arrival_time_ms": "arrival_ms", "time_ms": "arrival_ms"})
    df["source_dc"] = pd.to_numeric(df["source_dc"], errors="coerce").fillna(0).astype(int)
    df["model"] = df["model"].astype(str)
    df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    df["arrival_ms"] = pd.to_numeric(df.get("arrival_ms", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    return df

def _dc_token_capacity(sim: Rate_Flow_Sim.LLM_Simulator, epoch_length: int) -> dict:
    epoch_ms = float(epoch_length) * 1000.0
    caps = {}
    for dc_id, dc in sim.datacenters.items():
        total = 0.0
        for unit in getattr(dc, "units", []):
            best_tpm = 0.0
            for rec in getattr(unit, "model_perf", {}).values():
                mpt = float(rec.get("ms_per_token", 0.0))
                if mpt > 0: best_tpm = max(best_tpm, 1.0 / mpt)
            total += best_tpm * epoch_ms
        caps[int(dc_id)] = max(total, 1.0)
    return caps

def _normalize_sim_output(sim_out):
    metrics, details, leftovers = sim_out
    avg_ttft = float(metrics.get("avg_ttft", metrics.get("avg_ttft_sec", 0.0)))
    energy_kwh = float(metrics.get("total_energy", metrics.get("energy_kwh", 0.0)))
    stats = {
        "avg_ttft_sec": avg_ttft,
        "energy_kwh": energy_kwh,
        "carbon_emissions": float(metrics.get("carbon_emissions", 0.0)),
        "water_usage": float(metrics.get("water_usage", 0.0)),
        "energy_cost": float(metrics.get("energy_cost", 0.0)),
    }
    return stats, details, leftovers

def milp_optimizer(epoch_data: pd.DataFrame, epoch_idx: int, **kwargs):
    global _AGENT, _last_state
    summary = kwargs.get('epoch_summary', {})
    spec_dir = summary.get('spec_dir', 'sim_specs')
    epoch_length = int(summary.get('epoch_length', 900))
    node_types = list(summary.get("node_types", DEFAULT_NODE_TYPES))

    sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_length, debug=False)
    sim_data = _prepare_sim_data(epoch_data)
    dc_ids = sorted(int(dc_id) for dc_id in sim.datacenters.keys()) or sorted(sim_data["source_dc"].unique().tolist() or [0])

    num_dcs = len(dc_ids)
    if _AGENT is None or _AGENT.num_dcs != num_dcs:
        _AGENT = QLearningAgent(num_dcs)

    if epoch_idx < 20:
        epsilon, alpha = max(0.01, 1.0 * (0.95 ** epoch_idx)), max(0.01, 1.0 * (0.95 ** epoch_idx))
    else:
        epsilon, alpha = 0.05, 0.01

    current_state = _last_state if _last_state is not None else 0
    action_idx = int(_AGENT.get_action(current_state, epsilon))
    target_dc = int(dc_ids[action_idx])

    dc_capacity = _dc_token_capacity(sim, epoch_length)
    target_cap = dc_capacity.get(target_dc, 1.0)

    plan_map = {}
    req_rows = []
    routed_tokens = {dc: 0.0 for dc in dc_ids}

    for idx, row in enumerate(sim_data.itertuples(index=False)):
        tokens = max(1, int(row.tokens))
        if routed_tokens[target_dc] + tokens <= target_cap:
            tgt = target_dc
        else:
            available_caps = {d: dc_capacity.get(d, 0.0) - routed_tokens[d] for d in dc_ids if d != target_dc}
            tgt = max(available_caps, key=available_caps.get) if available_caps and max(available_caps.values()) > 0 else target_dc

        routed_tokens[tgt] += tokens
        plan_map[idx] = tgt
        req_rows.append({
            "source_dc": int(row.source_dc),
            "model": f"{row.model}{FIXED_VARIANT}",
            "arrival_ms": float(row.arrival_ms),
            "tokens": tokens,
        })

    requests_df = pd.DataFrame(req_rows)
    schedule_plan = {"map": plan_map}

    active_dcs = {dc for dc, load in routed_tokens.items() if load > 0}
    power_plan = {}
    for dc in dc_ids:
        dc_power = {}
        for nt in node_types:
            dc_power[nt] = "ON" if dc in active_dcs else "OFF"
        power_plan[dc] = {"unit": dc_power}

    sim_out = sim.run_epoch(epoch_idx, requests_df, schedule_plan, power_plan)
    stats, results, leftovers = _normalize_sim_output(sim_out)

    n_active = len(active_dcs)
    n_migrated = len(sim_data[sim_data['source_dc'] != target_dc])
    reward = -(_AGENT.theta * n_active + (1 - _AGENT.theta) * n_migrated)

    # Discretized network stress state for Q-table based on active DCs
    next_state = n_active
    _last_state = next_state

    _AGENT.update(current_state, action_idx, reward, next_state, alpha)

    return stats, results, leftovers