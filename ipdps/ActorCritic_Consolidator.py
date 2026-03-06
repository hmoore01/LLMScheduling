import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import Rate_Flow_Sim

FIXED_VARIANT = "_FP16 (Base)_B16"
DEFAULT_NODE_TYPES = [0, 1, 2, 3, 4, 5]


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.affine = nn.Linear(state_dim, 128)
        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values


class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = None
        self.state_value = None

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs = m.log_prob(action)
        self.state_value = state_value
        return action.item()

    def update_policy(self, reward, next_state, done):
        _, next_value = self.model(torch.from_numpy(next_state).float())
        returns = reward + (self.gamma * next_value.item() * (1 - done))
        advantage = returns - self.state_value.item()
        action_loss = -self.log_probs * advantage
        value_loss = F.mse_loss(self.state_value, torch.tensor([[returns]]))

        loss = action_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


_AGENT = None
_last_state = None


def _prepare_sim_data(epoch_data: pd.DataFrame) -> pd.DataFrame:
    df = epoch_data.copy() if isinstance(epoch_data, pd.DataFrame) else pd.DataFrame(epoch_data)
    df = df.rename(
        columns={"source_dc_id": "source_dc", "src_dc": "source_dc", "model_type": "model", "num_tokens": "tokens",
                 "arrival_time_ms": "arrival_ms", "time_ms": "arrival_ms"})
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
    dc_ids = sorted(int(dc_id) for dc_id in sim.datacenters.keys()) or sorted(
        sim_data["source_dc"].unique().tolist() or [0])

    num_dcs = len(dc_ids)
    state_dim = num_dcs
    action_dim = num_dcs

    if _AGENT is None or _AGENT.state_dim != state_dim:
        _AGENT = A2CAgent(state_dim, action_dim)

    current_state = _last_state if _last_state is not None and len(_last_state) == state_dim else np.zeros(state_dim,
                                                                                                           dtype=float)

    action_idx = int(_AGENT.select_action(current_state))
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
            tgt = max(available_caps, key=available_caps.get) if available_caps and max(
                available_caps.values()) > 0 else target_dc

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

    n_migrated = len(sim_data[sim_data['source_dc'] != target_dc])
    reward = -(stats.get('energy_cost', 0) + (n_migrated * 0.1))

    next_state = np.array([min(1.0, routed_tokens.get(dc, 0.0) / dc_capacity.get(dc, 1.0)) for dc in dc_ids],
                          dtype=float)
    _last_state = next_state

    _AGENT.update_policy(reward, next_state, False)

    return stats, results, leftovers