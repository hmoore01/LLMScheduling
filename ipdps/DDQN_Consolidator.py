import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import collections
import Rate_Flow_Sim_v2 as Rate_Flow_Sim

REPLAY_BUFFER_MAXLEN = 10_000
FIXED_VARIANT = "_FP16 (Base)_B16"
DEFAULT_NODE_TYPES = [0, 1, 2, 3, 4, 5]


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = collections.deque(maxlen=REPLAY_BUFFER_MAXLEN)
        self.batch_size = 32

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state)
            return self.policy_net(state_t).argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones)

        current_q = self.policy_net(states_t).gather(1, actions_t)
        next_actions = self.policy_net(next_states_t).argmax(1).unsqueeze(1)
        next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)

        target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q.detach())
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
        _AGENT = DDQNAgent(state_dim, action_dim)

    epsilon = max(0.01, 1.0 - (epoch_idx / 100))
    current_state = _last_state if _last_state is not None and len(_last_state) == state_dim else np.zeros(state_dim,
                                                                                                           dtype=float)

    action_idx = int(_AGENT.select_action(current_state, epsilon))
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

    # FIX: Valid Power Plan schema addressing hardware node units
    active_dcs = {dc for dc, load in routed_tokens.items() if load > 0}
    power_plan = {}
    for dc in dc_ids:
        dc_power = {}
        for nt in node_types:
            dc_power[nt] = "ON" if dc in active_dcs else "OFF"
        power_plan[dc] = {"unit": dc_power}

    # FIX: Valid simulator tuple unpacking and normalization
    sim_out = sim.run_epoch(epoch_idx, requests_df, schedule_plan, power_plan)
    stats, results, leftovers = _normalize_sim_output(sim_out)

    reward = -stats.get('energy_cost', 0) - (stats.get('avg_ttft_sec', 0) * 100)

    # FIX: Real representation of environment state (Actual DC Utilization)
    next_state = np.array([min(1.0, routed_tokens.get(dc, 0.0) / dc_capacity.get(dc, 1.0)) for dc in dc_ids],
                          dtype=float)
    _last_state = next_state

    _AGENT.memory.append((current_state, action_idx, reward, next_state, False))
    _AGENT.train_step()

    if epoch_idx % 5 == 0:
        _AGENT.target_net.load_state_dict(_AGENT.policy_net.state_dict())

    return stats, results, leftovers