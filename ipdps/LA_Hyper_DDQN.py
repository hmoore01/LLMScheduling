import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import Rate_Flow_Sim

# --- CONFIGURATION ---
BATCH_SIZE = 64
OPTIM_STEPS = 500
LR_ACTOR = 0.0005
LR_CRITIC = 0.001
# [FIX] Set GAMMA to 0.0 for contextual bandit/repeated epoch setup
GAMMA = 0.0
TAU = 0.005
MEMORY_SIZE = 20000
NUM_NODE_TYPES = 6
NUM_MODEL_CLASSES = 2
MIN_TRAFFIC_PERCENT = 0.05


# --- REPLAY BUFFER ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, pref, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, pref, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, pref, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state), torch.FloatTensor(pref), torch.FloatTensor(action),
                torch.FloatTensor(reward), torch.FloatTensor(next_state), torch.FloatTensor(done))

    def __len__(self):
        return len(self.buffer)


# --- DUAL-CONDITIONED PSL-MORL ARCHITECTURE ---
class AttentionPSLActor(nn.Module):
    def __init__(self, num_dcs, state_dim, pref_dim=4, hidden_dim=64, n_heads=4, n_layers=2):
        super(AttentionPSLActor, self).__init__()
        self.num_dcs = num_dcs
        self.state_dim = state_dim + pref_dim

        self.state_embed = nn.Linear(self.state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # [FIX] FiLM Modulation Layer: Forcefully alters features based on preference
        self.pref_modulation = nn.Sequential(
            nn.Linear(pref_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + pref_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )

        self.out_small = nn.Linear(64, 1)
        self.out_large = nn.Linear(64, 1)
        self.out_power = nn.Linear(64, 1)

    def forward(self, state, pref):
        B = state.size(0)
        state_seq = state.view(B, self.num_dcs, -1)

        pref_expanded = pref.unsqueeze(1).expand(-1, self.num_dcs, -1)
        state_pref_seq = torch.cat([state_seq, pref_expanded], dim=-1)

        emb = torch.relu(self.state_embed(state_pref_seq))
        enc_out = self.transformer(emb)

        # Apply Preference Modulation (Multiplication instead of just concatenation)
        pref_gate = self.pref_modulation(pref).unsqueeze(1).expand(-1, self.num_dcs, -1)
        modulated_enc = enc_out * (pref_gate * 2.0)  # *2.0 keeps expected mean at 1.0

        enc_pref = torch.cat([modulated_enc, pref_expanded], dim=-1)

        h = self.action_head(enc_pref)

        logit_s = self.out_small(h).squeeze(2)
        logit_l = self.out_large(h).squeeze(2)
        logit_p = self.out_power(h).squeeze(2)

        p_small = torch.softmax(logit_s, dim=1)
        p_large = torch.softmax(logit_l, dim=1)
        p_power = torch.sigmoid(logit_p + 2.0)

        return torch.cat([p_small, p_large, p_power], dim=1)


class AttentionPSLCritic(nn.Module):
    def __init__(self, num_dcs, state_dim, action_dim, pref_dim=4, hidden_dim=64, n_heads=4, n_layers=2):
        super(AttentionPSLCritic, self).__init__()
        self.num_dcs = num_dcs
        self.state_dim = state_dim + pref_dim

        self.state_embed = nn.Linear(self.state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # [FIX] Feature Modulation for Preference Conditioning
        self.sa_net = nn.Sequential(
            nn.Linear((num_dcs * hidden_dim) + action_dim, 256), nn.ReLU()
        )
        self.pref_net = nn.Sequential(
            nn.Linear(pref_dim, 256), nn.ReLU()
        )
        self.out_net = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, pref, action):
        B = state.size(0)
        state_seq = state.view(B, self.num_dcs, -1)

        pref_expanded = pref.unsqueeze(1).expand(-1, self.num_dcs, -1)
        state_pref_seq = torch.cat([state_seq, pref_expanded], dim=-1)

        emb = torch.relu(self.state_embed(state_pref_seq))
        enc_out = self.transformer(emb)
        flat_enc = enc_out.view(B, -1)

        # Process State+Action and Preference separately
        sa_feat = self.sa_net(torch.cat([flat_enc, action], dim=1))
        pref_feat = self.pref_net(pref)

        # Multiply them to forcefully gate the Q-value by the preference vector
        return self.out_net(sa_feat * pref_feat)


# --- AGENT WRAPPER ---
class PSLAgent:
    def __init__(self, num_dcs, state_feat_per_dc):
        self.num_dcs = num_dcs
        self.action_dim = (num_dcs * NUM_MODEL_CLASSES) + num_dcs

        self.actor = AttentionPSLActor(num_dcs, 4)
        self.actor_target = AttentionPSLActor(num_dcs, 4)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = AttentionPSLCritic(num_dcs, 4, self.action_dim)
        self.critic_target = AttentionPSLCritic(num_dcs, 4, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.buffer = ReplayBuffer(MEMORY_SIZE)
        self.noise_std = 0.3

    def select_action(self, state, pref, exploration=True):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        pref_t = torch.FloatTensor(pref).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_t, pref_t).cpu().numpy()[0]

        if exploration:
            # 1. Routing Exploration: Dirichlet Noise
            dirichlet_alpha = 0.3
            noise_weight = min(0.5, self.noise_std)

            for k in range(NUM_MODEL_CLASSES):
                start, end = k * self.num_dcs, (k + 1) * self.num_dcs
                noise = np.random.dirichlet([dirichlet_alpha] * self.num_dcs)
                action[start:end] = (1 - noise_weight) * action[start:end] + noise_weight * noise

            # 2. Power Plan Exploration: Gaussian Noise
            power_start = 2 * self.num_dcs
            action[power_start:] += np.random.normal(0, self.noise_std, size=self.num_dcs)
            action[power_start:] = np.clip(action[power_start:], 0.0, 1.0)

            # 3. Safe Structural Exploration (Datacenter Drops)
            if random.random() < 0.20:
                num_drop = random.randint(1, min(4, self.num_dcs - 2))
                drop_indices = random.sample(range(self.num_dcs), num_drop)

                for idx in drop_indices:
                    action[idx] = 0.0
                    action[self.num_dcs + idx] = 0.0
                    action[2 * self.num_dcs + idx] = 0.0

                # Renormalize the routing arrays so they sum to 1.0 again
                for k in range(NUM_MODEL_CLASSES):
                    start, end = k * self.num_dcs, (k + 1) * self.num_dcs
                    segment_sum = np.sum(action[start:end])
                    if segment_sum > 0:
                        action[start:end] /= segment_sum
                    else:
                        action[start] = 1.0

        return action

    def train(self, batch_size=BATCH_SIZE):
        if len(self.buffer) < batch_size: return None
        state, pref, action, reward, next_state, done = self.buffer.sample(batch_size)
        reward, done = reward.unsqueeze(1), done.unsqueeze(1)

        with torch.no_grad():
            next_action = self.actor_target(next_state, pref)
            target_value = reward + (1 - done) * GAMMA * self.critic_target(next_state, pref, next_action)

        critic_loss = nn.MSELoss()(self.critic(state, pref, action), target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, pref, self.actor(state, pref)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        self.noise_std = max(0.05, self.noise_std * 0.995)
        return critic_loss.item()


# --- PARETO TRACKER ---
class ParallelParetoTracker:
    def __init__(self):
        self.epoch_solutions = []
        self.epoch_count = 0
        self.last_sim_ref = None

    def clear_epoch(self):
        self.epoch_solutions = []

    def set_sim_ref(self, sim):
        self.last_sim_ref = sim

    def record_solution(self, metrics, weights, power_plan, mode_name="Scan"):
        active_nodes = sum(
            1 for p in power_plan.values() for st in p.get("unit", {}).values() if str(st).upper() in ["IDLE", "ON"])
        self.epoch_solutions.append({
            "mode": mode_name, "ttft": metrics.get('avg_ttft', 0.0),
            "carbon": metrics.get('carbon_emissions', 0.0) / 1000.0, "water": metrics.get('water_usage', 0.0) / 100.0,
            "cost": metrics.get('energy_cost', 0.0), "total_energy": metrics.get('total_energy', 0.0),
            "active_nodes": active_nodes, "weights": weights, "power_plan": power_plan
        })

    def increment_epoch(self):
        self.epoch_count += 1

    def get_report(self):
        if not self.epoch_solutions: return "No data."

        report = []
        report.append(f"\n=== EPOCH {self.epoch_count - 1} PARETO FRONT EVALUATION ===")
        report.append(
            f"{'Mode':<18} | {'TTFT(s)':<8} | {'Carb(kg)':<8} | {'Wat(L)':<8} | {'Cost($)':<8} | {'Energy(kWh)':<11} | {'ActTypes'}")
        report.append("-" * 95)

        sorted_sols = sorted(self.epoch_solutions, key=lambda x: x["carbon"])
        for s in sorted_sols:
            report.append(
                f"{s['mode']:<18} | {s['ttft']:.4f}   | {s['carbon']:.3f}     | {s['water']:.3f}    | {s['cost']:.3f}    | {s['total_energy']:.3f}       | {s['active_nodes']}")
        return "\n".join(report)


# --- GLOBALS ---
_PARETO_TRACKER = ParallelParetoTracker()
_LAST_DC_STATE = None
_GLOBAL_AGENT = None


def get_rich_state(sim, num_dcs, dc_usage):
    state = np.zeros((num_dcs, 4))
    for i in range(num_dcs):
        ci, cost, water_val, util = 400.0, 0.10, 1.18, 0.0
        if hasattr(sim, 'datacenters') and i in sim.datacenters:
            dc = sim.datacenters[i]
            ci = float(getattr(dc, 'carbon_intensity_g_per_kwh', 400.0))
            try:
                cost = float(getattr(dc, 'tou_price', [0.10])[0])
            except:
                pass
            water_val = float(getattr(dc, 'pue_value', 1.18))
        if dc_usage and i in dc_usage: util = dc_usage[i].get('utilization', 0.0)
        state[i] = [ci / 1000.0, cost * 5.0, water_val / 2.0, util]
    return state


def apply_min_percentage_and_reroute(w_small, w_large, latency_matrix, num_dcs):
    w_total = (w_small + w_large) / 2.0
    active_mask = w_total >= MIN_TRAFFIC_PERCENT
    if not np.any(active_mask): active_mask[np.argmax(w_total)] = True
    active_indices, pruned_indices = np.where(active_mask)[0], np.where(~active_mask)[0]
    for p_idx in pruned_indices:
        if w_total[p_idx] <= 0: continue
        valid_dists = np.full_like(latency_matrix[p_idx], float('inf'))
        valid_dists[active_indices] = latency_matrix[p_idx][active_indices]
        nearest_neighbor = np.argmin(valid_dists)
        w_small[nearest_neighbor] += w_small[p_idx];
        w_large[nearest_neighbor] += w_large[p_idx]
        w_small[p_idx] = w_large[p_idx] = 0.0
    return w_small, w_large


def build_schedule_map(sim_data, num_dcs, w_small, w_large):
    if len(sim_data) == 0: return {"map": {}}
    indices = np.arange(num_dcs)
    models = sim_data["model"].astype(str).str.lower() if "model" in sim_data.columns else sim_data[
        "model_type"].astype(str).str.lower()
    mask_small = models.str.contains("7b") | models.str.contains("8b") | models.str.contains("small")

    def safe_sample(w): return np.random.choice(indices, p=w / np.sum(w)) if np.sum(w) > 0 else 0

    return {
        "map": {sim_data.index[i]: safe_sample(w_small) if mask_small.at[sim_data.index[i]] else safe_sample(w_large)
                for i in range(len(sim_data))}}


def build_power_plan_sliding(num_dcs, slider_values, routing_weights):
    plan = {}
    for i in range(num_dcs):
        if routing_weights[i] <= 0.001:
            plan[i] = {"all": "OFF"}
        else:
            num_active = max(1, int(round(slider_values[i] * NUM_NODE_TYPES)))
            plan[i] = {"unit": {str(t): "IDLE" if t < num_active else "OFF" for t in range(NUM_NODE_TYPES)}}
    return plan


def milp_optimizer(epoch_data, epoch_idx, node_properties, epoch_summary):
    global _LAST_DC_STATE, _PARETO_TRACKER, _GLOBAL_AGENT

    spec_dir = epoch_summary.get('spec_dir', 'sim_specs')
    temp_sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, debug=False)
    _PARETO_TRACKER.set_sim_ref(temp_sim)

    real_num_dcs = len(temp_sim.datacenters) or 1

    if hasattr(temp_sim, 'network'):
        if hasattr(temp_sim.network, 'lat'):
            latency_mat = np.array(temp_sim.network.lat)
        elif hasattr(temp_sim.network, 'latency_matrix'):
            latency_mat = np.array(temp_sim.network.latency_matrix)
        else:
            latency_mat = np.ones((real_num_dcs, real_num_dcs))
    else:
        latency_mat = np.ones((real_num_dcs, real_num_dcs))

    if _LAST_DC_STATE is None or _LAST_DC_STATE.shape[0] != real_num_dcs:
        _LAST_DC_STATE = get_rich_state(temp_sim, real_num_dcs, {})
    current_state = _LAST_DC_STATE

    if _GLOBAL_AGENT is None:
        print("[INIT] Booting Modulated Hyper-Agent...")
        _GLOBAL_AGENT = PSLAgent(real_num_dcs, 4)
    agent = _GLOBAL_AGENT

    _PARETO_TRACKER.increment_epoch()
    _PARETO_TRACKER.clear_epoch()

    clean_data = epoch_data.copy().rename(
        columns={"source_dc_id": "source_dc", "model_type": "model", "num_tokens": "tokens"})
    if "model" not in clean_data.columns: clean_data["model"] = "Llama7b"
    if "tokens" not in clean_data.columns: clean_data["tokens"] = 1024
    if "source_dc" not in clean_data.columns: clean_data["source_dc"] = 0

    population = [
        {"mode": "time_agent", "pref": [1.0, 0.0, 0.0, 0.0], "constraints": {}},
        {"mode": "carbon_agent", "pref": [0.0, 1.0, 0.0, 0.0], "constraints": {}},
        {"mode": "water_agent", "pref": [0.0, 0.0, 1.0, 0.0], "constraints": {}},
        {"mode": "cost_agent", "pref": [0.0, 0.0, 0.0, 1.0], "constraints": {}},
        {"mode": "Balanced", "pref": [0.25, 0.25, 0.25, 0.25], "constraints": {}},

        {"mode": "green_perf", "pref": [0.6, 0.3, 0.0, 0.1],
         "constraints": {"carbon": {"budget": 4000.0 / 96.0, "penalty": 0.5}}},

        {"mode": "cost_guard", "pref": [0.7, 0.0, 0.0, 0.3],
         "constraints": {"cost": {"budget": 2800.0 / 96.0, "penalty": 0.5}}},

        {"mode": "water_saver", "pref": [0.7, 0.0, 0.3, 0.0],
         "constraints": {"water": {"budget": 2500.0 / 96.0, "penalty": 0.5}}},

        {"mode": "peak_power_guard", "pref": [0.8, 0.0, 0.0, 0.2],
         "constraints": {"total_energy": {"budget": 25051.0 / 96.0, "penalty": 0.3}}}
    ]

    # --- 1. EXPLORATION / TRAINING PHASE ---
    for step in range(OPTIM_STEPS):
        config = random.choice(population)
        pref_vec = np.array(config["pref"], dtype=np.float32)

        full_action = agent.select_action(current_state, pref_vec, exploration=True)

        w_small, w_large = np.maximum(0, full_action[0:real_num_dcs]), np.maximum(0, full_action[
            real_num_dcs:2 * real_num_dcs])
        if np.sum(w_small) == 0: w_small[0] = 1.0
        if np.sum(w_large) == 0: w_large[0] = 1.0
        w_small /= np.sum(w_small);
        w_large /= np.sum(w_large)

        w_small, w_large = apply_min_percentage_and_reroute(w_small, w_large, latency_mat, real_num_dcs)
        w_total = (w_small + w_large) / 2.0

        schedule_plan = build_schedule_map(clean_data, real_num_dcs, w_small, w_large)
        power_plan = build_power_plan_sliding(real_num_dcs, full_action[2 * real_num_dcs:], w_total)

        metrics, results, dc_usage = temp_sim.run_epoch(epoch_idx, clean_data, schedule_plan, power_plan)

        ttft = metrics.get('avg_ttft', 0.0)
        carbon = metrics.get('carbon_emissions', 0.0) / 1000.0
        water = metrics.get('water_usage', 0.0) / 100.0
        cost = metrics.get('energy_cost', 0.0)
        total_energy = metrics.get('total_energy', 0.0)

        is_empty_epoch = len(clean_data) == 0

        # --- 1. ALWAYS calculate Lagrangian penalties (close the loophole) ---
        lagrangian_penalty = 0.0
        constraints = config.get("constraints", {})

        if "carbon" in constraints and constraints["carbon"]["budget"] > 0:
            viol_pct = max(0.0, (carbon - constraints["carbon"]["budget"]) / constraints["carbon"]["budget"])
            lagrangian_penalty += min(50.0, constraints["carbon"]["penalty"] * viol_pct * 100.0)

        if "water" in constraints and constraints["water"]["budget"] > 0:
            viol_pct = max(0.0, (water - constraints["water"]["budget"]) / constraints["water"]["budget"])
            lagrangian_penalty += min(50.0, constraints["water"]["penalty"] * viol_pct * 100.0)

        if "cost" in constraints and constraints["cost"]["budget"] > 0:
            viol_pct = max(0.0, (cost - constraints["cost"]["budget"]) / constraints["cost"]["budget"])
            lagrangian_penalty += min(50.0, constraints["cost"]["penalty"] * viol_pct * 100.0)

        if "total_energy" in constraints and constraints["total_energy"]["budget"] > 0:
            viol_pct = max(0.0, (total_energy - constraints["total_energy"]["budget"]) / constraints["total_energy"][
                "budget"])
            lagrangian_penalty += min(50.0, constraints["total_energy"]["penalty"] * viol_pct * 100.0)

        # --- 2. Calculate Dense Rewards ---
        capped_ttft = min(ttft, 150.0)

        norm_ttft = capped_ttft / 50.0
        norm_carbon = carbon / 90.0
        norm_water = water / 40.0
        norm_cost = cost / 30.0

        w_perf, w_carb, w_wat, w_cost = pref_vec
        eco_focus = w_carb + w_wat + w_cost

        weighted_penalty = (w_perf * norm_ttft) + (w_carb * norm_carbon) + (w_wat * norm_water) + (w_cost * norm_cost)

        if is_empty_epoch:
            num_dcs_off = sum(1 for p in power_plan.values() if str(p.get("all")).upper() == "OFF")
            base_reward = (num_dcs_off * 20.0) - (weighted_penalty * 50.0)
        else:
            num_dcs_off = sum(1 for p in power_plan.values() if str(p.get("all")).upper() == "OFF")

            # Unchained Shutdown Bonus
            shutdown_bonus = num_dcs_off * 60.0 * eco_focus

            # Penalize underutilized servers (Zombies) heavily for eco agents
            raw_zombie_penalty = sum(10.0 for i, usage in dc_usage.items() if
                                     str(power_plan.get(i, {}).get("all")).upper() != "OFF" and usage.get("utilization",
                                                                                                          0.0) < 0.25)
            zombie_penalty = raw_zombie_penalty * eco_focus * 2.0

            # Penalize overloaded servers (Under-provisioning) heavily for time agents
            raw_overload_penalty = sum(20.0 for i, usage in dc_usage.items() if usage.get("utilization", 0.0) > 0.85)
            overload_penalty = raw_overload_penalty * w_perf * 3.0

            base_reward = shutdown_bonus - (weighted_penalty * 80.0) - zombie_penalty - overload_penalty

            # Massive fanatical penalty for the time agent if TTFT is bad
            if ttft > 5.0:
                base_reward -= (capped_ttft * 5.0) * w_perf

        reward = base_reward - lagrangian_penalty

        # [FIX] Scale reward so gradients don't permanently hit the clip threshold
        reward = reward * 0.01

        next_state = get_rich_state(temp_sim, real_num_dcs, dc_usage)
        agent.buffer.push(current_state, pref_vec, full_action, reward, next_state, False)

        # [FIX] Multi-update: Force the Critic to learn the TTFT cliff
        for _ in range(4):
            agent.train()

        current_state = next_state

    # --- 2. EXPLOITATION / REPORTING PHASE ---
    best_balanced_metrics, best_balanced_results = None, None

    for config in population:
        mode_name = config["mode"]
        pref_vec = np.array(config["pref"], dtype=np.float32)

        full_action = agent.select_action(current_state, pref_vec, exploration=False)

        w_small, w_large = np.maximum(0, full_action[0:real_num_dcs]), np.maximum(0, full_action[
            real_num_dcs:2 * real_num_dcs])
        if np.sum(w_small) == 0: w_small[0] = 1.0
        if np.sum(w_large) == 0: w_large[0] = 1.0
        w_small /= np.sum(w_small);
        w_large /= np.sum(w_large)

        w_small, w_large = apply_min_percentage_and_reroute(w_small, w_large, latency_mat, real_num_dcs)
        w_total = (w_small + w_large) / 2.0

        schedule_plan = build_schedule_map(clean_data, real_num_dcs, w_small, w_large)
        power_plan = build_power_plan_sliding(real_num_dcs, full_action[2 * real_num_dcs:], w_total)

        metrics, results, dc_usage = temp_sim.run_epoch(epoch_idx, clean_data, schedule_plan, power_plan)

        _PARETO_TRACKER.record_solution(metrics, w_total, power_plan, mode_name)

        if mode_name == "Balanced":
            best_balanced_metrics, best_balanced_results = metrics, results
            _LAST_DC_STATE = get_rich_state(temp_sim, real_num_dcs, dc_usage)

    print(_PARETO_TRACKER.get_report())

    return best_balanced_metrics, best_balanced_results, []