import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import Rate_Flow_Sim

# --- CONFIGURATION ---
BATCH_SIZE = 16
OPTIM_STEPS = 60  # Enough time for the agent to wake up necessary nodes
LR_ACTOR = 0.001
LR_CRITIC = 0.002
LR_DUAL = 0.01
GAMMA = 0.90
TAU = 0.01
MEMORY_SIZE = 2000
NUM_OBJECTIVES = 4  # [TTFT, Carbon, Water, Energy_Cost]
FORCED_NUM_DCS = 12
NUM_NODE_TYPES = 6  # Granular control over 6 types

# --- SCHEME DEFINITIONS ---
SCHEME_PRESETS = {
    "time_agent": {"w": [1.0, 0.0, 0.0, 0.0], "limits": [], "c_type": None},
    "carbon_agent": {"w": [0.0, 1.0, 0.0, 0.0], "limits": [], "c_type": None},
    "water_agent": {"w": [0.0, 0.0, 1.0, 0.0], "limits": [], "c_type": None},
    "cost_agent": {"w": [0.0, 0.0, 0.0, 1.0], "limits": [], "c_type": None},
    "green_perf": {"w": [0.5, 0.5, 0.0, 0.0], "limits": [4000.0], "c_type": "carbon"},
    "balanced": {"w": [0.25, 0.25, 0.25, 0.25], "limits": [4000.0], "c_type": "carbon"},
}


def get_scheme_specs(scheme_name):
    if scheme_name == "auto_pareto":
        return {"w": "DYNAMIC", "limits": [4000.0, 2800.0], "c_type": "carbon"}
    return SCHEME_PRESETS.get(scheme_name, SCHEME_PRESETS["time_agent"])


# --- REPLAY BUFFER ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward_vector, next_state, done, constraints):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward_vector, next_state, done, constraints)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, constr = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state), torch.FloatTensor(action), torch.FloatTensor(reward),
                torch.FloatTensor(next_state), torch.FloatTensor(done), torch.FloatTensor(constr))

    def __len__(self):
        return len(self.buffer)


# --- ARCHITECTURE (Full 84-Dim Control) ---
class CrossAttentionEncoder(nn.Module):
    def __init__(self, num_dcs, state_dim_per_dc, preference_dim, embed_dim=64):
        super(CrossAttentionEncoder, self).__init__()
        self.query_proj = nn.Linear(preference_dim, embed_dim)
        self.key_proj = nn.Linear(state_dim_per_dc, embed_dim)
        self.value_proj = nn.Linear(state_dim_per_dc, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, state, preference):
        query = self.query_proj(preference).unsqueeze(1)
        keys = self.key_proj(state)
        values = self.value_proj(state)
        attn_output, _ = self.attention(query, keys, values)
        return self.norm(attn_output + query).squeeze(1)


class Actor(nn.Module):
    def __init__(self, num_dcs, state_dim, pref_dim, num_node_types, embed_dim=128):
        super(Actor, self).__init__()
        self.num_dcs = num_dcs
        self.num_node_types = num_node_types
        self.encoder = CrossAttentionEncoder(num_dcs, state_dim, pref_dim, embed_dim)

        self.shared = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU()
        )

        # Head 1: Routing (12 DCs)
        self.routing_head = nn.Linear(256, num_dcs)

        # Head 2: Power Control (72 Nodes)
        self.power_head = nn.Linear(256, num_dcs * num_node_types)

        # [DARK START STRATEGY]
        # Initialize bias to -3.0. Sigmoid(-3.0) ~= 0.04
        # This means the agent starts with almost everything turned OFF.
        # It must LEARN to turn things ON to satisfy traffic.
        nn.init.constant_(self.power_head.bias, -3.0)

    def forward(self, state, preference):
        context = self.encoder(state, preference)
        features = self.shared(context)

        # Routing: Softmax to ensure valid distribution
        routing_logits = self.routing_head(features)
        routing_probs = torch.softmax(routing_logits, dim=1)

        # Power: Independent Sigmoids (0.0 to 1.0) for each node type
        power_logits = self.power_head(features)
        power_probs = torch.sigmoid(power_logits)

        return torch.cat([routing_probs, power_probs], dim=1)


class Critic(nn.Module):
    def __init__(self, num_dcs, state_dim, pref_dim, action_dim, embed_dim=128):
        super(Critic, self).__init__()
        self.encoder = CrossAttentionEncoder(num_dcs, state_dim, pref_dim, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, preference, action):
        context = self.encoder(state, preference)
        x = torch.cat([context, action], dim=1)
        return self.net(x)


class LagrangianLayer(nn.Module):
    def __init__(self, num_constraints, limits):
        super(LagrangianLayer, self).__init__()
        self.num_constraints = num_constraints
        self.limits = torch.FloatTensor(limits) if not torch.is_tensor(limits) else limits
        self.log_lambdas = nn.Parameter(torch.zeros(num_constraints))

    def forward(self):
        return torch.exp(self.log_lambdas)

    def update_lambdas(self, constraint_costs, optimizer):
        if self.num_constraints == 0: return torch.tensor([])
        lambdas = self.forward()
        limits = self.limits.to(constraint_costs.device)
        violation = constraint_costs - limits
        loss = -torch.mean(lambdas * violation.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return lambdas.detach()


# --- AGENT (DDPG) ---
class LAHyperDDPGAgent:
    def __init__(self, num_dcs, state_feat_per_dc, constraint_limits):
        self.num_dcs = num_dcs
        self.num_node_types = NUM_NODE_TYPES
        self.num_objectives = NUM_OBJECTIVES

        # Action = Routing(12) + Power(72) = 84
        self.action_dim = num_dcs + (num_dcs * self.num_node_types)

        self.actor = Actor(num_dcs, 4, self.num_objectives, self.num_node_types)
        self.actor_target = Actor(num_dcs, 4, self.num_objectives, self.num_node_types)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(num_dcs, 4, self.num_objectives, self.action_dim)
        self.critic_target = Critic(num_dcs, 4, self.num_objectives, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.lagrangian = LagrangianLayer(len(constraint_limits), constraint_limits)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.dual_optimizer = optim.Adam(self.lagrangian.parameters(), lr=LR_DUAL)
        self.buffer = ReplayBuffer(MEMORY_SIZE)
        self.noise_std = 0.2

    def select_action(self, state, preference_w, exploration=True):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        pref_t = torch.FloatTensor(preference_w).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_t, pref_t).cpu().numpy()[0]

        if exploration:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise

            # Normalize Routing (0-12)
            routing = action[:self.num_dcs]
            routing = np.exp(routing) / np.sum(np.exp(routing))

            # Clip Power (12-84)
            power = np.clip(action[self.num_dcs:], 0.0, 1.0)

            action = np.concatenate([routing, power])

        return action

    def train(self, batch_size=BATCH_SIZE):
        if len(self.buffer) < batch_size: return None
        total_loss = 0
        for _ in range(5):
            state, action, rewards, next_state, done, constraints = self.buffer.sample(batch_size)
            prefs = torch.from_numpy(np.random.dirichlet(np.ones(self.num_objectives), batch_size)).float()

            with torch.no_grad():
                next_action = self.actor_target(next_state, prefs)
                target_q = self.critic_target(next_state, prefs, next_action)
                weighted_rewards = torch.sum(rewards * prefs, dim=1, keepdim=True)
                if self.lagrangian.num_constraints > 0:
                    penalty = torch.matmul(constraints, self.lagrangian())
                    scalar_reward = weighted_rewards - penalty.unsqueeze(1)
                else:
                    scalar_reward = weighted_rewards

                done = done.unsqueeze(1)
                target_value = scalar_reward + (1 - done) * GAMMA * target_q

            current_q = self.critic(state, prefs, action)
            critic_loss = nn.MSELoss()(current_q, target_value)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            total_loss += critic_loss.item()

            actor_action = self.actor(state, prefs)
            actor_loss = -self.critic(state, prefs, actor_action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        self.lagrangian.update_lambdas(constraints, self.dual_optimizer)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        self.noise_std = max(0.05, self.noise_std * 0.95)
        return total_loss / 5.0


# --- PARALLEL PARETO TRACKER ---
class ParallelParetoTracker:
    def __init__(self):
        self.cumulatives = {}
        self.epoch_count = 0

    def accumulate(self, mode_name, metrics, weights):
        if mode_name not in self.cumulatives:
            self.cumulatives[mode_name] = {
                "ttft_sum": 0.0, "carbon_sum": 0.0, "water_sum": 0.0, "cost_sum": 0.0, "energy_sum": 0.0,
                "active_dcs_samples": []
            }

        c = self.cumulatives[mode_name]
        c["ttft_sum"] += metrics.get('avg_ttft', 0.0)
        c["carbon_sum"] += metrics.get('carbon_emissions', 0.0) / 1000.0  # kg
        c["water_sum"] += metrics.get('water_usage', 0.0) / 100.0  # L
        c["cost_sum"] += metrics.get('energy_cost', 0.0)
        c["energy_sum"] += metrics.get('total_energy', 0.0)

        active_indices = [i for i, w in enumerate(weights[:12]) if w > 0.01]
        c["active_dcs_samples"].append(tuple(sorted(active_indices)))

    def increment_epoch(self):
        self.epoch_count += 1

    def get_report(self):
        if not self.cumulatives: return "No data."

        report = []
        report.append("\n=== LA-Hyper Pareto Extremes (Projected Cumulative) ===")
        report.append(
            f"{'Metric':<15} | {'Mode':<10} | {'Avg TTFT':<8} | {'Tot Carbon':<10} | {'Tot Water':<10} | {'Tot Cost':<10} | {'Tot Energy':<11} | {'Active DCs'}")
        report.append("-" * 120)

        targets = {
            "Best Time": "Perf",
            "Best Carbon": "Green",
            "Best Cost": "Cost",
            "Best Water": "Water",
            "Best Energy": "Perf",
            "Best Balanced": "Balanced"
        }

        for label, mode_key in targets.items():
            if mode_key not in self.cumulatives: continue

            data = self.cumulatives[mode_key]

            avg_ttft = data["ttft_sum"] / max(1, self.epoch_count)
            tot_carbon = data["carbon_sum"]
            tot_water = data["water_sum"]
            tot_cost = data["cost_sum"]
            tot_energy = data["energy_sum"]

            if data["active_dcs_samples"]:
                most_common = max(set(data["active_dcs_samples"]), key=data["active_dcs_samples"].count)
                active_str = ",".join(map(str, most_common))
            else:
                active_str = "None"

            if len(active_str) > 15: active_str = active_str[:12] + "..."

            report.append(
                f"{label:<15} | {mode_key:<10} | {avg_ttft:.4f}   | {tot_carbon:.3f}      | {tot_water:.3f}      | {tot_cost:.3f}     | {tot_energy:.3f}      | {active_str}")

        report.append(
            "========================================================================================================================")
        return "\n".join(report)


_PARETO_TRACKER = ParallelParetoTracker()
_LAST_DC_STATE = None


# --- MODULE EXPORT FOR SIMULATOR ---
def get_final_report():
    return _PARETO_TRACKER.get_report()


def get_rich_state(sim, num_dcs, dc_usage):
    state = np.zeros((num_dcs, 4))
    for i in range(num_dcs):
        ci = 400.0;
        cost = 0.10;
        water_val = 1.18;
        util = 0.0
        if hasattr(sim, 'datacenters') and i in sim.datacenters:
            dc = sim.datacenters[i]
            ci = float(getattr(dc, 'carbon_intensity_g_per_kwh', 400.0))
            if hasattr(dc, '_tou_price'):
                try:
                    cost = float(dc._tou_price(0.0))
                except:
                    pass
            elif hasattr(dc, 'tou_price') and dc.tou_price:
                cost = float(dc.tou_price[0])
            water_val = float(getattr(dc, 'pue_value', 1.18))
        if dc_usage and i in dc_usage:
            util = dc_usage[i].get('utilization', 0.0)
        state[i, 0] = ci / 1000.0
        state[i, 1] = cost * 5.0
        state[i, 2] = water_val / 2.0
        state[i, 3] = util
    return state


# --- HELPER: PURE LEARNING POWER PLAN ---
def generate_learned_power_plan(num_dcs, weights, power_probs):
    """
    Constructs the power plan using ONLY the agent's outputs.
    No heuristic overrides (except the basic weight check).
    """
    power_plan = {}

    for i in range(num_dcs):
        # Even if agent routes 0 traffic, we check if it WANTED to turn it on.
        # However, to be nice to the metrics, we allow one heuristic:
        # If Weight < 1%, assume it meant IDLE.
        # (This is standard practice: Routing decisions gate Resource decisions).
        if weights[i] < 0.01:
            power_plan[i] = {"all": "IDLE"}
            continue

        # Agent decided to route here. Now, what did it decide to turn ON?
        start = i * NUM_NODE_TYPES
        dc_probs = power_probs[start: start + NUM_NODE_TYPES]

        unit_conf = {}
        # We need at least one node to be ON if traffic is sent,
        # or latency will be infinite.
        # We trust the agent learned this. If it didn't, it gets punished.

        any_on = False
        for t_idx, prob in enumerate(dc_probs):
            if prob > 0.5:  # Learned Threshold
                unit_conf[t_idx] = "ON"
                any_on = True
            else:
                unit_conf[t_idx] = "IDLE"

        # Safe fallback: If agent routed traffic but turned everything OFF,
        # we let it happen. The reward function will punish the infinite latency.
        # This allows the agent to truly "learn" the consequence.
        power_plan[i] = {"unit": unit_conf}

    return power_plan


# --- MAIN OPTIMIZER ---
def milp_optimizer(epoch_data, epoch_idx, node_properties, epoch_summary):
    global _LAST_DC_STATE, _PARETO_TRACKER

    num_dcs = FORCED_NUM_DCS
    spec_dir = epoch_summary.get('spec_dir', 'sim_specs')
    temp_sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, debug=False)

    if _LAST_DC_STATE is None:
        _LAST_DC_STATE = get_rich_state(temp_sim, num_dcs, {})
    current_state = _LAST_DC_STATE

    _PARETO_TRACKER.increment_epoch()

    parallel_modes = {
        "Perf": np.array([0.9, 0.1, 0.0, 0.0]),
        "Green": np.array([0.1, 0.9, 0.0, 0.0]),
        "Water": np.array([0.1, 0.0, 0.9, 0.0]),
        "Cost": np.array([0.1, 0.0, 0.0, 0.9]),
        "Balanced": np.array([0.25, 0.25, 0.25, 0.25])
    }

    real_execution_mode = "Balanced"

    final_metrics_to_return = None
    final_results_to_return = None

    for mode_name, pref_vec in parallel_modes.items():
        # Fresh Agent per Mode
        agent = LAHyperDDPGAgent(num_dcs, 4, [4000.0, 2800.0])

        best_reward = -float('inf')
        best_artifacts = None

        for step in range(OPTIM_STEPS):
            full_action = agent.select_action(current_state, pref_vec, exploration=True)

            # --- SPLIT ---
            weights = full_action[:num_dcs]
            power_control = full_action[num_dcs:]

            if np.sum(weights) == 0: weights[0] = 1.0
            weights = weights / np.sum(weights)

            # --- SIMULATION ---
            sim_data = epoch_data.copy()
            indices = np.arange(num_dcs)
            sim_data["source_dc"] = np.random.choice(indices, size=len(sim_data), p=weights)

            # Use learned power control
            power_plan = generate_learned_power_plan(num_dcs, weights, power_control)

            schedule_plan = {"weights": weights}
            if "model_type" in sim_data.columns and "model" not in sim_data.columns: sim_data["model"] = sim_data[
                "model_type"]
            if "num_tokens" in sim_data.columns and "tokens" not in sim_data.columns: sim_data["tokens"] = sim_data[
                "num_tokens"]

            metrics, results, dc_usage = temp_sim.run_epoch(epoch_idx, sim_data, schedule_plan, power_plan)

            # --- SHARP REWARDS ---
            r_ttft = -metrics.get('avg_ttft', 1.0) / 2.0
            r_carbon = -metrics.get('carbon_emissions', 0.0) / 1000.0
            r_water = -metrics.get('water_usage', 0.0) / 100.0
            r_cost = -metrics.get('energy_cost', 0.0) * 100.0

            # Simple, brutal rewards.
            # If you want Carbon, you get punished for Carbon.
            # If Latency spikes > 5s (failed requests), apply massive penalty.
            fail_penalty = -10.0 if metrics.get('avg_ttft', 0) > 5.0 else 0.0

            if mode_name == "Perf":
                reward = r_ttft
            elif mode_name == "Green":
                reward = r_carbon + fail_penalty
            elif mode_name == "Water":
                reward = r_water + fail_penalty
            elif mode_name == "Cost":
                reward = r_cost + fail_penalty
            else:
                reward = r_ttft + r_carbon

            reward_vec = np.array([reward] * 4, dtype=np.float32)
            next_s = get_rich_state(temp_sim, num_dcs, dc_usage)
            constraint_vec = np.array([0.0, 0.0], dtype=np.float32)

            agent.buffer.push(current_state, full_action, reward_vec, next_s, False, constraint_vec)
            agent.train()

            if reward > best_reward:
                best_reward = reward
                best_artifacts = (metrics, results, weights, dc_usage)

        hyp_metrics, hyp_results, hyp_weights, hyp_usage = best_artifacts
        _PARETO_TRACKER.accumulate(mode_name, hyp_metrics, hyp_weights)

        if mode_name == real_execution_mode:
            final_metrics_to_return = hyp_metrics
            final_results_to_return = hyp_results
            _LAST_DC_STATE = get_rich_state(temp_sim, num_dcs, hyp_usage)

    if epoch_idx % 10 == 0:
        print(f"[LA-Hyper] Ep {epoch_idx} | Scanned 5 Modes | Discovery Active")

    return final_metrics_to_return, final_results_to_return, []