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
LR_ACTOR = 0.001
LR_DUAL = 0.01
GAMMA = 0.90
MEMORY_SIZE = 10000
NUM_OBJECTIVES = 4  # [TTFT, Carbon, Water, Energy_Cost]

# --- SCHEME DEFINITIONS ---
SCHEME_PRESETS = {
    "time_agent": {"w": [1.0, 0.0, 0.0, 0.0], "limits": [], "c_type": None},
    "carbon_agent": {"w": [0.0, 1.0, 0.0, 0.0], "limits": [], "c_type": None},
    "water_agent": {"w": [0.0, 0.0, 1.0, 0.0], "limits": [], "c_type": None},
    "cost_agent": {"w": [0.0, 0.0, 0.0, 1.0], "limits": [], "c_type": None},
    "green_perf": {"w": [0.5, 0.5, 0.0, 0.0], "limits": [4000.0], "c_type": "carbon"},
    "cost_guard": {"w": [0.5, 0.0, 0.0, 0.5], "limits": [2800.0], "c_type": "cost"},
    "water_saver": {"w": [0.5, 0.0, 0.5, 0.0], "limits": [2500.0], "c_type": "water"},
    "peak_power_guard": {"w": [0.8, 0.0, 0.0, 0.2], "limits": [25051.0], "c_type": "energy"}
}


def get_scheme_specs(scheme_name):
    # DYNAMIC means auto-pareto logic handles weights
    if scheme_name == "auto_pareto":
        return {"w": "DYNAMIC", "limits": [4000.0, 2800.0], "c_type": "carbon"}
    return SCHEME_PRESETS.get(scheme_name, SCHEME_PRESETS["time_agent"])


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
        return (torch.FloatTensor(state), torch.LongTensor(action), torch.FloatTensor(reward),
                torch.FloatTensor(next_state), torch.FloatTensor(done), torch.FloatTensor(constr))

    def __len__(self):
        return len(self.buffer)


# --- ARCHITECTURE ---
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


class HyperNetwork(nn.Module):
    def __init__(self, preference_dim, hidden_dim, output_shape):
        super(HyperNetwork, self).__init__()
        self.output_shape = output_shape
        self.flattened_dim = output_shape[0] * output_shape[1]
        self.net = nn.Sequential(
            nn.Linear(preference_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.flattened_dim)
        )

    def forward(self, preference):
        return self.net(preference).view(-1, self.output_shape[1], self.output_shape[0])


class LAHyperQNetwork(nn.Module):
    def __init__(self, num_dcs, state_feat, num_objectives, embed_dim=64):
        super(LAHyperQNetwork, self).__init__()
        self.encoder = CrossAttentionEncoder(num_dcs, state_feat, num_objectives, embed_dim)
        self.shared = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU())
        self.hypernet = HyperNetwork(num_objectives, 128, (128, num_dcs))

    def forward(self, state, preference):
        context = self.encoder(state, preference)
        features = self.shared(context)
        dynamic_weights = self.hypernet(preference)
        return torch.bmm(dynamic_weights, features.unsqueeze(2)).squeeze(2)


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

        # constraint_costs shape: [Batch, Num_Constraints]
        # limits shape: [Num_Constraints]
        # violation shape: [Batch, Num_Constraints]
        violation = constraint_costs - limits

        # Dual Ascent: We want to maximize lambda * violation
        # PyTorch minimizes loss, so we minimize -lambda * violation
        loss = -torch.mean(lambdas * violation.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return lambdas.detach()


# --- AGENT ---
class LAHyperAgent:
    def __init__(self, num_dcs, state_feat_per_dc, constraint_limits):
        self.num_dcs = num_dcs
        self.num_objectives = NUM_OBJECTIVES
        self.policy_net = LAHyperQNetwork(num_dcs, 4, self.num_objectives)
        self.target_net = LAHyperQNetwork(num_dcs, 4, self.num_objectives)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.lagrangian = LagrangianLayer(len(constraint_limits), constraint_limits)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=LR_ACTOR)
        self.optimizer_dual = optim.Adam(self.lagrangian.parameters(), lr=LR_DUAL)
        self.buffer = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = 0.5

    def select_action(self, state, preference_w, debug=False):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_dcs - 1)
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0)
            w_t = torch.FloatTensor(preference_w).unsqueeze(0)
            q_values = self.policy_net(s_t, w_t)
            if debug:
                print(f"    [DEBUG] Q-Values: {np.round(q_values.numpy(), 2)} | Pref: {preference_w}")
            return q_values.argmax(dim=1).item()

    def train(self, batch_size=BATCH_SIZE):
        if len(self.buffer) < batch_size: return None
        total_loss = 0
        for _ in range(5):
            state, action, rewards, next_state, done, constraints = self.buffer.sample(batch_size)
            prefs = torch.from_numpy(np.random.dirichlet(np.ones(self.num_objectives), batch_size)).float()

            weighted_rewards = torch.sum(rewards * prefs, dim=1, keepdim=True)
            current_lambdas = self.lagrangian()

            # Constraint Penalty Calculation
            if self.lagrangian.num_constraints > 0:
                # constraints: [Batch, 2], current_lambdas: [2] -> [Batch]
                penalty = torch.matmul(constraints, current_lambdas)
                final_reward = weighted_rewards.squeeze() - penalty
            else:
                final_reward = weighted_rewards.squeeze()

            q_current = self.policy_net(state, prefs).gather(1, action.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = self.policy_net(next_state, prefs).argmax(1).unsqueeze(1)
                q_next = self.target_net(next_state, prefs).gather(1, next_actions).squeeze(1)
                q_target = final_reward + (1 - done) * GAMMA * q_next

            loss = nn.MSELoss()(q_current, q_target)
            self.optimizer_policy.zero_grad()
            loss.backward()
            self.optimizer_policy.step()
            total_loss += loss.item()

        self.lagrangian.update_lambdas(constraints, self.optimizer_dual)
        self.epsilon = max(0.01, self.epsilon * 0.95)
        return total_loss / 5


# --- ROBUST STATE TRACKING ---
_LA_AGENT = None
_LAST_DC_STATE = None


def get_rich_state(sim, num_dcs, dc_usage):
    state = np.zeros((num_dcs, 4))
    for i in range(num_dcs):
        # Defaults
        ci = 400.0
        cost = 0.10
        water_val = 1.18
        util = 0.0

        # 1. Fetch Static Props from Simulator
        if hasattr(sim, 'datacenters') and i in sim.datacenters:
            dc = sim.datacenters[i]
            if hasattr(dc, 'carbon_intensity_g_per_kwh'):
                ci = float(dc.carbon_intensity_g_per_kwh)

            if hasattr(dc, '_tou_price'):
                try:
                    cost = float(dc._tou_price(0.0))
                except:
                    pass
            elif hasattr(dc, 'tou_price') and dc.tou_price:
                cost = float(dc.tou_price[0])

            if hasattr(dc, 'pue_value'):
                water_val = float(dc.pue_value)

        # 2. Fetch Dynamic Props
        if dc_usage and i in dc_usage:
            util = dc_usage[i].get('utilization', 0.0)

        # 3. Normalize
        state[i, 0] = ci / 1000.0  # Carbon
        state[i, 1] = cost * 5.0  # Cost
        state[i, 2] = water_val / 2.0  # Water
        state[i, 3] = util  # Utilization
    return state


# --- PRE-TRAINING LOOP ---
def pretrain_agent(agent, num_dcs, spec_dir):
    print(f"[LA-Hyper] Starting Pre-training (200 steps)...")
    sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, debug=False)
    dummy_data = pd.DataFrame({
        "source_dc": np.random.randint(0, num_dcs, 20),
        "model": ["Llama7b"] * 20,
        "tokens": np.random.randint(100, 1000, 20),
        "arrival_ms": [0] * 20
    })
    current_state = get_rich_state(sim, num_dcs, {})

    for i in range(200):
        # Random sweep during pretraining
        w_train = np.random.dirichlet(np.ones(4))
        target_dc = agent.select_action(current_state, w_train)
        schedule_plan = {"default_target_dc": target_dc}
        power_plan = {dc: {"all": "ON" if dc == target_dc else "IDLE"} for dc in range(num_dcs)}

        metrics, _, dc_usage = sim.run_epoch(0, dummy_data, schedule_plan, power_plan)
        next_state = get_rich_state(sim, num_dcs, dc_usage)

        r_ttft = -metrics.get('avg_ttft', 1.0)
        r_carbon = -metrics.get('carbon_emissions', 0.0) / 1000.0
        r_water = -metrics.get('water_usage', 0.0) / 100.0
        r_cost = -metrics.get('energy_cost', 0.0)
        reward_vec = np.array([r_ttft, r_carbon, r_water, r_cost], dtype=np.float32)

        # FIX: Ensure constraint vector has 2 elements [Carbon, Cost]
        # to match the 2 limits initialized in Agent [4000.0, 2800.0]
        constraint_vec = [metrics.get('carbon_emissions', 0.0), metrics.get('energy_cost', 0.0)]
        constraint_vec_np = np.array(constraint_vec, dtype=np.float32)

        agent.buffer.push(current_state, target_dc, reward_vec, next_state, False, constraint_vec_np)
        agent.train(BATCH_SIZE)
        current_state = next_state
    print(f"[LA-Hyper] Pre-training Complete.")


# --- SCAN HELPERS ---
def scan_hypotheticals(agent, state, num_dcs, sim):
    choices = {}
    dc_props = {}
    for i in range(num_dcs):
        if i in sim.datacenters:
            dc = sim.datacenters[i]
            ci = getattr(dc, 'carbon_intensity_g_per_kwh', 0.0)
            try:
                cost = float(dc._tou_price(0.0))
            except:
                cost = 0.0
            dc_props[i] = {"ci": ci, "cost": cost}

    for name, spec in SCHEME_PRESETS.items():
        w = np.array(spec["w"], dtype=np.float32)
        original_eps = agent.epsilon
        agent.epsilon = 0.0
        action = agent.select_action(state, w)
        agent.epsilon = original_eps

        choices[f"Hypothetical_{name}_DC"] = action
        choices[f"Hypothetical_{name}_EstCarbon"] = dc_props.get(action, {}).get("ci", 0)
        choices[f"Hypothetical_{name}_EstCost"] = dc_props.get(action, {}).get("cost", 0)

    return choices


# --- MAIN OPTIMIZER ---
def milp_optimizer(epoch_data, epoch_idx, node_properties, epoch_summary):
    global _LA_AGENT, _LAST_DC_STATE

    num_dcs = len(epoch_summary.get('datacenters', [0, 1, 2]))
    spec_dir = epoch_summary.get('spec_dir', 'sim_specs')

    # 1. Init
    temp_sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, debug=False)
    if _LA_AGENT is None:
        # Init with Auto-Pareto constraints [Carbon, Cost]
        _LA_AGENT = LAHyperAgent(num_dcs, 4, [4000.0, 2800.0])
        _LAST_DC_STATE = get_rich_state(temp_sim, num_dcs, {})
        pretrain_agent(_LA_AGENT, num_dcs, spec_dir)

    current_state = _LAST_DC_STATE

    # 2. HYPOTHETICAL SCAN
    pareto_data = scan_hypotheticals(_LA_AGENT, current_state, num_dcs, temp_sim)

    # 3. AUTO-PARETO SELECTION
    avg_ci = np.mean(current_state[:, 0])
    avg_cost = np.mean(current_state[:, 1])

    if avg_ci > 0.6:
        weights_np = np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32)  # Green
        mode = "Green"
    elif avg_cost > 0.8:
        weights_np = np.array([0.1, 0.0, 0.0, 0.9], dtype=np.float32)  # Cost
        mode = "Cost"
    elif avg_ci < 0.2 and avg_cost < 0.3:
        weights_np = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)  # Perf
        mode = "Perf"
    else:
        weights_np = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)  # Balanced
        mode = "Balanced"

    # 4. Select Actual Action
    target_dc = _LA_AGENT.select_action(current_state, weights_np)

    # 5. Run Sim
    schedule_plan = {"default_target_dc": target_dc}
    power_plan = {dc: {"all": "ON" if dc == target_dc else "IDLE"} for dc in range(num_dcs)}

    sim_data = epoch_data.copy()
    if "source_dc_id" in sim_data.columns and "source_dc" not in sim_data.columns: sim_data["source_dc"] = sim_data[
        "source_dc_id"]
    if "model_type" in sim_data.columns and "model" not in sim_data.columns: sim_data["model"] = sim_data["model_type"]
    if "num_tokens" in sim_data.columns and "tokens" not in sim_data.columns: sim_data["tokens"] = sim_data[
        "num_tokens"]

    metrics, results, dc_usage = temp_sim.run_epoch(epoch_idx, sim_data, schedule_plan, power_plan)

    # 6. Update State
    next_state = get_rich_state(temp_sim, num_dcs, dc_usage)
    _LAST_DC_STATE = next_state

    # 7. Train
    r_ttft = -metrics.get('avg_ttft', 1.0)
    r_carbon = -metrics.get('carbon_emissions', 0.0) / 1000.0
    r_water = -metrics.get('water_usage', 0.0) / 100.0
    r_cost = -metrics.get('energy_cost', 0.0)
    reward_vec = np.array([r_ttft, r_carbon, r_water, r_cost], dtype=np.float32)

    # Push 2 constraints: Carbon and Cost
    constraint_vec = [metrics.get('carbon_emissions', 0.0), metrics.get('energy_cost', 0.0)]
    constraint_vec_np = np.array(constraint_vec, dtype=np.float32)

    _LA_AGENT.buffer.push(current_state, target_dc, reward_vec, next_state, False, constraint_vec_np)
    loss = _LA_AGENT.train()

    if epoch_idx % 10 == 0:
        _LA_AGENT.target_net.load_state_dict(_LA_AGENT.policy_net.state_dict())
        print(f"[LA-Hyper] Ep {epoch_idx} | Mode: {mode} | Act: {target_dc} | Loss: {loss}")

    # 8. MERGE DATA
    metrics.update(pareto_data)
    metrics["AutoPareto_Mode"] = mode

    return metrics, results, []