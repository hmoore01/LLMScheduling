import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import Rate_Flow_Sim


# --- DDQN Neural Network Architecture ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Paper suggests deep architecture for complex energy-performance mapping
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

        # Two networks to prevent overestimation
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = []  # Simple replay buffer
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

        # DDQN Core Logic: select action with policy_net, evaluate with target_net
        current_q = self.policy_net(states_t).gather(1, actions_t)
        next_actions = self.policy_net(next_states_t).argmax(1).unsqueeze(1)
        next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)

        target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --- Simulator Integration ---
_AGENT = None


def milp_optimizer(epoch_data: pd.DataFrame, epoch_idx: int, **kwargs):
    global _AGENT
    summary = kwargs.get('epoch_summary', {})
    num_dcs = len(summary.get('datacenters', [0, 1, 2]))
    spec_dir = summary.get('spec_dir', 'sim_specs')

    # State: Current DC utilization levels
    state_dim = num_dcs
    action_dim = num_dcs  # Action: Which DC to consolidate to

    if _AGENT is None:
        _AGENT = DDQNAgent(state_dim, action_dim)

    # Column Mapping for Simulator
    sim_data = epoch_data.rename(columns={
        "source_dc_id": "source_dc",
        "model_type": "model",
        "num_tokens": "tokens"
    })

    # Epsilon decay
    epsilon = max(0.01, 1.0 - (epoch_idx / 100))

    # Simple state representation: 0 if empty, 1 if busy (placeholder for real util)
    current_state = np.random.rand(num_dcs)
    target_dc = _AGENT.select_action(current_state, epsilon)

    # Building Plans
    schedule_plan = {"default_target_dc": target_dc}
    power_plan = {dc: {"all": "ON" if dc == target_dc else "OFF"} for dc in range(num_dcs)}

    sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, debug=False)
    metrics, results, dc_usage = sim.run_epoch(epoch_idx, sim_data, schedule_plan, power_plan)

    # Reward: Balance Energy Cost and Performance (TTFT)
    energy_reward = -metrics.get('energy_cost', 0)
    perf_penalty = -metrics.get('avg_ttft', 0) * 100  # Scaling factor
    reward = energy_reward + perf_penalty

    # Store experience and train
    next_state = np.array([v['utilization'] for v in dc_usage.values()])
    _AGENT.memory.append((current_state, target_dc, reward, next_state, False))
    _AGENT.train_step()

    # Periodically update target network
    if epoch_idx % 5 == 0:
        _AGENT.target_net.load_state_dict(_AGENT.policy_net.state_dict())

    return metrics, results, []