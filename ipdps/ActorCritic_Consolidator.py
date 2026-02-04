import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import Rate_Flow_Sim


# --- Actor-Critic Neural Network ---
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.affine = nn.Linear(state_dim, 128)

        # Actor head: Outputs probability distribution over actions
        self.action_head = nn.Linear(128, action_dim)

        # Critic head: Outputs the value of the current state
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values


class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.model = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = None
        self.state_value = None

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)

        # Sample action from the probability distribution
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        self.log_probs = m.log_prob(action)
        self.state_value = state_value
        return action.item()

    def update_policy(self, reward, next_state, done):
        # Calculate Advantage: R + gamma * V(s') - V(s)
        _, next_value = self.model(torch.from_numpy(next_state).float())
        returns = reward + (self.gamma * next_value.item() * (1 - done))
        advantage = returns - self.state_value.item()

        # Actor loss (Policy Gradient)
        action_loss = -self.log_probs * advantage

        # Critic loss (MSE of state value)
        value_loss = F.mse_loss(self.state_value, torch.tensor([[returns]]))

        loss = action_loss + value_loss
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

    state_dim = num_dcs
    action_dim = num_dcs

    if _AGENT is None:
        _AGENT = A2CAgent(state_dim, action_dim)

    # Column Mapping for Rate_Flow_Sim expectations
    sim_data = epoch_data.rename(columns={
        "source_dc_id": "source_dc",
        "model_type": "model",
        "num_tokens": "tokens"
    })

    # State: Current DC utilizations
    # Note: Using random initialization for the first step
    current_state = np.zeros(num_dcs)
    target_dc = _AGENT.select_action(current_state)

    # Consolidation Plans
    schedule_plan = {"default_target_dc": target_dc}
    power_plan = {dc: {"all": "ON" if dc == target_dc else "OFF"} for dc in range(num_dcs)}

    sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, debug=False)
    metrics, results, dc_usage = sim.run_epoch(epoch_idx, sim_data, schedule_plan, power_plan)

    # Reward Definition: - (Energy + Migration Penalty)
    # N_migrated is approximated by requests not served at source
    n_migrated = len(sim_data[sim_data['source_dc'] != target_dc])
    reward = -(metrics.get('energy_cost', 0) + (n_migrated * 0.1))

    # Update policy using the Critic's evaluation
    next_state = np.array([v['utilization'] for v in dc_usage.values()])
    _AGENT.update_policy(reward, next_state, False)

    return metrics, results, []