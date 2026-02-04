import pandas as pd
import numpy as np
import Rate_Flow_Sim

# Algorithm constants from paper [cite: 494, 511, 513]
GAMMA = 0.9
THETA_DEFAULT = 0.87 # More learning towards energy efficiency


class QLearningAgent:
    def __init__(self, num_dcs, theta=THETA_DEFAULT):
        self.num_dcs = num_dcs
        self.theta = theta
        # State space is number of active nodes [cite: 202, 361]
        self.q_table = np.zeros((num_dcs + 1, num_dcs))
        self.best_placement = None
        self.min_obj = float('inf')

    def get_action(self, state, epsilon):
        # Epsilon-greedy strategy [cite: 191, 192]
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_dcs)
        return np.argmax(self.q_table[state])

    def update(self, s, a, r, s_next, alpha):
        # Bellman equation update [cite: 193]
        max_future_q = np.max(self.q_table[s_next])
        self.q_table[s, a] = (1 - alpha) * self.q_table[s, a] + \
                             alpha * (r + GAMMA * max_future_q)


# Static agent to persist across simulator epochs
_AGENT = None


def milp_optimizer(epoch_data: pd.DataFrame, epoch_idx: int, **kwargs):
    global _AGENT
    summary = kwargs.get('epoch_summary', {})
    num_dcs = len(summary.get('datacenters', [0, 1, 2]))
    spec_dir = summary.get('spec_dir', 'sim_specs')

    if _AGENT is None:
        _AGENT = QLearningAgent(num_dcs)

    # 1. FIX: Column Mapping for Rate_Flow_Sim
    sim_data = epoch_data.rename(columns={
        "source_dc_id": "source_dc",
        "model_type": "model",
        "num_tokens": "tokens"
    })

    # 2. Comprehensive Decay Strategy [cite: 511, 512, 513]
    # Fast decay for exploration, then slow decay for exploitation
    if epoch_idx < 20:
        epsilon = max(0.01, 1.0 * (0.95 ** epoch_idx))
        alpha = max(0.01, 1.0 * (0.95 ** epoch_idx))
    else:
        epsilon = 0.05
        alpha = 0.01

    # 3. Decision & Simulation
    current_state = num_dcs  # Start assuming max possible active nodes
    target_dc = _AGENT.get_action(current_state, epsilon)

    schedule_plan = {"default_target_dc": target_dc}
    power_plan = {dc: {"all": "ON" if dc == target_dc else "OFF"} for dc in range(num_dcs)}

    sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, debug=False)
    metrics, results, dc_usage = sim.run_epoch(epoch_idx, sim_data, schedule_plan, power_plan)

    # 4. Reward Calculation [cite: 210, 211]
    # Objective = theta * N_active + (1 - theta) * N_migrated
    n_active = sum(1 for dc in dc_usage.values() if dc['utilization'] > 0)
    # Number of migrated elements (requests routed away from source) [cite: 391]
    n_migrated = len(sim_data[sim_data['source_dc'] != target_dc])

    reward = -(_AGENT.theta * n_active + (1 - _AGENT.theta) * n_migrated)

    # 5. Online Learning Update
    _AGENT.update(current_state, target_dc, reward, n_active, alpha)

    return metrics, results, []