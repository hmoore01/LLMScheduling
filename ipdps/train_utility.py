import os
# --- 1. PYTORCH DEADLOCK FIX ---
# Must be set before importing PyTorch or SB3
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- 2. FAULT HANDLER (Just in case) ---
import faulthandler
import signal
faulthandler.enable()
if hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1)
# ---------------------------------------
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import your MARLIN grid files
import grid_topology
import utility_gnn


class MarlinUtilityEnv(gym.Env):
    """
    A Gymnasium wrapper for the MARLIN InteractiveGridNetwork.
    """

    def __init__(self, max_epochs=96):
        super(MarlinUtilityEnv, self).__init__()
        # Initialize with a dummy datacenter ID (0)
        self.grid = grid_topology.InteractiveGridNetwork(dc_ids=[0])
        self.max_epochs = max_epochs
        self.current_epoch = 0

        # Action Space: [0] Gas Dispatch (0-1), [1] Peaker Dispatch (0-1)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation Space: Flat 16-dimensional observation from get_graph_feature_vector()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_epoch = 0
        self.grid.reset()
        self.grid.set_epoch(self.current_epoch)

        obs = self.grid.get_graph_feature_vector()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        # 1. Randomize DC load to train the agent to react to spikes
        simulated_dc_load = np.random.uniform(100.0, 1500.0)

        # 2. Map the PPO action array to the dispatch dictionary your file expects
        dispatch_fracs = {
            "Gas": float(action[0]),
            "Peaker": float(action[1])
        }

        # 3. Step the grid physics
        economics = self.grid.step(p_dc_kw=simulated_dc_load, dispatch_fracs=dispatch_fracs)

        # 4. Calculate the reward using your file's built-in formula
        reward = self.grid.compute_utility_reward(economics)

        # 5. Advance time
        self.current_epoch += 1
        done = self.current_epoch >= self.max_epochs
        truncated = False

        # Prepare for next observation
        if not done:
            self.grid.set_epoch(self.current_epoch)

        obs = self.grid.get_graph_feature_vector()

        return np.array(obs, dtype=np.float32), float(reward), done, truncated, {}


def main():
    print("=== MARLIN Utility Agent Training ===")
    os.makedirs("models", exist_ok=True)

    env = MarlinUtilityEnv()
    check_env(env, warn=True)

    policy_kwargs = {}
    if hasattr(utility_gnn, 'GridGATExtractor'):
        policy_kwargs = {
            "features_extractor_class": utility_gnn.GridGATExtractor,
            "features_extractor_kwargs": {"features_dim": 64},
        }

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/utility_tensorboard/"
    )

    total_timesteps = 100_000
    print(f"Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    save_path = "models/utility_gnn_ppo.zip"
    model.save(save_path)
    print(f"\n[SUCCESS] Utility Agent trained and saved to: {save_path}")


if __name__ == "__main__":
    main()