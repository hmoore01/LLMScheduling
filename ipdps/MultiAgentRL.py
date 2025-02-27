import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import os
import supersuit
import ray
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.catalog import ModelCatalog

from pettingzoo.utils.env import ParallelEnv
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1, black_death_v3
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.wrappers import BaseWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

a100_startup_time = 15
h100_startup_time = 10
a100_idle_resource = 60
h100_idle_resource = 50
a100_startup_resource = 400
h100_startup_resource = 350
number_of_a100_nodes = 4
number_of_h100_nodes = 4
total_number_of_nodes = number_of_a100_nodes + number_of_h100_nodes
epoch_length = 900
real_time_node_load_arr = np.zeros((total_number_of_nodes, epoch_length))
number_of_core_per_a100_node = 512
number_of_core_per_h100_node = 768
max_gpu_time_a100 = number_of_a100_nodes * number_of_core_per_a100_node * 1000
max_gpu_time_h100 = number_of_h100_nodes * number_of_core_per_h100_node * 1000
max_gpu_time = max_gpu_time_a100 + max_gpu_time_h100
this_interval_avail_resource_time = np.ones(epoch_length, dtype=float) * max_gpu_time
next_interval_avail_resource_time = np.ones(epoch_length, dtype=float) * max_gpu_time
resource_per_request_a100 = 8
resource_per_request_h100 = 6
vm_startup_time = 0
vm_shutdown_time = 30

idle_res_usage_a100 = 4
idle_res_usage_h100 = 3

carbon_density_list = [241.7, 221.5, 210.5, 202.1, 199.4, 199.8, 203.9, 223.1,
                           232.0, 244.1, 229.6, 228.9, 229.9, 227.0, 229.9, 249.0,
                           246.2, 259.6, 273.2, 280.9, 282.4, 279.3,274.4, 260.4] #gram/kWh

water_density = 0.53 #L/KWh
cop_factor_list = [0.85, 0.92, 1.02, 1.10, 1.14, 1.20, 1.21, 1.22,
                    1.21, 1.20, 1.15, 1.11, 1.03, 0.96, 0.91, 0.816,
                    0.83, 0.82, 0.82, 0.81, 0.81, 0.80, 0.80, 0.80]
cop = 1.5
price_list = [10.0, 10.0 ,10.0 ,10.0, 10.0, 10.0, 11.5, 11.5,
                11.5, 11.5, 11.5, 11.5, 11.5, 18.7, 18.7, 18.7,
                18.7, 18.7, 11.5, 11.5, 11.5, 11.5, 11.5, 10.0]


def simulation(
        vm_distribution_arr,
        new_vm_distribution_arr,
        func_distribution_arr,
        leftover_resource_time_arr,
        func_priority_list,
        lut_list
):


    # Number of function types
    total_number_of_func_type = func_distribution_arr.shape[0]

    # (A) Start with the resource usage array
    cumulative_resource_time_arr = np.zeros((total_number_of_nodes, (2 * epoch_length) + 1))
    # Include leftover usage from previous epoch
    cumulative_resource_time_arr[:, :epoch_length + 1] += leftover_resource_time_arr

    # Track carbon, water, TTF, and total served
    total_carbon_emissions = 0.0
    total_water_usage = 0.0
    sum_of_ttf_times_invocations = 0.0
    total_invocation_served = 0

    # (B) Shutdown unnecessary VMs
    for func_id in range(total_number_of_func_type):
        if func_id not in func_priority_list:
            # This func was pre-warmed but is no longer in the new list
            previous_prewarm_locs = list(np.nonzero(vm_distribution_arr[func_id, :])[0])
            for node in previous_prewarm_locs:
                if node >= number_of_a100_nodes:
                    # H100 node
                    cumulative_resource_time_arr[node, :vm_shutdown_time] += h100_idle_resource
                else:
                    # A100 node
                    cumulative_resource_time_arr[node, :vm_shutdown_time] += a100_idle_resource

    # (C) Calculate per-node startup time
    vm_startup_time_list = []
    number_of_container_per_node = np.count_nonzero(new_vm_distribution_arr[:, :], axis=0)

    for node_id in range(total_number_of_nodes):
        num_colocated = number_of_container_per_node[node_id]
        base_startup = (h100_startup_time if node_id >= number_of_a100_nodes
                        else a100_startup_time)
        vm_startup_time_list.append(base_startup * num_colocated)

    # Update the VM distribution
    vm_distribution_arr = copy.deepcopy(new_vm_distribution_arr)

    # (D) Main Simulation Loop
    for node_id in range(total_number_of_nodes):

        queued_func_id_list = []
        queued_func_req_list = []
        queued_func_delay_list = []

        for second_time in range(epoch_length):
            # 1) Process currently queued requests
            updated_q_ids = []
            updated_q_reqs = []
            updated_q_delays = []

            for i in range(len(queued_func_id_list)):
                func_id = queued_func_id_list[i]
                inv_count = queued_func_req_list[i]
                current_delay = queued_func_delay_list[i]

                if second_time >= vm_startup_time_list[node_id]:
                    # Build load_list for each node
                    load_list = [
                        max(cumulative_resource_time_arr[nn, second_time:second_time + 15])
                        for nn in range(total_number_of_nodes)
                    ]

                    (resource_time_arr,
                     time_to_first_token,
                     disused_cold_startup,
                     carbon_emissions,
                     water_usage) = calculate_resource_time_v5(
                        load_list, lut_list, func_id, inv_count,
                        node_id, current_delay,
                        node_type='H100' if node_id >= number_of_a100_nodes else 'A100'
                    )

                    # How many were actually served?
                    served_invocations = inv_count - disused_cold_startup

                    # TTF accumulation
                    sum_of_ttf_times_invocations += time_to_first_token * served_invocations
                    total_invocation_served += served_invocations

                    # Track carbon & water
                    total_carbon_emissions += carbon_emissions
                    total_water_usage += water_usage

                    end_idx = min(second_time + 15, cumulative_resource_time_arr.shape[1])
                    slice_len = end_idx - second_time

                    cumulative_resource_time_arr[node_id, second_time: end_idx] += resource_time_arr[:slice_len]

                    # If leftover remains, queue them
                    if disused_cold_startup > 0:
                        updated_q_ids.append(func_id)
                        updated_q_reqs.append(disused_cold_startup)
                        updated_q_delays.append(time_to_first_token)
                else:
                    # Node not ready yet => re-queue
                    updated_q_ids.append(func_id)
                    updated_q_reqs.append(inv_count)
                    updated_q_delays.append(current_delay)

            # Refresh the queue after processing
            queued_func_id_list = updated_q_ids
            queued_func_req_list = updated_q_reqs
            queued_func_delay_list = updated_q_delays

            # 2) Process new arrivals for pre-warmed functions
            for func_id in func_priority_list:
                inv_count = func_distribution_arr[func_id, second_time]
                if inv_count > 0 and second_time >= vm_startup_time_list[node_id]:
                    load_list = [
                        max(cumulative_resource_time_arr[nn, second_time:second_time + 15])
                        for nn in range(total_number_of_nodes)
                    ]

                    (resource_time_arr,
                     time_to_first_token,
                     disused_cold_startup,
                     carbon_emissions,
                     water_usage) = calculate_resource_time_v5(
                        load_list, lut_list, func_id, inv_count,
                        node_id, 0,
                        node_type='H100' if node_id >= number_of_a100_nodes else 'A100'
                    )

                    served_invocations = inv_count - disused_cold_startup

                    # Accumulate TTF, carbon, water, etc.
                    sum_of_ttf_times_invocations += time_to_first_token * served_invocations
                    total_invocation_served += served_invocations
                    total_carbon_emissions += carbon_emissions
                    total_water_usage += water_usage

                    # Update usage
                    cumulative_resource_time_arr[node_id, second_time:second_time + 15] += resource_time_arr

                    # If partial leftover, queue them
                    if disused_cold_startup > 0:
                        queued_func_id_list.append(func_id)
                        queued_func_req_list.append(disused_cold_startup)
                        queued_func_delay_list.append(time_to_first_token)


    # (E) Update leftover usage for next epoch
    leftover_resource_time_arr[:, :epoch_length + 1] = cumulative_resource_time_arr[:, epoch_length:]


    # (F) Final aggregator results
    # Compute average TTF
    if total_invocation_served > 0:
        average_time_to_first_token = sum_of_ttf_times_invocations / float(total_invocation_served)
    else:
        average_time_to_first_token = 0.0

    return (
        cumulative_resource_time_arr,  # 1
        average_time_to_first_token,  # 2
        total_carbon_emissions,  # 3
        total_water_usage,  # 4
        total_invocation_served,  # 5
        vm_distribution_arr  # 6
    )


def calculate_resource_time_v5(load_list, lut_list, func_id, number_of_invocation, node_id, delay_time, node_type=None):
    node_properties = {
        'A100': {
            'max_capacity': 512,
            'resource_per_token': 8,
            'startup_time': 15,
            'carbon_rate': 0.02,
            'energy_per_token': 0.0003,
        },
        'H100': {
            'max_capacity': 768,
            'resource_per_token': 6,
            'startup_time': 10,
            'carbon_rate': 0.015,
            'energy_per_token': 0.00025,
        }
    }

    if node_type not in node_properties:
        raise ValueError(f"Invalid node type: {node_type}")

    # Extract node properties
    max_capacity = node_properties[node_type]['max_capacity']
    resource_per_token = node_properties[node_type]['resource_per_token']
    carbon_rate = node_properties[node_type]['carbon_rate']
    energy_per_token = node_properties[node_type]['energy_per_token']

    # Water density in liters/kWh
    water_density = 0.53

    # Prepare output structures
    resource_time_arr = np.zeros(15)
    number_of_invocation = int(number_of_invocation)  # ensure integer

    # Compute total resource requirement and available capacity
    total_required_resources = number_of_invocation * resource_per_token
    available_resources = max_capacity - load_list[node_id]

    # Initialize return variables
    disused_cold_startup = 0
    carbon_emissions = 0.0
    water_usage = 0.0
    time_to_first_token = delay_time

    if available_resources >= total_required_resources:
        # All invocations can be served
        resource_time_arr[:number_of_invocation] = resource_per_token

        # Emissions & water usage for ALL allocated resources
        carbon_emissions = total_required_resources * carbon_rate
        # Convert resource -> kWh, then multiply by water density
        water_usage = (total_required_resources * energy_per_token) * water_density

    else:
        # Partial service (some are unprocessed)
        allocatable_invocations = int(available_resources // resource_per_token)
        allocatable_invocations = max(0, allocatable_invocations)

        disused_cold_startup = number_of_invocation - allocatable_invocations
        resource_time_arr[:allocatable_invocations] = resource_per_token

        # Calculate resource usage for the portion we can serve
        allocated_resource_units = allocatable_invocations * resource_per_token
        carbon_emissions = allocated_resource_units * carbon_rate
        water_usage = (allocated_resource_units * energy_per_token) * water_density

        time_to_first_token = delay_time + 1

    return resource_time_arr, time_to_first_token, disused_cold_startup, carbon_emissions, water_usage



class ResourceEnv(ParallelEnv):
    metadata = {
        "render_mode": ["human"],
        "name": "ResourceEnv",
        "is_parallelizable": True,
    }
    def __init__(self, config):
        if config is None:
            config = {}
        super().__init__()
        obs_shape = (2 + 2 * total_number_of_nodes,)
        self.agents = ["carbon_agent","time_agent","water_agent"]
        self.action_spaces = {agent: spaces.MultiDiscrete([2] * config["number_of_nodes"]) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32) for agent in self.agents}
        self.observation_space = self.get_observation_space(self.agents[0])
        self.action_space = self.get_action_space(self.agents[0])
        self.possible_agents = self.agents
        self.render_mode = None

        self.epoch_idx = config.get("epoch_idx",0)
        self.num_nodes = config.get("number_of_nodes", 8)
        self.max_steps = config.get("max_steps", 20)

        self.vm_distribution_arr = config.get("vm_distribution_arr", np.zeros((901, 8)))
        self.new_distribution_arr = copy.deepcopy(self.vm_distribution_arr)
        self.func_distribution_arr = config.get("func_distribution_arr", np.zeros((2, 901)))
        self.leftover_resource_time_arr = config.get("leftover_resource_time_arr", np.zeros((8, 901)))
        self.func_priority_list = config.get("func_priority_list", [0, 1])
        self.lut_list = config.get("lut_list", [])

        self.current_step = 0
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.dones["__all__"] = False
        self.infos = {agent: {} for agent in self.agents}
        self.total_carbon_emissions = 0
        self.total_water_usage = 0
        self.average_time_to_first_token = 0
        self.agent_selection = self.agents[0]

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.agents = ["carbon_agent", "time_agent", "water_agent"]
        self.dones = {agent: False for agent in self.agents}
        self.dones["__all__"] = False
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        obs_dict = {agent: self.observe(agent) for agent in self.agents}
        obs_array = np.array([obs_dict[agent] for agent in self.agents], dtype=np.float32)
        return obs_dict, self.infos

    def step(self, actions):
        self.current_step += 1

        print(f"Step {self.current_step}: Agents list: {self.agents}")

        for agent, action in actions.items():
            print(f"DEBUG: Agent {agent} - Action type {action.dtype}, Shape: {action.shape}")
            self.new_distribution_arr[self.epoch_idx, :] = action



        (
            cumulative_resource_time_arr,
            average_time_to_first_token,
            total_carbon_emissions,
            total_water_usage,
            total_invocation_served,
            updated_vm_distribution_arr,
        ) = simulation(
            vm_distribution_arr=self.vm_distribution_arr,
            new_vm_distribution_arr=self.new_distribution_arr,
            func_distribution_arr=self.func_distribution_arr,
            leftover_resource_time_arr=self.leftover_resource_time_arr,
            func_priority_list=self.func_priority_list,
            lut_list=self.lut_list,
        )

        self.vm_distribution_arr = updated_vm_distribution_arr
        self.leftover_resource_time_arr = cumulative_resource_time_arr[:, epoch_length:]
        self.total_carbon_emissions = total_carbon_emissions
        self.total_water_usage = total_water_usage
        self.average_time_to_first_token = average_time_to_first_token

        # self.total_carbon_emissions = np.random.uniform(0, 100)
        # self.average_time_to_first_token = np.random.uniform(0, 100)
        # self.total_water_usage = np.random.uniform(0, 100)

        self.rewards["carbon_agent"] = self.normalize_carbon(self.total_carbon_emissions)
        self.rewards["time_agent"] = self.normalize_time(self.average_time_to_first_token)
        self.rewards["water_agent"] = self.normalize_water(self.total_water_usage)

        done_flag = self.current_step >= self.max_steps
        dones = {agent: done_flag for agent in self.agents}
        dones["__all__"] = done_flag
        self.infos = {agent: {} for agent in self.agents}

        observations = {agent: self.observe(agent) for agent in self.agents}
        obs_array = np.array([observations[agent] for agent in self.agents], dtype=np.float32)

        return observations, self.rewards, self.dones, self.infos

    def observe(self, agent):
        past_timesteps = 10
        epoch_idx = self.epoch_idx

        start_idx = max(0, epoch_idx - past_timesteps)
        recent_vm_alloc = np.mean(self.new_distribution_arr[start_idx:epoch_idx + 1], axis=0)

        agent_specific_value = {
            "carbon_agent": self.normalize_carbon(self.total_carbon_emissions),
            "time_agent": self.normalize_time(self.average_time_to_first_token),
            "water_agent": self.normalize_water(self.total_water_usage)
        }.get(agent, 0)
        obs = np.concatenate([
            np.array([
                self.current_step,
                agent_specific_value
            ]),
            self.vm_distribution_arr[epoch_idx],
            recent_vm_alloc,
        ]).astype(np.float32)

        return obs

    def normalize_carbon(self, total_carbon_emissions):
        normalized_carbon = total_carbon_emissions/100000
        return normalized_carbon

    def normalize_time(self, average_time_to_first_token):
        normalized_time = average_time_to_first_token/100
        return normalized_time

    def normalize_water(self, total_water_usage):
        normalized_water = total_water_usage/1000
        return normalized_water
    def render(self, mode='human'):
        pass

    def get_observation_space(self, agent):
        return self.observation_spaces[agent]

    def get_action_space(self, agent):
        return self.action_spaces[agent]



def train_marl_agents(out_env, total_timesteps=100000, save_path="trained_models/marl_agents"):
    os.makedirs(save_path, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    env_name = "multi_agent_env"


    register_env(env_name, lambda env: out_env)

    agent_list = out_env.agents
    policies = {agent: (None, out_env.get_observation_space(agent), out_env.get_action_space(agent), {}) for agent in agent_list}

    # Policy mapping function (maps each agent to its corresponding policy)
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return agent_id  # Each agent gets its own policy

    config = PPOConfig().training(
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        train_batch_size=2048,
        num_epochs=10,
    ).environment(
        env=env_name,
        disable_env_checking=True
    ).framework("torch").env_runners(num_env_runners=1
    ).multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn
    ).api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    trainer = config.build()

    for i in range(total_timesteps // 10000):  # Save every 10k steps
        results = trainer.train()
        print(f"Iteration {i}: reward = {results['episode_reward_mean']}")

        # Save checkpoint every 10k steps
        if (i + 1) % 10 == 0:
            checkpoint_path = trainer.save(save_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # Final save
    final_checkpoint = trainer.save(save_path)
    print(f"Final checkpoint saved at {final_checkpoint}")

    # Shutdown
    ray.shutdown()

    return trainer

# def train_marl_agents(env, total_timesteps=100000, save_path="trained_models/marl_agents"):
#     os.makedirs(save_path, exist_ok=True)
#     vec_env = supersuit.pettingzoo_env_to_vec_env_v1(env)
#     concat_env = supersuit.concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=6, base_class="stable_baselines3")
#
#     agent_models = {}
#     for agent in env.agents:
#         model = PPO(
#             policy="MlpPolicy",
#             env=concat_env,
#             learning_rate=3e-4,
#             n_steps=2048,
#             batch_size=64,
#             n_epochs=10,
#             gamma=0.99,
#             gae_lambda=0.95,
#             clip_range=0.2,
#         )
#
#         obs, info = env.reset()
#
#         # for _ in range(1):
#         #     action, _ = model.predict(obs, deterministic=True)
#         #
#         #     # Debug: Check Action Dtype Before Passing to Step Function
#         #     print(f"DEBUG: {agent} - SB3 Raw Action Before Step: {action}, dtype: {action.dtype}")
#         #
#         #     # Ensure Action is `int8` before sending to step
#         #     action = action.astype(np.int8)
#         #     print(f"DEBUG: {agent} - Action After Cast: {action}, dtype: {action.dtype}")
#         #
#         #     obs, rewards, dones, infos = env.step(action)
#         #
#         #     obs_dict = {agent: obs[i] for i, agent in enumerate(env.agents)}
#         #
#         #     # 🚀 Debug: Check Observation dtype After Step
#         #     for agent_id, obs_val in obs_dict.items():
#         #         print(f"DEBUG: {agent_id} - Observation dtype: {obs_val.dtype}, Shape: {obs_val.shape}")
#
#         model.learn(total_timesteps=total_timesteps)
#
#         model_path = os.path.join(save_path, f"{agent}.zip")
#         model.save(model_path)
#
#         agent_models[agent] = model
#
#     return agent_models

def main():
    # Test Configuration
    fake_func_invocations = np.array([
        np.linspace(10, 100, 901),
        np.linspace(5, 50, 901)
    ])
    config = {
        "number_of_nodes": 8,
        "max_steps": 10,
        "epoch_idx": 0,
        "func_distribution_arr": fake_func_invocations
    }

    env = ResourceEnv(config)

    observations = env.reset()
    print("\n Initial Observations:", observations)

    # Run a few test steps
    for step in range(config["max_steps"]):
        print(f"\n Step {step + 1}")

        # Generate a random action (MultiBinary action per agent)
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}

        # Step the environment
        observations, rewards, dones, infos = env.step(actions)

        print("Actions:", actions)
        print("Observations:", observations)
        print("Rewards:", rewards)
        print("Dones:", dones)

        if all(dones.values()):
            print("\n Finished")
            break

if __name__ == "__main__":
    main()

#####################
#   TimeEnv         #
#####################
# class TimeEnv(gym.Env):
#     def __init__(self, config=None):
#         super().__init__()
#         if config is None:
#             config = {}
#
#         self.rng = seed
#         self.num_nodes = config.get("number_of_nodes", 8)
#
#         self.action_space = spaces.MultiBinary(self.num_nodes)
#
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
#         )
#
#         self.epoch_idx = config.get("epoch_idx", 0)
#         # Internal counters / arrays
#         self.current_step = 0
#         self.prev_action = 0
#         self.prev_reward = 0.0
#         self.max_steps = config.get("max_steps", 20)
#
#         self.vm_distribution_arr = config.get("vm_distribution_arr", np.zeros((8, 901)))
#         self.new_vm_distribution_arr = copy.deepcopy(self.vm_distribution_arr)
#         self.func_distribution_arr = config.get("func_distribution_arr", np.zeros((3, 901)))
#         self.leftover_resource_time_arr = config.get("leftover_resource_time_arr", np.zeros((8, 901)))
#         self.func_priority_list = config.get("func_priority_list", [0,1])
#         self.lut_list = config.get("lut_list", [])
#
#     def seed(self, seed=None):
#         self.rng = seed
#
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_step = 0
#
#         self.leftover_resource_time_arr.fill(0.0)
#         obs = self._get_obs()
#         info = {}
#         return obs, info
#
#     def step(self, action):
#         self.current_step += 1
#
#         self.new_vm_distribution_arr[self.epoch_idx, :] = action
#
#         (
#             cumulative_resource_time_arr,
#             average_time_to_first_token,
#             total_carbon_emissions,
#             total_water_usage,
#             total_invocation_served,
#             updated_vm_distribution_arr
#         ) = simulation(
#             vm_distribution_arr=self.vm_distribution_arr,
#             new_vm_distribution_arr=self.new_vm_distribution_arr,
#             func_distribution_arr=self.func_distribution_arr,
#             leftover_resource_time_arr=self.leftover_resource_time_arr,
#             func_priority_list=self.func_priority_list,
#             lut_list=self.lut_list
#         )
#
#
#         self.vm_distribution_arr = updated_vm_distribution_arr
#         # leftover_resource_time_arr for next step
#         self.leftover_resource_time_arr = cumulative_resource_time_arr[:, epoch_length:]
#
#
#
#         reward = -float(average_time_to_first_token)
#         terminated = (self.current_step >= self.max_steps)
#         truncated = False
#
#         self.prev_action = float(self.multi_binary_to_int(action))
#         self.prev_reward = float(reward)
#
#         obs = self._get_obs()
#
#         info = {"time_val": average_time_to_first_token}
#         return obs, reward, terminated, truncated, info
#
#     def _get_obs(self):
#
#         obs = np.array([
#             self.prev_action,
#             self.prev_reward,
#             self.current_step
#         ], dtype=np.float32)
#         return obs
#
#     def multi_binary_to_int(self, action):
#         bit_string = ''.join(str(int(a)) for a in action)
#         return int(bit_string, 2)
#
#
# ########################
# #  CarbonEnv           #
# ########################
# class CarbonEnv(gym.Env):
#     def __init__(self, config=None):
#         super().__init__()
#         if config is None:
#             config = {}
#
#         self.rng = seed
#         self.num_nodes = config.get("number_of_nodes", 8)
#
#         self.action_space = spaces.MultiBinary(self.num_nodes)
#
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
#         )
#
#         self.epoch_idx = config.get("epoch_idx", 0)
#
#         self.current_step = 0
#         self.prev_action = 0
#         self.prev_reward = 0.0
#         self.max_steps = config.get("max_steps", 20)
#
#         self.vm_distribution_arr = config.get("vm_distribution_arr", np.zeros((8, 901)))
#         self.new_vm_distribution_arr = copy.deepcopy(self.vm_distribution_arr)
#         self.func_distribution_arr = config.get("func_distribution_arr", np.zeros((3, 901)))
#         self.leftover_resource_time_arr = config.get("leftover_resource_time_arr", np.zeros((8, 901)))
#         self.func_priority_list = config.get("func_priority_list", [0,1])
#         self.lut_list = config.get("lut_list", [])
#
#     def seed(self, seed=None):
#         self.rng = seed
#
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_step = 0
#         self.leftover_resource_time_arr.fill(0.0)
#         obs = self._get_obs()
#         info = {}
#         return obs, info
#
#     def step(self, action):
#         self.current_step += 1
#         # print("epoch")
#         # print(self.epoch_idx)
#         # print("Action taken: ")
#         # print(action)
#         self.new_vm_distribution_arr[self.epoch_idx, :] = action
#
#         (
#             cumulative_resource_time_arr,
#             average_time_to_first_token,
#             total_carbon_emissions,
#             total_water_usage,
#             total_invocation_served,
#             updated_vm_distribution_arr
#         ) = simulation(
#             vm_distribution_arr=self.vm_distribution_arr,
#             new_vm_distribution_arr=self.new_vm_distribution_arr,
#             func_distribution_arr=self.func_distribution_arr,
#             leftover_resource_time_arr=self.leftover_resource_time_arr,
#             func_priority_list=self.func_priority_list,
#             lut_list=self.lut_list
#         )
#
#         self.vm_distribution_arr = updated_vm_distribution_arr
#         self.leftover_resource_time_arr = cumulative_resource_time_arr[:, epoch_length:]
#         # print("vm distribution: ")
#         # print(self.vm_distribution_arr)
#         # print("cumulative resource array: ")
#         # print(cumulative_resource_time_arr)
#         # print("leftover resource array")
#         # print(self.leftover_resource_time_arr)
#         # print("Carbon Emissions")
#         # print(total_carbon_emissions)
#         reward = -float(total_carbon_emissions)
#
#         terminated = (self.current_step >= self.max_steps)
#         truncated = False
#
#         self.prev_action = float(self.multi_binary_to_int(action))
#         self.prev_reward = float(reward)
#         obs = self._get_obs()
#         # print("observation space")
#         # print(obs)
#         info = {"carbon_val": total_carbon_emissions}
#         return obs, reward, terminated, truncated, info
#
#     def _get_obs(self):
#
#         obs = np.array([
#             self.prev_action,
#             self.prev_reward,
#             self.current_step
#         ], dtype=np.float32)
#         return obs
#
#     def multi_binary_to_int(self, action):
#         bit_string = ''.join(str(int(a)) for a in action)
#         return int(bit_string, 2)
#
#
#
# ########################
# #  WaterEnv            #
# ########################
# class WaterEnv(gym.Env):
#     def __init__(self, config=None):
#         super().__init__()
#         if config is None:
#             config = {}
#
#         self.rng = seed
#         self.num_nodes = config.get("number_of_nodes", 8)
#
#         self.action_space = spaces.MultiBinary(self.num_nodes)
#
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
#         )
#
#         self.epoch_idx = config.get("epoch_idx", 0)
#
#         self.current_step = 0
#         self.prev_action = 0
#         self.prev_reward = 0.0
#         self.max_steps = config.get("max_steps", 20)
#
#         self.vm_distribution_arr = config.get("vm_distribution_arr", np.zeros((901, 8)))
#         self.new_vm_distribution_arr = copy.deepcopy(self.vm_distribution_arr)
#         self.func_distribution_arr = config.get("func_distribution_arr", np.zeros((3, 901)))
#         self.leftover_resource_time_arr = config.get("leftover_resource_time_arr", np.zeros((8, 901)))
#         self.func_priority_list = config.get("func_priority_list", [0,1])
#         self.lut_list = config.get("lut_list", [])
#
#     def seed(self, seed=None):
#         self.rng = seed
#
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_step = 0
#         self.leftover_resource_time_arr.fill(0.0)
#         obs = self._get_obs()
#         info = {}
#         return obs, info
#
#     def step(self, action):
#         self.current_step += 1
#         self.new_vm_distribution_arr[self.epoch_idx, :] = action
#
#         (
#             cumulative_resource_time_arr,
#             average_time_to_first_token,
#             total_carbon_emissions,
#             total_water_usage,
#             total_invocation_served,
#             updated_vm_distribution_arr
#         ) = simulation(
#             vm_distribution_arr=self.vm_distribution_arr,
#             new_vm_distribution_arr=self.new_vm_distribution_arr,
#             func_distribution_arr=self.func_distribution_arr,
#             leftover_resource_time_arr=self.leftover_resource_time_arr,
#             func_priority_list=self.func_priority_list,
#             lut_list=self.lut_list
#         )
#
#         self.vm_distribution_arr = updated_vm_distribution_arr
#         self.leftover_resource_time_arr = cumulative_resource_time_arr[:, epoch_length:]
#
#         reward = float(total_water_usage)
#         terminated = (self.current_step >= self.max_steps)
#         truncated = False
#
#         self.prev_action = float(self.multi_binary_to_int(action))
#         self.prev_reward = float(reward)
#
#         obs = self._get_obs()
#         info = {"water_val": total_water_usage}
#         return obs, reward, terminated, truncated, info
#
#     def _get_obs(self):
#
#         obs = np.array([
#             self.prev_action,
#             self.prev_reward,
#             self.current_step
#         ], dtype=np.float32)
#         return obs
#
#     def multi_binary_to_int(self, action):
#         bit_string = ''.join(str(int(a)) for a in action)
#         return int(bit_string, 2)
#

# ##############################
# #  Train three PPO models    #
# ##############################
# def train_marl_agents(config=None, total_timesteps=20000, n_envs=3):
#     save_dir = "trained_models"
#     if config is None:
#         config = {}
#
#     env_classes = {
#         "carbon_model": CarbonEnv,
#         "time_model": TimeEnv,
#         "water_model": WaterEnv
#     }
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     trained_models = {}
#     for agent_name, env_class in env_classes.items():
#         print(f"Starting training for {agent_name}...")
#
#         # Create parallel environments
#         parallel_env = make_parallel_env(env_class, config, n_envs)
#         model_path = os.path.join(save_dir, f"{agent_name}.zip")
#         if os.path.exists(model_path):
#             # Load the previously trained model
#             print(f"Loading previously trained model from {model_path}")
#             model = PPO.load(model_path)
#             # Attach parallel environments to the loaded model
#             parallel_env = make_parallel_env(env_class, config, n_envs)
#             model.set_env(parallel_env)
#         else:
#             # Initialize a new model if none exists
#             print("No previously trained model found. Initializing a new model.")
#             parallel_env = make_parallel_env(env_class, config, n_envs)
#             model = PPO("MlpPolicy", parallel_env, verbose=1, device='cpu')
#
#         # Train with progress bar
#         progress_callback = TQDMProgressBarCallback(total_timesteps=total_timesteps)
#         step_tracker_callback = StepTrackingCallback(agent_max_steps=100)
#         model.learn(total_timesteps=total_timesteps, callback=[progress_callback, step_tracker_callback])
#
#         # Store trained model
#         trained_models[agent_name] = model
#
#         parallel_env.close()
#
#         model_path = os.path.join(save_dir, f"{agent_name}.zip")
#         model.save(model_path)
#
#         print(f"Finished training for {agent_name}.\n")
#
#     return trained_models
#
#
# def make_parallel_env(env_class, config, n_env=3):
#     def make_env(seed):
#         def _init():
#             env = env_class(config)
#             env.seed(seed)
#             return env
#         return _init
#     return SubprocVecEnv([make_env(i) for i in range(n_env)])
#
#
# class TQDMProgressBarCallback(BaseCallback):
#     def __init__(self, total_timesteps, verbose=0):
#         super().__init__(verbose)
#         self.total_timesteps = total_timesteps
#         self.pbar = None
#
#     def _on_training_start(self):
#         self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="steps")
#
#     def _on_step(self):
#         self.pbar.update(self.model.n_envs)  # Update progress bar with the number of environments
#         return True
#
#     def _on_training_end(self):
#         self.pbar.close()
#
# class StepTrackingCallback(BaseCallback):
#     def __init__(self, agent_max_steps, verbose=0):
#         super().__init__(verbose)
#         self.agent_max_steps = agent_max_steps
#         self.steps_taken = 0
#
#     def _on_step(self) -> bool:
#         self.steps_taken += self.training_env.num_envs
#         if self.steps_taken >= self.agent_max_steps:
#             print(f"Stopping training: Agent reached {self.steps_taken} steps (max: {self.agent_max_steps}).")
#             return False  # Stops training
#         return True
#
#
# if __name__ == "__main__":
#     config = {
#         "max_steps": 20
#     }
#     models = train_marl_agents(config, total_timesteps=5000)
#     print("Trained 3 separate PPO models using real simulation() logic.")

