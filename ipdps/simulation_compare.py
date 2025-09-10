import pandas as pd
import numpy as np
import os
import random

EPOCH_LENGTH = 900
NUM_EPOCHS = 3
REQUESTS_PER_EPOCH = 1000
TOTAL_REQUESTS = NUM_EPOCHS * REQUESTS_PER_EPOCH
NUM_DATACENTERS = 12
WORKLOAD_FILE = "simulator_ready_trace.csv"

def my_scheduler(epoch_df: pd.DataFrame, epoch_idx: int) -> tuple:

    # Schedule Plan (Round Robin)
    schedule_plan = []
    for i, row in epoch_df.iterrows():
        target_dc_id = i % NUM_DATACENTERS
        source_dc_id = int(row["source_dc_id"])

        request = {
            "target_dc_id": target_dc_id,
            "model_type": row["model_type"],
            "num_tokens": row["num_tokens"],
            "batch_size": row["batch_size"],
            "source_dc_id": source_dc_id,
            "time_index": row["time_index"]
        }
        schedule_plan.append(request)

    # Power Plan (Simple: Node types 0 and 1 Idle)
    power_plan = {
        dc_id: {
            0: "Idle",
            1: "Idle",
            2: "Idle",
            3: "Idle",
            4: "Idle",
            5: "Idle"
        } for dc_id in range(NUM_DATACENTERS)
    }

    return schedule_plan, power_plan


model_types = {
    "Llama7b": (128, 1024),
    "Llama70b": (512, 2048)
}

def generate_random_workload():
    rows = []
    for i in range(TOTAL_REQUESTS):
        epoch = i // REQUESTS_PER_EPOCH
        timestamp = epoch * EPOCH_LENGTH + random.randint(0, EPOCH_LENGTH - 1)
        model_type = random.choice(list(model_types.keys()))
        token_range = model_types[model_type]
        num_tokens = random.randint(*token_range)
        batch_size = random.choice([1, 2, 4])
        source_dc_index = random.randint(0, NUM_DATACENTERS - 1)

        rows.append({
            "epoch": epoch,
            "model_type": model_type,
            "num_tokens": num_tokens,
            "time_index": timestamp % EPOCH_LENGTH,
            "source_dc_id": source_dc_index,
            "batch_size": batch_size
        })

    df = pd.DataFrame(rows)
    df.to_csv(WORKLOAD_FILE, index=False)
    print(f"Generated {len(df)} requests → {WORKLOAD_FILE}")


def LLM_Simulator(
    epoch_idx: int,
    workload_df: pd.DataFrame,
    schedule_plan: list,
    power_plan: dict
) -> tuple:

    # Initialize the Geo_Network with default files
    geo_network = Geo_Network.load_network()

    # Prepare the simulation environment
    geo_network.reset_all()
    geo_network.apply_power_plan(power_plan)

    # Route and simulate all requests
    results = geo_network.apply_schedule_plan(schedule_plan)

    # Aggregate results
    hour = ((epoch_idx * 15) // 60) % 23
    stats = geo_network.report_global_stats(current_hour=hour)

    ttft_values = []
    for dc in geo_network.datacenters:
        for node in dc.nodes:
            for processor in node.processors:
                proc_stats = processor.report_epoch_stats()
                if proc_stats["avg_ttft_second"] > 0:
                    ttft_values.append(proc_stats["avg_ttft_second"])

    avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0

    return {
        "avg_ttft": avg_ttft,
        "energy_cost": stats["total_cost_usd"],
        "carbon_emissions": stats["carbon_emitted_kg"],
        "water_usage": stats["water_used_liters"],
        "total_energy": stats["total_energy_kWh"],
    }, results



class Processor:
    PROCESSOR_STATE_ACTIVE = 'Active'
    PROCESSOR_STATE_IDLE = 'Idle'
    PROCESSOR_STATE_OFF = 'Off'
    def __init__(self, processor_type, processor_id, performance_file="H100_GPU.csv", model_file="Llama7b.csv", epoch_length=900):
        self.state = Processor.PROCESSOR_STATE_OFF
        self.processor_id = processor_id
        self.processor_type = processor_type
        base_path = '/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/'

        self.performance_file = performance_file if os.path.isabs(performance_file) else os.path.join(base_path,performance_file)
        self.model_file = model_file if os.path.isabs(model_file) else os.path.join(base_path, model_file)
        self.performance_metrics = {}
        self.model_metrics = {}
        self.loaded_models = set()

        # Default values
        self.default_metrics = {
            "num_GPUs": 1,
            "Mem_Size": "80GB",
            "Llama8b_Process": 2000,  # Default to 2s processing time per batch
            "Llama70b_Process": 10000,  # Default to 10s processing time per batch
            "base_token_size": 200,  # Default base token size per batch
            "batch_size": 1,  # Default batch size
        }

        self.epoch_length = epoch_length
        # Columns: [executing, power_used, model_load, ttft_flag, idle_flag, off_flag, failed_attempt]
        self.resource_usage_log = np.zeros((self.epoch_length, 7), dtype=float)
        self.failed_attempts = 0

        # Load processor and model configurations
        self.load_processor_config()
        self.load_model_config()
        self.total_ttft_time = 0.0
        self.ttft_events = 0

    def reset_epoch_stats(self):
        self.resource_usage_log.fill(0.0)
        self.loaded_models.clear()
        self.failed_attempts = 0
        self.total_ttft_time = 0.0
        self.ttft_events = 0

    def load_processor_config(self):
        if os.path.exists(self.performance_file):
            df = pd.read_csv(self.performance_file)
            if "num_GPUs" in df.columns:
                matched_rows = df[df["num_GPUs"] == int(self.processor_type)]
                if not matched_rows.empty:
                    self.performance_metrics = matched_rows.iloc[0].to_dict()
                    if isinstance(self.performance_metrics["Mem_Size"], str):
                        self.performance_metrics["Mem_Size"] = int(self.performance_metrics["Mem_Size"].replace("GB", ""))
                    if "TDP" in self.performance_metrics:
                        self.performance_metrics["TDP"] = float(self.performance_metrics["TDP"])
                    for key in ["Llama7b_Process", "Llama70b_Process"]:
                        if key in self.performance_metrics and not pd.isna(self.performance_metrics[key]):
                            self.performance_metrics[key] = float(self.performance_metrics[key]) / 1000
                else:
                    print(f"Warning: No match for processor type {self.processor_type}. Using defaults.")
                    self.performance_metrics = self.default_metrics.copy()
            else:
                print(f"Warning: Malformed CSV structure in {self.performance_file}. Using defaults.")
                self.performance_metrics = self.default_metrics.copy()
        else:
            print(f"Warning: Processor config file {self.performance_file} missing. Using defaults.")
            self.performance_metrics = self.default_metrics.copy()

    def load_model_config(self):
        if os.path.exists(self.model_file):
            df = pd.read_csv(self.model_file)
            if "Parameters" in df.columns:
                self.model_metrics = {row["Model_Name"]: row.to_dict() for _, row in df.iterrows()}
            else:
                print(f"Warning: Malformed CSV structure in {self.model_file}. Model data unavailable.")
        else:
            print(f"Warning: Model config file {self.model_file} missing. Model data unavailable.")

    def load_model(self, model_type, node):
        self.state = Processor.PROCESSOR_STATE_IDLE

        if model_type in self.loaded_models:
            return 0  # Model already loaded, no load time

        model_info = self.model_metrics.get(model_type, {})
        model_size_gb = float(model_info.get("Parameter_Mem", 13))  # Default to 13GB
        load_time_pcie = model_size_gb / float(node.load_bandwidth_pcie.replace("GB/s", ""))
        load_time_nvlink = model_size_gb / float(node.inter_gpu_bandwidth.replace("GB/s", ""))
        total_load_time = load_time_pcie + load_time_nvlink

        self.loaded_models.add(model_type)
        return total_load_time

    def estimate_power_usage(self,execution_time, num_processors=1):
        tdp = self.performance_metrics.get("TDP", 300)  # Default to 300W if not in CSV

        if self.state == Processor.PROCESSOR_STATE_ACTIVE:
            return tdp * 1.0 * num_processors * execution_time  # Full TDP usage
        elif self.state == Processor.PROCESSOR_STATE_IDLE:
            return tdp * 0.15 * num_processors * execution_time  # 15% of TDP for idle power
        else:
            return 0  # No power usage when Off

    def log_usage_over_time(self, start_time, duration, tdp, model_load_time=None, ttft_time=None):
        end = min(int(start_time + duration), self.epoch_length)
        for t in range(int(start_time), end):
            self.resource_usage_log[t][0] = 1
            self.resource_usage_log[t][1] = tdp
        if model_load_time:
            ml_end = min(int(start_time + model_load_time), self.epoch_length)
            for t in range(int(start_time), ml_end):
                self.resource_usage_log[t][2] = 1
        if ttft_time:
            ttft_idx = int(start_time + ttft_time)
            if 0 <= ttft_idx < self.epoch_length:
                self.resource_usage_log[ttft_idx][3] = 1

    def update_state_log(self, start_time, duration, state):
        tdp = self.performance_metrics.get("TDP", 300)
        end = min(int(start_time + duration), self.epoch_length)
        for t in range(int(start_time), end):
            if self.resource_usage_log[t][0] == 0:
                if state == Processor.PROCESSOR_STATE_IDLE:
                    self.resource_usage_log[t][4] = 1
                    self.resource_usage_log[t][1] = tdp * 0.15
                elif state == Processor.PROCESSOR_STATE_OFF:
                    self.resource_usage_log[t][5] = 1
                    self.resource_usage_log[t][1] = 0


    def execute_task(self, model_type, num_tokens, requested_batch_size, node, migration_latency):

        self.state = Processor.PROCESSOR_STATE_ACTIVE

        if model_type not in self.model_metrics:
            raise ValueError(f"Unsupported model type: {model_type}")

        load_time = self.load_model(model_type, node)

        model_info = self.model_metrics[model_type]
        model_mem_per_token = float(model_info.get("Mem_per_Token", 0.262))
        model_param_mem = float(model_info.get("Parameter_Mem", 13))
        processor_mem = self.performance_metrics.get("Mem_Size", 80)

        if model_param_mem > processor_mem:
            print(f"Processor {self.processor_type} (ID: {self.processor_id}) - Not enough memory to load model {model_type}.")
            return None, False

        base_processing_time = self.performance_metrics.get(f"{model_type}_Process", 2)
        base_token_size = self.performance_metrics.get("base_token_size", 200)
        default_batch_size = self.performance_metrics.get("batch_size", 1)

        token_scaling_factor = num_tokens / base_token_size
        batch_scaling_factor = default_batch_size / requested_batch_size

        # Track KV-cache memory growth
        kv_cache_growth = model_mem_per_token * num_tokens
        total_memory_usage = model_param_mem + kv_cache_growth

        # Check if the processor can handle the growing KV-cache
        if total_memory_usage > processor_mem:
            self.failed_attempts += 1
            t = self.get_current_time_index()
            if t < self.epoch_length:
                self.resource_usage_log[t][6] = 1
            return None, False

        execution_time = base_processing_time * token_scaling_factor * batch_scaling_factor + load_time
        first_token_processing_time = base_processing_time * token_scaling_factor * batch_scaling_factor / num_tokens
        time_to_first_token = load_time + first_token_processing_time + 2 * migration_latency

        self.total_ttft_time += time_to_first_token
        self.ttft_events += 1
        tdp = self.performance_metrics.get("TDP", 300)
        self.log_usage_over_time(self.get_current_time_index(), execution_time, tdp, load_time, time_to_first_token)

        return execution_time, True

    def get_current_time_index(self):
        for i in range(self.epoch_length):
            if np.all(self.resource_usage_log[i, :6] == 0):
                return i
        return self.epoch_length - 1

    def fill_remaining_time(self):
        for t in range(self.epoch_length):
            if np.all(self.resource_usage_log[t, :6] == 0):
                if self.state == Processor.PROCESSOR_STATE_IDLE:
                    self.update_state_log(t, 1, Processor.PROCESSOR_STATE_IDLE)
                elif self.state == Processor.PROCESSOR_STATE_OFF:
                    self.update_state_log(t, 1, Processor.PROCESSOR_STATE_OFF)

    def report_epoch_stats(self):
        self.fill_remaining_time()
        exec_time = np.sum(self.resource_usage_log[:, 0])
        idle_time = np.sum(self.resource_usage_log[:, 4])
        off_time = np.sum(self.resource_usage_log[:, 5])
        active_power = np.sum(self.resource_usage_log[self.resource_usage_log[:, 0] == 1, 1])
        idle_power = np.sum(self.resource_usage_log[self.resource_usage_log[:, 4] == 1, 1])
        total_energy_kWh = np.sum(self.resource_usage_log[:, 1]) / 3600000
        avg_ttft = self.total_ttft_time / self.ttft_events if self.ttft_events > 0 else 0
        model_loads = int(np.sum(self.resource_usage_log[:, 2]))

        return {
            "processor_id": self.processor_id,
            "active_seconds": int(exec_time),
            "idle_seconds": int(idle_time),
            "off_seconds": int(off_time),
            "active_energy_kWh": active_power / 3600000,
            "idle_energy_kWh": idle_power / 3600000,
            "total_energy_kWh": total_energy_kWh,
            "avg_ttft_second": avg_ttft,
            "model_load_events": model_loads,
            "failed_requests": int(np.sum(self.resource_usage_log[:, 6]))
        }

class Node:
    def __init__(self, node_id, node_type, inter_gpu_bandwidth, load_bandwidth_pcie, load_delay_nvlink, processors):
        self.node_id = node_id
        self.node_type = node_type
        self.inter_gpu_bandwidth = inter_gpu_bandwidth
        self.load_bandwidth_pcie = load_bandwidth_pcie
        self.load_delay_nvlink = load_delay_nvlink
        self.processors = processors  # List of Processor instances
        self.loaded_models = set()

    @classmethod
    def load_nodes_from_csv(cls, node_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Node_Specs.csv'):
        nodes = []
        if os.path.exists(node_file):
            df = pd.read_csv(node_file)
            for _, row in df.iterrows():
                num_processors = int(row["Num_GPUs"]) if "Num_GPUs" in row else 1
                node_type = row["Node_Type"]
                processors = [
                    Processor(
                        processor_type=node_type.split("_")[0],
                        processor_id=f"{row['Node_Num']}_{i}",
                        performance_file=row.get("Performance_File", "H100_GPU.csv"),
                        model_file=row.get("Model_File", "Llama7b.csv")
                    ) for i in range(num_processors)
                ]
                nodes.append(cls(
                    node_id=row["Node_Num"],
                    node_type=node_type,
                    inter_gpu_bandwidth=row["Inter_GPU_Bandwidth"],
                    load_bandwidth_pcie=row["Load_Bandwidth_PCIE"],
                    load_delay_nvlink=row["Load_Delay_NVLinkNum_GPUs"],
                    processors=processors
                ))
        else:
            print(f"Warning: Node configuration file {node_file} missing.")
        return nodes

    def reset_epoch(self):
        self.loaded_models.clear()
        for p in self.processors:
            p.reset_epoch_stats()

    def set_node_state(self, state):
        for processor in self.processors:
            processor.state = state
            processor.update_state_log(0, processor.epoch_length, state)

    def execute_task_on_available_processor(self, model_type, num_tokens, batch_size, migration_latency):
        """Assigns a task to the first available processor with memory capacity."""
        for processor in self.processors:
            execution_time, success = processor.execute_task(model_type, num_tokens, batch_size, self, migration_latency)
            if success:
                self.loaded_models.add(model_type)
                return execution_time, processor.processor_id
        return None, None  # No processor succeeded

    def report_epoch_stats(self, cop):
        total_processor_energy = 0
        processor_stats = []

        for p in self.processors:
            stats = p.report_epoch_stats()
            total_processor_energy += stats["total_energy_kWh"]
            processor_stats.append(stats)

        other_hardware_energy = 0.13 * total_processor_energy
        cooling_energy = 3 * (total_processor_energy / cop) if cop > 0 else 0
        total_node_energy = total_processor_energy + other_hardware_energy + cooling_energy

        return {
            "node_id": self.node_id,
            "total_node_energy_kWh": total_node_energy,
            "processor_energy_kWh": total_processor_energy,
            "other_hardware_energy_kWh": other_hardware_energy,
            "cooling_energy_kWh": cooling_energy,
            "processors": processor_stats
        }



class Datacenter:
    def __init__(self, datacenter_id, location, energy_cost, carbon_intensity, nodes,
                 water_usage=None, cop_profile=None, water_cycling_density=0.1, solids_ratio=0.3,
                 potable_energy_intensity=0.005, wastewater_energy_intensity=0.01):
        self.datacenter_id = datacenter_id
        self.location = location

        self.energy_cost_profile = [float(x) for x in energy_cost.split(";")]
        self.carbon_intensity = float(carbon_intensity)
        self.water_usage = float(water_usage) if water_usage else 0.0

        self.cop_profile = [float(x) for x in cop_profile.split(";")] if cop_profile else [3.0] * 24

        self.water_cycling_density = float(water_cycling_density)
        self.solids_ratio = float(solids_ratio)
        self.potable_energy_intensity = float(potable_energy_intensity)
        self.wastewater_energy_intensity = float(wastewater_energy_intensity)

        self.nodes = nodes
        self.loaded_models = set()

    @classmethod
    def load_datacenters_from_csv(
            cls,
            datacenter_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Datacenter_Specs.csv',
            node_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Node_Specs.csv'
    ):
        datacenters = []
        if os.path.exists(datacenter_file):
            df = pd.read_csv(datacenter_file)
            all_node_templates = Node.load_nodes_from_csv(node_file)
            node_type_map = {node.node_id: node for node in all_node_templates}  # node_id = node_type_id

            for _, row in df.iterrows():
                dc_nodes = []
                dc_id = row["DC_Num"]
                total_nodes = int(row.get("Total_Nodes", 1000))  # default fallback

                # Parse Node_Types column like "0:2;1:1;3:1"
                raw_types = str(row["Node_Types"]).split(";")
                type_weights = []
                for entry in raw_types:
                    parts = entry.split(":")
                    type_id = int(parts[0])
                    weight = int(parts[1]) if len(parts) > 1 else 1
                    if type_id in node_type_map:
                        type_weights.append((type_id, weight))
                    else:
                        print(f"Warning: Node type {type_id} not found in template.")

                if not type_weights:
                    print(f"Warning: No valid node types for datacenter {dc_id}")
                    continue

                total_weight = sum(w for _, w in type_weights)
                node_allocations = {
                    type_id: (total_nodes * weight) // total_weight
                    for type_id, weight in type_weights
                }

                # Add remainder to last type to make up exact total_nodes
                assigned = sum(node_allocations.values())
                if assigned < total_nodes:
                    node_allocations[type_weights[-1][0]] += total_nodes - assigned

                node_counter = 0
                for node_type_id, count in node_allocations.items():
                    base_node = node_type_map[node_type_id]

                    for _ in range(count):
                        new_processors = []
                        for p in base_node.processors:
                            processor_id = f"{dc_id}_{node_counter}_{p.processor_id.split('_')[-1]}"
                            new_p = Processor(
                                processor_type=p.processor_type,
                                processor_id=processor_id,
                                performance_file=p.performance_file,
                                model_file=p.model_file
                            )
                            new_p.dc_id = dc_id
                            new_p.node_id = node_counter
                            new_processors.append(new_p)

                        new_node = Node(
                            node_id=node_counter,
                            node_type=base_node.node_type,
                            inter_gpu_bandwidth=base_node.inter_gpu_bandwidth,
                            load_bandwidth_pcie=base_node.load_bandwidth_pcie,
                            load_delay_nvlink=base_node.load_delay_nvlink,
                            processors=new_processors
                        )
                        new_node.dc_id = dc_id
                        dc_nodes.append(new_node)
                        node_counter += 1

                datacenters.append(cls(
                    datacenter_id=dc_id,
                    location=row["Location"],
                    carbon_intensity=row["Carbon_Intensity"],
                    energy_cost=row["Time_of_Use(24_Hours)"],
                    water_usage=row.get("Water_Static", None),
                    cop_profile=row.get("COP_Profile(24_Hours)", None),
                    water_cycling_density=row.get("Water_Cycling_Density", 0.1),
                    solids_ratio=row.get("Solids_Ratio", 0.3),
                    potable_energy_intensity=row.get("Potable_Energy_Intensity", 0.005),
                    wastewater_energy_intensity=row.get("Wastewater_Energy_Intensity", 0.01),
                    nodes=dc_nodes
                ))

        else:
            print(f"Warning: Datacenter configuration file {datacenter_file} missing.")
        return datacenters

    def reset_epoch(self):
        self.loaded_models.clear()
        for node in self.nodes:
            node.reset_epoch()

    def set_datacenter_state(self, node_type_plan: dict):
        for node in self.nodes:
            desired_state = node_type_plan.get(node.node_id, "Off")
            node.set_node_state(desired_state)

    def calculate_metrics(self, power_kwh, current_hour):
        price = self.energy_cost_profile[current_hour]
        # print(price)
        # print(self.energy_cost_profile)
        return {
            "energy_cost": power_kwh * price,
            "carbon_emitted": power_kwh * self.carbon_intensity,
            "water_used": power_kwh * self.water_usage
        }

    def schedule_request(self, model_type, num_tokens, batch_size, time_index, migration_latency, max_retries=None):
        # Prioritize nodes with model already loaded
        preferred_nodes = [n for n in self.nodes if model_type in n.loaded_models]
        fallback_nodes = [n for n in self.nodes if n not in preferred_nodes]
        all_nodes = preferred_nodes + fallback_nodes

        attempted_nodes = 0
        for node in all_nodes:
            exec_time, processor_id = node.execute_task_on_available_processor(model_type, num_tokens, batch_size, migration_latency)
            attempted_nodes += 1
            if exec_time is not None:
                self.loaded_models.add(model_type)
                return {
                    "scheduled": True,
                    "datacenter_id": self.datacenter_id,
                    "location": self.location,
                    "node_id": node.node_id,
                    "processor_id": processor_id,
                    "model_type": model_type,
                    "execution_time": exec_time,
                    "ttft_seconds": None,
                    "model_already_loaded": model_type in node.loaded_models,
                    "retry_attempts": attempted_nodes - 1
                }
            if max_retries is not None and attempted_nodes >= max_retries:
                break

        return {
            "scheduled": False,
            "datacenter_id": self.datacenter_id,
            "location": self.location,
            "model_type": model_type,
            "retry_attempts": attempted_nodes
        }

    def report_epoch_stats(self, current_hour):
        total_processor_energy = 0
        processor_stats_all = []
        cop = self.cop_profile[current_hour]

        for node in self.nodes:
            node_stats = node.report_epoch_stats(cop=cop)
            total_processor_energy += node_stats["total_node_energy_kWh"]
            processor_stats_all.extend(node_stats["processors"])

        # Cooling water usage based on heat load
        heat_load_mj = total_processor_energy * 3.6  # kWh → MJ
        cooling_water_evaporated = heat_load_mj * self.water_cycling_density

        # Total drawn water & treated blowdown
        cooling_water_drawn = cooling_water_evaporated / self.solids_ratio
        cooling_water_treated = cooling_water_drawn - cooling_water_evaporated

        # Energy used to support water movement and treatment
        potable_energy = self.potable_energy_intensity * (cooling_water_drawn - cooling_water_treated)
        wastewater_energy = self.wastewater_energy_intensity * cooling_water_treated
        total_water_energy = potable_energy + wastewater_energy

        # Carbon emissions associated with water
        water_carbon_kg = total_water_energy * self.carbon_intensity

        # Base cost and static environmental impact
        impact = self.calculate_metrics(total_processor_energy, current_hour)

        return {
            "datacenter_id": self.datacenter_id,
            "location": self.location,
            "total_energy_kWh": total_processor_energy + total_water_energy,
            "total_cost_usd": impact["energy_cost"],
            "carbon_emitted_kg": impact["carbon_emitted"] + water_carbon_kg,
            "water_used_liters": impact["water_used"] + cooling_water_drawn,
            "cooling_water_evaporated_liters": cooling_water_evaporated,
            "cooling_water_treated_liters": cooling_water_treated,
            "potable_energy_kWh": potable_energy,
            "wastewater_energy_kWh": wastewater_energy,
            "water_carbon_kg": water_carbon_kg,
            "processor_stats": processor_stats_all
        }


class Geo_Network:
    def __init__(self, datacenters, latency_matrix, datacenter_ids, locations):
        self.datacenters = datacenters
        self.latency_matrix = latency_matrix
        self.datacenter_ids = datacenter_ids
        self.locations = locations
        self.id_to_index = {dc_id: i for i, dc_id in enumerate(datacenter_ids)}

    @classmethod
    def load_network(cls,
                     latency_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Geo_Latencies.csv',
                     dc_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Datacenter_specs.csv',
                     node_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Node_Specs.csv'):
        datacenters = Datacenter.load_datacenters_from_csv(dc_file, node_file)
        dc_ids = [dc.datacenter_id for dc in datacenters]
        locations = [dc.location for dc in datacenters]

        if os.path.exists(latency_file):
            latency_df = pd.read_csv(latency_file, index_col=0)
            latency_df.columns = latency_df.columns.astype(int)
            latency_df.index = latency_df.index.astype(int)
            latency_matrix = latency_df.loc[dc_ids, dc_ids].to_numpy()
        else:
            latency_matrix = np.zeros((len(datacenters), len(datacenters)))

        return cls(datacenters, latency_matrix, dc_ids, locations)

    def reset_all(self):
        for dc in self.datacenters:
            dc.reset_epoch()

    def apply_power_plan(self, power_plan):
        for dc in self.datacenters:
            dc_plan = power_plan.get(dc.datacenter_id)
            if dc_plan:
                dc.set_datacenter_state(dc_plan)

    def get_latency(self, source_id, target_id):
        i = self.id_to_index[source_id]
        j = self.id_to_index[target_id]
        return self.latency_matrix[i][j]

    def apply_schedule_plan(self, schedule_plan):
        results = []
        for req in schedule_plan:
            result = self.route_request(
                target_dc_id=req["target_dc_id"],
                model_type=req["model_type"],
                num_tokens=req["num_tokens"],
                batch_size=req["batch_size"],
                source_dc_id=req["source_dc_id"],
                time_index=req["time_index"]
            )
            results.append(result)
        return results

    def route_request(self, target_dc_id, model_type, num_tokens, batch_size, source_dc_id, time_index):
        target_dc = next((dc for dc in self.datacenters if dc.datacenter_id == target_dc_id), None)
        migration_latency = float(self.get_latency(source_dc_id, target_dc_id)) / 1000.0
        if not target_dc:
            return {
                "scheduled": False,
                "error": f"Datacenter {target_dc_id} not found",
                "source_dc": source_dc_id
            }

        result = target_dc.schedule_request(model_type, num_tokens, batch_size, time_index, migration_latency)
        result["migration_latency"] = self.get_latency(source_dc_id, target_dc_id)
        result["source_dc"] = source_dc_id
        result["target_dc"] = target_dc_id
        return result

    def report_global_stats(self, current_hour):
        total_energy = 0
        total_cost = 0
        total_carbon = 0
        total_water = 0
        water_carbon = 0
        dc_reports = []

        for dc in self.datacenters:
            stats = dc.report_epoch_stats(current_hour)
            total_energy += stats["total_energy_kWh"]
            total_cost += stats["total_cost_usd"]
            total_carbon += stats["carbon_emitted_kg"]
            total_water += stats["water_used_liters"]
            water_carbon += stats.get("water_carbon_kg", 0)
            dc_reports.append(stats)

        return {
            "total_energy_kWh": total_energy,
            "total_cost_usd": total_cost,
            "carbon_emitted_kg": total_carbon,
            "water_used_liters": total_water,
            "water_carbon_kg": water_carbon,
            "datacenter_stats": dc_reports
        }


def main():
    if not os.path.exists(WORKLOAD_FILE):
        print("No workload found, generating synthetic trace...")
        generate_random_workload()

    workload_df = pd.read_csv(WORKLOAD_FILE)

    epoch_idx = 0
    epoch_df = workload_df[workload_df["epoch"] == epoch_idx]

    print(f"\n--- Running Simulation for Epoch {epoch_idx} ---")
    schedule_plan, power_plan = my_scheduler(epoch_df, epoch_idx)
    stats, results = LLM_Simulator(epoch_idx, workload_df, schedule_plan, power_plan)

    print("\n=== Epoch Stats ===")
    for k, v in stats.items():
        if k != "datacenter_stats":
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Sample Request Results ===")
    for r in results[:5]:
        print(r)


if __name__ == "__main__":
    main()


