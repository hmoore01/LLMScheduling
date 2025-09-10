import pandas as pd
import numpy as np
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import njit
import math

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


def LLM_Simulator(epoch_idx, workload_df, schedule_plan, power_plan):
    t0 = time.time()
    geo_network = Geo_Network.load_network()
    # print(f"[Timer] load_network: {time.time() - t0:.2f}s")

    t1 = time.time()
    geo_network.reset_all()
    geo_network.apply_power_plan(power_plan)
    # print(f"[Timer] reset + power plan: {time.time() - t1:.2f}s")

    t2 = time.time()
    results = geo_network.apply_schedule_plan(schedule_plan)
    # print(f"[Timer] schedule plan applied: {time.time() - t2:.2f}s")

    leftover_requests_out = getattr(geo_network, "leftover_request_arr", [])
    # Ensure policy: leftovers arrive at t=0 next epoch
    if leftover_requests_out:
        for r in leftover_requests_out:
            r["time_index"] = 0

    t3 = time.time()
    hour = ((epoch_idx * 15) // 60) % 24
    stats = geo_network.report_global_stats(current_hour=hour)
    # print(f"[Timer] report global stats: {time.time() - t3:.2f}s")

    t4 = time.time()
    # print(f"[Timer] aggregate TTFT: {time.time() - t4:.2f}s")

    # print(f"[LLM_Simulator] Total: {time.time() - t0:.2f}s")

    return {
        "avg_ttft": stats["avg_ttft"],
        "energy_cost": stats["total_cost_usd"],
        "carbon_emissions": stats["carbon_emitted_kg"],
        "water_usage": stats["water_used_liters"],
        "total_energy": stats["total_energy_kWh"],
        "network_load": stats["network_load"]
    }, results, leftover_requests_out


from dataclasses import dataclass
from typing import Literal

CoolingType = Literal["air", "liquid", "immersion"]

@dataclass(frozen=True)
class CoolingConfig:
    cooling_type: CoolingType
    base_pue_35c: float
    water_intensity_l_per_kwh_35c: float
    pue_slope_per_c: float
    water_slope_per_c: float = 0.0

    def pue_at_setpoint(self, setpoint_c: float) -> float:
        pue = self.base_pue_35c + self.pue_slope_per_c * (setpoint_c - 35.0)
        return max(1.01, float(pue))

    def water_intensity_at_setpoint(self, setpoint_c: float) -> float:
        w = self.water_intensity_l_per_kwh_35c + self.water_slope_per_c * (setpoint_c - 35.0)
        return max(0.0, float(w))

# Made-up but reasonable defaults (you can change at DC init)
DEFAULT_COOLING_PROFILES = {
    "air": CoolingConfig("air", base_pue_35c=1.45, water_intensity_l_per_kwh_35c=0.80,
                         pue_slope_per_c=-0.007, water_slope_per_c=-0.01),
    "liquid": CoolingConfig("liquid", base_pue_35c=1.20, water_intensity_l_per_kwh_35c=0.40,
                            pue_slope_per_c=-0.004, water_slope_per_c=-0.004),
    "immersion": CoolingConfig("immersion", base_pue_35c=1.08, water_intensity_l_per_kwh_35c=0.05,
                               pue_slope_per_c=-0.002, water_slope_per_c=-0.001),
}


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
        self.next_available_time = 0.0

        # Load processor and model configurations
        self.load_processor_config()
        self.load_model_config()
        self.total_ttft_time = 0.0
        self.ttft_events = 0
        self.energy_used_total = 0.0
        self.ttft_total = 0.0
        self.ttft_count = 0
        self.total_active_time = 0.0

    # --- NEW: deterministic effective start including warmups/residency (no mutation here) ---
    def effective_start_time(self, arrival_t: int, model_type: str, warm_start_penalty_s: int,
                            power_on_warmup_s: int) -> int:
        st = max(int(arrival_t), int(self.next_available_time))
        if self.state == Processor.PROCESSOR_STATE_OFF:
            st += int(power_on_warmup_s)
        if model_type not in self.loaded_models:
            st += int(warm_start_penalty_s)
        return st

     # --- NEW: commit power+model residency so future requests see it loaded ---
    def commit_model_load_and_power(self, model_type: str):
        if self.state == Processor.PROCESSOR_STATE_OFF:
            self.state = Processor.PROCESSOR_STATE_IDLE
        self.loaded_models.add(model_type)


    def reset_epoch_stats(self):
        self.resource_usage_log.fill(0.0)
        self.loaded_models.clear()
        self.failed_attempts = 0
        self.total_ttft_time = 0.0
        self.ttft_events = 0
        self.total_active_time = 0.0

    def load_processor_config(self):
        if os.path.exists(self.performance_file):
            df = pd.read_csv(self.performance_file)
            if "num_GPUs" in df.columns:
                matched_rows = df[df["num_GPUs"] == int(self.processor_type)]
                if not matched_rows.empty:
                    self.performance_metrics = matched_rows.iloc[0].to_dict()

                    # Normalize Mem_Size like "80GB" -> 80
                    if isinstance(self.performance_metrics.get("Mem_Size"), str):
                        self.performance_metrics["Mem_Size"] = int(
                            self.performance_metrics["Mem_Size"].replace("GB", "")
                        )

                    # numeric TDP
                    if "TDP" in self.performance_metrics:
                        self.performance_metrics["TDP"] = float(self.performance_metrics["TDP"])

                    # *** DO NOT divide by 1000 here; keep MS ***
                    for key in ("Llama7b_Process", "Llama70b_Process"):
                        if key in self.performance_metrics and not pd.isna(self.performance_metrics[key]):
                            self.performance_metrics[key] = float(self.performance_metrics[key])  # ms

                    # map your CSV column(s) to the key your code reads
                    if "base_token_size" not in self.performance_metrics:
                        if "prefill_token_size" in self.performance_metrics and not pd.isna(
                                self.performance_metrics["prefill_token_size"]):
                            self.performance_metrics["base_token_size"] = int(
                                self.performance_metrics["prefill_token_size"])
                        elif "gen_token_size" in self.performance_metrics and not pd.isna(
                                self.performance_metrics["gen_token_size"]):
                            self.performance_metrics["base_token_size"] = int(
                                self.performance_metrics["gen_token_size"])
                        else:
                            self.performance_metrics["base_token_size"] = 200  # safe default

                    # batch_size is already present in your CSV
                    if "batch_size" in self.performance_metrics and not pd.isna(self.performance_metrics["batch_size"]):
                        self.performance_metrics["batch_size"] = int(self.performance_metrics["batch_size"])
                    else:
                        self.performance_metrics["batch_size"] = 1
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

    def load_model(self, model_type, node, simulate_only=False):
        t0 = time.time()

        # === Already loaded, fast return ===
        if model_type in self.loaded_models:
            return 0  # No additional load time

        model_info = self.model_metrics.get(model_type, {})
        model_size_gb = float(model_info.get("Parameter_Mem", 13))  # Default to 13GB

        try:
            pcie_bw = float(node.load_bandwidth_pcie.replace("GB/s", ""))
            nvlink_bw = float(node.inter_gpu_bandwidth.replace("GB/s", ""))
        except Exception as e:
            print(f"[Warning] Bandwidth parsing error: {e}")
            pcie_bw = 12  # Safe default
            nvlink_bw = 100  # Safe default

        load_time_pcie = model_size_gb / pcie_bw
        load_time_nvlink = model_size_gb / nvlink_bw
        total_load_time = load_time_pcie + load_time_nvlink

        if not simulate_only:
            self.loaded_models.add(model_type)
            self.state = Processor.PROCESSOR_STATE_IDLE

        # print(f"[Timer] Processor.load_model (simulate={simulate_only}): {time.time() - t0:.4f}s")
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
        self.fast_log_usage_over_time(
            self.resource_usage_log,
            self.epoch_length,
            start_time,
            duration,
            tdp,
            model_load_time if model_load_time is not None else -1,
            ttft_time if ttft_time is not None else -1
        )

    @staticmethod
    @njit
    def fast_log_usage_over_time(resource_log, epoch_length, start_time, duration, tdp, model_load_time=-1,
                                 ttft_time=-1):
        s = int(start_time)
        # choose your rounding; ceil tends to overcount when duration has fractional ticks
        dur = max(1, int(np.round(duration)))  # was: int(np.ceil(duration))
        e = min(s + dur, epoch_length)
        if s >= epoch_length or e <= s:
            return

        for t in range(s, e):
            # set executing; clear other state flags for this tick
            resource_log[t, 0] = 1.0  # executing
            resource_log[t, 4] = 0.0  # idle
            resource_log[t, 5] = 0.0  # off

            # OVERWRITE power with active watts (don't add on top of baseline)
            resource_log[t, 1] = float(tdp)  # <-- was "+="

        # model load window (doesn't affect power)
        if model_load_time is not None and model_load_time > 0:
            ml_end = min(s + int(model_load_time), epoch_length)
            for t in range(s, ml_end):
                resource_log[t, 2] = 1.0

        # TTFT event tick
        if ttft_time is not None and ttft_time > 0:
            ttft_idx = int(s + ttft_time)
            if 0 <= ttft_idx < epoch_length:
                resource_log[ttft_idx, 3] = 1.0

    @staticmethod
    @njit
    def fast_update_state_log(resource_log, start, duration, epoch_length, idle_watts, state_code):
        # state_code: 1 = Idle, 2 = Off
        s = int(start)
        e = s + int(duration)
        L = int(epoch_length)
        if s < 0:
            s = 0
        if e > L:
            e = L
        if e <= s:
            return

        for t in range(s, e):
            if resource_log[t, 0] != 0.0:  # executing flag at col 0
                continue

            # clear idle/off (cols 4,5)
            resource_log[t, 4] = 0.0
            resource_log[t, 5] = 0.0

            if state_code == 1:  # Idle
                resource_log[t, 4] = 1.0  # idle_flag
                resource_log[t, 1] = float(idle_watts)  # power_used
            else:  # Off (state_code == 2)
                resource_log[t, 5] = 1.0  # off_flag
                resource_log[t, 1] = 0.0  # power_used


    def update_state_log(self, start_time, duration, state):
        if state not in (Processor.PROCESSOR_STATE_IDLE, Processor.PROCESSOR_STATE_OFF):
            return  # only handle IDLE/OFF here

        state_code = 1 if state == Processor.PROCESSOR_STATE_IDLE else 2
        tdp = 0.15 * self.performance_metrics.get("TDP", 300)
        self.fast_update_state_log(
            self.resource_usage_log,
            int(start_time),
            int(duration),
            self.epoch_length,
            float(tdp),
            state_code
        )

    @staticmethod
    @njit
    def compute_execution_metrics(base_processing_time, base_token_size, default_batch_size,
                                  num_tokens, requested_batch_size,
                                  model_param_mem, model_mem_per_token, processor_mem,
                                  load_time, migration_latency):

        # -------- sanitize ints --------
        btok = base_token_size if base_token_size > 0 else 1
        db = default_batch_size if default_batch_size > 0 else 1
        rb = requested_batch_size if requested_batch_size > 0 else 1
        ntok = num_tokens if num_tokens > 0 else 1

        # -------- memory check with unit hardening --------
        kv_gb = model_mem_per_token * ntok
        total_gb = model_param_mem + kv_gb

        # Heuristic: if per-token memory is large and we overflow badly, assume it was MB/token
        if total_gb > processor_mem and model_mem_per_token > 1.0:
            kv_gb = (model_mem_per_token / 1024.0) * ntok  # MB/token -> GB/token
            total_gb = model_param_mem + kv_gb

        if total_gb > processor_mem:
            return -1.0, -1.0  # signal memory infeasibility

        # -------- time model --------
        # chunks = ceil(tokens / base_token_size)
        chunks = (ntok + btok - 1) // btok

        # naive batch scaling: time ∝ default / requested
        batch_scale = float(db) / float(rb)

        # Guard against absurdly tiny per-chunk time (ms): floor to a small realistic minimum
        per_chunk_ms = float(base_processing_time)
        if per_chunk_ms < 5.0:  # 5 ms minimum per chunk
            per_chunk_ms = 5.0

        proc_ms = per_chunk_ms * chunks * batch_scale
        proc_s = proc_ms * 0.001

        # Total time includes load & migration
        exec_time = float(load_time) + float(migration_latency) + proc_s

        # TTFT: first chunk + load + migration
        ttft = float(load_time) + float(migration_latency) + (per_chunk_ms * batch_scale) * 0.001

        return exec_time, ttft

    def execute_task(self, model_type, num_tokens, requested_batch_size, node, migration_latency):
        start_tick = int(self.next_available_time)

        # decide load treatment by baseline state
        baseline_state = self.state
        if baseline_state == Processor.PROCESSOR_STATE_OFF:
            load_time = float(self.load_model(model_type, node))  # seconds/ticks
            load_ticks = max(0, int(math.ceil(load_time)))  # <-- ceil
            if load_ticks > 0:
                self.update_state_log(start_tick, load_ticks, Processor.PROCESSOR_STATE_IDLE)
                start_tick += load_ticks
        else:
            load_time = 0.0
            load_ticks = 0

        # gather perf inputs
        model_info = self.model_metrics[model_type]
        model_mem_per_token = float(model_info.get("Mem_per_Token", 0.262))
        model_param_mem = float(model_info.get("Parameter_Mem", 13.0))
        processor_mem = float(self.performance_metrics.get("Mem_Size", 80))

        base_processing_time = float(self.performance_metrics.get(f"{model_type}_Process", 2.0))  # ms
        base_token_size = int(self.performance_metrics.get("base_token_size", 200))
        default_batch_size = int(self.performance_metrics.get("batch_size", 1))

        # compute exec + ttft (seconds/ticks)
        exec_time_total, ttft = self.compute_execution_metrics(
            base_processing_time, base_token_size, default_batch_size,
            int(num_tokens), int(requested_batch_size),
            model_param_mem, model_mem_per_token, processor_mem,
            float(load_time), float(migration_latency)
        )
        if exec_time_total <= 0:
            self.failed_attempts += 1
            return 0.0, False

        # Only the processing part should be ACTIVE; ensure >=1 tick if any processing
        proc_seconds = max(0.0, float(exec_time_total) - float(load_time))
        proc_only_ticks = int(math.ceil(proc_seconds)) if proc_seconds > 0.0 else 0  # <-- ceil

        if proc_only_ticks > 0:
            self.state = Processor.PROCESSOR_STATE_ACTIVE
            self.log_usage_over_time(
                start_tick, proc_only_ticks,
                tdp=self.performance_metrics["TDP"],
                model_load_time=None, ttft_time=None
            )
            start_tick += proc_only_ticks

        # advance time; leave as Idle after
        self.next_available_time = start_tick
        self.state = Processor.PROCESSOR_STATE_IDLE

        if ttft is not None and ttft > 0:
            self.total_ttft_time += float(ttft)  # seconds
            self.ttft_events += 1

        return exec_time_total, True

    def get_current_time_index(self):
        return self.fast_get_current_time_index(self.resource_usage_log, self.epoch_length)

    @staticmethod
    @njit
    def fast_get_current_time_index(log, epoch_length):
        for i in range(epoch_length):
            if log[i, 0] == 0:  # Check only the "executing" flag
                return i
        return epoch_length - 1

    @staticmethod
    @njit
    def fast_fill_remaining_time(resource_log, epoch_length, state_code, tdp):
        for t in range(epoch_length):
            # Only fill if no active, idle, or off marking already present
            if resource_log[t, 0] == 0 and resource_log[t, 4] == 0 and resource_log[t, 5] == 0:
                if state_code == 1:  # IDLE
                    resource_log[t, 4] = 1
                    resource_log[t, 1] = tdp * 0.15
                elif state_code == 2:  # OFF
                    resource_log[t, 5] = 1
                    resource_log[t, 1] = 0

    def fill_remaining_time(self):
        state_code = 0
        if self.state == Processor.PROCESSOR_STATE_IDLE:
            state_code = 1
        elif self.state == Processor.PROCESSOR_STATE_OFF:
            state_code = 2

        if state_code:
            tdp = self.performance_metrics.get("TDP", 300)
            self.fast_fill_remaining_time(self.resource_usage_log, self.epoch_length, state_code, tdp)

    def compute_execution_metrics_wrapper(self, *args, **kwargs):
        import math

        perf = getattr(self, "performance_metrics", {}) or {}
        model_metrics = getattr(self, "model_metrics", {}) or {}
        epoch_len = int(getattr(self, "epoch_length", 900))

        # -------- helpers --------
        def _defaults_for_proc():
            return (
                int(perf.get("base_token_size", 200)),
                int(perf.get("batch_size", 1)),
                float(perf.get("Mem_Size", 80.0)),
            )

        def _defaults_for_model(model_type: str):
            mm = model_metrics.get(str(model_type), {}) or {}
            return (
                float(mm.get("Parameter_Mem", 13.0)),  # GB
                float(mm.get("Mem_per_Token", 0.262)),
            # GB/token (or MB/token if that's your scale; adjust below if needed)
            )

        def _estimate_load_seconds():
            # Prefer a configured hint; caller can override via short/full args
            return float(perf.get("model_load_seconds", 0.0))

        def _to_full_from_short(model_type, base_ms, num_tokens, batch_size, load_seconds=0.0, migration_latency=0.0):
            base_token_size, default_batch_size, processor_mem = _defaults_for_proc()
            model_param_mem, model_mem_per_token = _defaults_for_model(model_type)
            return (
                float(base_ms), int(base_token_size), int(default_batch_size),
                int(num_tokens), int(batch_size),
                float(model_param_mem), float(model_mem_per_token), float(processor_mem),
                float(load_seconds), float(migration_latency)
            )

        def _parse_args():
            # kwargs: allow both forms
            if kwargs:
                if "model_type" in kwargs:  # short via kwargs
                    mt = kwargs["model_type"]
                    base_ms = kwargs.get("base_ms")
                    nt = kwargs.get("num_tokens")
                    bs = kwargs.get("batch_size")
                    if base_ms is None or nt is None or bs is None:
                        raise TypeError("Short-form kwargs require model_type, base_ms, num_tokens, batch_size.")
                    ls = kwargs.get("load_seconds", _estimate_load_seconds())
                    ml = kwargs.get("migration_latency", 0.0)
                    return _to_full_from_short(mt, base_ms, nt, bs, ls, ml)
                else:
                    # full via kwargs
                    keys = ["base_ms", "base_token_size", "default_batch_size", "num_tokens", "batch_size",
                            "model_param_mem", "model_mem_per_token", "processor_mem", "load_seconds",
                            "migration_latency"]
                    missing = [k for k in keys if k not in kwargs]
                    if missing:
                        raise TypeError(f"Full-form kwargs missing: {missing}")
                    return (
                        float(kwargs["base_ms"]), int(kwargs["base_token_size"]), int(kwargs["default_batch_size"]),
                        int(kwargs["num_tokens"]), int(kwargs["batch_size"]),
                        float(kwargs["model_param_mem"]), float(kwargs["model_mem_per_token"]),
                        float(kwargs["processor_mem"]),
                        float(kwargs["load_seconds"]), float(kwargs["migration_latency"])
                    )

            # positional: short or full
            if len(args) >= 5 and isinstance(args[0], str):
                mt = args[0]
                base_ms, nt, bs = args[1], args[2], args[3]
                ls = args[4] if len(args) > 4 else _estimate_load_seconds()
                ml = args[5] if len(args) > 5 else 0.0
                return _to_full_from_short(mt, base_ms, nt, bs, ls, ml)
            if len(args) == 10:
                return (
                    float(args[0]), int(args[1]), int(args[2]), int(args[3]), int(args[4]),
                    float(args[5]), float(args[6]), float(args[7]), float(args[8]), float(args[9])
                )
            raise TypeError(
                f"compute_execution_metrics_wrapper received unsupported args: len={len(args)}, kwargs={list(kwargs.keys())}"
            )

        def _normalize_seconds(exec_val, ttft_val):
            def _f(x):
                try:
                    x = float(x)
                except Exception:
                    return 0.0
                if math.isnan(x) or math.isinf(x):
                    return 0.0
                return x

            e, t = _f(exec_val), _f(ttft_val)

            # If absurd vs epoch -> probably ms
            if e > epoch_len * 10:
                e /= 1000.0
                t /= 1000.0
            # Clamp
            return max(e, 0.0), max(t, 0.0)

        def _fallback_seconds(base_ms, base_token_size, default_batch_size,
                              num_tokens, batch_size, model_param_mem, model_mem_per_token,
                              processor_mem, load_seconds, migration_latency):
            # Basic throughput model: chunked by base_token_size, scaled by batch, with a memory penalty if over capacity.
            chunks = max(1, int(math.ceil(num_tokens / float(max(base_token_size, 1)))))

            # Scale by batch vs default
            batch_scale = max(1.0, float(batch_size) / float(max(default_batch_size, 1)))
            core_ms = chunks * float(base_ms) / batch_scale

            # Memory pressure penalty (soft), if model+activations exceed capacity
            # Adjust the unit if your Mem_per_Token is MB/token instead of GB/token:
            activation_mem = num_tokens * float(model_mem_per_token)
            total_mem = float(model_param_mem) + activation_mem
            if total_mem > float(processor_mem) > 0:
                over = (total_mem - processor_mem) / processor_mem
                core_ms *= (1.0 + max(0.1, over))  # at least +10% if over

            exec_s = core_ms / 1000.0 + float(load_seconds) + float(migration_latency)
            ttft_s = float(load_seconds) + float(migration_latency)
            return max(exec_s, 0.0), max(ttft_s, 0.0)

        # -------- main --------
        full = _parse_args()
        base_ms, base_token_size, default_batch_size, num_tokens, batch_size, \
            model_param_mem, model_mem_per_token, processor_mem, load_seconds, migration_latency = full

        exec_val = ttft_val = None
        # Try the canonical implementation first
        if hasattr(self, "compute_execution_metrics"):
            try:
                exec_val, ttft_val = self.compute_execution_metrics(*full)
            except Exception:
                exec_val = ttft_val = None

        # Normalize + sanity
        if exec_val is not None:
            exec_s, ttft_s = _normalize_seconds(exec_val, ttft_val if ttft_val is not None else 0.0)
            # If it still looks implausibly tiny, use deterministic fallback
            if exec_s < 0.01:  # <10ms for any real job is suspicious
                exec_s, ttft_s = _fallback_seconds(*full)
            return exec_s, ttft_s

        # Deterministic fallback (never returns nonsense)
        return _fallback_seconds(*full)


    def report_epoch_stats(self):
        self.fill_remaining_time()

        # === Summarize flags ===
        exec_time = np.sum(self.resource_usage_log[:, 0])  # executing flag
        idle_time = np.sum(self.resource_usage_log[:, 4])  # idle flag
        off_time = np.sum(self.resource_usage_log[:, 5])  # off flag
        failed_attempts = np.sum(self.resource_usage_log[:, 6])

        # === Power stats ===
        active_power = np.sum(self.resource_usage_log[self.resource_usage_log[:, 0] == 1, 1])
        idle_power = np.sum(self.resource_usage_log[self.resource_usage_log[:, 4] == 1, 1])
        total_energy_kWh = np.sum(self.resource_usage_log[:, 1]) / 3_600_000  # mWh to kWh

        # === TTFT and model loads ===
        avg_ttft = self.total_ttft_time / self.ttft_events if self.ttft_events > 0 else 0
        model_loads = int(np.sum(self.resource_usage_log[:, 2]))

        # === Debug: Validate total time ===
        total_time_tracked = exec_time + idle_time + off_time
        expected_time = self.epoch_length  # for one processor

        return {
            "processor_id": self.processor_id,
            "active_seconds": int(exec_time),
            "idle_seconds": int(idle_time),
            "off_seconds": int(off_time),
            "active_energy_kWh": active_power / 3_600_000,
            "idle_energy_kWh": idle_power / 3_600_000,
            "total_energy_kWh": total_energy_kWh,
            "avg_ttft_second": avg_ttft,
            "model_load_events": model_loads,
            "failed_requests": int(failed_attempts)
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

    def pick_processor_scored(self, arrival_t: int, model_type: str,
                              warm_start_penalty_s: int = 3, power_on_warmup_s: int = 5):
        best = None
        for p in self.processors:
            st = p.effective_start_time(arrival_t, model_type, warm_start_penalty_s, power_on_warmup_s)
            residency = 1 if (model_type in p.loaded_models or model_type in self.loaded_models) else 0
            key = (st, -residency, str(p.processor_id))
            if best is None or key < best[0]:
                best = (key, p, st)
        return best[1], best[2]

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

    def set_node_state(self, state, current_idx):
        end = int(self.processors[0].epoch_length)
        for p in self.processors:
            if not hasattr(p, "_last_state_change_idx"):
                p._last_state_change_idx = 0
            # close previous state up to now
            prev = p._last_state_change_idx
            if current_idx > prev:
                p.update_state_log(prev, current_idx - prev, p.state)
            # start new state from now → end
            p.state = state
            p.update_state_log(int(current_idx), end - int(current_idx), state)
            p._last_state_change_idx = int(current_idx)

    def execute_task_on_available_processor(self, model_type, num_tokens, batch_size, migration_latency):
        best_processor = None
        best_exec_time = None
        best_time_index = float("inf")

        for processor in self.processors:
            try:
                # derive per-chunk baseline (ms per base_token_size) for this model
                perf = getattr(processor, "performance_metrics", {}) or {}
                base_ms = float(perf.get(f"{model_type}_Process", perf.get("process_ms_per_base", 2.0)))

                # estimate load seconds (only matters if the proc is OFF or model not resident)
                load_seconds = 0.0
                state_off = getattr(processor, "PROCESSOR_STATE_OFF", "OFF")
                node_loaded = getattr(self, "loaded_models", set())
                proc_loaded = getattr(processor, "loaded_models", set())
                needs_load = (getattr(processor, "state", None) == state_off) or (
                            model_type not in node_loaded and model_type not in proc_loaded)

                if needs_load:
                    # prefer a hint if you have one; otherwise 0.0 is fine for scoring
                    load_seconds = float(perf.get("model_load_seconds", 0.0))

                # === Correct call signature (SHORT form via kwargs) ===
                est_exec_time, _ = processor.compute_execution_metrics_wrapper(
                    model_type=model_type,
                    base_ms=base_ms,
                    num_tokens=int(num_tokens),
                    batch_size=int(batch_size),
                    load_seconds=float(load_seconds),
                    migration_latency=float(migration_latency),
                )
            except Exception:
                continue  # skip this processor on estimation failure

            if est_exec_time is None or est_exec_time <= 0:
                continue

            # choose earliest-available processor (use next_available_time if available)
            current_time = int(getattr(processor, "next_available_time", 0))
            if current_time < best_time_index:
                best_processor = processor
                best_exec_time = est_exec_time
                best_time_index = current_time

        if best_processor:
            final_exec_time, success = best_processor.execute_task(
                model_type, int(num_tokens), int(batch_size), self, float(migration_latency)
            )
            if success:
                return final_exec_time, best_processor.processor_id

        return None, None

    def report_epoch_stats(self, cop):
        total_processor_energy = 0
        processor_stats = []
        total_ttft = 0.0
        ttft_count = 0

        for p in self.processors:
            stats = p.report_epoch_stats()
            total_processor_energy += stats["total_energy_kWh"]
            processor_stats.append(stats)

            if stats["avg_ttft_second"] > 0:
                total_ttft += stats["avg_ttft_second"]
                ttft_count += 1

        other_hardware_energy = 0.13 * total_processor_energy
        cooling_energy = 3 * (total_processor_energy / cop) if cop > 0 else 0
        total_node_energy = total_processor_energy + other_hardware_energy + cooling_energy

        return {
            "node_id": self.node_id,
            "total_node_energy_kWh": total_node_energy,
            "processor_energy_kWh": total_processor_energy,
            "other_hardware_energy_kWh": other_hardware_energy,
            "cooling_energy_kWh": cooling_energy,
            "processors": processor_stats,
            "ttft_sum": total_ttft,
            "ttft_count": ttft_count
        }



class Datacenter:
    def __init__(
            self,
            datacenter_id,
            location,
            carbon_intensity,  # CSV "Carbon_Intensity"
            energy_cost,  # CSV "Time_of_Use(24_Hours)" (e.g., "0.1;0.1;...x24")
            nodes,
            # Legacy/optional CSV fields (accepted for compatibility):
            water_usage=None,
            cop_profile=None,  # "COP_Profile(24_Hours)"
            water_cycling_density=0.1,
            solids_ratio=0.3,
            potable_energy_intensity=0.005,
            wastewater_energy_intensity=0.01,
            **kwargs,  # swallow any other CSV columns without errors
    ):
        import numpy as np

        self.datacenter_id = datacenter_id
        self.location = location

        # Parse a 24h cost profile (semicolon/comma separated string → list[float] length 24)
        def _parse_profile(v, default_len=24, default_val=0.0):
            if v is None:
                return [default_val] * default_len
            if isinstance(v, (list, tuple, np.ndarray)):
                out = [float(x) for x in v]
            else:
                s = str(v).strip()
                if not s:
                    return [default_val] * default_len
                sep = ";" if ";" in s else ("," if "," in s else None)
                out = [float(x) for x in (s.split(sep) if sep else [s])]
            if len(out) < default_len:
                out = out + [out[-1]] * (default_len - len(out))
            elif len(out) > default_len:
                out = out[:default_len]
            return out

        self.energy_cost_profile = _parse_profile(energy_cost, default_len=24, default_val=0.0)
        # If your CSV is g/kWh, leave as float; if it's kg/kWh, remove any later /1000 conversion
        self.carbon_intensity = float(carbon_intensity)

        # Keep legacy fields (not used when PUE model is active, but accepted to avoid crashes)
        self.water_usage = float(water_usage) if water_usage not in (None, "") else 0.0
        self.cop_profile = _parse_profile(cop_profile, default_len=24,
                                          default_val=3.0) if cop_profile is not None else [3.0] * 24
        self.water_cycling_density = float(water_cycling_density)
        self.solids_ratio = float(solids_ratio)
        self.potable_energy_intensity = float(potable_energy_intensity)
        self.wastewater_energy_intensity = float(wastewater_energy_intensity)

        # Topology
        self.nodes = nodes
        self.loaded_models = set()
        self.loaded_model_to_nodes = {}

        # ---- Cooling (temperature-aware PUE) ----
        # If you already defined DEFAULT_COOLING_PROFILES earlier in the file, this import is not needed.

        self.cooling_cfg = DEFAULT_COOLING_PROFILES["air"]
        self.cooling_setpoint_c = 35.0  # knob (can be overridden after construction)

        # ---- Epoch accumulators (facility-based accounting) ----
        self._it_energy_kwh_epoch = 0.0
        self._facility_energy_kwh_epoch = 0.0
        self._cooling_energy_kwh_epoch = 0.0
        self._water_l_epoch = 0.0
        self._cost_usd_epoch = 0.0
        self._ttft_sum_epoch = 0.0
        self._ttft_cnt_epoch = 0

    @classmethod
    def load_datacenters_from_csv(
            cls,
            datacenter_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Datacenter_Specs.csv',
            node_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Node_Specs.csv'
    ):
        datacenters = []
        if not os.path.exists(datacenter_file):
            print(f"Warning: Datacenter configuration file {datacenter_file} missing.")
            return datacenters

        df = pd.read_csv(datacenter_file)
        all_node_templates = Node.load_nodes_from_csv(node_file)

        # IMPORTANT: In templates, node.node_id is the *type id*. Keep that mapping.
        node_type_map = {node.node_id: node for node in all_node_templates}

        for _, row in df.iterrows():
            dc_nodes = []
            dc_id = int(row["DC_Num"])
            total_nodes = int(row.get("Total_Nodes", 1000))  # default fallback

            # Parse Node_Types like "0:2;1:1;3:1"
            raw_types = str(row["Node_Types"]).split(";")
            type_weights = []
            for entry in raw_types:
                if not entry:
                    continue
                parts = entry.split(":")
                try:
                    type_id = int(parts[0])
                    weight = int(parts[1]) if len(parts) > 1 else 1
                except ValueError:
                    print(f"Warning: Bad Node_Types entry '{entry}' in DC {dc_id}")
                    continue
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
            assigned = sum(node_allocations.values())
            if assigned < total_nodes:
                # add remainder to the last listed type
                node_allocations[type_weights[-1][0]] += (total_nodes - assigned)

            node_counter = 0
            for node_type_id, count in node_allocations.items():
                base_node = node_type_map[node_type_id]
                # Keep epoch length consistent with template; fall back to 900
                base_epoch_len = (
                    base_node.processors[0].epoch_length if base_node.processors else 900
                )

                for _ in range(count):
                    new_processors = []
                    # Give each processor a fresh ID and fresh log/cursors
                    for p_idx, p in enumerate(base_node.processors):
                        processor_id = f"{dc_id}_{node_counter}_{p_idx}"
                        new_p = Processor(
                            processor_type=p.processor_type,
                            processor_id=processor_id,
                            performance_file=p.performance_file,
                            model_file=p.model_file,
                            epoch_length=base_epoch_len,
                        )
                        # Attach DC/node refs and initialize state/cursors
                        new_p.dc_id = dc_id
                        new_p.node_id = node_counter
                        new_p.state = Processor.PROCESSOR_STATE_OFF
                        new_p.next_available_time = 0.0
                        # state-change cursor used by set_datacenter_state()
                        new_p._last_state_change_idx = 0
                        # resource_usage_log is already a fresh np.zeros(...) per instance
                        new_processors.append(new_p)

                    new_node = Node(
                        node_id=node_counter,  # per-DC unique runtime node id
                        node_type=base_node.node_type,
                        inter_gpu_bandwidth=base_node.inter_gpu_bandwidth,
                        load_bandwidth_pcie=base_node.load_bandwidth_pcie,
                        load_delay_nvlink=base_node.load_delay_nvlink,
                        processors=new_processors,
                    )
                    new_node.dc_id = dc_id
                    # Preserve the template/type id so power plans can target types
                    new_node.type_id = node_type_id

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
                nodes=dc_nodes,
            ))

        return datacenters

    def reset_epoch(self):
        self._it_energy_kwh_epoch = 0.0
        self._facility_energy_kwh_epoch = 0.0
        self._cooling_energy_kwh_epoch = 0.0
        self._water_l_epoch = 0.0
        self._cost_usd_epoch = 0.0
        self._ttft_sum_epoch = 0.0
        self._ttft_cnt_epoch = 0
        for n in self.nodes:
            for p in n.processors:
                p.reset_epoch_stats()

    def set_datacenter_state(self, node_type_plan: dict, current_idx: int):
        # Short-circuit
        if not self.nodes:
            return

        # Derive epoch length and Processor constants
        first_p = self.nodes[0].processors[0]
        epoch_len = int(getattr(first_p, "epoch_length", 900))
        PROC = first_p.__class__
        STATE_OFF = getattr(PROC, "PROCESSOR_STATE_OFF", "OFF")
        STATE_IDLE = getattr(PROC, "PROCESSOR_STATE_IDLE", "IDLE")
        STATE_ON = getattr(PROC, "PROCESSOR_STATE_ON", STATE_IDLE)  # alias

        # Normalize external labels -> internal constants
        def _norm_state(label):
            s = str(label).strip().lower()
            if s in ("idle", "on", "ready"):
                return STATE_IDLE
            if s in ("off", "powered_off", "down"):
                return STATE_OFF
            # If already using internal enum/constant, pass through
            return label

        for node in self.nodes:
            desired_label = node_type_plan.get(node.node_id, "Off")
            desired_state = _norm_state(desired_label)

            # Ensure residency containers exist
            if not hasattr(node, "loaded_models"):
                node.loaded_models = set()

            for p in node.processors:
                if not hasattr(p, "_last_state_change_idx"):
                    p._last_state_change_idx = 0
                if not hasattr(p, "loaded_models"):
                    p.loaded_models = set()

                prev_idx = int(p._last_state_change_idx)

                # Close the previous state interval from last change -> current_idx
                if current_idx > prev_idx:
                    try:
                        # Log whatever state was active for [prev_idx, current_idx)
                        p.update_state_log(prev_idx, current_idx - prev_idx, getattr(p, "state", STATE_OFF))
                    except Exception:
                        # Logging is best-effort; don't crash the scheduler
                        pass
                    p._last_state_change_idx = int(current_idx)

                # Apply the new state (use internal constants, not strings)
                if getattr(p, "state", None) != desired_state:
                    p.state = desired_state

            # ---------------- Residency policy ----------------
            if desired_state == STATE_IDLE:
                # "Preloaded" residency: mark models available without calling load_model()
                supported = set()
                for p in node.processors:
                    supported |= set(getattr(p, "model_metrics", {}).keys())
                # You may choose to keep this empty if you want real loads later:
                # if not supported: supported = {"Llama7b", "Llama70b"}

                # Seed residency on node and processors (executor/scheduler may check either)
                if supported:
                    node.loaded_models |= supported
                    for p in node.processors:
                        p.loaded_models |= node.loaded_models

            elif desired_state == STATE_OFF:
                # Powering off clears residency so future requests pay load/TTFT
                if node.loaded_models:
                    node.loaded_models.clear()
                for p in node.processors:
                    p.loaded_models.clear()


    def calculate_metrics(self, power_kwh, current_hour):
        price = self.energy_cost_profile[current_hour]
        # print(price)
        # print(self.energy_cost_profile)
        return {
            "energy_cost": power_kwh * price,
            "carbon_emitted": power_kwh * self.carbon_intensity,
            "water_used": power_kwh * self.water_usage
        }

    def _energy_cost_per_kwh(self, current_hour: int) -> float:
        idx = int(current_hour) % len(self.energy_cost_profile)
        return float(self.energy_cost_profile[idx])

    def schedule_request(
            self,
            model_type,
            num_tokens,
            batch_size,
            time_index,
            migration_latency,
            max_retries=None,
            request_id=None,
    ):
        """
        DC-local scheduler (no cross-DC moves).
        - Do NOT add warm-up penalties here; lower layers account for them.
        - Start time = max(arrival_t, processor.next_available_time).
        - Choose (full-fit this epoch) over (overflow), then earliest finish, then earliest start.
        - If no lane fully fits before epoch end, defer whole job to next epoch.
        """
        import math

        model_type = str(model_type)
        arrival_t = int(time_index)
        num_tokens = int(num_tokens)
        batch_size = int(batch_size)
        mig_lat = float(migration_latency) if migration_latency is not None else 0.0

        # Epoch length (derive from first processor safely)
        if self.nodes and self.nodes[0].processors:
            epoch_len = int(getattr(self.nodes[0].processors[0], "epoch_length", 900))
        else:
            epoch_len = 900

        # Optional: sticky model affinity inside this DC (helps reduce repeated warm-ups downstream)
        if not hasattr(self, "_model_affinity"):
            self._model_affinity = {}  # model_type -> (node_id:int, proc_id:str)
        aff_node_id, aff_proc_id = self._model_affinity.get(model_type, (None, None))

        def _estimate_exec_seconds_on(p):
            """
            Delegate to lower layers. They already include warm-ups (power/model) in exec time.
            Try wrapper first; fall back to canonical compute if needed.
            """
            # Prefer wrapper signature
            try:
                exec_seconds, _ = p.compute_execution_metrics_wrapper(
                    model_type=model_type,
                    num_tokens=num_tokens,
                    batch_size=batch_size,
                    migration_latency=float(mig_lat),
                )
                return float(exec_seconds) if exec_seconds is not None else None
            except Exception:
                pass

            # Fallback: canonical compute signature if available
            try:
                perf = getattr(p, "performance_metrics", {}) or {}
                base_ms = float(perf.get(f"{model_type}_Process", perf.get("process_ms_per_base", 2000.0)))
                base_token_size = int(perf.get("base_token_size", 200))
                default_batch_size = int(perf.get("batch_size", 1))
                mm_all = getattr(p, "model_metrics", {}) or {}
                mm = mm_all.get(model_type, {}) or {}
                model_param_mem = float(mm.get("Parameter_Mem", 13.0))
                model_mem_per_token = float(mm.get("Mem_per_Token", 0.262))
                processor_mem = float(perf.get("Mem_Size", 80.0))
                exec_seconds, _ = p.compute_execution_metrics(
                    base_ms, base_token_size, default_batch_size,
                    num_tokens, batch_size,
                    model_param_mem, model_mem_per_token, processor_mem,
                    0.0,  # load_seconds handled internally by lower layers; don’t add here
                    float(mig_lat),
                )
                return float(exec_seconds) if exec_seconds is not None else None
            except Exception:
                return None

        best = None  # (full_fit_flag, finish_or_epoch, start, affinity_penalty, node_id, proc_id_str, node, proc, exec_secs)

        # Scan all processors in this DC
        for node in self.nodes:
            for p in node.processors:
                # Start time = max(arrival, processor availability). No warm-up math here.
                st = int(max(arrival_t, int(getattr(p, "next_available_time", 0))))

                exec_secs = _estimate_exec_seconds_on(p)
                if not exec_secs or exec_secs <= 0:
                    continue

                exec_ticks = int(math.ceil(exec_secs))
                finish = st + exec_ticks

                # Prefer lanes that fully finish within this epoch
                full_fit_flag = 0 if finish <= epoch_len else 1

                # Affinity bias (0 = prefer sticky lane; helps lower layers avoid re-warm)
                proc_id_str = str(getattr(p, "processor_id", ""))
                affinity_penalty = 0 if (aff_node_id == int(node.node_id) and aff_proc_id == proc_id_str) else 1

                key = (full_fit_flag, min(finish, epoch_len), st, affinity_penalty, int(node.node_id), proc_id_str)
                if best is None or key < best[0]:
                    best = (key, node, p, st, exec_secs)

        # If nothing can even start before epoch end, defer whole job to next epoch
        if best is None:
            if not hasattr(self, "leftover_requests"):
                self.leftover_requests = []
            self.leftover_requests.append({
                "request_id": request_id,
                "model_type": model_type,
                "num_tokens": num_tokens,
                "batch_size": batch_size,
                "target_dc_id": self.datacenter_id,
                "time_index": 0,
                "original_time_index": arrival_t,
            })
            return {
                "scheduled": False,
                "datacenter_id": self.datacenter_id,
                "location": self.location,
                "model_type": model_type,
                "retry_attempts": 0,
                "leftover": True,
                "request_id": request_id,
            }

        # Commit the chosen lane and execute (lower layers handle warm-ups & residency)
        (_, node, proc, start_tick, exec_secs) = best

        # Optional: remember the sticky lane for this model
        self._model_affinity[model_type] = (int(node.node_id), str(getattr(proc, "processor_id", "")))

        final_exec_time, success = proc.execute_task(
            model_type, num_tokens, batch_size, node, float(mig_lat)
        )

        if not success:
            if not hasattr(self, "leftover_requests"):
                self.leftover_requests = []
            self.leftover_requests.append({
                "request_id": request_id,
                "model_type": model_type,
                "num_tokens": num_tokens,
                "batch_size": batch_size,
                "target_dc_id": self.datacenter_id,
                "time_index": 0,
                "original_time_index": arrival_t,
            })
            return {
                "scheduled": False,
                "datacenter_id": self.datacenter_id,
                "location": self.location,
                "model_type": model_type,
                "retry_attempts": 0,
                "leftover": True,
                "request_id": request_id,
            }

        return {
            "scheduled": True,
            "datacenter_id": self.datacenter_id,
            "location": self.location,
            "node_id": node.node_id,
            "processor_id": getattr(proc, "processor_id", None),
            "model_type": model_type,
            "execution_time": float(final_exec_time),
            "ttft_seconds": None,  # keep None if Processor tracks TTFT internally
            "model_already_loaded": True,  # after execution, lower layers keep residency state
            "retry_attempts": 0,
            "request_id": request_id,
        }

    def report_epoch_stats(self, current_hour: int):
        processor_stats_all = []
        ttft_total = 0.0
        ttft_events = 0
        total_it_kwh = 0.0

        for node in self.nodes:
            for p in node.processors:
                if hasattr(p, "report_epoch_stats"):
                    stats = p.report_epoch_stats()
                else:
                    stats = {}
                processor_stats_all.append(stats)
                total_it_kwh += stats.get("total_energy_kWh", 0.0)
                if stats.get("avg_ttft_second", 0.0) > 0:
                    ttft_total += stats["avg_ttft_second"]
                    ttft_events += 1

        pue = self.cooling_cfg.pue_at_setpoint(self.cooling_setpoint_c)
        water_int = self.cooling_cfg.water_intensity_at_setpoint(self.cooling_setpoint_c)

        facility_kwh = pue * total_it_kwh
        cooling_kwh = max(0.0, facility_kwh - total_it_kwh)

        cost = self._energy_cost_per_kwh(current_hour) * facility_kwh
        # If your carbon_intensity is g/kWh, convert to kg; if already kg/kWh, remove /1000.
        carbon_kg = (self.carbon_intensity * facility_kwh) / 1000.0
        water_l = water_int * facility_kwh

        num_processors = len(processor_stats_all)
        total_capacity_seconds = num_processors * 900
        total_active_seconds = sum(p.get("active_seconds", 0.0) for p in processor_stats_all)
        load_ratio = (total_active_seconds / total_capacity_seconds) if total_capacity_seconds > 0 else 0.0

        self._it_energy_kwh_epoch = total_it_kwh
        self._facility_energy_kwh_epoch = facility_kwh
        self._cooling_energy_kwh_epoch = cooling_kwh
        self._water_l_epoch = water_l
        self._cost_usd_epoch = cost
        self._ttft_sum_epoch = ttft_total
        self._ttft_cnt_epoch = ttft_events

        return {
            "datacenter_id": self.datacenter_id,
            "location": self.location,
            "total_energy_kWh": facility_kwh,
            "total_cost_usd": cost,
            "carbon_emitted_kg": carbon_kg,
            "water_used_liters": water_l,
            "processor_stats": processor_stats_all,
            "ttft_total": ttft_total,
            "ttft_events": ttft_events,
            "load_ratio": load_ratio,
            "active_seconds": total_active_seconds,
            "capacity_seconds": total_capacity_seconds,
            "pue": pue,
            "setpoint_c": float(self.cooling_setpoint_c),
        }

import heapq
from typing import List, Dict, Tuple

class Geo_Network:
    _cached_instance = None
    def __init__(self, datacenters, latency_matrix, datacenter_ids, locations):
        self.datacenters = datacenters
        self.latency_matrix = latency_matrix
        self.datacenter_ids = datacenter_ids
        self.locations = locations
        self.id_to_index = {dc_id: i for i, dc_id in enumerate(datacenter_ids)}
        self.datacenter_map = {dc.datacenter_id: dc for dc in self.datacenters}

        self._ring_order = [0, 1, 4, 5, 3, 10, 11, 2, 9, 8, 7, 6]
        self._adj = self._build_ring_adjacency(self.datacenter_ids)

    # --- ring helpers ---
    def _build_ring_adjacency(self, dc_ids_list):
        order = [dc for dc in self._ring_order if dc in dc_ids_list]
        n = len(order)
        assert n >= 3, "Ring needs ≥ 3 DCs"
        adj: Dict[int, List[int]] = {dc: [] for dc in order}
        for i, dc in enumerate(order):
            adj[dc] = sorted({order[(i - 1) % n], order[(i + 1) % n]})
        return adj

    def _neighbors(self, a_id: int):
        return self._adj.get(a_id, [])

    def _edge_latency_ms(self, a_id: int, b_id: int) -> float:
        i = self.id_to_index[a_id];
        j = self.id_to_index[b_id]
        return float(self.latency_matrix[i][j])

    # --- Dijkstra shortest path over ring edges ---
    def shortest_path_ms(self, src_id: int, dst_id: int) -> Tuple[List[int], float]:
        if src_id == dst_id:
            return [src_id], 0.0
        dist = {d: float("inf") for d in self.datacenter_ids}
        prev: Dict[int, int] = {}
        dist[src_id] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, src_id)]
        seen = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == dst_id:
                break
            for v in self._neighbors(u):
                nd = d + self._edge_latency_ms(u, v)
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[dst_id] == float("inf"):
            raise ValueError(f"No path between {src_id} and {dst_id}")
        # reconstruct
        path = [dst_id]
        cur = dst_id
        while cur != src_id:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path, dist[dst_id]

    def network_delay_seconds_shortest(self, source_id: int, target_id: int) -> int:
        _, total_ms = self.shortest_path_ms(source_id, target_id)
        return int(round(total_ms / 1000.0))


    @classmethod
    def load_network(cls,
                     latency_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Geo_Latencies.csv',
                     dc_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Datacenter_specs.csv',
                     node_file='/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Node_Specs.csv'):

        if cls._cached_instance is not None:
            return cls._cached_instance

        start = time.time()
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

        cls._cached_instance = cls(datacenters, latency_matrix, dc_ids, locations)
        # print(f"[Timer] Geo network loaded in {time.time() - start:.2f}s")
        return cls._cached_instance

    def reset_all(self):
        for dc in self.datacenters:
            dc.reset_epoch()

    def apply_power_plan(self, power_plan):
        if not power_plan:
            return

        for dc in self.datacenters:
            dc_plan = power_plan.get(dc.datacenter_id)
            if not dc_plan:
                continue

            # --- normalize keys to ints when possible ---
            norm_plan = {}
            for k, v in dc_plan.items():
                try:
                    k_int = int(k)
                except (TypeError, ValueError):
                    k_int = k
                norm_plan[k_int] = v

            # --- decide whether plan is by node type or node id ---
            dc_nodes = dc.nodes if hasattr(dc, "nodes") else []
            dc_type_ids = {getattr(n, "type_id", None) for n in dc_nodes}
            plan_uses_type_ids = any((k in dc_type_ids) for k in norm_plan.keys() if isinstance(k, int))

            if plan_uses_type_ids:
                # Expand type-level plan → per-node plan
                expanded = {n.node_id: norm_plan.get(getattr(n, "type_id", None), "Off") for n in dc_nodes}
            else:
                # Assume plan is by node_id; default unspecified nodes to Off
                expanded = {n.node_id: norm_plan.get(n.node_id, "Off") for n in dc_nodes}

            # --- infer current tick index ---
            current_idx = 0
            # Prefer a DC method if present
            if hasattr(dc, "current_time_index") and callable(getattr(dc, "current_time_index")):
                try:
                    current_idx = int(dc.current_time_index())
                except Exception:
                    pass
            # Else derive from processors' next_available_time
            else:
                try:
                    current_idx = min(
                        int(p.next_available_time)
                        for n in dc_nodes
                        for p in getattr(n, "processors", [])
                    )
                except ValueError:
                    current_idx = 0  # no processors
                except Exception:
                    current_idx = 0

            # --- apply without painting the whole epoch ---
            try:
                # Preferred: updated signature (plan, current_idx)
                dc.set_datacenter_state(expanded, current_idx)
            except TypeError:
                # Fallback to legacy signature (may blanket the epoch if not updated)
                dc.set_datacenter_state(expanded)

    def get_latency(self, source_id, target_id):
        i = self.id_to_index[source_id]
        j = self.id_to_index[target_id]
        return self.latency_matrix[i][j]

    def apply_schedule_plan(self, schedule_plan):
        # ensure leftover buffer exists
        if not hasattr(self, "leftover_request_arr"):
            self.leftover_request_arr = []

        route = self.route_request
        get = lambda r, k: r.get(k)

        def process_request(i):
            return route(
                get(i, "target_dc_id"),
                get(i, "model_type"),
                get(i, "num_tokens"),
                get(i, "batch_size"),
                get(i, "source_dc_id"),
                get(i, "time_index"),
            )

        t1 = time.time()
        results = []
        leftovers = []

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=4) as executor:  # tune as needed
            futures = [executor.submit(process_request, i) for i in schedule_plan]
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                # collect leftovers for next epoch
                if (not r.get("scheduled")) and r.get("leftover"):
                    # prefer explicit leftover_request if provided by route_request
                    lo = r.get("leftover_request")
                    if lo is None:
                        lo = {
                            "model_type": r.get("model_type"),
                            "num_tokens": int(r.get("num_tokens", 0)),
                            "batch_size": int(r.get("batch_size", 1)),
                            "source_dc_id": r.get("source_dc"),
                            "target_dc_id": r.get("target_dc"),
                            "time_index": 0,
                            "original_time_index": int(r.get("time_index", 0)),
                        }
                    leftovers.append(lo)

        # stash for the next epoch
        if leftovers:
            self.leftover_request_arr.extend(leftovers)
            print(f"[LEFTOVER] queued {len(leftovers)} request(s) for next epoch @ t=0")

        # (optional) timing
        # print(f"[Timer] total apply_schedule time (parallel): {time.time() - t1:.2f}s")
        return results

    def route_request(self, target_dc_id, model_type, num_tokens, batch_size, source_dc_id, time_index):
        target_dc = self.datacenter_map.get(target_dc_id)

        # NEW: shortest-path RTT across the ring, returned in SECONDS
        migration_latency = float(self.network_delay_seconds_shortest(source_dc_id, target_dc_id))

        if not target_dc:
            return {
                "scheduled": False,
                "error": f"Datacenter {target_dc_id} not found",
                "source_dc": source_dc_id,
                "target_dc": target_dc_id,
                "model_type": model_type,
                "num_tokens": int(num_tokens),
                "batch_size": int(batch_size),
                "time_index": int(time_index),
                "migration_latency": migration_latency,
            }

        res = target_dc.schedule_request(
            model_type, int(num_tokens), int(batch_size), int(time_index), migration_latency
        )

        # annotate with routing + original request fields (handy for leftovers/metrics)
        res["migration_latency"] = migration_latency
        res["source_dc"] = source_dc_id
        res["target_dc"] = target_dc_id
        res["model_type"] = model_type
        res["num_tokens"] = int(num_tokens)
        res["batch_size"] = int(batch_size)
        res["time_index"] = int(time_index)

        # if DC says it's a leftover, attach the exact envelope we want next epoch
        if (not res.get("scheduled")) and res.get("leftover"):
            res["leftover_request"] = {
                "model_type": model_type,
                "num_tokens": int(num_tokens),
                "batch_size": int(batch_size),
                "source_dc_id": source_dc_id,
                "target_dc_id": target_dc_id,
                "time_index": 0,  # next epoch arrival
                "original_time_index": int(time_index),
            }

        return res


    def report_global_stats(self, current_hour):
        total_energy = 0
        total_cost = 0
        total_carbon = 0
        total_water = 0
        water_carbon = 0
        total_ttft = 0.0
        total_ttft_events = 0
        dc_reports = []
        total_active = 0
        total_capacity = 0
        load_by_dc = []

        for dc in self.datacenters:
            stats = dc.report_epoch_stats(current_hour)
            total_energy += stats["total_energy_kWh"]
            total_cost += stats["total_cost_usd"]
            total_carbon += stats["carbon_emitted_kg"]
            total_water += stats["water_used_liters"]
            water_carbon += stats.get("water_carbon_kg", 0)
            total_ttft += stats.get("ttft_total", 0)
            total_ttft_events += stats.get("ttft_events", 0)
            total_active += stats.get("active_seconds", 0)
            total_capacity += stats.get("capacity_seconds", 0)
            load_by_dc.append({
                "datacenter_id": stats["datacenter_id"],
                "location": stats["location"],
                "load_ratio": stats.get("load_ratio", 0),
                "active_seconds": stats.get("active_seconds", 0),
                "capacity_seconds": stats.get("capacity_seconds", 0),
            })
            dc_reports.append(stats)

        avg_load_ratio = total_active / total_capacity if total_capacity > 0 else 0

        return {
            "total_energy_kWh": total_energy,
            "total_cost_usd": total_cost,
            "carbon_emitted_kg": total_carbon,
            "water_used_liters": total_water,
            "water_carbon_kg": water_carbon,
            "datacenter_stats": dc_reports,
            "avg_ttft": total_ttft / total_ttft_events if total_ttft_events > 0 else 0,
            "network_load": {
                "total_active_seconds": total_active,
                "total_capacity_seconds": total_capacity,
                "avg_load_ratio": avg_load_ratio,
                "per_datacenter": load_by_dc
            }
        }


import time


def main():
    if not os.path.exists(WORKLOAD_FILE):
        print("No workload found, generating synthetic trace...")
        generate_random_workload()

    workload_df = pd.read_csv(WORKLOAD_FILE)

    total_epochs = 10
    for epoch_idx in range(0,9):
        epoch_df = workload_df[workload_df["epoch"] == epoch_idx]
        print(f"\n--- Running Simulation for Epoch {epoch_idx} ---")

        start = time.time()
        schedule_plan, power_plan = my_scheduler(epoch_df, epoch_idx)
        stats, results, leftovers = LLM_Simulator(epoch_idx, workload_df, schedule_plan, power_plan)
        print(f"[Timer] Epoch {epoch_idx} took {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()