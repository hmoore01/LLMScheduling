#!/usr/bin/env python3
"""
Experiment Runner for LLM Simulator
===================================

Supports three experiment types:
1. SCALABILITY: Vary number of DCs and nodes/node types per DC
2. MISPREDICTION: Test multiple misprediction (error) rates
3. DISTRIBUTION: Compare even vs population-weighted request origin distributions

Usage:
    python run_experiments.py --experiment scalability --frameworks Helix NSGA2
    python run_experiments.py --experiment misprediction --error-rates 0.0 0.1 0.2 0.3
    python run_experiments.py --experiment distribution --frameworks Helix

Author: Auto-generated for LLM Simulation Framework
"""

from __future__ import annotations
import argparse
import subprocess
import os
import sys
import json
import itertools
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import shutil

# ==============================================================================
# Configuration Defaults
# ==============================================================================

DEFAULT_FRAMEWORKS = ["Helix", "NSGA2", "PerLLM", "Splitwise"]
DEFAULT_EPOCHS = 96
DEFAULT_TRACE = "simulator_ready_trace.csv"
DEFAULT_SPEC_DIR = "sim_specs"
DEFAULT_OUTPUT_DIR = "experiment_results"

# Scalability configurations
SCALABILITY_CONFIGS = {
    "small": {"num_dcs": 4, "nodes_per_dc": 100, "node_type_dist": {0: 20, 1: 20, 2: 20, 3: 20, 4: 10, 5: 10}},
    "medium": {"num_dcs": 8, "nodes_per_dc": 500, "node_type_dist": {0: 100, 1: 100, 2: 100, 3: 100, 4: 50, 5: 50}},
    "large": {"num_dcs": 12, "nodes_per_dc": 1000, "node_type_dist": {0: 167, 1: 167, 2: 167, 3: 167, 4: 166, 5: 166}},
}

# Misprediction rates to test
DEFAULT_ERROR_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Population weights for major regions (normalized)
# Based on approximate global internet user distribution
POPULATION_WEIGHTS = {
    0: 0.12,  # US East
    1: 0.10,  # US West
    2: 0.08,  # US Central
    3: 0.15,  # Europe West
    4: 0.10,  # Europe Central
    5: 0.05,  # Europe North
    6: 0.18,  # Asia Pacific (China region)
    7: 0.08,  # Asia Pacific (Japan/Korea)
    8: 0.06,  # Asia Pacific (Southeast)
    9: 0.04,  # South America
    10: 0.02,  # Middle East
    11: 0.02,  # Africa
}


# ==============================================================================
# Utility Functions
# ==============================================================================

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def timestamp_str() -> str:
    """Return current timestamp as string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_results_file(filepath: str) -> Dict[str, float]:
    """Parse a results .txt file into a dictionary."""
    results = {}
    if not os.path.exists(filepath):
        return results

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    results[key] = float(val)
                except ValueError:
                    results[key] = val
    return results


# ==============================================================================
# Datacenter Spec Modification
# ==============================================================================

def modify_dc_specs(
        spec_dir: str,
        num_dcs: int,
        node_type_counts: Dict[int, int],
        output_dir: str
) -> str:
    """
    Create modified DC specs with specified number of DCs and node counts.
    Returns path to the new spec directory.
    """
    new_spec_dir = os.path.join(output_dir, "sim_specs")
    ensure_dir(new_spec_dir)

    # Copy base specs
    for fname in ["Node_specs.csv", "Geo_Latencies.csv", "A100_specs.csv", "H100_specs.csv"]:
        src = os.path.join(spec_dir, fname)
        dst = os.path.join(new_spec_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)

    # Modify DC specs
    dc_specs_path = os.path.join(spec_dir, "Datacenter_specs.csv")
    if os.path.exists(dc_specs_path):
        df = pd.read_csv(dc_specs_path)

        # Filter to requested number of DCs
        df = df[df["DC_Num"] < num_dcs].copy()

        # Update node type counts
        counts_str = ";".join(f"{k}:{v}" for k, v in sorted(node_type_counts.items()))
        total_nodes = sum(node_type_counts.values())

        df["Node_Type_Counts"] = counts_str
        df["Total_Nodes"] = total_nodes

        # If we need more DCs than available, duplicate with offset
        if len(df) < num_dcs:
            base_df = df.copy()
            while len(df) < num_dcs:
                add_df = base_df.copy()
                offset = len(df)
                add_df["DC_Num"] = add_df["DC_Num"] + offset
                df = pd.concat([df, add_df], ignore_index=True)
            df = df[df["DC_Num"] < num_dcs]

        df.to_csv(os.path.join(new_spec_dir, "Datacenter_specs.csv"), index=False)

    # Modify latency matrix for new DC count
    lat_path = os.path.join(spec_dir, "Geo_Latencies.csv")
    if os.path.exists(lat_path):
        lat_df = pd.read_csv(lat_path)
        # Ensure matrix is square for num_dcs
        if len(lat_df) >= num_dcs:
            lat_df = lat_df.iloc[:num_dcs, :num_dcs + 1]  # +1 for label column
        else:
            # Extend matrix with synthetic latencies
            while len(lat_df) < num_dcs:
                new_row = lat_df.iloc[len(lat_df) % len(lat_df)].copy()
                lat_df = pd.concat([lat_df, pd.DataFrame([new_row])], ignore_index=True)
            lat_df = lat_df.iloc[:num_dcs]

        lat_df.to_csv(os.path.join(new_spec_dir, "Geo_Latencies.csv"), index=False)

    return new_spec_dir


# ==============================================================================
# Trace Modification for Distribution Experiments
# ==============================================================================

def apply_population_distribution(
        trace_path: str,
        num_dcs: int,
        output_path: str,
        weights: Optional[Dict[int, float]] = None
) -> str:
    """
    Modify trace to use population-weighted source DC distribution.
    """
    df = pd.read_csv(trace_path)

    if weights is None:
        weights = POPULATION_WEIGHTS

    # Normalize weights to available DCs
    available_weights = {k: v for k, v in weights.items() if k < num_dcs}
    total = sum(available_weights.values())
    if total > 0:
        available_weights = {k: v / total for k, v in available_weights.items()}
    else:
        # Fallback to even distribution
        available_weights = {i: 1.0 / num_dcs for i in range(num_dcs)}

    # Assign source DCs based on weights
    dc_ids = list(available_weights.keys())
    dc_probs = list(available_weights.values())

    # Stratified assignment per epoch for consistency
    if "epoch" in df.columns:
        new_src_dcs = []
        for epoch_idx, group in df.groupby("epoch"):
            n = len(group)
            assigned = np.random.choice(dc_ids, size=n, p=dc_probs)
            new_src_dcs.extend(assigned)
        df["source_dc_id"] = new_src_dcs
    else:
        df["source_dc_id"] = np.random.choice(dc_ids, size=len(df), p=dc_probs)

    df.to_csv(output_path, index=False)
    return output_path


def apply_even_distribution(
        trace_path: str,
        num_dcs: int,
        output_path: str
) -> str:
    """
    Modify trace to use even (round-robin) source DC distribution.
    """
    df = pd.read_csv(trace_path)

    # Round-robin per epoch
    if "epoch" in df.columns:
        new_src_dcs = []
        for epoch_idx, group in df.groupby("epoch"):
            n = len(group)
            assigned = np.arange(n) % num_dcs
            new_src_dcs.extend(assigned)
        df["source_dc_id"] = new_src_dcs
    else:
        df["source_dc_id"] = np.arange(len(df)) % num_dcs

    df.to_csv(output_path, index=False)
    return output_path


# ==============================================================================
# Experiment Runners
# ==============================================================================

@dataclass
class ExperimentResult:
    """Container for experiment results."""
    experiment_type: str
    config_name: str
    framework: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    runtime_seconds: float = 0.0


def run_single_experiment(
        framework: str,
        num_dcs: int,
        num_epochs: int,
        error_rate: float = 0.0,
        trace_path: str = DEFAULT_TRACE,
        spec_dir: str = DEFAULT_SPEC_DIR,
        extra_args: Optional[List[str]] = None
) -> Tuple[Dict[str, float], float]:
    """
    Run a single experiment with the given configuration.
    Returns (metrics_dict, runtime_seconds).
    """
    import time

    cmd = [
        sys.executable, "simulator_LLM.py",
        "-f", framework,
        "--num-dcs", str(num_dcs),
        "-e", str(num_epochs),
        "--error-rate", str(error_rate),
    ]

    if extra_args:
        cmd.extend(extra_args)

    # Set environment for spec directory if modified
    env = os.environ.copy()
    if spec_dir != DEFAULT_SPEC_DIR:
        # The simulator reads from sim_specs by default; we need to symlink or copy
        if os.path.exists("sim_specs"):
            shutil.rmtree("sim_specs", ignore_errors=True)
        shutil.copytree(spec_dir, "sim_specs")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            env=env
        )

        if result.returncode != 0:
            print(f"  [WARNING] Command failed with return code {result.returncode}")
            print(f"  STDERR: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Experiment timed out after 1 hour")
        return {}, time.time() - start_time
    except Exception as e:
        print(f"  [ERROR] Exception during experiment: {e}")
        return {}, time.time() - start_time

    runtime = time.time() - start_time

    # Parse results
    results_file = f"LLM_Results/{framework}_final.txt"
    metrics = parse_results_file(results_file)

    return metrics, runtime


def run_scalability_experiment(
        frameworks: List[str],
        configs: Dict[str, Dict] = None,
        num_epochs: int = DEFAULT_EPOCHS,
        output_dir: str = DEFAULT_OUTPUT_DIR
) -> List[ExperimentResult]:
    """
    Run scalability experiments across different DC/node configurations.
    """
    if configs is None:
        configs = SCALABILITY_CONFIGS

    results = []
    exp_dir = ensure_dir(os.path.join(output_dir, f"scalability_{timestamp_str()}"))

    print("=" * 60)
    print("SCALABILITY EXPERIMENT")
    print("=" * 60)

    for config_name, config in configs.items():
        print(f"\n--- Configuration: {config_name} ---")
        print(f"    DCs: {config['num_dcs']}, Nodes/DC: {config['nodes_per_dc']}")

        # Modify specs for this configuration
        config_dir = ensure_dir(os.path.join(exp_dir, config_name))
        new_spec_dir = modify_dc_specs(
            DEFAULT_SPEC_DIR,
            config["num_dcs"],
            config["node_type_dist"],
            config_dir
        )

        for fw in frameworks:
            print(f"  Running {fw}...")

            metrics, runtime = run_single_experiment(
                framework=fw,
                num_dcs=config["num_dcs"],
                num_epochs=num_epochs,
                spec_dir=new_spec_dir
            )

            result = ExperimentResult(
                experiment_type="scalability",
                config_name=config_name,
                framework=fw,
                parameters={
                    "num_dcs": config["num_dcs"],
                    "nodes_per_dc": config["nodes_per_dc"],
                    "node_type_dist": config["node_type_dist"]
                },
                metrics=metrics,
                runtime_seconds=runtime
            )
            results.append(result)

            print(f"    TTFT: {metrics.get('Average TTFT (s)', 'N/A')}")
            print(f"    Carbon: {metrics.get('Total Carbon (g)', 'N/A')}")
            print(f"    Runtime: {runtime:.2f}s")

    # Save results
    save_results(results, os.path.join(exp_dir, "results.json"))
    save_results_csv(results, os.path.join(exp_dir, "results.csv"))

    return results


def run_misprediction_experiment(
        frameworks: List[str],
        error_rates: List[float] = None,
        num_dcs: int = 12,
        num_epochs: int = DEFAULT_EPOCHS,
        output_dir: str = DEFAULT_OUTPUT_DIR
) -> List[ExperimentResult]:
    """
    Run misprediction experiments with varying error rates.
    """
    if error_rates is None:
        error_rates = DEFAULT_ERROR_RATES

    results = []
    exp_dir = ensure_dir(os.path.join(output_dir, f"misprediction_{timestamp_str()}"))

    print("=" * 60)
    print("MISPREDICTION EXPERIMENT")
    print("=" * 60)

    for error_rate in error_rates:
        print(f"\n--- Error Rate: {error_rate:.0%} ---")

        for fw in frameworks:
            print(f"  Running {fw}...")

            metrics, runtime = run_single_experiment(
                framework=fw,
                num_dcs=num_dcs,
                num_epochs=num_epochs,
                error_rate=error_rate
            )

            result = ExperimentResult(
                experiment_type="misprediction",
                config_name=f"error_{error_rate:.2f}",
                framework=fw,
                parameters={
                    "error_rate": error_rate,
                    "num_dcs": num_dcs
                },
                metrics=metrics,
                runtime_seconds=runtime
            )
            results.append(result)

            print(f"    TTFT: {metrics.get('Average TTFT (s)', 'N/A')}")
            print(f"    Carbon: {metrics.get('Total Carbon (g)', 'N/A')}")

    # Save results
    save_results(results, os.path.join(exp_dir, "results.json"))
    save_results_csv(results, os.path.join(exp_dir, "results.csv"))

    return results


def run_distribution_experiment(
        frameworks: List[str],
        num_dcs: int = 12,
        num_epochs: int = DEFAULT_EPOCHS,
        trace_path: str = DEFAULT_TRACE,
        output_dir: str = DEFAULT_OUTPUT_DIR
) -> List[ExperimentResult]:
    """
    Run distribution experiments comparing even vs population-weighted.
    """
    results = []
    exp_dir = ensure_dir(os.path.join(output_dir, f"distribution_{timestamp_str()}"))

    print("=" * 60)
    print("DISTRIBUTION EXPERIMENT")
    print("=" * 60)

    distributions = {
        "even": {"apply_fn": apply_even_distribution, "weights": None},
        "population": {"apply_fn": apply_population_distribution, "weights": POPULATION_WEIGHTS},
    }

    for dist_name, dist_config in distributions.items():
        print(f"\n--- Distribution: {dist_name} ---")

        # Create modified trace
        dist_dir = ensure_dir(os.path.join(exp_dir, dist_name))
        modified_trace = os.path.join(dist_dir, "trace.csv")

        dist_config["apply_fn"](
            trace_path,
            num_dcs,
            modified_trace,
            **({"weights": dist_config["weights"]} if dist_name == "population" else {})
        )

        # Copy trace to expected location
        shutil.copy(modified_trace, "simulator_ready_trace.csv")

        for fw in frameworks:
            print(f"  Running {fw}...")

            metrics, runtime = run_single_experiment(
                framework=fw,
                num_dcs=num_dcs,
                num_epochs=num_epochs,
                trace_path=modified_trace
            )

            result = ExperimentResult(
                experiment_type="distribution",
                config_name=dist_name,
                framework=fw,
                parameters={
                    "distribution": dist_name,
                    "num_dcs": num_dcs,
                    "weights": dist_config["weights"] if dist_name == "population" else "even"
                },
                metrics=metrics,
                runtime_seconds=runtime
            )
            results.append(result)

            print(f"    TTFT: {metrics.get('Average TTFT (s)', 'N/A')}")
            print(f"    Carbon: {metrics.get('Total Carbon (g)', 'N/A')}")

    # Save results
    save_results(results, os.path.join(exp_dir, "results.json"))
    save_results_csv(results, os.path.join(exp_dir, "results.csv"))

    return results


# ==============================================================================
# Results Saving
# ==============================================================================

def save_results(results: List[ExperimentResult], filepath: str):
    """Save results to JSON."""
    data = []
    for r in results:
        data.append({
            "experiment_type": r.experiment_type,
            "config_name": r.config_name,
            "framework": r.framework,
            "parameters": r.parameters,
            "metrics": r.metrics,
            "runtime_seconds": r.runtime_seconds
        })

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[SAVED] Results to {filepath}")


def save_results_csv(results: List[ExperimentResult], filepath: str):
    """Save results to CSV for easy analysis."""
    rows = []
    for r in results:
        row = {
            "experiment_type": r.experiment_type,
            "config_name": r.config_name,
            "framework": r.framework,
            "runtime_seconds": r.runtime_seconds,
        }
        # Flatten parameters
        for k, v in r.parameters.items():
            if isinstance(v, dict):
                row[f"param_{k}"] = json.dumps(v)
            else:
                row[f"param_{k}"] = v
        # Flatten metrics
        for k, v in r.metrics.items():
            row[f"metric_{k.replace(' ', '_').replace('(', '').replace(')', '')}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"[SAVED] CSV to {filepath}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for LLM Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run scalability experiment with default configs
  python run_experiments.py --experiment scalability --frameworks Helix NSGA2

  # Run misprediction experiment with custom error rates
  python run_experiments.py --experiment misprediction --error-rates 0.0 0.1 0.2 0.3

  # Run distribution experiment
  python run_experiments.py --experiment distribution --frameworks Helix

  # Run all experiments
  python run_experiments.py --experiment all --frameworks Helix NSGA2 PerLLM Splitwise
        """
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["scalability", "misprediction", "distribution", "all"],
        help="Type of experiment to run"
    )

    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=DEFAULT_FRAMEWORKS,
        help=f"Frameworks to test (default: {DEFAULT_FRAMEWORKS})"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs to run (default: {DEFAULT_EPOCHS})"
    )

    parser.add_argument(
        "--num-dcs",
        type=int,
        default=12,
        help="Number of datacenters for misprediction/distribution experiments (default: 12)"
    )

    parser.add_argument(
        "--error-rates",
        nargs="+",
        type=float,
        default=DEFAULT_ERROR_RATES,
        help=f"Error rates for misprediction experiment (default: {DEFAULT_ERROR_RATES})"
    )

    parser.add_argument(
        "--scalability-configs",
        nargs="+",
        default=list(SCALABILITY_CONFIGS.keys()),
        choices=list(SCALABILITY_CONFIGS.keys()),
        help=f"Scalability configurations to test (default: all)"
    )

    parser.add_argument(
        "--trace",
        type=str,
        default=DEFAULT_TRACE,
        help=f"Input trace file (default: {DEFAULT_TRACE})"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
    )

    parser.add_argument(
        "--spec-dir",
        type=str,
        default=DEFAULT_SPEC_DIR,
        help=f"Spec directory (default: {DEFAULT_SPEC_DIR})"
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    all_results = []

    if args.experiment in ["scalability", "all"]:
        configs = {k: v for k, v in SCALABILITY_CONFIGS.items() if k in args.scalability_configs}
        results = run_scalability_experiment(
            frameworks=args.frameworks,
            configs=configs,
            num_epochs=args.epochs,
            output_dir=args.output_dir
        )
        all_results.extend(results)

    if args.experiment in ["misprediction", "all"]:
        results = run_misprediction_experiment(
            frameworks=args.frameworks,
            error_rates=args.error_rates,
            num_dcs=args.num_dcs,
            num_epochs=args.epochs,
            output_dir=args.output_dir
        )
        all_results.extend(results)

    if args.experiment in ["distribution", "all"]:
        results = run_distribution_experiment(
            frameworks=args.frameworks,
            num_dcs=args.num_dcs,
            num_epochs=args.epochs,
            trace_path=args.trace,
            output_dir=args.output_dir
        )
        all_results.extend(results)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total experiments run: {len(all_results)}")

    if all_results:
        # Group by experiment type
        by_type = {}
        for r in all_results:
            by_type.setdefault(r.experiment_type, []).append(r)

        for exp_type, exp_results in by_type.items():
            print(f"\n{exp_type.upper()}:")
            for r in exp_results:
                ttft = r.metrics.get("Average TTFT (s)", "N/A")
                carbon = r.metrics.get("Total Carbon (g)", "N/A")
                print(f"  {r.config_name}/{r.framework}: TTFT={ttft}, Carbon={carbon}")


if __name__ == "__main__":
    main()