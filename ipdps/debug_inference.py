#!/usr/bin/env python3
"""
Diagnostic Script for Synthetic Environment

This script verifies that:
1. The synthetic CSV files are properly formatted
2. The LLM_Simulator correctly loads the DC characteristics
3. Different DCs produce different metrics when routing to them

Run this BEFORE training to ensure the environment is set up correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

SPEC_DIR = "./sim_specs"  # Should contain Datacenter_specs.csv, etc.


def check_csv_files():
    """Verify all required CSV files exist and are properly formatted."""
    print("\n" + "=" * 60)
    print("STEP 1: Checking CSV Files")
    print("=" * 60)

    # Check for both standard and _synthetic suffix versions
    file_pairs = [
        ("Datacenter_specs.csv", "Datacenter_specs_synthetic.csv"),
        ("Geo_Latencies.csv", "Geo_Latencies_synthetic.csv"),
        ("Node_Specs.csv", None),
        ("A100_GPU.csv", None),
        ("H100_GPU.csv", None),
    ]

    all_ok = True
    actual_dc_file = None
    actual_latency_file = None

    for primary, alternate in file_pairs:
        path = os.path.join(SPEC_DIR, primary)
        alt_path = os.path.join(SPEC_DIR, alternate) if alternate else None

        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ {primary}: {len(df)} rows, columns: {list(df.columns)[:5]}...")
            if "Datacenter" in primary:
                actual_dc_file = alternate
            if "Latencies" in primary:
                actual_latency_file = alternate
        elif alt_path and os.path.exists(alt_path):
            df = pd.read_csv(alt_path)
            print(f"✓ {alternate}: {len(df)} rows, columns: {list(df.columns)[:5]}...")
            print(f"  ⚠️  NOTE: Using '{alternate}' instead of '{primary}'")
            print(f"      The simulator may not find this file automatically!")
            print(f"      Consider renaming to '{primary}'")
            if "Datacenter" in alternate:
                actual_dc_file = alternate
            if "Latencies" in alternate:
                actual_latency_file = alternate
        else:
            print(f"✗ {primary}: NOT FOUND!")
            all_ok = False

    # CRITICAL WARNING about file naming
    if actual_dc_file and "_synthetic" in actual_dc_file:
        print("\n" + "!" * 60)
        print("CRITICAL WARNING: Datacenter specs file has '_synthetic' suffix!")
        print(f"  Found: {actual_dc_file}")
        print(f"  Expected: Datacenter_specs.csv")
        print()
        print("The simulator looks for 'Datacenter_specs.csv' by default.")
        print("Your synthetic environment will NOT be loaded unless you:")
        print(f"  1. Rename '{actual_dc_file}' to 'Datacenter_specs.csv', OR")
        print("  2. Pass the correct filename to the simulator")
        print("!" * 60 + "\n")

    return all_ok, actual_dc_file, actual_latency_file


def check_dc_specs(dc_file=None):
    """Verify Datacenter_specs.csv has correct values for synthetic env."""
    print("\n" + "=" * 60)
    print("STEP 2: Checking DC Specifications")
    print("=" * 60)

    if dc_file is None:
        dc_file = "Datacenter_specs_synthetic.csv"

    path = os.path.join(SPEC_DIR, dc_file)
    if not os.path.exists(path):
        print(f"✗ Cannot find {dc_file}")
        return False

    df = pd.read_csv(path)

    print(f"\nFound {len(df)} datacenters:\n")

    expected = {
        0: {"Carbon_Intensity": 50.0, "name": "Carbon Best"},
        1: {"Carbon_Intensity": 500.0, "name": "Water Best"},
        2: {"Carbon_Intensity": 500.0, "name": "Cost Best"},
    }

    for _, row in df.iterrows():
        dc_num = int(row["DC_Num"])
        ci = float(row["Carbon_Intensity"])
        ws = float(row["Water_Static"])
        wcd = float(row["Water_Cycling_Density"])

        # Parse ToU prices
        tou_str = str(row["Time_of_Use(24_Hours)"])
        tou_prices = [float(x) for x in tou_str.split(";")]
        avg_price = np.mean(tou_prices)

        print(f"DC {dc_num} ({row['DC_Name']}):")
        print(f"  Carbon Intensity: {ci} g/kWh")
        print(f"  Water Static: {ws}")
        print(f"  Water Cycling: {wcd}")
        print(f"  Avg Energy Price: ${avg_price:.3f}/kWh")
        print()

    # Verify expected characteristics
    print("Expected Synthetic Environment:")
    print("  DC 0: LOWEST carbon (50 g/kWh)")
    print("  DC 1: LOWEST water (0.5 static, 0.01 cycling)")
    print("  DC 2: LOWEST cost ($0.02/kWh)")
    print()

    # Check if values match
    dc0 = df[df["DC_Num"] == 0].iloc[0]
    dc1 = df[df["DC_Num"] == 1].iloc[0]
    dc2 = df[df["DC_Num"] == 2].iloc[0]

    issues = []

    # DC 0 should have lowest carbon
    if dc0["Carbon_Intensity"] >= dc1["Carbon_Intensity"]:
        issues.append("DC 0 carbon NOT lower than DC 1!")
    if dc0["Carbon_Intensity"] >= dc2["Carbon_Intensity"]:
        issues.append("DC 0 carbon NOT lower than DC 2!")

    # DC 1 should have lowest water
    if dc1["Water_Static"] >= dc0["Water_Static"]:
        issues.append("DC 1 water NOT lower than DC 0!")
    if dc1["Water_Cycling_Density"] >= dc0["Water_Cycling_Density"]:
        issues.append("DC 1 water cycling NOT lower than DC 0!")

    # DC 2 should have lowest price
    tou0 = np.mean([float(x) for x in str(dc0["Time_of_Use(24_Hours)"]).split(";")])
    tou1 = np.mean([float(x) for x in str(dc1["Time_of_Use(24_Hours)"]).split(";")])
    tou2 = np.mean([float(x) for x in str(dc2["Time_of_Use(24_Hours)"]).split(";")])

    if tou2 >= tou0:
        issues.append(f"DC 2 price (${tou2:.3f}) NOT lower than DC 0 (${tou0:.3f})!")
    if tou2 >= tou1:
        issues.append(f"DC 2 price (${tou2:.3f}) NOT lower than DC 1 (${tou1:.3f})!")

    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✓ DC specifications look correct for synthetic environment!")
        return True


def check_latency_matrix(latency_file=None):
    """Verify latencies are equal (isolated variable)."""
    print("\n" + "=" * 60)
    print("STEP 3: Checking Latency Matrix")
    print("=" * 60)

    if latency_file is None:
        latency_file = "Geo_Latencies_synthetic.csv"

    path = os.path.join(SPEC_DIR, latency_file)
    if not os.path.exists(path):
        print(f"✗ Cannot find {latency_file}")
        return False

    df = pd.read_csv(path)

    print(f"\nLatency matrix ({len(df)}x{len(df.columns) - 1}):\n")
    print(df.to_string())
    print()

    # Check if all non-diagonal entries are equal
    latencies = []
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                col = str(j)
                if col in df.columns:
                    latencies.append(float(df.iloc[i][col]))

    unique_latencies = set(latencies)
    if len(unique_latencies) == 1:
        print(f"✓ All latencies are equal: {unique_latencies.pop()} ms")
        return True
    else:
        print(f"⚠️  Latencies vary: {unique_latencies}")
        print("   This may confuse the time_agent optimization!")
        return False


def test_simulator_routing():
    """Test that routing to different DCs produces different metrics."""
    print("\n" + "=" * 60)
    print("STEP 4: Testing Simulator Routing")
    print("=" * 60)

    try:
        from Rate_Flow_Sim import LLM_Simulator
    except ImportError:
        print("✗ Could not import LLM_Simulator")
        print("  Make sure Rate_Flow_Sim.py is in the current directory")
        return False

    # Create simulator - use debug=False to avoid broken debug prints in Rate_Flow_Sim
    print("\nInitializing LLM_Simulator...")
    sim = LLM_Simulator(spec_dir=SPEC_DIR, epoch_length=3600, debug=False)

    print(f"Loaded {len(sim.datacenters)} datacenters")

    # Print DC characteristics from simulator
    print("\nDC characteristics loaded by simulator:")
    for dc_id, dc in sim.datacenters.items():
        ci = getattr(dc, "carbon_intensity_g_per_kwh", "?")
        ws = getattr(dc, "water_static", "?")
        wcd = getattr(dc, "water_cycling_density", "?")
        tou = getattr(dc, "tou_price", None)
        avg_price = np.mean(tou) if tou is not None else "?"
        print(f"  DC {dc_id}: carbon={ci} g/kWh, water_static={ws}, water_cycling={wcd}, avg_price=${avg_price}")

    # Create LARGER test workload to see real differentiation
    # Use 500 requests with realistic token counts
    print("\nCreating larger test workload (500 requests)...")
    np.random.seed(42)
    test_rows = []
    for i in range(500):
        model = "Llama7b_FP16 (Base)_B1" if np.random.random() < 0.6 else "Llama70b_FP16 (Base)_B1"
        tokens = int(np.random.exponential(500) + 100)
        test_rows.append({
            "source_dc": 0,
            "model": model,
            "arrival_ms": i * 7000,  # Spread over epoch
            "tokens": tokens,
        })
    test_workload = pd.DataFrame(test_rows)
    total_tokens = test_workload["tokens"].sum()
    print(f"  Total requests: {len(test_workload)}")
    print(f"  Total tokens: {total_tokens:,}")

    results = {}

    # CRITICAL: Only turn ON the target DC, turn OFF others
    # This isolates the effect of routing decisions
    for target_dc in range(min(3, len(sim.datacenters))):
        print(f"\nRouting ALL traffic to DC {target_dc} (others OFF)...")

        # Route everything to target DC
        schedule = {"default_target_dc": target_dc}

        # Power plan: ONLY target DC is ON, others are OFF
        # This eliminates idle power from non-target DCs
        power = {}
        for dc_id in sim.datacenters.keys():
            if dc_id == target_dc:
                power[dc_id] = {"all": "ON"}
            else:
                power[dc_id] = {"all": "OFF"}

        # Reset simulator state
        for dc_id, dc in sim.datacenters.items():
            if hasattr(dc, 'reset_epoch'):
                dc.reset_epoch()  # Reinitialize for clean state

        metrics, details, usage = sim.run_epoch(0, test_workload, schedule, power)

        results[target_dc] = {
            "ttft": metrics.get("avg_ttft", 0),
            "carbon": metrics.get("carbon_emissions", 0),
            "water": metrics.get("water_usage", 0),
            "cost": metrics.get("energy_cost", 0),
            "energy": metrics.get("total_energy", 0),
        }

        print(f"  TTFT: {results[target_dc]['ttft']:.4f} s")
        print(f"  Carbon: {results[target_dc]['carbon']:.2f} g")
        print(f"  Water: {results[target_dc]['water']:.6f} m³")
        print(f"  Cost: ${results[target_dc]['cost']:.4f}")
        print(f"  Energy: {results[target_dc]['energy']:.2f} kWh")

    # Analyze results
    print("\n" + "-" * 40)
    print("ROUTING ANALYSIS:")
    print("-" * 40)

    if len(results) >= 3:
        # Find best DC for each metric
        best_carbon_dc = min(results.keys(), key=lambda x: results[x]["carbon"])
        best_water_dc = min(results.keys(), key=lambda x: results[x]["water"])
        best_cost_dc = min(results.keys(), key=lambda x: results[x]["cost"])

        print(f"\nBest DC for CARBON: DC {best_carbon_dc} ({results[best_carbon_dc]['carbon']:.2f} g)")
        print(f"Best DC for WATER:  DC {best_water_dc} ({results[best_water_dc]['water']:.6f} m³)")
        print(f"Best DC for COST:   DC {best_cost_dc} (${results[best_cost_dc]['cost']:.4f})")

        # Check if metrics differ significantly
        carbons = [r["carbon"] for r in results.values()]
        waters = [r["water"] for r in results.values()]
        costs = [r["cost"] for r in results.values()]

        carbon_min, carbon_max = min(carbons), max(carbons)
        water_min, water_max = min(waters), max(waters)
        cost_min, cost_max = min(costs), max(costs)

        carbon_diff = carbon_max / max(carbon_min, 0.001)
        water_diff = water_max / max(water_min, 0.0001)
        cost_diff = cost_max / max(cost_min, 0.001)

        print(f"\nMetric variation ratios (higher = more differentiation):")
        print(f"  Carbon: {carbon_diff:.2f}x  (range: {carbon_min:.0f} - {carbon_max:.0f} g)")
        print(f"  Water:  {water_diff:.2f}x  (range: {water_min:.4f} - {water_max:.4f} m³)")
        print(f"  Cost:   {cost_diff:.2f}x  (range: ${cost_min:.2f} - ${cost_max:.2f})")

        # For synthetic env, we expect:
        # - Carbon: 10x difference (50 vs 500 g/kWh)
        # - Water: 10x difference (0.5+0.01 vs 5.0+0.1)
        # - Cost: 5x difference ($0.02 vs $0.10)

        expected_carbon_ratio = 10.0
        expected_water_ratio = 10.0
        expected_cost_ratio = 5.0

        issues = []

        if carbon_diff < expected_carbon_ratio * 0.5:
            issues.append(f"Carbon ratio {carbon_diff:.1f}x is less than expected ~{expected_carbon_ratio}x")
        if water_diff < expected_water_ratio * 0.5:
            issues.append(f"Water ratio {water_diff:.1f}x is less than expected ~{expected_water_ratio}x")
        if cost_diff < expected_cost_ratio * 0.5:
            issues.append(f"Cost ratio {cost_diff:.1f}x is less than expected ~{expected_cost_ratio}x")

        if best_carbon_dc != 0:
            issues.append(f"Expected DC 0 best for carbon, got DC {best_carbon_dc}")
        if best_water_dc != 1:
            issues.append(f"Expected DC 1 best for water, got DC {best_water_dc}")
        if best_cost_dc != 2:
            issues.append(f"Expected DC 2 best for cost, got DC {best_cost_dc}")

        if issues:
            print("\n⚠️  ISSUES FOUND:")
            for issue in issues:
                print(f"   - {issue}")

            # Additional debugging
            print("\n[DEBUG] Checking if energy usage is the same across DCs...")
            energies = [r["energy"] for r in results.values()]
            if max(energies) - min(energies) < 0.01 * max(energies):
                print("  Energy is nearly identical across all DCs.")
                print("  This means the simulator computes carbon/water/cost")
                print("  based on the same energy regardless of DC characteristics.")
                print("\n  POSSIBLE CAUSES:")
                print("  1. All DCs have the same node configurations")
                print("  2. The synthetic CSV file names don't match what's being loaded")
                print("  3. There's a bug in how DC metrics are computed")

            return False
        else:
            print("\n✓ Synthetic environment produces expected metric differentiation!")
            print(f"  Carbon agent should prefer DC 0 (lowest carbon)")
            print(f"  Water agent should prefer DC 1 (lowest water)")
            print(f"  Cost agent should prefer DC 2 (lowest cost)")
            return True

    return False


def main():
    print("=" * 60)
    print("SYNTHETIC ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    print(f"Spec directory: {os.path.abspath(SPEC_DIR)}")

    csv_ok, actual_dc_file, actual_latency_file = check_csv_files()

    results = {
        "csv_files": csv_ok,
        "dc_specs": check_dc_specs(actual_dc_file),
        "latency": check_latency_matrix(actual_latency_file),
        "simulator": test_simulator_routing(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed

    if all_pass:
        print("\n✓ Synthetic environment is correctly configured!")
        print("  You can proceed with MARL training.")
    else:
        print("\n✗ Issues found! Fix them before training.")
        print("  The MARL agents will not learn properly with the current setup.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
