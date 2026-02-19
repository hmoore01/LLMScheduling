import Rate_Flow_Sim
import numpy as np


def run_checks():
    print("=" * 80)
    print("=== PHYSICS EXPLOIT DETECTION SUITE ===")
    print("=" * 80)

    # Initialize Simulator
    try:
        sim = Rate_Flow_Sim.LLM_Simulator(spec_dir="sim_specs", epoch_length=900, debug=False)
    except Exception as e:
        print(f"CRITICAL: Could not load simulator. {e}")
        return

    # --- TEST 1: GHOST COMPUTING (Routing to OFF nodes) ---
    print("\n[TEST 1] 'Ghost Computing' Check")
    dc0 = sim.datacenters[0]
    dc0.apply_power_plan({"all": "OFF"})
    res = dc0.schedule_request(model="Llama7b", arrival=0, net_latency_ms=50, tokens=100)

    if res['ttft_s'] >= 50.0:
        print("✅ PASS: Request to OFF node penalized (TTFT > 50s).")
    else:
        print(f"🚨 FAIL: Request to OFF node processed instantly (TTFT={res['ttft_s']}s).")
        print("   -> Fix: Ensure 'schedule_request' returns a timeout if available_units is empty.")

    # --- TEST 2: ZOMBIE POWER (Free Idle Energy) ---
    print("\n[TEST 2] 'Zombie Power' Check (Do idle servers burn carbon?)")
    dc0.reset_epoch()
    dc0.apply_power_plan({"all": "ON"})  # Turn everything ON
    # Do NOT send any requests. Just finalize.
    dc0.finalize_epoch()

    energy_idle = dc0.energy_it_kwh + dc0.energy_other_kwh

    if energy_idle > 0.1:
        print(f"✅ PASS: Idle servers consume power ({energy_idle:.4f} kWh).")
    else:
        print(f"🚨 FAIL: Idle servers consume ZERO power ({energy_idle:.4f} kWh).")
        print("   -> Fix: Check 'finalize_epoch' logic for base_idle_frac calculation.")

    # --- TEST 3: THE 'CLOWN CAR' (Infinite Capacity) ---
    print("\n[TEST 3] 'Clown Car' Check (Infinite Capacity Exploit)")
    # This is the most dangerous exploit. Can 1 node handle 1 million requests instantly?
    dc0.reset_epoch()
    dc0.apply_power_plan({"all": "OFF"})

    # Enable exactly 1 node
    if dc0.units:
        dc0.units[0].state = "ON"
        target_node = dc0.units[0]
        # Calculate theoretical max capacity for 900s epoch
        # Assume 1000 tokens ~ 0.5s processing
        est_ms = target_node.estimate_exec_ms(tokens=1000, model="Llama7b", kwargs={})
        capacity = (900.0 * 1000.0) / max(1.0, est_ms)

        print(f"   Node Capacity: ~{int(capacity)} requests/epoch")

        # Send 100x Capacity
        overload_count = int(capacity * 100)
        print(f"   Sending {overload_count} requests to 1 node...")

        ttfts = []
        for i in range(100):  # Sample first 100
            r = dc0.schedule_request(model="Llama7b", arrival=i * 10, tokens=1000)
            ttfts.append(r['ttft_s'])

        # In a rate-flow sim without queuing, TTFT often stays flat even if utilization > 100%
        # This is a common simplification that RL agents abuse.
        avg_ttft = sum(ttfts) / len(ttfts)

        dc0.finalize_epoch()
        util = dc0.report_utilization()

        print(f"   Utilization: {util * 100:.1f}%")
        print(f"   Avg TTFT   : {avg_ttft:.4f}s")

        if util > 1.0 and avg_ttft < 1.0:
            print("🚨 FAIL: EXPLOIT CONFIRMED.")
            print("   The node handled 100x capacity with NO latency penalty.")
            print("   The RL agent will route EVERYTHING to a single node to save power.")
            print("   -> Fix: Add a 'Congestion Penalty' in the Reward Function or Simulator.")
        elif util > 1.0 and avg_ttft > 5.0:
            print("✅ PASS: Congestion correctly spikes latency.")
        else:
            print("⚠️ WARN: Utilization calculation might be capped or skewed.")

    # --- TEST 4: TELEPORTATION (Network Latency) ---
    print("\n[TEST 4] 'Teleportation' Check (Network Latency)")
    # Route from DC 0 to DC 11 (assuming they are far apart)
    src, dst = 0, 11
    if len(sim.datacenters) > 11:
        net_lat = sim.network._ring_path_latency_ms(src, dst)
        dc_dst = sim.datacenters[dst]
        dc_dst.apply_power_plan({"all": "ON"})

        res = dc_dst.schedule_request(
            model="Llama7b", arrival=0, net_latency_ms=net_lat, tokens=100
        )

        if res['ttft_s'] * 1000.0 >= net_lat:
            print(f"✅ PASS: Latency enforced ({res['ttft_s']:.3f}s >= {net_lat / 1000.0:.3f}s).")
        else:
            print(f"🚨 FAIL: Traffic moved faster than light ({res['ttft_s']:.3f}s < {net_lat / 1000.0:.3f}s).")
    else:
        print("   Skipping (Not enough DCs).")

    # --- TEST 5: COLD FUSION (Free Cooling) ---
    print("\n[TEST 5] 'Cold Fusion' Check (Cooling Overhead)")
    dc0.reset_epoch()
    dc0.apply_power_plan({"all": "ON"})
    # Send moderate load
    for i in range(100):
        dc0.schedule_request(model="Llama7b", arrival=i, tokens=1000)

    dc0.finalize_epoch()

    it_power = dc0.energy_it_kwh
    cooling_power = dc0.energy_cooling_kwh

    if cooling_power > 0 and cooling_power < it_power:  # Cooling shouldn't be 0, but usually less than IT
        print(f"✅ PASS: Cooling consumes energy ({cooling_power:.4f} kWh for {it_power:.4f} IT kWh).")
    elif cooling_power == 0:
        print("🚨 FAIL: Cooling is FREE (0 kWh). Check COP/PUE logic.")
    else:
        print(f"⚠️ NOTE: Cooling is higher than IT? ({cooling_power:.4f} > {it_power:.4f}). Check simulation specs.")


if __name__ == "__main__":
    run_checks()