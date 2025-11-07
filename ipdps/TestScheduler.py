#!/usr/bin/env python3
# TestScheduler.py — diagnostic harness for updated LLM_Simulator (token-per-request path)
import os, json
from typing import Dict, List, Tuple
import pandas as pd

from Rate_Flow_Sim import LLM_Simulator  # expects updated simulator with run_epoch()

# =========================
# Config
# =========================
SPEC_DIR        = "sim_specs"                 # folder with CSV specs
WORKLOAD_CSV    = "simulator_ready_trace.csv" # aggregated workload: epoch, source_dc_id, model_type, num_tokens
N_EPOCHS_RUN    = 10                          # how many epochs to test per policy
DIAG_DIR        = "diagnostics_out"

os.makedirs(DIAG_DIR, exist_ok=True)

# =========================
# Workload loader (aggregate-safe, column-normalizing)
# =========================
def load_workload(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing workload file: {path}")
    df = pd.read_csv(path)

    # Normalize common column aliases -> canonical names we use below
    col_map = {
        "src_dc": "source_dc_id",
        "src": "source_dc_id",
        "model": "model_type",
        "tokens": "num_tokens",
        "total_tokens": "num_tokens",
        "epoch_idx": "epoch",
        "epoch_id": "epoch",
    }
    for k, v in col_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    needed = {"source_dc_id", "model_type", "num_tokens", "epoch"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Workload CSV missing required columns: {sorted(missing)}")

    # Aggregate in case upstream split rows exist
    g = df.groupby(["epoch", "source_dc_id", "model_type"], as_index=False)["num_tokens"].sum()
    g = g.sort_values(["epoch", "source_dc_id", "model_type"]).reset_index(drop=True)
    return g

# =========================
# Helpers against the new simulator
# =========================
def dc_ids_in_sim(sim: LLM_Simulator) -> List[int]:
    # simulator.datacenters is a dict {dc_id: Datacenter}
    return sorted(sim.datacenters.keys())

def get_dc_intensity_map_grams(sim: LLM_Simulator) -> Dict[int, float]:
    # grams CO2 per kWh from each DC
    out = {}
    for dc_id, dc in sim.datacenters.items():
        out[int(dc_id)] = float(getattr(dc, "carbon_intensity_g_per_kwh", 0.0))
    return out

def build_epoch_requests(ep_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-request dataframe expected by Geo_Network.apply_schedule_plan:
      columns: source_dc, model, arrival_ms
    We create one request per (source_dc_id, model_type) group (arrival=0ms).
    NOTE: num_tokens is currently unused by Datacenter (exec time comes from CSV perf).
    """
    data = []
    for _, r in ep_rows.iterrows():
        data.append({
            "source_dc": int(r["source_dc_id"]),
            "model": str(r["model_type"]),
            "arrival_ms": 0,  # simple single-burst arrival at start of epoch
        })
    return pd.DataFrame(data)

def choose_target_from_fractions(frac_map: Dict[int, float]) -> int:
    # Deterministic: pick argmax(frac), tie-break on smallest dc_id
    return sorted(frac_map.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

def per_row_schedule_map(sim: LLM_Simulator, ep_rows: pd.DataFrame, planner) -> Dict[int, int]:
    """
    Use a schedule builder that returns (src, model) -> {dc: frac}.
    Convert to a row-index -> target_dc map compatible with simulator's schedule 'map'.
    """
    plan_frac = planner(sim, ep_rows)  # Dict[(src, model)] -> {dc: frac}
    row_map: Dict[int, int] = {}
    for ridx, (_, r) in enumerate(ep_rows.iterrows()):
        key = (int(r["source_dc_id"]), str(r["model_type"]))
        targets = plan_frac.get(key, {key[0]: 1.0})
        tgt = choose_target_from_fractions(targets)
        row_map[ridx] = int(tgt)
    return row_map

# =========================
# Schedules (deterministic, expressed as fraction maps)
#   Each returns: Dict[(src_dc, model), Dict[target_dc, frac]]
# =========================
def build_local_schedule(sim: LLM_Simulator, rows: pd.DataFrame) -> Dict[Tuple[int, str], Dict[int, float]]:
    plan: Dict[Tuple[int, str], Dict[int, float]] = {}
    for _, r in rows.iterrows():
        src = int(r["source_dc_id"]); model = str(r["model_type"])
        plan[(src, model)] = {src: 1.0}
    return plan

def build_rr_schedule(sim: LLM_Simulator, rows: pd.DataFrame) -> Dict[Tuple[int, str], Dict[int, float]]:
    dcs = dc_ids_in_sim(sim)
    idx = {dc: i for i, dc in enumerate(dcs)}
    plan: Dict[Tuple[int, str], Dict[int, float]] = {}
    for _, r in rows.iterrows():
        src = int(r["source_dc_id"]); model = str(r["model_type"])
        j = (idx[src] + 1) % len(dcs)
        tgt = dcs[j]
        plan[(src, model)] = {tgt: 1.0}
    return plan

def build_allto_minCI_schedule(sim: LLM_Simulator, rows: pd.DataFrame) -> Dict[Tuple[int, str], Dict[int, float]]:
    ci_map = get_dc_intensity_map_grams(sim)
    min_dc = min(ci_map, key=lambda d: (ci_map[d], d))
    plan: Dict[Tuple[int, str], Dict[int, float]] = {}
    for _, r in rows.iterrows():
        src = int(r["source_dc_id"]); model = str(r["model_type"])
        plan[(src, model)] = {min_dc: 1.0}
    return plan

def build_uniform_schedule(sim: LLM_Simulator, rows: pd.DataFrame) -> Dict[Tuple[int, str], Dict[int, float]]:
    dcs = dc_ids_in_sim(sim)
    frac = 1.0 / float(len(dcs))
    fan = {dc: frac for dc in dcs}
    plan: Dict[Tuple[int, str], Dict[int, float]] = {}
    for _, r in rows.iterrows():
        src = int(r["source_dc_id"]); model = str(r["model_type"])
        plan[(src, model)] = dict(fan)
    return plan

# =========================
# Carbon checker from detailed results + CI map
# =========================
def recompute_carbon_grams_from_details(details: List[dict], ci_map_g: Dict[int, float]) -> float:
    e_per_dc: Dict[int, float] = {}
    for rec in details:
        dc = int(rec.get("target_dc", rec.get("dc_id", -1)))
        if dc < 0:
            continue
        e = float(rec.get("energy_kwh", 0.0))
        e_per_dc[dc] = e_per_dc.get(dc, 0.0) + e
    s = 0.0
    for dc_id, ekwh in e_per_dc.items():
        s += ekwh * float(ci_map_g.get(dc_id, 0.0))
    return s

# =========================
# Runner for one policy over N epochs
# =========================
def run_policy(
    sim: LLM_Simulator,
    name: str,
    workload: pd.DataFrame,
    schedule_builder,
    n_epochs: int,
    out_csv: str,
) -> Dict[str, float]:
    dcs = set(dc_ids_in_sim(sim))
    models = set(workload["model_type"].unique())
    print(f"\n===== {name} (first {n_epochs} epochs) =====")
    print(f"[TEST] DCs in workload subset (for schedules): {len(dcs)}")
    print(f"[TEST] Models: {', '.join(sorted(models))}\n")

    ci_map_g = get_dc_intensity_map_grams(sim)

    tot_ttft = 0.0
    tot_c = tot_w = tot_e = 0.0
    rows_out: List[Dict[str, object]] = []

    epochs = sorted(workload["epoch"].unique())[:n_epochs]

    # Simple determinism check on first 5 epochs for local-only
    if name.lower().startswith("local"):
        ok = True
        for ep in epochs[:5]:
            sub = workload[workload["epoch"] == ep]
            # build per-row schedule map two times
            reqs = build_epoch_requests(sub)
            plan_map_1 = per_row_schedule_map(sim, sub, build_local_schedule)
            plan_map_2 = per_row_schedule_map(sim, sub, build_local_schedule)

            m1, d1, _ = sim.run_epoch(ep, reqs, {"map": plan_map_1}, power_plan={})
            m2, d2, _ = sim.run_epoch(ep, reqs, {"map": plan_map_2}, power_plan={})
            if json.dumps(m1, sort_keys=True) != json.dumps(m2, sort_keys=True):
                ok = False
                break
        print("[DET] Determinism check on first 5 epochs (Local-only)")
        print("[DET] PASS" if ok else "[DET] FAIL")

    for i, ep in enumerate(epochs, 1):
        sub = workload[workload["epoch"] == ep]

        # Requests for this epoch + per-row target mapping
        reqs = build_epoch_requests(sub)
        plan_map = per_row_schedule_map(sim, sub, schedule_builder)
        schedule_plan = {"map": plan_map}

        metrics, details, _dc_usage = sim.run_epoch(ep, reqs, schedule_plan, power_plan={})

        # metrics (units: TTFT seconds, carbon grams, water m^3, energy kWh)
        ttft   = float(metrics.get("avg_ttft", 0.0))
        carbon = float(metrics.get("carbon_emissions", 0.0))
        water  = float(metrics.get("water_usage", 0.0))
        energy = float(metrics.get("total_energy", 0.0))
        energy_costs = float(metrics.get("energy_cost", 0.0))

        # independent carbon recompute
        sigma_e_ci = recompute_carbon_grams_from_details(details, ci_map_g)
        if carbon > 0.0 or sigma_e_ci > 0.0:
            rel_err = 0.0 if sigma_e_ci == 0 else abs((carbon - sigma_e_ci) / max(sigma_e_ci, 1e-12)) * 100.0
            if rel_err > 0.05:  # >0.05% relative error (tight)
                print(f"[WARN] Check failed: Carbon mismatch at epoch {ep}: reported={carbon:.6f}, Σ(E_dc*CI_dc)={sigma_e_ci:.6f}, rel_err={rel_err:.3f}%")

        print(f"[{i:02d}/{len(epochs)}] epoch={ep:6d}  TTFT={ttft:0.6f}s  C={carbon:.6f}  W={water:.6f}  E={energy:.6f} EC={energy_costs:.6f}")

        tot_ttft += ttft
        tot_c += carbon
        tot_w += water
        tot_e += energy

        rows_out.append({
            "epoch": ep,
            "ttft_s": ttft,
            "carbon_g": carbon,
            "water_m3": water,
            "energy_kwh": energy,
            "energy_costs": energy_costs,
        })

    # write CSV
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"\n[WROTE] {out_csv}")

    return {
        "ttft": tot_ttft,
        "carbon": tot_c,
        "water": tot_w,
        "energy": tot_e,
        "energy_costs": energy_costs,
    }

# =========================
# Latency scaling test (×3 off-diagonal) for RR
# =========================
def scale_offdiag_latency(lat: List[List[float]], scale: float) -> List[List[float]]:
    n = len(lat)
    out = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = lat[i][j] if i == j else lat[i][j] * scale
    return out

def run_latency_test(sim: LLM_Simulator, workload: pd.DataFrame, epochs: int) -> None:
    # base RR
    base = run_policy(sim, "Round-robin (src→src+1)", workload, build_rr_schedule, epochs,
                      os.path.join(DIAG_DIR, "test_Round-robin_srctosrc+1_10epochs.csv"))

    # scale the in-memory matrix used by Geo_Network (ring uses self.lat for edges)
    lat = sim.network.lat
    if isinstance(lat, list) and len(lat) > 0:
        sim.network.lat = scale_offdiag_latency(lat, 3.0)
        scaled = run_policy(sim, "RR (latency ×3 off-diag)", workload, build_rr_schedule, epochs,
                            os.path.join(DIAG_DIR, "test_RR_latencyx3_10epochs.csv"))

        base_sum = base["ttft"]
        scaled_sum = scaled["ttft"]
        status = "OK" if scaled_sum > base_sum + 1e-9 else "WARN"
        if status != "OK":
            print(f"[WARN] Check failed: Latency scaling did not increase TTFT enough: base={base_sum:.6f}, ×3.0={scaled_sum:.6f}")
        print(f"[LAT] base_ttft_sum={base_sum:.6f}, scaled_ttft_sum={scaled_sum:.6f}  —  {status}")
    else:
        print("[LAT] Skipped (simulator did not expose latency matrix)")

# =========================
# Main
# =========================
def main():
    # Build simulator (loads all CSV specs internally)
    sim = LLM_Simulator(spec_dir=SPEC_DIR, epoch_length=900, debug=True)

    print("=== DC inventory ===")
    for dc_id, dc in sim.datacenters.items():
        # 'units' is the new combined Node+Processor list (back-compat: .nodes property may alias)
        units = getattr(dc, "units", getattr(dc, "nodes", []))
        print(f"DC {dc_id}: units={len(units)}")

    # Load workload
    wl = load_workload(WORKLOAD_CSV)
    all_epochs = sorted(wl["epoch"].unique())
    if not all_epochs:
        raise RuntimeError("Workload has no epochs.")
    n_epochs = min(N_EPOCHS_RUN, len(all_epochs))
    wl_use = wl[wl["epoch"].isin(all_epochs[:n_epochs])].copy()

    # Policies
    totals_local   = run_policy(sim, "Local-only (src→src)",       wl_use, build_local_schedule,   n_epochs, os.path.join(DIAG_DIR, "test_Local-only_srctosrc_10epochs.csv"))
    totals_rr      = run_policy(sim, "Round-robin (src→src+1)",    wl_use, build_rr_schedule,      n_epochs, os.path.join(DIAG_DIR, "test_Round-robin_srctosrc+1_10epochs.csv"))
    totals_allto   = run_policy(sim, "All-to-one (src→minDC)",     wl_use, build_allto_minCI_schedule, n_epochs, os.path.join(DIAG_DIR, "test_All-to-one_srctominDC_10epochs.csv"))
    totals_uniform = run_policy(sim, "Uniform-split (src→all eq.)", wl_use, build_uniform_schedule, n_epochs, os.path.join(DIAG_DIR, "test_Uniform-split_srctoall_eq._10epochs.csv"))

    # Latency scaling smoke test
    print("\n[LAT] Latency scaling test on 10 RR epochs (×3.0 off-diag)")
    run_latency_test(sim, wl_use, n_epochs)

    # Tiny dashboard
    def fmt(t): return f"TTFTavg={t['ttft']/n_epochs:.6f}s  C={t['carbon']:.3f}  W={t['water']:.3f}  E={t['energy']:.3f}"
    print("\n=== Diagnostic Dashboard ===")
    print(f"Local-only    : {fmt(totals_local)}")
    print(f"Round-robin   : {fmt(totals_rr)}")
    print(f"All-to-one    : {fmt(totals_allto)}")
    print(f"Uniform-split : {fmt(totals_uniform)}")
    print("============================")

if __name__ == "__main__":
    main()







