#!/usr/bin/env python3
"""
memory_test_lahyper.py  —  find the row-count breaking point for LAHyper.

LAHyper's per-epoch memory is dominated by three things that all scale with the
workload row count:

  * clean_data            — the epoch DataFrame (one shared copy)
  * the schedule map      — build_schedule_map's {request_idx: dc_id} dict
  * run_epoch's details   — the per-request result list (~1 KB/request);
                            this is the big transient and the usual OOM cause

Phase 2 runs several run_epoch evaluations concurrently, so the transient is
multiplied by the worker count.  This script sweeps increasing row counts,
measures peak RSS at each, and extrapolates where peak memory crosses the RAM
available on THIS machine — both for a single epoch and for the concurrent
Phase-2 pattern.

HOW IT WORKS
------------
Each scale runs in its own subprocess (this script re-invoked with --worker).
That isolates OOM kills (a crash at one scale doesn't abort the sweep) and
guarantees memory is reclaimed between scales.  The parent detects an OOM kill
by the subprocess dying on SIGKILL (return code -9 / 137).

USAGE  (run in the project directory — needs the sim spec CSVs)
---------------------------------------------------------------
    python memory_test_lahyper.py --spec-dir sim_specs
    python memory_test_lahyper.py --spec-dir sim_specs --max-rows 16000000
    python memory_test_lahyper.py --spec-dir sim_specs --rows 1000000,4000000,8000000

It is read-only: it builds synthetic workloads and runs the sim; it never
writes to your sources.
"""

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd

# Mirrors PHASE2_MAX_WORKERS in LA_Hyper_DDQN.py — keep in sync.
PHASE2_CPU_CAP = 4
TAG = "MEMTEST"


# ──────────────────────────────────────────────────────────────────────────────
# Memory measurement helpers
# ──────────────────────────────────────────────────────────────────────────────
def system_ram_mb():
    """(total_mb, available_mb) for this machine."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 1024 / 1024, vm.available / 1024 / 1024
    except Exception:
        pass
    try:  # /proc/meminfo — works on Linux / WSL
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                info[k.strip()] = int(rest.strip().split()[0]) * 1024
        total = info.get("MemTotal", 0) / 1024 / 1024
        avail = info.get("MemAvailable", info.get("MemFree", 0)) / 1024 / 1024
        return total, avail
    except Exception:
        return 0.0, 0.0


def cur_rss_mb():
    """Current resident set size of this process, in MB."""
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * resource.getpagesize() / 1024 / 1024
    except Exception:
        # ru_maxrss is a high-water mark, not current, but a usable fallback.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def process_peak_rss_mb():
    """Lifetime peak RSS of this process, in MB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class PeakSampler(threading.Thread):
    """Background thread that records the max RSS seen while it runs."""
    def __init__(self, interval=0.04):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = cur_rss_mb()
        self._stop = False

    def run(self):
        while not self._stop:
            self.peak = max(self.peak, cur_rss_mb())
            time.sleep(self.interval)

    def stop(self):
        self._stop = True
        try:
            self.join(timeout=1.0)
        except Exception:
            pass
        self.peak = max(self.peak, cur_rss_mb())
        return self.peak


def adaptive_workers(n_rows):
    """Phase-2 worker count for a given epoch size — mirrors the adaptive
    concurrency logic added to milp_optimizer in LA_Hyper_DDQN.py."""
    base = min(max(1, os.cpu_count() or 1), PHASE2_CPU_CAP)
    if n_rows > 8_000_000:
        return 1
    if n_rows > 4_000_000:
        return min(base, 2)
    if n_rows > 1_000_000:
        return min(base, max(1, base // 2))
    return base


# ──────────────────────────────────────────────────────────────────────────────
# WORKER — measures one row-count scale (runs in its own subprocess)
# ──────────────────────────────────────────────────────────────────────────────
def emit(obj):
    """Stream one tagged JSON line so the parent sees progress even if the
    process is later OOM-killed mid-run."""
    print(f"{TAG} {json.dumps(obj)}", flush=True)


def build_workload(n_rows, models, dc_ids, epoch_length, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.integers(200, 2000, size=n_rows).astype(np.int64)
    return pd.DataFrame({
        "source_dc_id":    rng.choice(dc_ids, size=n_rows),
        "model_type":      rng.choice(models, size=n_rows),
        "base_num_tokens": base,
        "num_tokens":      (base * 25).astype(np.int64),
        "arrival_ms":      rng.uniform(0, epoch_length * 1000.0, size=n_rows),
    })


def build_schedule_plan(LAH, df, sim, dc_ids):
    """Build the schedule map the way LAHyper does, falling back to a
    memory-equivalent plain dict if the LAHyper builders are unavailable."""
    n = len(df)
    if LAH is not None:
        try:
            try:
                LAH._compute_dc_capacity(sim)
            except Exception:
                pass
            small, large = LAH._precompute_request_split(df)
            n_dc = len(dc_ids)
            uni = np.ones(n_dc, dtype=np.float32) / n_dc
            sliders = np.ones(n_dc, dtype=np.float32)
            return LAH.build_schedule_map(
                small, large, dc_ids, uni, uni, sliders, epoch_idx=0,
                token_counts=df["num_tokens"].to_numpy(dtype=np.float32),
                source_dc=df["source_dc_id"].to_numpy(dtype=np.int64),
                lat_matrix=None, dc_to_idx={int(d): i for i, d in enumerate(dc_ids)},
            )
        except Exception as e:
            emit({"phase": "schedule_map_fallback", "reason": str(e)})
    # memory-equivalent fallback
    return {"map": {i: int(dc_ids[i % len(dc_ids)]) for i in range(n)}}


def run_worker(n_rows, spec_dir, epoch_length, streaming=False):
    """Measure the LAHyper per-epoch memory path for one row count.

    streaming=True exercises run_epoch(collect_details=False) — the
    streaming-aggregation path that skips the per-request detail list."""
    _ce = {"collect_details": False} if streaming else {}
    total_mb, avail_mb = system_ram_mb()
    rss0 = cur_rss_mb()
    emit({"phase": "start", "rows": n_rows, "streaming": streaming,
          "ram_total_mb": round(total_mb, 1), "ram_available_mb": round(avail_mb, 1),
          "baseline_rss_mb": round(rss0, 1)})

    # Imports (heavy: torch via LA_Hyper).  RSS after import is the fixed
    # interpreter+library footprint, independent of row count.
    import Rate_Flow_Sim_v2 as RFS
    try:
        import LA_Hyper_DDQN as LAH
    except Exception as e:
        emit({"phase": "warn", "msg": f"LA_Hyper_DDQN import failed: {e}"})
        LAH = None
    rss_imported = cur_rss_mb()
    emit({"phase": "imported", "rss_mb": round(rss_imported, 1),
          "import_cost_mb": round(rss_imported - rss0, 1)})

    # Reference sim (for DC ids + model names).
    ref = RFS.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_length, debug=False)
    dc_ids = sorted(int(d) for d in ref.datacenters.keys())
    models = sorted({m for dc in ref.datacenters.values()
                     for u in getattr(dc, "units", [])
                     for m in getattr(u, "model_perf", {}).keys()}) or ["Llama7b"]
    del ref
    gc.collect()

    # 1) clean_data
    df = build_workload(n_rows, models, dc_ids, epoch_length)
    clean_data_mb = float(df.memory_usage(deep=True).sum()) / 1024 / 1024
    rss_df = cur_rss_mb()
    emit({"phase": "workload_built", "rows": n_rows,
          "clean_data_mb": round(clean_data_mb, 1), "rss_mb": round(rss_df, 1)})

    # 2) sim graph + schedule map
    sim = RFS.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_length, debug=False)
    rss_sim = cur_rss_mb()
    sched = build_schedule_plan(LAH, df, sim, dc_ids)
    power = {int(d): {"all": "IDLE"} for d in dc_ids}
    rss_sched = cur_rss_mb()
    emit({"phase": "plan_built", "rows": n_rows,
          "sim_graph_mb": round(rss_sim - rss_df, 1),
          "schedule_map_mb": round(rss_sched - rss_sim, 1),
          "rss_mb": round(rss_sched, 1)})

    # 3) single run_epoch — measure the per-request detail-list transient.
    sampler = PeakSampler()
    sampler.start()
    single_status = "ok"
    try:
        metrics, details, dc_usage = sim.run_epoch(0, df, sched, power, **_ce)
        n_details = len(details) if details is not None else 0
        del details, dc_usage
    except MemoryError:
        single_status = "MemoryError"
        n_details = 0
    except TypeError as e:
        single_status = ("deployed Rate_Flow_Sim_v2.py lacks the collect_details "
                          "patch") if "collect_details" in str(e) else "error: TypeError"
        n_details = 0
    except Exception as e:
        single_status = f"error: {type(e).__name__}"
        n_details = 0
    single_peak = sampler.stop()
    gc.collect()
    emit({"phase": "single_run_epoch", "rows": n_rows, "status": single_status,
          "peak_rss_mb": round(single_peak, 1),
          "run_epoch_transient_mb": round(single_peak - rss_sched, 1),
          "n_detail_records": n_details})

    # 4) concurrent run_epoch — the realistic Phase-2 pattern.  K worker
    #    threads, each with its own sim, all on the shared clean_data.
    K = adaptive_workers(n_rows)
    del sim, sched
    gc.collect()
    sampler = PeakSampler()
    sampler.start()
    conc_status = "ok"

    def _one_eval(_):
        s = RFS.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_length, debug=False)
        sp = build_schedule_plan(LAH, df, s, dc_ids)
        pp = {int(d): {"all": "IDLE"} for d in dc_ids}
        m, det, u = s.run_epoch(0, df, sp, pp, **_ce)
        n = len(det) if det is not None else 0
        del det, u, sp, s
        return n

    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=K) as ex:
            list(ex.map(_one_eval, range(K)))
    except MemoryError:
        conc_status = "MemoryError"
    except Exception as e:
        conc_status = f"error: {type(e).__name__}"
    conc_peak = sampler.stop()

    emit({"phase": "concurrent_run_epoch", "rows": n_rows, "status": conc_status,
          "workers": K, "peak_rss_mb": round(conc_peak, 1)})

    emit({"phase": "done", "rows": n_rows, "streaming": streaming,
          "ram_total_mb": round(total_mb, 1),
          "ram_available_mb": round(avail_mb, 1),
          "import_cost_mb": round(rss_imported - rss0, 1),
          "clean_data_mb": round(clean_data_mb, 1),
          "schedule_map_mb": round(rss_sched - rss_sim, 1),
          "single_peak_mb": round(single_peak, 1),
          "single_status": single_status,
          "concurrent_workers": K,
          "concurrent_peak_mb": round(conc_peak, 1),
          "concurrent_status": conc_status,
          "process_peak_rss_mb": round(process_peak_rss_mb(), 1)})


# ──────────────────────────────────────────────────────────────────────────────
# PARENT — sweeps row counts, each in its own subprocess
# ──────────────────────────────────────────────────────────────────────────────
def run_one_scale(n_rows, spec_dir, epoch_length, streaming=False):
    """Spawn a worker subprocess for one row count; return its parsed result."""
    cmd = [sys.executable, os.path.abspath(__file__),
           "--worker", str(n_rows),
           "--spec-dir", spec_dir,
           "--epoch-length", str(epoch_length)]
    if streaming:
        cmd.append("--streaming")
    records = []
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return {"rows": n_rows, "status": "TIMEOUT", "records": []}

    for line in proc.stdout.splitlines():
        if line.startswith(TAG + " "):
            try:
                records.append(json.loads(line[len(TAG) + 1:]))
            except Exception:
                pass

    done = next((r for r in records if r.get("phase") == "done"), None)
    rc = proc.returncode

    if done is not None and rc == 0:
        done["status"] = "OK"
        done["records"] = records
        return done

    # Non-zero exit / killed → classify.
    killed_oom = rc in (-9, 137)              # SIGKILL — almost always the OOM killer
    last = records[-1] if records else {}
    result = {"rows": n_rows, "records": records,
              "ram_total_mb": (records[0].get("ram_total_mb", 0.0) if records else 0.0),
              "ram_available_mb": (records[0].get("ram_available_mb", 0.0) if records else 0.0)}
    if killed_oom:
        result["status"] = "OOM-KILLED"
        result["died_at_phase"] = last.get("phase", "?")
    else:
        result["status"] = f"CRASHED (rc={rc})"
        result["died_at_phase"] = last.get("phase", "?")
        result["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-6:])
    return result


def fit_and_predict(xs, ys, target):
    """Linear fit y = a + b·x; return (a, b, x_at_target)."""
    if len(xs) < 2:
        return None
    b, a = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
    if b <= 0:
        return (a, b, None)
    return (a, b, (target - a) / b)


def run_sweep(row_counts, spec_dir, epoch_length, streaming=False):
    total_mb, avail_mb = system_ram_mb()
    mode = "STREAMING (collect_details=False)" if streaming else "buffered (default path)"
    print("═" * 80)
    print("  memory_test_lahyper.py  —  LAHyper row-count breaking point")
    print(f"  machine RAM: total {total_mb/1024:.1f} GB   available {avail_mb/1024:.1f} GB")
    print(f"  spec-dir={spec_dir}   mode={mode}")
    print(f"  sweep={[f'{n:,}' for n in row_counts]}")
    print("═" * 80)

    results = []
    for n in row_counts:
        print(f"\n→ scale {n:,} rows  (Phase-2 workers: {adaptive_workers(n)}) ...",
              flush=True)
        r = run_one_scale(n, spec_dir, epoch_length, streaming)
        results.append(r)
        st = r.get("status", "?")
        if st == "OK":
            print(f"   clean_data {r.get('clean_data_mb',0):,.0f} MB | "
                  f"schedule_map {r.get('schedule_map_mb',0):,.0f} MB | "
                  f"single peak {r.get('single_peak_mb',0):,.0f} MB "
                  f"({r.get('single_status')}) | "
                  f"concurrent x{r.get('concurrent_workers',0)} peak "
                  f"{r.get('concurrent_peak_mb',0):,.0f} MB "
                  f"({r.get('concurrent_status')})")
        else:
            print(f"   {st} — died during phase '{r.get('died_at_phase','?')}'")
            if r.get("stderr_tail"):
                print("   stderr tail:")
                for ln in r["stderr_tail"].splitlines():
                    print(f"     {ln}")
            print("   stopping sweep — larger scales would also fail.")
            break

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("RESULTS")
    print("═" * 80)
    hdr = f"  {'rows':>12} | {'clean_data':>11} | {'sched_map':>10} | " \
          f"{'single peak':>12} | {'concurrent peak':>16} | status"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    ok = []
    for r in results:
        if r.get("status") == "OK":
            ok.append(r)
            print(f"  {r['rows']:>12,} | {r.get('clean_data_mb',0):>9,.0f} MB | "
                  f"{r.get('schedule_map_mb',0):>8,.0f} MB | "
                  f"{r.get('single_peak_mb',0):>10,.0f} MB | "
                  f"{r.get('concurrent_peak_mb',0):>11,.0f} MB x{r.get('concurrent_workers',1)} | OK")
        else:
            print(f"  {r['rows']:>12,} | {'—':>11} | {'—':>10} | {'—':>12} | "
                  f"{'—':>16} | {r.get('status')}")

    if len(ok) < 2:
        print("\n  Not enough successful scales to extrapolate a breaking point.")
        if ok:
            print(f"  Largest measured OK scale: {ok[-1]['rows']:,} rows.")
        return results

    # Memory attribution at the largest successful scale.
    big = ok[-1]
    print("\n  Memory breakdown at the largest successful scale "
          f"({big['rows']:,} rows):")
    cd = big.get("clean_data_mb", 0)
    sm = big.get("schedule_map_mb", 0)
    tr = max(0.0, big.get("single_peak_mb", 0) - big.get("import_cost_mb", 0) - cd - sm)
    imp = big.get("import_cost_mb", 0)
    for label, val in [("interpreter + libraries (torch etc.)", imp),
                       ("clean_data DataFrame", cd),
                       ("schedule map", sm),
                       ("run_epoch per-request detail list", tr)]:
        print(f"    {label:42s} {val:>9,.0f} MB")
    print("    (the detail list is the transient that dominates at scale and "
          "is the usual OOM cause)")

    # Extrapolate breaking points against available RAM.
    xs = [r["rows"] for r in ok]
    avail = avail_mb if avail_mb > 0 else total_mb
    print(f"\n  Extrapolating against {avail/1024:.1f} GB available RAM:")

    s_fit = fit_and_predict(xs, [r["single_peak_mb"] for r in ok], avail)
    if s_fit and s_fit[2]:
        a, b, xmax = s_fit
        print(f"    single epoch       : peak ≈ {a:,.0f} MB + {b*1e6:,.1f} MB per "
              f"1M rows  →  breaks at ~{xmax:,.0f} rows")
    c_fit = fit_and_predict(xs, [r["concurrent_peak_mb"] for r in ok], avail)
    if c_fit and c_fit[2]:
        a, b, xmax = c_fit
        print(f"    Phase-2 concurrent : peak ≈ {a:,.0f} MB + {b*1e6:,.1f} MB per "
              f"1M rows  →  breaks at ~{xmax:,.0f} rows")
        safe = int(xmax * 0.85)
        print(f"\n  RECOMMENDATION: keep epochs under ~{safe:,} rows on this "
              f"machine (15% headroom).")
        print(f"  Your autoscaler cap is MAX_ROWS = 16,000,000 in "
              f"overnight_baseline_results.py /")
        print(f"  --autoscale-max-rows.  If {safe:,} < 16,000,000, lower that cap "
              f"to ~{safe:,}")
        print(f"  to stop the crashes — or add more RAM / reduce PHASE2_MAX_WORKERS.")

    crashed = [r for r in results if r.get("status") != "OK"]
    if crashed:
        first = crashed[0]
        print(f"\n  Observed: first failure at {first['rows']:,} rows "
              f"({first['status']}), during phase '{first.get('died_at_phase','?')}'.")
    return results


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="LAHyper memory breaking-point sweep.")
    ap.add_argument("--spec-dir", default="sim_specs")
    ap.add_argument("--epoch-length", type=int, default=900)
    ap.add_argument("--worker", type=int, default=None,
                    help="(internal) run as the worker for N rows.")
    ap.add_argument("--rows", default=None,
                    help="Comma-separated row counts to sweep (overrides default).")
    ap.add_argument("--max-rows", type=int, default=16_000_000,
                    help="Upper bound for the default sweep.")
    ap.add_argument("--streaming", action="store_true",
                    help="Measure run_epoch(collect_details=False) — the "
                         "streaming-aggregation path.")
    args = ap.parse_args()

    if args.worker is not None:
        try:
            run_worker(args.worker, args.spec_dir, args.epoch_length,
                       streaming=args.streaming)
        except MemoryError:
            emit({"phase": "fatal", "error": "MemoryError"})
            sys.exit(2)
        except Exception as e:
            emit({"phase": "fatal", "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
            sys.exit(1)
        return

    if args.rows:
        row_counts = sorted(int(x.strip()) for x in args.rows.split(",") if x.strip())
    else:
        default = [250_000, 500_000, 1_000_000, 2_000_000,
                   4_000_000, 8_000_000, 16_000_000]
        row_counts = [n for n in default if n <= args.max_rows]

    run_sweep(row_counts, args.spec_dir, args.epoch_length, streaming=args.streaming)


if __name__ == "__main__":
    main()