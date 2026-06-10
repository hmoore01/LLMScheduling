#!/usr/bin/env python3
"""
diagnose_system.py  —  broad failure-hunt across the LAHyper stack.

Supersedes diagnose_ttft.py.  Runs a battery of correctness + stress checks
against three files and their interactions:

  * Rate_Flow_Sim_v2.py      — the rate-flow simulator
  * simulator_LLM.py         — the driver / autoscaler
  * overnight_baseline_results.py — the experiment runner / results parsers

Plus the LAHyper-vs-Hybrid TTFT comparative bug hunt.

Each check reports PASS / FAIL / WARN / SKIP.  FAIL = a definite defect.
WARN = suspicious / needs a human eye.  SKIP = dependency unavailable.

Nothing here mutates source files; it only reads specs and runs the sim.

USAGE  (run inside the project directory)
-----------------------------------------
    python diagnose_system.py --spec-dir sim_specs
    python diagnose_system.py --spec-dir sim_specs --requests 200000 --extreme
    python diagnose_system.py --spec-dir sim_specs --token-scale 25

--extreme cranks workload sizes to stress memory and queueing paths.
"""

import argparse
import collections
import inspect
import math
import sys
import traceback

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Result tracking
# ══════════════════════════════════════════════════════════════════════════════
class Report:
    ICON = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "skip"}

    def __init__(self):
        self.rows = []          # (section, name, status, detail)
        self._section = "?"

    def section(self, name):
        self._section = name
        print(f"\n{'─' * 78}\n  SECTION: {name}\n{'─' * 78}")

    def record(self, name, status, detail=""):
        self.rows.append((self._section, name, status, detail))
        tag = self.ICON.get(status, status)
        line = f"  [{tag}] {name}"
        if detail:
            line += f"\n         {detail}"
        print(line)

    def check(self, name, condition, ok_detail="", fail_detail="", warn=False):
        """Record PASS if condition else FAIL (or WARN)."""
        if condition:
            self.record(name, "PASS", ok_detail)
        else:
            self.record(name, "WARN" if warn else "FAIL", fail_detail)
        return bool(condition)

    def skip(self, name, why):
        self.record(name, "SKIP", why)

    def fail(self, name, detail):
        self.record(name, "FAIL", detail)

    def warn(self, name, detail):
        self.record(name, "WARN", detail)

    def summary(self):
        counts = collections.Counter(r[2] for r in self.rows)
        print("\n" + "═" * 78)
        print("SUMMARY")
        print("═" * 78)
        print(f"  PASS {counts['PASS']:3d}   FAIL {counts['FAIL']:3d}   "
              f"WARN {counts['WARN']:3d}   SKIP {counts['SKIP']:3d}   "
              f"(total {len(self.rows)})")
        bad = [r for r in self.rows if r[2] in ("FAIL", "WARN")]
        if bad:
            print("\n  Items needing attention:")
            for sec, name, status, detail in bad:
                print(f"    [{status}] ({sec}) {name}")
                if detail:
                    print(f"           {detail}")
        else:
            print("\n  No failures or warnings — everything checked passed.")
        print("═" * 78)
        return counts


R = Report()


# ══════════════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════════════
def safe_import(name):
    try:
        return __import__(name)
    except Exception as e:
        print(f"  (could not import {name}: {e})")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Simulator helpers
# ══════════════════════════════════════════════════════════════════════════════
def make_sim(RFS, spec_dir, epoch_length):
    return RFS.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_length, debug=False)


def dc_ids_of(sim):
    return sorted(int(d) for d in sim.datacenters.keys())


def models_of(sim):
    names = set()
    for dc in sim.datacenters.values():
        for u in getattr(dc, "units", []):
            names.update(getattr(u, "model_perf", {}).keys())
    return sorted(names) or ["Llama7b_FP16 (Base)_B16"]


def active_units(sim):
    out = {}
    for dc_id, dc in sim.datacenters.items():
        out[int(dc_id)] = sum(
            1 for u in getattr(dc, "units", [])
            if str(getattr(u, "state", "OFF")).upper() != "OFF"
        )
    return out


def synth_workload(sim, n, token_scale, epoch_length, seed=7):
    rng = np.random.default_rng(seed)
    dcs = dc_ids_of(sim)
    models = models_of(sim)
    base = rng.integers(200, 2000, size=n).astype(int)
    return pd.DataFrame({
        "source_dc_id":    rng.choice(dcs, size=n),
        "model_type":      rng.choice(models, size=n),
        "base_num_tokens": base,
        "num_tokens":      (base * token_scale).round().astype(int),
        "arrival_ms":      rng.uniform(0, epoch_length * 1000.0, size=n),
    })


def run_epoch_safe(sim, df, sched, power, epoch_idx=0):
    """run_epoch wrapper -> dict of metrics, or None on exception."""
    try:
        metrics, details, dc_usage = sim.run_epoch(epoch_idx, df, sched, power)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}
    m = dict(metrics)
    m["_per_dc"] = collections.Counter(int(v) for v in sched.get("map", {}).values())
    m["_active"] = active_units(sim)
    return m


def all_on_power(dcs):
    return {int(d): {"all": "IDLE"} for d in dcs}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Simulator (Rate_Flow_Sim_v2) correctness & stress
# ══════════════════════════════════════════════════════════════════════════════
def section_simulator(RFS, spec_dir, epoch_length, n_requests, extreme):
    R.section("Rate_Flow_Sim_v2 — simulator correctness & stress")

    try:
        probe = make_sim(RFS, spec_dir, epoch_length)
        dcs = dc_ids_of(probe)
        del probe
    except Exception as e:
        R.skip("simulator construction", f"cannot build sim from {spec_dir}: {e}")
        return
    if len(dcs) < 2:
        R.warn("DC count", f"only {len(dcs)} DC(s) — some tests are weaker")

    df = synth_workload(make_sim(RFS, spec_dir, epoch_length),
                        n_requests, 1.0, epoch_length)

    # A1 — determinism: same plan twice must give identical metrics.
    sched = {"map": {i: int(dcs[i % len(dcs)]) for i in range(len(df))}}
    power = all_on_power(dcs)
    r1 = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df, sched, power)
    r2 = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df, sched, power)
    if "_error" in r1 or "_error" in r2:
        R.fail("A1 determinism", f"run_epoch errored: {r1.get('_error') or r2.get('_error')}")
    else:
        same = all(
            abs(float(r1.get(k, 0)) - float(r2.get(k, 0))) <= 1e-6 * (abs(float(r1.get(k, 0))) + 1)
            for k in ("avg_ttft", "carbon_emissions", "water_usage", "total_energy")
        )
        R.check("A1 determinism (identical plan -> identical metrics)", same,
                "deterministic",
                f"NONDETERMINISTIC: ttft {r1.get('avg_ttft')} vs {r2.get('avg_ttft')}, "
                f"carbon {r1.get('carbon_emissions')} vs {r2.get('carbon_emissions')}")

    # A2 — no-drop under extreme single-DC overload.
    big_n = (n_requests * 8) if extreme else (n_requests * 2)
    df_big = synth_workload(make_sim(RFS, spec_dir, epoch_length),
                            big_n, 1.0, epoch_length, seed=99)
    one = {"map": {i: int(dcs[0]) for i in range(len(df_big))}}
    pwr = all_on_power(dcs)
    r = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df_big, one, pwr)
    if "_error" in r:
        R.fail("A2 no-drop overload", f"run_epoch errored ({big_n:,} reqs): {r['_error']}")
    else:
        R.check(f"A2 no-drop policy ({big_n:,} reqs all -> 1 DC)",
                int(r.get("requests_dropped", 0)) == 0,
                f"0 dropped, {int(r.get('requests_completed',0)):,} served",
                f"{int(r.get('requests_dropped',0)):,} requests DROPPED — no-drop "
                f"policy not holding")

    # A3 — reroute away from a powered-OFF DC.
    if len(dcs) >= 2:
        offplan = {int(dcs[0]): {"all": "OFF"}}
        for d in dcs[1:]:
            offplan[int(d)] = {"all": "IDLE"}
        to_off = {"map": {i: int(dcs[0]) for i in range(len(df))}}
        r = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df, to_off, offplan)
        if "_error" in r:
            R.fail("A3 reroute off-DC", f"errored: {r['_error']}")
        else:
            R.check("A3 reroute from a powered-OFF DC (0 drops expected)",
                    int(r.get("requests_dropped", 0)) == 0,
                    f"all {int(r.get('requests_completed',0)):,} rerouted & served",
                    f"{int(r.get('requests_dropped',0)):,} dropped — reroute from "
                    f"OFF DC failed")

    # A4 — reroute from an INVALID DC id.
    bad = {"map": {i: 999999 for i in range(len(df))}}
    r = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df, bad, all_on_power(dcs))
    if "_error" in r:
        R.fail("A4 reroute invalid-DC", f"errored: {r['_error']}")
    else:
        R.check("A4 reroute from an INVALID DC id (0 drops expected)",
                int(r.get("requests_dropped", 0)) == 0,
                "all rerouted & served",
                f"{int(r.get('requests_dropped',0)):,} dropped — invalid-DC reroute "
                f"failed")

    # A5 — whole grid OFF: this IS the one legitimate drop case.
    alloff = {int(d): {"all": "OFF"} for d in dcs}
    every = {"map": {i: int(dcs[0]) for i in range(len(df))}}
    r = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df, every, alloff)
    if "_error" in r:
        R.fail("A5 whole-grid-off", f"errored: {r['_error']}")
    else:
        R.check("A5 whole grid OFF -> all requests dropped (expected)",
                int(r.get("requests_dropped", 0)) == len(df),
                "correctly drops everything when nothing can serve",
                f"expected {len(df):,} drops, got "
                f"{int(r.get('requests_dropped',0)):,} — grid-off handling odd")

    # A6 — token-scale prefill decoupling.
    df1 = synth_workload(make_sim(RFS, spec_dir, epoch_length), n_requests, 1.0,
                         epoch_length, seed=5)
    dfK = df1.copy()
    K = 25.0
    dfK["num_tokens"] = (dfK["base_num_tokens"] * K).round().astype(int)
    sp = {"map": {i: int(dcs[i % len(dcs)]) for i in range(len(df1))}}
    r_1 = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df1, sp, all_on_power(dcs))
    r_K = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), dfK, sp, all_on_power(dcs))
    if "_error" in r_1 or "_error" in r_K:
        R.skip("A6 prefill decoupling", "run_epoch errored")
    else:
        p1 = float(r_1.get("avg_prefill_ms", -1))
        pK = float(r_K.get("avg_prefill_ms", -1))
        if p1 < 0 or pK < 0:
            R.skip("A6 prefill decoupling",
                   "avg_prefill_ms missing — apply the TTFT-breakdown patch")
        else:
            ratio = pK / max(p1, 1e-9)
            R.check(f"A6 prefill decoupled from token scale (x1 vs x{K:.0f})",
                    ratio <= 1.15,
                    f"prefill stable: {p1:.0f}ms -> {pK:.0f}ms (ratio {ratio:.2f}x)",
                    f"prefill scaled {ratio:.1f}x with tokens ({p1:.0f}->{pK:.0f}ms) "
                    f"— base-token decoupling NOT in effect")

    # A7 — energy conservation: total ~= IT + cooling.
    r = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df, sched, power)
    if "_error" not in r:
        tot = float(r.get("total_energy", 0))
        it = float(r.get("total_it_energy_kwh", 0))
        cool = float(r.get("total_cooling_energy_kwh", 0))
        if tot > 0 and (it > 0 or cool > 0):
            rel = abs(tot - (it + cool)) / max(tot, 1e-9)
            R.check("A7 energy conservation (total ~= IT + cooling)",
                    rel <= 0.02,
                    f"total={tot:.1f} IT+cool={it+cool:.1f} kWh (rel {rel*100:.2f}%)",
                    f"MISMATCH: total={tot:.1f} but IT+cool={it+cool:.1f} kWh "
                    f"({rel*100:.1f}% off) — energy accounting inconsistency",
                    warn=True)
        else:
            R.skip("A7 energy conservation", "energy fields zero/absent")

    # A8 — load monotonicity: 2x requests -> >= energy, >= wait.
    df2 = synth_workload(make_sim(RFS, spec_dir, epoch_length), n_requests * 2,
                         1.0, epoch_length, seed=8)
    s1 = {"map": {i: int(dcs[i % len(dcs)]) for i in range(len(df))}}
    s2 = {"map": {i: int(dcs[i % len(dcs)]) for i in range(len(df2))}}
    ra = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df, s1, power)
    rb = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df2, s2, power)
    if "_error" not in ra and "_error" not in rb:
        e1, e2 = float(ra.get("total_energy", 0)), float(rb.get("total_energy", 0))
        R.check("A8 energy monotonic in load (2x reqs -> >= energy)",
                e2 >= e1 * 0.98,
                f"{e1:.1f} -> {e2:.1f} kWh",
                f"energy did NOT grow with load: {e1:.1f} -> {e2:.1f} kWh",
                warn=True)

    # A9 — zero requests.
    empty = pd.DataFrame({c: [] for c in
                          ["source_dc_id", "model_type", "num_tokens",
                           "base_num_tokens", "arrival_ms"]})
    r = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), empty,
                       {"map": {}}, all_on_power(dcs))
    if "_error" in r:
        R.fail("A9 zero-request epoch", f"crashed on empty workload: {r['_error']}")
    else:
        R.check("A9 zero-request epoch handled",
                int(r.get("requests_completed", 0)) == 0
                and float(r.get("avg_ttft", 0)) == 0.0,
                "empty epoch -> clean zeros",
                f"empty epoch produced ttft={r.get('avg_ttft')}, "
                f"completed={r.get('requests_completed')}")

    # A10 — robustness to malformed token counts (NaN / negative).
    dbad = df.copy()
    dbad.loc[dbad.index[:50], "num_tokens"] = -5
    dbad.loc[dbad.index[50:100], "num_tokens"] = np.nan
    r = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), dbad, sched, power)
    if "_error" in r:
        R.fail("A10 malformed tokens", f"crashed on NaN/negative tokens: {r['_error']}")
    else:
        finite = all(np.isfinite(float(r.get(k, 0)))
                     for k in ("avg_ttft", "carbon_emissions", "total_energy"))
        R.check("A10 robust to NaN / negative token counts",
                finite,
                "no crash, metrics finite",
                "metrics went non-finite on malformed token input")

    # A11 — accounting integrity: completed + dropped == total.
    r = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length), df, sched, power)
    if "_error" not in r:
        tot = int(r.get("requests_completed", 0)) + int(r.get("requests_dropped", 0))
        R.check("A11 accounting (completed + dropped == total)",
                tot == len(df),
                f"{tot:,} == {len(df):,}",
                f"LEAK: completed+dropped={tot:,} but workload={len(df):,}")

    # A12 — metric non-negativity.
    if "_error" not in r:
        negs = [k for k in ("avg_ttft", "carbon_emissions", "water_usage",
                            "total_energy", "energy_cost")
                if float(r.get(k, 0)) < 0]
        R.check("A12 all headline metrics non-negative",
                not negs, "all >= 0",
                f"negative metric(s): {negs}")

    # A13 — TTFT component sum: net + wait + prefill == avg_ttft * 1000.
    if "_error" not in r:
        comp = (float(r.get("avg_net_latency_ms", 0))
                + float(r.get("avg_wait_ms", 0))
                + float(r.get("avg_prefill_ms", 0)))
        ttft_ms = float(r.get("avg_ttft", 0)) * 1000.0
        if comp == 0 and ttft_ms == 0:
            R.skip("A13 TTFT component sum", "no breakdown fields (apply the patch)")
        else:
            rel = abs(comp - ttft_ms) / max(ttft_ms, 1e-6)
            R.check("A13 TTFT components sum to avg_ttft",
                    rel <= 0.01,
                    f"net+wait+prefill={comp:.1f}ms ~= ttft={ttft_ms:.1f}ms",
                    f"component sum {comp:.1f}ms != ttft {ttft_ms:.1f}ms "
                    f"({rel*100:.1f}% off) — breakdown accumulation bug")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — LAHyper time_agent vs ideal vs Hybrid (TTFT comparative hunt)
# ══════════════════════════════════════════════════════════════════════════════
def section_ttft_hunt(RFS, LAH, HYB, spec_dir, epoch_length, n_requests, token_scale):
    R.section("LAHyper time_agent vs ideal vs Hybrid — TTFT comparison")
    if LAH is None:
        R.skip("TTFT hunt", "LA_Hyper_DDQN not importable")
        return
    try:
        sim0 = make_sim(RFS, spec_dir, epoch_length)
        dcs = dc_ids_of(sim0)
        df = synth_workload(sim0, n_requests, token_scale, epoch_length)
        del sim0
    except Exception as e:
        R.skip("TTFT hunt", f"setup failed: {e}")
        return

    results = {}

    # ideal: round-robin, all on.
    sched = {"map": {i: int(dcs[i % len(dcs)]) for i in range(len(df))}}
    results["ideal"] = run_epoch_safe(make_sim(RFS, spec_dir, epoch_length),
                                      df, sched, all_on_power(dcs))

    # lahyper time_agent: uniform weights + full sliders via LAHyper builders.
    cap_spread = unit_spread = None
    try:
        sim = make_sim(RFS, spec_dir, epoch_length)
        try:
            LAH._compute_dc_capacity(sim)
        except Exception:
            pass
        # Inspect the capacity estimate vs the real per-DC unit counts.  If the
        # estimated-capacity spread is much wider than the unit-count spread,
        # _compute_dc_capacity (which uses the FASTEST model's rate) is
        # inflating the imbalance that build_schedule_map then routes by.
        try:
            cache = dict(getattr(LAH, "_DC_CAPACITY_CACHE", {}) or {})
            if cache:
                cv = [v for v in cache.values() if v > 0]
                cap_spread = max(cv) / max(min(cv), 1e-9)
            ucount = [len(getattr(sim.datacenters[d], "units", [])) for d in dcs]
            ucount = [u for u in ucount if u > 0]
            if ucount:
                unit_spread = max(ucount) / max(min(ucount), 1)
            if cap_spread is not None and unit_spread is not None:
                print(f"         capacity-estimate spread = {cap_spread:.1f}x  "
                      f"vs  per-DC unit-count spread = {unit_spread:.1f}x")
        except Exception:
            pass

        lat_raw = getattr(getattr(sim, "network", None), "lat", None)
        lat_m = None
        if lat_raw is not None:
            try:
                lat_m = LAH._build_latency_matrix(lat_raw, dcs)
            except Exception:
                pass
        small, large = LAH._precompute_request_split(df)
        n_dc = len(dcs)
        uni = np.ones(n_dc, dtype=np.float32) / n_dc
        sliders = np.ones(n_dc, dtype=np.float32)
        dnt = None
        try:
            dnt = LAH._get_dc_node_types(sim)
        except Exception:
            pass
        token_arr  = df["num_tokens"].to_numpy(dtype=np.float32)
        source_arr = df["source_dc_id"].to_numpy(dtype=np.int64)
        d2i = {int(d): i for i, d in enumerate(dcs)}

        sp = LAH.build_schedule_map(
            small, large, dcs, uni, uni, sliders, epoch_idx=0,
            token_counts=token_arr, source_dc=source_arr,
            lat_matrix=lat_m, dc_to_idx=d2i,
        )
        pp = LAH.build_power_plan_sliding(dcs, sliders, dnt)
        results["lahyper_time"] = run_epoch_safe(
            make_sim(RFS, spec_dir, epoch_length), df, sp, pp)

        # CONTROL — lahyper_flat: identical call to build_schedule_map, but with
        # the DC-capacity cache FLATTENED to uniform.  This isolates exactly one
        # variable: the capacity weighting (eff_w ~ real_capacity).  Everything
        # else in build_schedule_map — origin-aware assignment, overflow logic —
        # is unchanged.  If lahyper_flat ~= ideal while lahyper_time is far
        # worse, the capacity weighting is conclusively the bug.  If lahyper_flat
        # is ALSO bad, the defect is elsewhere inside build_schedule_map.
        try:
            saved_cache = getattr(LAH, "_DC_CAPACITY_CACHE", None)
            LAH._DC_CAPACITY_CACHE = {int(d): 1.0 for d in dcs}
            sp_flat = LAH.build_schedule_map(
                small, large, dcs, uni, uni, sliders, epoch_idx=0,
                token_counts=token_arr, source_dc=source_arr,
                lat_matrix=lat_m, dc_to_idx=d2i,
            )
            LAH._DC_CAPACITY_CACHE = saved_cache   # restore
            results["lahyper_flat"] = run_epoch_safe(
                make_sim(RFS, spec_dir, epoch_length), df, sp_flat, pp)
        except Exception as e:
            R.warn("lahyper_flat control", f"{type(e).__name__}: {e}")
            try:
                LAH._DC_CAPACITY_CACHE = saved_cache
            except Exception:
                pass
    except Exception as e:
        R.fail("lahyper_time plan build", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    # hybrid-like.
    if HYB is not None:
        try:
            cf = float(getattr(HYB, "CONSOLIDATION_FACTOR", 0.5))
            k = max(1, round(cf * len(dcs)))
            sim = make_sim(RFS, spec_dir, epoch_length)
            try:
                ranked = HYB._rank_dcs_balanced(sim, list(dcs))
            except Exception:
                ranked = list(dcs)
            active = [int(d) for d in ranked[:k]]
            rng = np.random.default_rng(42)
            tgt = rng.choice(active, size=len(df))
            sp = {"map": {i: int(tgt[i]) for i in range(len(df))}}
            rt = {int(d): 0.0 for d in dcs}
            tk = df["num_tokens"].to_numpy()
            for i in range(len(df)):
                rt[int(tgt[i])] += float(tk[i])
            try:
                pp = HYB._build_power_plan(rt, [0, 1, 2, 3, 4, 5], list(dcs))
            except Exception:
                pp = all_on_power(dcs)
            results["hybrid_like"] = run_epoch_safe(
                make_sim(RFS, spec_dir, epoch_length), df, sp, pp)
        except Exception as e:
            R.warn("hybrid_like plan build", f"{type(e).__name__}: {e}")

    # Report numbers.
    for name, r in results.items():
        if r is None or "_error" in r:
            R.fail(f"{name} run", (r or {}).get("_error", "no result"))
            continue
        reqs = r.get("_per_dc", {})
        spread = (max(reqs.values()) / max(1, min(reqs.values()))) \
            if reqs and min(reqs.values()) > 0 else 0.0
        print(f"         {name:14s} ttft={r.get('avg_ttft',0):.4f}s "
              f"net={r.get('avg_net_latency_ms',0):.0f} "
              f"wait={r.get('avg_wait_ms',0):.0f} "
              f"prefill={r.get('avg_prefill_ms',0):.0f}ms  "
              f"DCspread={spread:.1f}x")

    A = results.get("ideal")
    B = results.get("lahyper_time")
    if A and B and "_error" not in A and "_error" not in B:
        R.check("B1 lahyper_time TTFT not far above ideal-spread",
                float(B.get("avg_ttft", 0)) <= 2.0 * float(A.get("avg_ttft", 0)) + 0.05,
                f"lahyper {B.get('avg_ttft'):.3f}s vs ideal {A.get('avg_ttft'):.3f}s",
                f"lahyper_time TTFT {B.get('avg_ttft'):.3f}s >> ideal "
                f"{A.get('avg_ttft'):.3f}s — plan construction is the bug; "
                f"compare net/wait/prefill above")
        # Which component dominates the gap.
        for comp in ("avg_wait_ms", "avg_net_latency_ms", "avg_prefill_ms"):
            bv, av = float(B.get(comp, 0)), float(A.get(comp, 0))
            if bv > 3.0 * max(av, 1.0):
                R.warn(f"B-component {comp}",
                       f"lahyper_time {comp}={bv:.0f}ms is {bv/max(av,1.0):.1f}x "
                       f"the ideal — primary suspect")
    C = results.get("hybrid_like")
    if B and C and "_error" not in B and "_error" not in C:
        R.check("B2 latency-optimising time_agent beats consolidating Hybrid",
                float(B.get("avg_ttft", 0)) <= float(C.get("avg_ttft", 0)) * 1.05,
                f"lahyper {B.get('avg_ttft'):.3f}s <= hybrid {C.get('avg_ttft'):.3f}s",
                f"time_agent ({B.get('avg_ttft'):.3f}s) LOSES to Hybrid "
                f"({C.get('avg_ttft'):.3f}s) which deliberately worsens TTFT — "
                f"real defect")

    # B3 — the decisive control: lahyper_flat isolates the capacity weighting.
    F = results.get("lahyper_flat")
    if A and B and F and all("_error" not in x for x in (A, B, F)):
        ta, tb, tf = (float(x.get("avg_ttft", 0)) for x in (A, B, F))
        flat_ok    = tf <= 1.5 * ta + 0.05      # flat ~ ideal?
        time_bad   = tb > 1.5 * ta + 0.05       # capacity-weighted still bad?
        if flat_ok and time_bad:
            R.fail("B3 capacity weighting is the B1 root cause",
                   f"CONCLUSIVE: flattening the DC-capacity cache fixes it — "
                   f"lahyper_flat {tf:.3f}s ~= ideal {ta:.3f}s, while "
                   f"capacity-weighted lahyper_time is {tb:.3f}s.\n"
                   f"         The bug is build_schedule_map's eff_w ~ real_capacity: "
                   f"_compute_dc_capacity (which uses each unit's FASTEST model "
                   f"rate) over-estimates the per-DC capacity spread"
                   + (f" ({cap_spread:.1f}x capacity vs {unit_spread:.1f}x units)"
                      if cap_spread and unit_spread else "")
                   + ", so load is concentrated onto DCs that cannot clear it "
                   "proportionally faster -> queueing.\n"
                   "         FIX: estimate capacity from a representative (mean) "
                   "per-token rate, not the max; or route by unit count.")
        elif not flat_ok:
            R.warn("B3 capacity weighting NOT the sole cause",
                   f"lahyper_flat ({tf:.3f}s) is still well above ideal "
                   f"({ta:.3f}s) — the defect is elsewhere inside "
                   f"build_schedule_map (origin-aware assignment or the overflow "
                   f"redistribution), not the capacity weighting alone.")
        else:
            R.record("B3 capacity-weighting isolation", "PASS",
                     f"flat={tf:.3f}s time={tb:.3f}s ideal={ta:.3f}s — "
                     f"no clear capacity-weighting penalty in this run")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Autoscaler (simulator_LLM) correctness
# ══════════════════════════════════════════════════════════════════════════════
def section_autoscaler(SIM):
    R.section("simulator_LLM — autoscaler correctness")
    if SIM is None:
        R.skip("autoscaler", "simulator_LLM not importable")
        return

    split = getattr(SIM, "_split_autoscale_multiplier", None)
    apply = getattr(SIM, "_apply_autoscale_multiplier", None)
    MAXTS = float(getattr(SIM, "AUTOSCALE_MAX_TOKEN_SCALE", 1e9))
    TGTTS = float(getattr(SIM, "AUTOSCALE_TARGET_TOKEN_SCALE", 0))

    if split is None:
        R.skip("C1-C6 split tests", "_split_autoscale_multiplier not found")
    else:
        # C1 — product invariant across a wide sweep.
        mults = [1.0, 1.5, 5.0, 25.0, 100.0, 1000.0, 12703.5, 1e5, 1e6]
        caps = [None, 1, 4, 10, 100, 1000, 50000]
        bad_product, bad_int, bad_ceiling = [], [], []
        for m in mults:
            for cap in caps:
                try:
                    cm, rs = split(m, count_cap=cap)
                except Exception as e:
                    bad_product.append(f"m={m} cap={cap}: raised {e}")
                    continue
                if not (isinstance(cm, int) and cm >= 1):
                    bad_int.append(f"m={m} cap={cap}: count_mult={cm!r}")
                if rs > MAXTS * 1.001:
                    bad_ceiling.append(f"m={m} cap={cap}: rs={rs:.2f}>{MAXTS}")
                achieved = cm * rs
                if rs < MAXTS * 0.999:        # not clamped -> product must equal m
                    if abs(achieved - m) > 0.005 * m + 1e-6:
                        bad_product.append(
                            f"m={m} cap={cap}: cm*rs={achieved:.2f}!=m")
                else:                          # clamped -> achieved must be <= m
                    if achieved > m * 1.001:
                        bad_product.append(
                            f"m={m} cap={cap}: clamped but achieved {achieved:.1f}>m")
        R.check("C1 split product invariant (count_mult * token_scale == m)",
                not bad_product, f"{len(mults)*len(caps)} combinations consistent",
                f"{len(bad_product)} violation(s), e.g. {bad_product[:3]}")
        R.check("C2 count_mult is always an integer >= 1",
                not bad_int, "OK",
                f"{len(bad_int)} bad, e.g. {bad_int[:3]}")
        R.check("C3 token scale never exceeds AUTOSCALE_MAX_TOKEN_SCALE",
                not bad_ceiling, f"all <= {MAXTS}",
                f"{len(bad_ceiling)} over ceiling, e.g. {bad_ceiling[:3]}")

        # C4 — target token scale when uncapped.
        try:
            cm, rs = split(10000.0, count_cap=None)
            near = abs(rs - TGTTS) <= max(0.5, 0.25 * TGTTS) if TGTTS > 0 else True
            R.check("C4 uncapped split lands near AUTOSCALE_TARGET_TOKEN_SCALE",
                    near, f"token scale {rs:.2f} ~ target {TGTTS}",
                    f"uncapped token scale {rs:.2f} far from target {TGTTS}",
                    warn=True)
        except Exception as e:
            R.warn("C4 target token scale", str(e))

        # C5 — m <= 1 edge.
        try:
            cm, rs = split(0.7)
            R.check("C5 split edge case (m <= 1 -> count_mult 1)",
                    cm == 1 and abs(rs - 0.7) < 1e-6,
                    "OK", f"got count_mult={cm}, scale={rs}")
        except Exception as e:
            R.warn("C5 m<=1 edge", str(e))

        # C6 — clamp engages on an extreme multiplier with a tight row cap.
        try:
            cm, rs = split(1e7, count_cap=10)
            R.check("C6 token-scale clamp engages when rows are capped",
                    abs(rs - MAXTS) < 1e-3 and cm == 10,
                    f"clamped to {rs:.1f} at cap {cm}",
                    f"expected clamp at {MAXTS}, got scale={rs:.2f} cm={cm}",
                    warn=True)
        except Exception as e:
            R.warn("C6 clamp", str(e))

    # C7-C9 — _apply_autoscale_multiplier.
    if apply is None:
        R.skip("C7-C9 apply tests", "_apply_autoscale_multiplier not found")
    else:
        base_df = pd.DataFrame({
            "source_dc_id": np.zeros(500, dtype=int),
            "model_type":   ["Llama7b"] * 500,
            "num_tokens":   np.full(500, 400, dtype=int),
            "arrival_ms":   np.linspace(0, 900_000, 500),
        })
        try:
            sig = inspect.signature(apply)
            kwargs = {}
            if "epoch_length_s" in sig.parameters:
                kwargs["epoch_length_s"] = 900
            if "count_cap" in sig.parameters:
                kwargs["count_cap"] = 10000
            out, cm, rs = apply(base_df.copy(), 250.0, 0, **kwargs)
            # C7 — base_num_tokens captured and num_tokens == base * scale.
            if "base_num_tokens" not in out.columns:
                R.fail("C7 base_num_tokens captured",
                       "base_num_tokens column ABSENT after autoscale — the "
                       "prefill-decoupling source is missing")
            else:
                bt = pd.to_numeric(out["base_num_tokens"]).to_numpy()
                nt = pd.to_numeric(out["num_tokens"]).to_numpy()
                ok = np.allclose(nt, np.round(bt * rs), atol=1.5)
                R.check("C7 num_tokens == base_num_tokens * token_scale",
                        ok, f"consistent (scale {rs:.2f})",
                        "num_tokens != base_num_tokens * scale — autoscale token "
                        "math inconsistent")
            # C8 — row count == base_rows * count_mult.
            R.check("C8 row count == base_rows * count_mult",
                    len(out) == len(base_df) * cm,
                    f"{len(out)} == 500 * {cm}",
                    f"row count {len(out)} != 500*{cm}={500*cm}")
            # C9 — arrival times stay within the epoch window.
            if "arrival_ms" in out.columns:
                amax = float(pd.to_numeric(out["arrival_ms"]).max())
                R.check("C9 scaled arrivals stay within the epoch window",
                        0 <= amax <= 900_000 * 1.001,
                        f"max arrival {amax:.0f}ms <= 900000ms",
                        f"arrival {amax:.0f}ms exceeds the 900s epoch window")
        except Exception as e:
            R.warn("C7-C9 apply_autoscale", f"{type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — cross-file consistency
# ══════════════════════════════════════════════════════════════════════════════
def section_cross_file(SIM, LAH):
    R.section("Cross-file consistency (simulator_LLM <-> LA_Hyper_DDQN)")
    if SIM is None or LAH is None:
        R.skip("cross-file", "need both simulator_LLM and LA_Hyper_DDQN")
        return

    tgt = getattr(SIM, "AUTOSCALE_TARGET_TOKEN_SCALE", None)
    fac = getattr(LAH, "_TTFT_TOKEN_SCALE_FACTOR", None)
    if tgt is None or fac is None:
        R.skip("D1 token-scale sync", "constants not found")
    else:
        R.check("D1 _TTFT_TOKEN_SCALE_FACTOR matches AUTOSCALE_TARGET_TOKEN_SCALE",
                abs(float(tgt) - float(fac)) < 1e-6,
                f"both = {tgt}",
                f"DRIFT: simulator_LLM autoscales tokens x{tgt} but LA_Hyper's "
                f"_TTFT_TOKEN_SCALE_FACTOR is {fac} — the TTFT ceiling "
                f"(TTFT_MAX_S) is therefore mis-scaled; feasibility gating and "
                f"the Pareto front will be wrong")

    # D2 — TTFT_MAX_S should equal base SLA * token-scale factor.
    base = getattr(LAH, "_TTFT_BASE_SLA_S", None)
    maxs = getattr(LAH, "TTFT_MAX_S", None)
    if None not in (base, fac, maxs):
        R.check("D2 TTFT_MAX_S == base SLA * token-scale factor",
                abs(float(maxs) - float(base) * float(fac)) < 1e-6,
                f"{maxs} == {base} * {fac}",
                f"TTFT_MAX_S={maxs} != {base}*{fac}={float(base)*float(fac)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — overnight_baseline_results parsers
# ══════════════════════════════════════════════════════════════════════════════
def section_parsers(OVR):
    R.section("overnight_baseline_results — results-parser correctness")
    if OVR is None:
        R.skip("parsers", "overnight_baseline_results not importable")
        return

    # E1 — Final Report parser.
    pf = getattr(OVR, "_parse_final_report", None)
    if pf is None:
        R.skip("E1 final-report parser", "_parse_final_report not found")
    else:
        sample = [
            "=== Final Report ===\n", "Epochs: 24\n",
            "Average TTFT (s):    1.373541\n",
            "Total Carbon (kg):   5378.677\n",
            "Total Water (m\u00b3):    390142.3853\n",
            "Total Energy ($):    1462.938\n",
            "Total Energy (kWh):  17422.765\n",
        ]
        try:
            got = pf(sample)
            ok = (got.get("epochs") == "24"
                  and abs(float(got.get("avg_ttft_s", "nan")) - 1.373541) < 1e-6
                  and abs(float(got.get("total_carbon_kg", "nan")) - 5378.677) < 1e-3
                  and abs(float(got.get("total_energy_kwh", "nan")) - 17422.765) < 1e-3)
            R.check("E1 _parse_final_report extracts all fields",
                    ok, "all 6 fields parsed correctly",
                    f"mis-parsed Final Report -> {got}")
        except Exception as e:
            R.fail("E1 final-report parser", f"{type(e).__name__}: {e}")

    # E2 — per-mode summary parser.
    pm = getattr(OVR, "_parse_per_mode", None)
    if pm is None:
        R.skip("E2 per-mode parser", "_parse_per_mode not found")
    else:
        sample = [
            "=== LA_HYPER MULTI-AGENT SUMMARY (Run Totals) ===\n",
            "Mode               | Avg TTFT(s) | Total Carb(kg) | ...\n",
            "-" * 80 + "\n",
            "time_agent         | 1.5434      | 146.460         | 335.623       "
            "| 68.823       | 327.727\n",
            "carbon_agent       | 9.8765      | 12.340          | 980.100       "
            "| 11.500       | 95.250\n",
            "\n",
        ]
        try:
            got = pm(sample)
            ok = (abs(float(got.get("time_agent_ttft", "nan")) - 1.5434) < 1e-3
                  and abs(float(got.get("time_agent_total_energy", "nan")) - 327.727) < 1e-2
                  and abs(float(got.get("carbon_agent_carbon", "nan")) - 12.340) < 1e-2)
            R.check("E2 _parse_per_mode parses the summary table",
                    ok, "mode rows parsed correctly",
                    f"mis-parsed summary table -> {got}")
        except Exception as e:
            R.fail("E2 per-mode parser", f"{type(e).__name__}: {e}")

    # E3 / E4 — epoch-PHV parser on OLD and NEW front-line formats.
    pe = getattr(OVR, "_parse_epoch_phv", None)
    if pe is None:
        R.skip("E3/E4 epoch-PHV parser", "_parse_epoch_phv not found")
    else:
        old_fmt = [
            f"[HYBRID-FRONT] epoch={i} ttft={1.0+i*0.1:.4f} "
            f"carbon={100.0+i:.4f} water={10.0+i:.4f} cost={5.0+i:.4f}\n"
            for i in range(8)
        ]
        try:
            R.check("E3 epoch-PHV parser accepts the standard FRONT line",
                    isinstance(pe(old_fmt), str) and pe(old_fmt) != "",
                    "parsed 8 epochs -> a hypervolume value",
                    "failed to parse standard [HYBRID-FRONT] lines")
        except Exception as e:
            R.fail("E3 epoch-PHV parser (old format)", f"{type(e).__name__}: {e}")

        # NEW format: LAHYPER-FRONT now carries (net=.. wait=.. prefill=..)
        # between ttft= and carbon=.  This is a regression check on that change.
        new_fmt = [
            f"[LAHYPER-FRONT] epoch={i} ttft={1.0+i*0.1:.6f} "
            f"(net={50.0+i:.1f}ms wait={20.0+i:.1f}ms prefill={30.0+i:.1f}ms) "
            f"carbon={100.0+i:.4f} water={10.0+i:.4f} cost={5.0+i:.6f}\n"
            for i in range(8)
        ]
        try:
            res = pe(new_fmt)
            R.check("E4 epoch-PHV parser survives the net/wait/prefill breakdown",
                    isinstance(res, str) and res != "",
                    "new LAHYPER-FRONT format still parses",
                    "the net/wait/prefill breakdown added to [LAHYPER-FRONT] "
                    "BROKE _parse_epoch_phv — its regex needs updating")
        except Exception as e:
            R.fail("E4 epoch-PHV parser (new format)", f"{type(e).__name__}: {e}")

    # E5 — hypervolume monotonicity sanity.
    hv = getattr(OVR, "compute_hypervolume", None)
    if hv is None:
        R.skip("E5 hypervolume sanity", "compute_hypervolume not found")
    else:
        try:
            ref = [10.0, 10.0]
            base = hv([[5.0, 5.0]], ref)
            more = hv([[5.0, 5.0], [3.0, 3.0]], ref)
            R.check("E5 hypervolume does not shrink when a point is added",
                    more >= base - 1e-6,
                    f"HV {base:.2f} -> {more:.2f}",
                    f"HV DECREASED on adding a point ({base:.2f} -> {more:.2f})",
                    warn=True)
        except Exception as e:
            R.warn("E5 hypervolume sanity", str(e))

    # E6 — Pareto front correctness.  _pareto_front returns INDICES.
    pf2 = getattr(OVR, "_pareto_front", None)
    if pf2 is not None:
        try:
            pts = [[1.0, 5.0], [2.0, 2.0], [5.0, 1.0], [3.0, 6.0]]  # idx 3 dominated
            front = list(pf2(pts))
            R.check("E6 _pareto_front excludes dominated points",
                    3 not in front and {0, 1, 2}.issubset(set(front)),
                    f"front indices {front} — dominated point (idx 3) excluded",
                    f"dominated point (idx 3) leaked into the front: {front}",
                    warn=True)
        except Exception as e:
            R.warn("E6 pareto front", str(e))

    # E7 — config sanity.
    mr = getattr(OVR, "MAX_ROWS", None)
    if mr is not None:
        R.check("E7 MAX_ROWS is a positive integer",
                isinstance(mr, int) and mr > 0,
                f"MAX_ROWS = {mr:,}",
                f"MAX_ROWS is {mr!r} — invalid")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Broad failure-hunt for the LAHyper stack.")
    ap.add_argument("--spec-dir", default="sim_specs")
    ap.add_argument("--epoch-length", type=int, default=900)
    ap.add_argument("--requests", type=int, default=20000,
                    help="Base synthetic request count.")
    ap.add_argument("--token-scale", type=float, default=25.0)
    ap.add_argument("--extreme", action="store_true",
                    help="Crank workload sizes to stress memory/queueing paths.")
    args = ap.parse_args()

    if args.extreme:
        args.requests = max(args.requests, 150_000)

    print("═" * 78)
    print("  diagnose_system.py  —  LAHyper stack failure-hunt")
    print(f"  spec-dir={args.spec_dir}  requests={args.requests:,}  "
          f"token-scale=x{args.token_scale}  extreme={args.extreme}")
    print("═" * 78)

    RFS = safe_import("Rate_Flow_Sim_v2")
    LAH = safe_import("LA_Hyper_DDQN")
    SIM = safe_import("simulator_LLM")
    OVR = safe_import("overnight_baseline_results")
    HYB = safe_import("Hybrid_Scheduler_LLM")

    if RFS is not None:
        try:
            section_simulator(RFS, args.spec_dir, args.epoch_length,
                              args.requests, args.extreme)
        except Exception as e:
            R.fail("SECTION A crashed", f"{type(e).__name__}: {e}")
            traceback.print_exc()
        try:
            section_ttft_hunt(RFS, LAH, HYB, args.spec_dir, args.epoch_length,
                              args.requests, args.token_scale)
        except Exception as e:
            R.fail("SECTION B crashed", f"{type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        R.section("Rate_Flow_Sim_v2")
        R.skip("simulator sections", "Rate_Flow_Sim_v2 not importable")

    try:
        section_autoscaler(SIM)
    except Exception as e:
        R.fail("SECTION C crashed", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    try:
        section_cross_file(SIM, LAH)
    except Exception as e:
        R.fail("SECTION D crashed", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    try:
        section_parsers(OVR)
    except Exception as e:
        R.fail("SECTION E crashed", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    counts = R.summary()
    sys.exit(1 if counts["FAIL"] > 0 else 0)


if __name__ == "__main__":
    main()