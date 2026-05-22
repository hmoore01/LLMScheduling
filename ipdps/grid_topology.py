"""
grid_topology.py — MARLIN Grid Topology (Physics Constraints Layer)

Replaces the static CSV-based carbon/cost lookup in Rate_Flow_Sim_v2 with a
dynamic NetworkX distribution graph.  The graph simulates power flow, line
congestion, and locational marginal pricing across generator, datacenter, and
residential load nodes.

Key reference:
  Claeys et al. (2023) for stochastic residential load wavelet decomposition.
  Denholm et al. (2015) "Duck Chart" NREL report for evening ramp validation.
  Williams (2025) for mid-day solar price cannibalization.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    print("[grid_topology] WARNING: networkx not found — falling back to stub topology.")


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR PROFILES
# Each entry: (name, carbon_g_per_kwh, base_cost_usd_per_kwh, capacity_kw,
#              ramp_rate_kw_per_min, dispatchable)
# ─────────────────────────────────────────────────────────────────────────────
GENERATOR_SPECS = {
    "Solar":  dict(carbon=0.0,   cost=0.00,  capacity_kw=5_000.0,  ramp=500.0,  dispatchable=False),
    "Gas":    dict(carbon=490.0, cost=0.065, capacity_kw=20_000.0, ramp=200.0,  dispatchable=True),
    "Peaker": dict(carbon=700.0, cost=0.18,  capacity_kw=8_000.0,  ramp=800.0,  dispatchable=True),
}

# Transmission line impedance proxy: lower = cheaper to flow power through
LINE_IMPEDANCE = {
    ("Solar",   "Bus_A"): 0.02,
    ("Gas",     "Bus_A"): 0.03,
    ("Peaker",  "Bus_A"): 0.04,
    ("Bus_A",   "Bus_B"): 0.05,   # Inter-bus backbone
    ("Bus_B",   "Residential"): 0.01,
    ("Bus_B",   "Datacenter"): 0.02,
}

# Rated capacity per line (kW) — triggers congestion penalty above 90%
LINE_CAPACITY_KW = {
    ("Solar",   "Bus_A"): 5_500.0,
    ("Gas",     "Bus_A"): 22_000.0,
    ("Peaker",  "Bus_A"): 9_000.0,
    ("Bus_A",   "Bus_B"): 30_000.0,
    ("Bus_B",   "Residential"): 15_000.0,
    ("Bus_B",   "Datacenter"): 20_000.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# DUCK CURVE — hourly residential load shape (normalised to 1.0 at peak)
# Reproduces the characteristic "duck" with mid-day solar suppression and
# steep evening ramp (Denholm et al. 2015).
# ─────────────────────────────────────────────────────────────────────────────
_DUCK_CURVE_24H = np.array([
    0.52, 0.48, 0.45, 0.44, 0.45, 0.50,   # 00–05  night trough
    0.58, 0.68, 0.72, 0.70, 0.65, 0.58,   # 06–11  morning + solar suppression
    0.55, 0.53, 0.52, 0.55, 0.65, 0.85,   # 12–17  belly + start of ramp
    1.00, 0.97, 0.90, 0.80, 0.70, 0.60,   # 18–23  evening peak + decline
], dtype=np.float64)

# Residential peak load (kW) — scales the duck curve
RESIDENTIAL_PEAK_KW = 12_000.0


# ─────────────────────────────────────────────────────────────────────────────
# WAVELET NOISE MODEL  (Claeys et al. 2023)
# Stochastic variability added on top of the deterministic duck curve.
# Uses three frequency bands: daily, intra-hour, minute-level.
# ─────────────────────────────────────────────────────────────────────────────
def _wavelet_noise(hour: float, seed: int, amplitude: float = 0.06) -> float:
    """
    Generate a single stochastic noise sample from three wavelet bands.

    Args:
        hour:      Fractional hour of day [0, 24).
        seed:      Epoch-level RNG seed for reproducibility.
        amplitude: Peak-to-peak noise as fraction of peak load.

    Returns:
        Additive noise in normalised units (add to duck-curve value before
        multiplying by RESIDENTIAL_PEAK_KW).
    """
    rng = np.random.default_rng(seed + int(hour * 100))
    # Band 1: slow daily drift (period ~8 h)
    b1 = 0.5 * math.sin(2 * math.pi * hour / 8.0 + rng.uniform(0, 2 * math.pi))
    # Band 2: intra-hour fluctuation (period ~1 h)
    b2 = 0.3 * math.sin(2 * math.pi * hour / 1.0 + rng.uniform(0, 2 * math.pi))
    # Band 3: minute-level white noise
    b3 = rng.normal(0, 1.0)
    return amplitude * (b1 + b2 + b3) / 1.8   # /1.8 normalises to ≈ amplitude


def residential_load_kw(hour: float, epoch_seed: int = 0) -> float:
    """
    Return stochastic residential load (kW) at a given hour.

    Combines the NREL duck-curve baseline with Claeys-style wavelet noise.

    Args:
        hour:       Hour of day [0, 24).
        epoch_seed: Seed for the noise RNG — pass epoch_idx for reproducibility.

    Returns:
        Residential demand in kW.
    """
    h_idx   = int(hour) % 24
    frac    = hour - math.floor(hour)
    h_next  = (h_idx + 1) % 24
    base    = (1.0 - frac) * _DUCK_CURVE_24H[h_idx] + frac * _DUCK_CURVE_24H[h_next]
    noise   = _wavelet_noise(hour, seed=epoch_seed)
    return max(0.0, (base + noise) * RESIDENTIAL_PEAK_KW)


def solar_generation_kw(hour: float) -> float:
    """
    Clear-sky solar generation (kW) following Williams (2025) cannibalization
    pattern: negative/zero LMP contribution during mid-day glut.

    Uses a clipped cosine model peaking at solar noon (hour 12).
    """
    if hour < 5.5 or hour > 19.5:
        return 0.0
    radians = math.pi * (hour - 5.5) / 14.0   # sunrise-to-sunset span
    raw_kw  = GENERATOR_SPECS["Solar"]["capacity_kw"] * math.sin(radians) ** 1.5
    return max(0.0, raw_kw)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE GRID NETWORK
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GridNode:
    """Represents a node in the local distribution graph."""
    name:        str
    node_type:   str          # "generator", "bus", "load", "datacenter"
    load_kw:     float = 0.0  # current demand at this node
    gen_kw:      float = 0.0  # current generation at this node
    lmp:         float = 0.0  # locational marginal price ($/kWh)
    carbon_g_kwh: float = 0.0 # marginal carbon intensity at this node


class InteractiveGridNetwork:
    """
    NetworkX-based local distribution graph for MARLIN demand response.

    Topology (single-feeder radial approximation):

        [Solar] ──┐
        [Gas]   ──┤── Bus_A ── Bus_B ──┬── Residential Load
        [Peaker]──┘                    └── Datacenter Load(s)

    Responsibilities:
      • Track instantaneous power balance at each epoch step.
      • Compute line flows and flag congestion on the DC-feeder branch.
      • Emit Locational Marginal Prices that the Utility Agent uses to
        signal the Datacenter Agent to throttle or shift load.
      • Provide a flat feature vector for the GAT feature extractor.
    """

    def __init__(self, dc_ids: List[int], epoch_length_s: float = 900.0):
        """
        Args:
            dc_ids:          Active datacenter IDs (one load node per DC).
            epoch_length_s:  Duration of one simulation epoch in seconds.
        """
        self.dc_ids         = list(dc_ids)
        self.epoch_length_s = float(epoch_length_s)
        self._build_graph()
        self.reset()

    # ── Graph construction ─────────────────────────────────────────────────

    def _build_graph(self) -> None:
        """Construct the NetworkX DiGraph with generator, bus, and load nodes."""
        if not _NX_AVAILABLE:
            self._graph = None
            return

        G = nx.DiGraph()

        # Generator nodes
        for gen_name, spec in GENERATOR_SPECS.items():
            G.add_node(gen_name,
                       ntype="generator",
                       capacity_kw=spec["capacity_kw"],
                       carbon=spec["carbon"],
                       base_cost=spec["cost"],
                       dispatchable=spec["dispatchable"],
                       gen_kw=0.0,
                       dispatch_frac=0.0)

        # Bus nodes (aggregation points)
        G.add_node("Bus_A", ntype="bus", load_kw=0.0, gen_kw=0.0)
        G.add_node("Bus_B", ntype="bus", load_kw=0.0, gen_kw=0.0)

        # Residential load node
        G.add_node("Residential", ntype="load", load_kw=0.0)

        # Datacenter load node(s) — one aggregate node represents all DCs
        G.add_node("Datacenter", ntype="datacenter",
                   load_kw=0.0, dc_ids=list(self.dc_ids))

        # Transmission lines (directed: source → sink)
        for (src, dst), imp in LINE_IMPEDANCE.items():
            cap = LINE_CAPACITY_KW.get((src, dst), 10_000.0)
            G.add_edge(src, dst, impedance=imp, capacity_kw=cap,
                       flow_kw=0.0, congested=False)

        self._graph = G

    # ── State management ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset per-epoch state (call at the start of each epoch)."""
        self._hour:         float = 0.0
        self._epoch_seed:   int   = 0
        self._p_res_kw:     float = 0.0
        self._p_dc_kw:      float = 0.0
        self._dispatch:     Dict[str, float] = {g: 0.0 for g in GENERATOR_SPECS}
        self._lmp_by_node:  Dict[str, float] = {}
        self._carbon_mix_g_kwh: float = 400.0
        self._congested_lines:  List[Tuple[str, str]] = []
        self._unmet_kw:     float = 0.0

        if self._graph:
            for n, d in self._graph.nodes(data=True):
                d["gen_kw"] = 0.0
                d["load_kw"] = 0.0
                d.pop("lmp", None)
            for *_, d in self._graph.edges(data=True):
                d["flow_kw"] = 0.0
                d["congested"] = False

    def set_epoch(self, epoch_idx: int, epoch_length_s: float = None) -> None:
        """Configure the epoch's time window and random seed."""
        if epoch_length_s:
            self.epoch_length_s = float(epoch_length_s)
        epoch_start_s = epoch_idx * self.epoch_length_s
        self._hour       = (epoch_start_s % 86400.0) / 3600.0
        self._epoch_seed = int(epoch_idx)
        # Pre-populate residential load so the Utility Agent has a signal
        # for its first obs before step() is called this epoch.
        self._p_res_kw = residential_load_kw(self._hour, self._epoch_seed)

    # ── Core physics step ──────────────────────────────────────────────────

    def step(self, p_dc_kw: float, dispatch_fracs: Dict[str, float]) -> Dict[str, float]:
        """
        Execute one grid physics step for the current epoch.

        Pipeline (mirrors MARLIN §2.3 parliament_env step):
          1. Compute residential background load (duck curve + wavelet noise).
          2. Apply solar generation (non-dispatchable).
          3. Apply agent's dispatch fractions to Gas and Peaker generators.
          4. Compute line flows using a DC power-flow approximation.
          5. Flag congested lines and compute LMP uplift.
          6. Settle the generation mix and return economics.

        Args:
            p_dc_kw:        Total datacenter power demand this epoch (kW).
            dispatch_fracs: {"Gas": frac, "Peaker": frac} in [0, 1].

        Returns:
            economics: {
                "p_res_kw", "p_dc_kw", "p_solar_kw", "p_gas_kw", "p_peaker_kw",
                "total_gen_kw", "unmet_kw", "congestion_pct",
                "lmp_dc", "lmp_res",
                "carbon_mix_g_kwh", "gen_cost_usd",
                "blackout_penalty_usd",
            }
        """
        self._p_dc_kw = max(0.0, float(p_dc_kw))

        # ── 1. Residential load ───────────────────────────────────────────
        p_res = residential_load_kw(self._hour, self._epoch_seed)
        self._p_res_kw = p_res

        # ── 2. Solar (non-dispatchable) ───────────────────────────────────
        p_solar = solar_generation_kw(self._hour)
        p_solar = min(p_solar, GENERATOR_SPECS["Solar"]["capacity_kw"])

        # ── 3. Dispatchable generators ────────────────────────────────────
        p_gas = float(dispatch_fracs.get("Gas", 0.0)) * GENERATOR_SPECS["Gas"]["capacity_kw"]
        p_gas = max(0.0, min(p_gas, GENERATOR_SPECS["Gas"]["capacity_kw"]))

        p_peaker = float(dispatch_fracs.get("Peaker", 0.0)) * GENERATOR_SPECS["Peaker"]["capacity_kw"]
        p_peaker = max(0.0, min(p_peaker, GENERATOR_SPECS["Peaker"]["capacity_kw"]))

        total_gen = p_solar + p_gas + p_peaker
        total_load = p_res + self._p_dc_kw
        unmet = max(0.0, total_load - total_gen)
        self._unmet_kw = unmet
        self._dispatch = {"Solar": p_solar, "Gas": p_gas, "Peaker": p_peaker}

        # ── 4. DC power flow approximation ────────────────────────────────
        # Simplified linearised flow: each line carries proportional share
        # of load assigned to its downstream subtree.
        flow_bus_a   = total_gen           # all generation feeds Bus_A
        flow_bus_b   = total_load          # all load draws from Bus_B
        flow_dc_line = self._p_dc_kw       # dedicated DC feeder
        flow_res_line = p_res

        flows = {
            ("Bus_A", "Bus_B"):       flow_bus_a,
            ("Bus_B", "Datacenter"):  flow_dc_line,
            ("Bus_B", "Residential"): flow_res_line,
            ("Solar",  "Bus_A"):      p_solar,
            ("Gas",    "Bus_A"):      p_gas,
            ("Peaker", "Bus_A"):      p_peaker,
        }

        # ── 5. Congestion detection and LMP uplift ─────────────────────────
        congested = []
        max_congestion = 0.0
        if self._graph:
            for (src, dst), flow in flows.items():
                if self._graph.has_edge(src, dst):
                    cap = self._graph[src][dst]["capacity_kw"]
                    cong_pct = flow / max(cap, 1.0)
                    self._graph[src][dst]["flow_kw"]   = flow
                    self._graph[src][dst]["congested"] = cong_pct > 0.90
                    if cong_pct > 0.90:
                        congested.append((src, dst))
                    max_congestion = max(max_congestion, cong_pct)

        self._congested_lines = congested
        dc_line_congested = ("Bus_B", "Datacenter") in congested

        # ── 6. LMP calculation ────────────────────────────────────────────
        # Base LMP = weighted average generation cost by dispatch share.
        # Congestion uplift on the DC feeder → higher price signal to DC agent.
        weighted_cost = 0.0
        total_g = max(total_gen, 1.0)
        for gen, kw in [("Solar", p_solar), ("Gas", p_gas), ("Peaker", p_peaker)]:
            weighted_cost += (kw / total_g) * GENERATOR_SPECS[gen]["cost"]

        # Mid-day solar cannibalization: if solar > 40% of load, LMP → 0 or negative
        solar_frac = p_solar / max(total_load, 1.0)
        if solar_frac > 0.40:
            cannibal_discount = (solar_frac - 0.40) / 0.60   # 0→1 as solar fraction grows
            weighted_cost = max(-0.005, weighted_cost * (1.0 - cannibal_discount * 1.5))

        # Scarcity adder when supply is tight
        scarcity_adder = 0.0
        if unmet > 0:
            scarcity_adder = min(0.50, unmet / max(total_load, 1.0) * 0.80)

        # Congestion uplift for DC feeder
        congestion_uplift = 0.0
        if dc_line_congested:
            dc_flow    = flow_dc_line
            dc_cap     = LINE_CAPACITY_KW.get(("Bus_B", "Datacenter"), 20_000.0)
            cong_ratio = dc_flow / max(dc_cap, 1.0)
            congestion_uplift = max(0.0, (cong_ratio - 0.90) / 0.10) * 0.12

        lmp_dc  = max(-0.01, weighted_cost + scarcity_adder + congestion_uplift)
        lmp_res = max(-0.01, weighted_cost + scarcity_adder)

        self._lmp_by_node = {"Datacenter": lmp_dc, "Residential": lmp_res, "Bus_A": weighted_cost}

        # ── 7. Carbon mix ─────────────────────────────────────────────────
        carbon_mix = 0.0
        for gen, kw in [("Solar", p_solar), ("Gas", p_gas), ("Peaker", p_peaker)]:
            carbon_mix += (kw / total_g) * GENERATOR_SPECS[gen]["carbon"]
        self._carbon_mix_g_kwh = carbon_mix

        # ── 8. Economics ──────────────────────────────────────────────────
        epoch_h  = self.epoch_length_s / 3600.0
        gen_cost = sum(
            (kw * epoch_h) * GENERATOR_SPECS[gen]["cost"]
            for gen, kw in [("Gas", p_gas), ("Peaker", p_peaker)]
        )
        blackout_penalty = unmet * epoch_h * 10.0   # $10/kWh for unmet demand

        return {
            "p_res_kw":           p_res,
            "p_dc_kw":            self._p_dc_kw,
            "p_solar_kw":         p_solar,
            "p_gas_kw":           p_gas,
            "p_peaker_kw":        p_peaker,
            "total_gen_kw":       total_gen,
            "unmet_kw":           unmet,
            "congestion_pct":     max_congestion,
            "lmp_dc":             lmp_dc,
            "lmp_res":            lmp_res,
            "carbon_mix_g_kwh":   carbon_mix,
            "gen_cost_usd":       gen_cost,
            "blackout_penalty_usd": blackout_penalty,
            "solar_cannibal_frac": solar_frac,
        }

    # ── Feature vector for GAT ─────────────────────────────────────────────

    def get_graph_feature_vector(self) -> np.ndarray:
        """
        Return a flat feature vector representing the current graph state.

        The GAT feature extractor in utility_gnn.py uses this as its observation.
        Features (per node/edge, concatenated):
          • Generator dispatch fractions (3 values)
          • Generator capacity utilisation (3 values)
          • Residential load normalised (1 value)
          • DC load normalised (1 value)
          • Line flow/capacity ratios for key lines (4 values)
          • LMP at Bus_A, Datacenter nodes (2 values)
          • Carbon mix normalised (1 value)
          • Unmet demand fraction (1 value)

        Total: 16 features.
        """
        total_cap_gen = sum(s["capacity_kw"] for s in GENERATOR_SPECS.values())
        total_load    = max(self._p_res_kw + self._p_dc_kw, 1.0)

        # Generator dispatch fractions
        f_solar  = self._dispatch.get("Solar",  0.0) / GENERATOR_SPECS["Solar"]["capacity_kw"]
        f_gas    = self._dispatch.get("Gas",    0.0) / GENERATOR_SPECS["Gas"]["capacity_kw"]
        f_peaker = self._dispatch.get("Peaker", 0.0) / GENERATOR_SPECS["Peaker"]["capacity_kw"]

        # Capacity utilisation
        total_gen_kw = sum(self._dispatch.values())
        u_solar  = self._dispatch.get("Solar",  0.0) / max(total_cap_gen, 1.0)
        u_gas    = self._dispatch.get("Gas",    0.0) / max(total_cap_gen, 1.0)
        u_peaker = self._dispatch.get("Peaker", 0.0) / max(total_cap_gen, 1.0)

        # Load normalised
        p_res_n = self._p_res_kw  / RESIDENTIAL_PEAK_KW
        p_dc_n  = self._p_dc_kw   / max(sum(s["capacity_kw"] for s in GENERATOR_SPECS.values()), 1.0)

        # Line flow ratios
        def _line_flow(src, dst):
            if self._graph and self._graph.has_edge(src, dst):
                d = self._graph[src][dst]
                return d["flow_kw"] / max(d["capacity_kw"], 1.0)
            return 0.0

        lr_bus    = _line_flow("Bus_A", "Bus_B")
        lr_dc     = _line_flow("Bus_B", "Datacenter")
        lr_res    = _line_flow("Bus_B", "Residential")
        lr_solar  = _line_flow("Solar", "Bus_A")

        # Prices and carbon
        lmp_bus   = self._lmp_by_node.get("Bus_A", 0.0) / 0.30   # normalise to ≈[0,1]
        lmp_dc    = self._lmp_by_node.get("Datacenter", 0.0) / 0.50
        carbon_n  = self._carbon_mix_g_kwh / 700.0
        unmet_n   = self._unmet_kw / max(total_load, 1.0)

        return np.array([
            f_solar, f_gas, f_peaker,
            u_solar, u_gas, u_peaker,
            p_res_n, p_dc_n,
            lr_bus, lr_dc, lr_res, lr_solar,
            lmp_bus, lmp_dc,
            carbon_n, unmet_n,
        ], dtype=np.float32)

    # ── Reward for utility agent ───────────────────────────────────────────

    def compute_utility_reward(self, economics: Dict[str, float]) -> float:
        """
        Utility Agent reward (Eq. from MARLIN §1.2):

            R_grid = −(Σ c_i * P_i) − λ * max(0, P_dc + P_res − Σ P_i)

        Penalises:
          • Total generation cost (Gas + Peaker; Solar is free).
          • Unmet demand (blackout risk), weighted by λ=10.
          • Peaker plant usage specifically (high carbon, high cost).

        Returns a scalar reward (higher is better, so maximise).
        """
        epoch_h  = self.epoch_length_s / 3600.0
        gen_cost = float(economics.get("gen_cost_usd",       0.0))
        unmet    = float(economics.get("unmet_kw",           0.0))
        p_peaker = float(economics.get("p_peaker_kw",        0.0))
        total_load = max(float(economics.get("p_res_kw", 0)) +
                         float(economics.get("p_dc_kw",  0)), 1.0)

        # Normalise generation cost to a per-epoch scale
        cost_penalty    = gen_cost / max(total_load * epoch_h * 0.20, 1e-4)
        scarcity_penalty = (unmet / total_load) * 10.0
        peaker_penalty   = (p_peaker / GENERATOR_SPECS["Peaker"]["capacity_kw"]) * 2.0

        return -(cost_penalty + scarcity_penalty + peaker_penalty)

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def lmp_dc(self) -> float:
        """Current LMP at the datacenter bus ($/kWh)."""
        return self._lmp_by_node.get("Datacenter", 0.10)

    @property
    def carbon_mix_g_kwh(self) -> float:
        """Current marginal carbon intensity of the generation mix (g CO2/kWh)."""
        return self._carbon_mix_g_kwh

    @property
    def congested(self) -> bool:
        """True if the DC feeder line is currently congested."""
        return ("Bus_B", "Datacenter") in self._congested_lines

    def get_lmp_by_dc(self, dc_ids: List[int]) -> Dict[int, float]:
        """Return per-DC LMP mapping (all DCs share the same aggregate node)."""
        lmp = self.lmp_dc
        return {dc_id: lmp for dc_id in dc_ids}