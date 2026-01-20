#!/usr/bin/env python3
"""
MARL Training Dashboard - Enhanced with Parallel Training Support & IPC
"""
import os, json, time, threading, tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

try:
    from flask import Flask, render_template_string, jsonify
    from flask_cors import CORS
except ImportError:
    import subprocess

    subprocess.run(["pip", "install", "flask", "flask-cors", "--break-system-packages", "-q"])
    from flask import Flask, render_template_string, jsonify
    from flask_cors import CORS

# IPC via temp files for cross-process updates
STATE_DIR = Path(tempfile.gettempdir()) / "marl_dashboard"
STATE_DIR.mkdir(exist_ok=True)


def _state_path(pid): return STATE_DIR / f"{pid.replace('/', '_')}.json"


def _write_state(pid, data):
    try:
        with open(_state_path(pid), 'w') as f:
            json.dump(data, f)
    except:
        pass


def _read_state(pid):
    try:
        p = _state_path(pid)
        if p.exists():
            with open(p) as f: return json.load(f)
    except:
        pass
    return None


@dataclass
class ProfileMetrics:
    profile_id: str
    timesteps: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    rewards_smoothed: List[float] = field(default_factory=list)  # EMA smoothed
    ttft: List[float] = field(default_factory=list)
    carbon: List[float] = field(default_factory=list)
    water: List[float] = field(default_factory=list)
    cost: List[float] = field(default_factory=list)
    routing_dist: List[float] = field(default_factory=list)
    power_plan: Dict = field(default_factory=dict)

    reward_sum: float = 0.0
    reward_count: int = 0
    reward_min: float = float('inf')
    reward_max: float = float('-inf')
    reward_ema: float = 0.0  # Exponential moving average
    best_ttft: float = float('inf')
    best_carbon: float = float('inf')
    best_water: float = float('inf')
    best_cost: float = float('inf')

    agent_type: str = "unknown"
    primary_metric: str = ""
    reward_weights: Dict = field(default_factory=dict)
    constraints: Dict = field(default_factory=dict)

    total_timesteps: int = 0
    current_timesteps: int = 0
    start_time: float = field(default_factory=time.time)
    is_training: bool = False
    is_completed: bool = False
    steps_per_second: float = 0.0
    _last_t: float = field(default_factory=time.time)
    _last_s: int = 0

    def add_reward(self, r):
        self.reward_sum += r
        self.reward_count += 1
        self.reward_min = min(self.reward_min, r)
        self.reward_max = max(self.reward_max, r)
        # Update EMA with alpha=0.05 for smooth display
        alpha = 0.05
        if self.reward_count == 1:
            self.reward_ema = r
        else:
            self.reward_ema = (1 - alpha) * self.reward_ema + alpha * r
        self.rewards_smoothed.append(self.reward_ema)

    def update_best(self, **kw):
        for k, v in kw.items():
            if v and v > 0:
                cur = getattr(self, f'best_{k}', float('inf'))
                setattr(self, f'best_{k}', min(cur, v))

    def update_sps(self):
        now, elapsed = time.time(), time.time() - self._last_t
        if elapsed > 1:
            self.steps_per_second = (self.current_timesteps - self._last_s) / elapsed
            self._last_t, self._last_s = now, self.current_timesteps

    @property
    def reward_mean(self):
        return self.reward_sum / max(1, self.reward_count)

    @property
    def reward_std(self):
        if self.reward_count < 2: return 0
        return max(0, (self.reward_sum ** 2 / self.reward_count) - self.reward_mean ** 2) ** 0.5

    @property
    def progress_pct(self):
        return 100 * self.current_timesteps / max(1, self.total_timesteps)

    @property
    def eta_seconds(self):
        if self.steps_per_second <= 0: return 0
        return (self.total_timesteps - self.current_timesteps) / self.steps_per_second

    def to_dict(self):
        trim = lambda x: x[-300:] if len(x) > 300 else x
        inf = float('inf')
        return {
            "profile_id": self.profile_id,
            "timesteps": trim(self.timesteps),
            "rewards": trim(self.rewards),
            "rewards_smoothed": trim(self.rewards_smoothed),  # Add smoothed
            "ttft": trim(self.ttft), "carbon": trim(self.carbon),
            "water": trim(self.water), "cost": trim(self.cost),
            "current": {
                "reward": self.rewards[-1] if self.rewards else 0,
                "reward_smoothed": self.reward_ema,  # Add smoothed current
                "ttft": self.ttft[-1] if self.ttft else 0,
                "carbon": self.carbon[-1] if self.carbon else 0,
                "water": self.water[-1] if self.water else 0,
                "cost": self.cost[-1] if self.cost else 0,
            },
            "best": {
                "ttft": 0 if self.best_ttft == inf else self.best_ttft,
                "carbon": 0 if self.best_carbon == inf else self.best_carbon,
                "water": 0 if self.best_water == inf else self.best_water,
                "cost": 0 if self.best_cost == inf else self.best_cost,
            },
            "stats": {
                "reward_mean": self.reward_mean, "reward_std": self.reward_std,
                "reward_min": 0 if self.reward_min == inf else self.reward_min,
                "reward_max": 0 if self.reward_max == -inf else self.reward_max,
            },
            "routing_dist": self.routing_dist, "power_plan": self.power_plan,
            "agent_type": self.agent_type, "primary_metric": self.primary_metric,
            "reward_weights": self.reward_weights, "constraints": self.constraints,
            "current_timesteps": self.current_timesteps,
            "total_timesteps": self.total_timesteps,
            "progress_pct": self.progress_pct, "eta_seconds": self.eta_seconds,
            "elapsed_seconds": time.time() - self.start_time,
            "steps_per_second": self.steps_per_second,
            "is_training": self.is_training, "is_completed": self.is_completed,
        }


class TrainingDashboard:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self._initialized = True
        self.profiles: Dict[str, ProfileMetrics] = {}
        self.global_start_time = time.time()
        self.training_active = False
        self._lock = threading.Lock()
        self.stop_requested: Dict[str, bool] = {}
        # Clear old state
        for f in STATE_DIR.glob("*.json"):
            try:
                f.unlink()
            except:
                pass
        # Poll thread for IPC
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        while True:
            try:
                for f in STATE_DIR.glob("*.json"):
                    if "_stop" in f.stem: continue
                    s = _read_state(f.stem)
                    if s: self._merge(f.stem, s)
            except:
                pass
            time.sleep(0.5)

    def _merge(self, pid, s):
        with self._lock:
            if pid not in self.profiles:
                self.profiles[pid] = ProfileMetrics(profile_id=pid)
            p = self.profiles[pid]
            p.current_timesteps = s.get("current_timesteps", p.current_timesteps)
            p.total_timesteps = s.get("total_timesteps", p.total_timesteps)
            p.is_training = s.get("is_training", p.is_training)
            p.is_completed = s.get("is_completed", p.is_completed)
            # Merge time series
            for ts, rw in zip(s.get("timesteps", []), s.get("rewards", [])):
                if not p.timesteps or ts > p.timesteps[-1]:
                    p.timesteps.append(ts)
                    p.rewards.append(rw)
                    p.add_reward(rw)
            for m in ['ttft', 'carbon', 'water', 'cost']:
                v = s.get(m)
                if v and v > 0:
                    getattr(p, m).append(v)
                    p.update_best(**{m: v})
            if "routing_dist" in s: p.routing_dist = s["routing_dist"]
            if "power_plan" in s: p.power_plan = s["power_plan"]
            p.update_sps()
            self.training_active = any(x.is_training for x in self.profiles.values())

    def register_profile(self, pid, total, config=None):
        with self._lock:
            p = ProfileMetrics(profile_id=pid, total_timesteps=total)
            if config:
                w, c = config.get("weights", {}), config.get("constraints", {})
                p.reward_weights, p.constraints = w, c
                if len(w) == 1 and not c:
                    p.agent_type, p.primary_metric = "single_metric", list(w.keys())[0]
                else:
                    p.agent_type = "constrained"
                    if w: p.primary_metric = max(w.items(), key=lambda x: x[1])[0]
            self.profiles[pid] = p
            self.stop_requested[pid] = False

    def start_training(self, pid):
        with self._lock:
            if pid in self.profiles:
                self.profiles[pid].is_training = True
                self.profiles[pid].start_time = time.time()
                self.training_active = True

    def end_training(self, pid):
        with self._lock:
            if pid in self.profiles:
                self.profiles[pid].is_training = False
                self.profiles[pid].is_completed = True
                self.training_active = any(x.is_training for x in self.profiles.values())

    def should_stop(self, pid):
        return self.stop_requested.get(pid, False)

    def request_stop(self, pid):
        with self._lock:
            self.stop_requested[pid] = True
            try:
                with open(_state_path(f"{pid}_stop"), 'w') as f:
                    json.dump({"stop": True}, f)
            except:
                pass

    def log_step(self, profile_id, timestep, reward, metrics=None, power_plan=None, **kw):
        with self._lock:
            if profile_id not in self.profiles:
                self.profiles[profile_id] = ProfileMetrics(profile_id=profile_id)
            p = self.profiles[profile_id]
            p.timesteps.append(timestep)
            p.rewards.append(reward)
            p.current_timesteps = timestep
            p.add_reward(reward)
            p.update_sps()
            if metrics:
                for m in ['ttft', 'carbon', 'water', 'cost']:
                    if m in metrics:
                        getattr(p, m).append(metrics[m])
                        p.update_best(**{m: metrics[m]})
                if "routing_dist" in metrics: p.routing_dist = metrics["routing_dist"]
            if power_plan: p.power_plan = power_plan
        # Write for IPC
        _write_state(profile_id, {
            "current_timesteps": timestep, "total_timesteps": p.total_timesteps,
            "is_training": p.is_training, "timesteps": p.timesteps[-50:],
            "rewards": p.rewards[-50:], "reward": reward,
            "ttft": p.ttft[-1] if p.ttft else 0, "carbon": p.carbon[-1] if p.carbon else 0,
            "water": p.water[-1] if p.water else 0, "cost": p.cost[-1] if p.cost else 0,
            "routing_dist": p.routing_dist, "power_plan": p.power_plan,
        })

    def get_all_data(self):
        with self._lock:
            return {
                "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
                "summary": {
                    "total": len(self.profiles),
                    "active": sum(1 for p in self.profiles.values() if p.is_training),
                    "completed": sum(1 for p in self.profiles.values() if p.is_completed),
                },
                "global_elapsed": time.time() - self.global_start_time,
                "training_active": self.training_active,
            }


dashboard = TrainingDashboard()


def log_to_dashboard(profile_id, timestep, reward, metrics=None, power_plan=None, **kw):
    """Cross-process logging - writes to shared file."""
    _write_state(profile_id, {
        "profile_id": profile_id, "current_timesteps": timestep,
        "is_training": True, "timesteps": [timestep], "rewards": [reward],
        "ttft": metrics.get("ttft", 0) if metrics else 0,
        "carbon": metrics.get("carbon", 0) if metrics else 0,
        "water": metrics.get("water", 0) if metrics else 0,
        "cost": metrics.get("cost", 0) if metrics else 0,
        "routing_dist": metrics.get("routing_dist", []) if metrics else [],
        "power_plan": power_plan or {},
    })


def check_stop_requested(pid): return _state_path(f"{pid}_stop").exists()


# Flask App
app = Flask(__name__)
CORS(app)

DASHBOARD_HTML = '''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><title>MARL Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--brd:#30363d;--txt:#e6edf3;--txt2:#8b949e;--blu:#58a6ff;--grn:#3fb950;--red:#f85149;--org:#d29922;--pur:#a371f7;--cyn:#39c5cf}
*{margin:0;padding:0;box-sizing:border-box}body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--txt);min-height:100vh}
.nav{background:var(--bg2);border-bottom:1px solid var(--brd);padding:12px 24px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:100}
.nav h1{font-size:1.1rem;background:linear-gradient(135deg,var(--blu),var(--pur));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav-stats{display:flex;gap:16px;font-size:.8rem}.stat{display:flex;align-items:center;gap:5px}.stat-lbl{color:var(--txt2)}.stat-val{font-family:monospace}
.dot{width:8px;height:8px;border-radius:50%;background:var(--grn);animation:pulse 2s infinite}.dot.idle{background:var(--txt2);animation:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.main{padding:16px;max-width:1600px;margin:0 auto}
.sum{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:16px}
.sum-c{background:var(--bg2);border:1px solid var(--brd);border-radius:8px;padding:12px}.sum-c .lbl{font-size:.65rem;text-transform:uppercase;color:var(--txt2)}.sum-c .val{font-size:1.4rem;font-weight:700;font-family:monospace}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}
.card{background:var(--bg2);border:1px solid var(--brd);border-radius:8px;overflow:hidden}.card.training{border-left:3px solid var(--blu)}.card.done{border-left:3px solid var(--grn)}
.card-hd{padding:12px;border-bottom:1px solid var(--brd);display:flex;justify-content:space-between;align-items:center}
.card-ti{display:flex;align-items:center;gap:6px}.card-nm{font-weight:600;font-size:.9rem}
.badge{font-size:.55rem;padding:2px 6px;border-radius:8px;font-weight:500;text-transform:uppercase}
.b-single{background:rgba(88,166,255,.15);color:var(--blu)}.b-multi{background:rgba(163,113,247,.15);color:var(--pur)}
.b-train{background:rgba(88,166,255,.15);color:var(--blu)}.b-done{background:rgba(63,185,80,.15);color:var(--grn)}
.prog{padding:8px 12px;background:var(--bg3)}.prog-bar{height:4px;background:var(--brd);border-radius:2px;overflow:hidden;margin-bottom:5px}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--blu),var(--pur));transition:width .3s}
.prog-info{display:flex;justify-content:space-between;font-size:.65rem;color:var(--txt2);font-family:monospace}
.met{padding:12px}.met-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-bottom:10px}
.met-i{background:var(--bg3);border-radius:5px;padding:6px;text-align:center}
.met-i .lbl{font-size:.55rem;text-transform:uppercase;color:var(--txt2)}.met-i .val{font-size:.85rem;font-weight:600;font-family:monospace}
.met-i .bst{font-size:.5rem;color:var(--txt2)}.val.ttft{color:var(--blu)}.val.carbon{color:var(--grn)}.val.water{color:var(--cyn)}.val.cost{color:var(--org)}
.ch-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
.ch-box{background:var(--bg3);border-radius:5px;padding:8px}.ch-ti{font-size:.6rem;text-transform:uppercase;color:var(--txt2);margin-bottom:4px}.ch-wrap{height:70px}
.pl-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.pl-box{background:var(--bg3);border-radius:5px;padding:8px}.pl-ti{font-size:.6rem;text-transform:uppercase;color:var(--txt2);margin-bottom:4px}
.pw-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:2px}
.pw-c{aspect-ratio:1.2;border-radius:2px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:monospace;font-size:.5rem}
.pw-c .dc{font-size:.4rem;opacity:.7}.pw-c.high{background:rgba(248,81,73,.25);color:var(--red)}.pw-c.med{background:rgba(210,153,34,.25);color:var(--org)}
.pw-c.low{background:rgba(63,185,80,.25);color:var(--grn)}.pw-c.off{background:var(--brd);color:var(--txt2)}
.rt-bar{display:flex;height:16px;border-radius:2px;overflow:hidden;background:var(--brd)}
.rt-seg{display:flex;align-items:center;justify-content:center;font-size:.5rem;font-weight:500;color:#fff;transition:width .3s}
.card-ft{padding:8px 12px;border-top:1px solid var(--brd);display:flex;justify-content:space-between;font-size:.65rem}
.st-grp{display:flex;gap:10px;color:var(--txt2)}.mini{display:flex;gap:2px}.mini span:first-child{color:var(--txt2)}
.btn-stop{padding:4px 8px;border-radius:4px;border:1px solid var(--red);background:transparent;color:var(--red);font-size:.65rem;cursor:pointer}
.btn-stop:hover:not(:disabled){background:var(--red);color:#fff}.btn-stop:disabled{opacity:.4}
.empty{text-align:center;padding:50px;color:var(--txt2)}.empty h2{margin-bottom:6px;font-size:1rem}
</style></head>
<body>
<nav class="nav"><h1>MARL Training Dashboard</h1>
<div class="nav-stats">
<div class="stat"><div class="dot" id="dot"></div><span class="stat-lbl">Status:</span><span class="stat-val" id="st">Init</span></div>
<div class="stat"><span class="stat-lbl">Active:</span><span class="stat-val" id="act">0</span></div>
<div class="stat"><span class="stat-lbl">Done:</span><span class="stat-val" id="dn">0</span></div>
<div class="stat"><span class="stat-lbl">Time:</span><span class="stat-val" id="tm">00:00:00</span></div>
</div></nav>
<main class="main">
<div class="sum">
<div class="sum-c"><div class="lbl">Profiles</div><div class="val" id="s-tot">0</div></div>
<div class="sum-c"><div class="lbl">Steps</div><div class="val" id="s-stp">0</div></div>
<div class="sum-c"><div class="lbl">Speed</div><div class="val" id="s-spd">0/s</div></div>
<div class="sum-c"><div class="lbl">Avg Reward</div><div class="val" id="s-rwd">-</div></div>
</div>
<div class="grid" id="grid"><div class="empty"><h2>Waiting for Training</h2><p>Start training to see metrics</p></div></div>
</main>
<script>
const charts={},DC_COL=['#ef4444','#f97316','#eab308','#84cc16','#22c55e','#14b8a6','#06b6d4','#0ea5e9','#3b82f6','#6366f1','#8b5cf6','#a855f7'];
const fmt=s=>{if(!s||!isFinite(s))return'--:--:--';const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),ss=Math.floor(s%60);return`${h.toString().padStart(2,'0')}:${m.toString().padStart(2,'0')}:${ss.toString().padStart(2,'0')}`};
const num=(v,d=2)=>{if(v==null||!isFinite(v))return'-';if(Math.abs(v)>=1e6)return(v/1e6).toFixed(1)+'M';if(Math.abs(v)>=1e3)return(v/1e3).toFixed(1)+'K';return Math.abs(v)>=100?v.toFixed(0):v.toFixed(d)};
const fmtM=(v,t)=>{if(!v||!isFinite(v))return'-';return t==='ttft'?v.toFixed(4):t==='cost'?'$'+v.toFixed(2):num(v)};
function mkCard(p){const st=p.is_training?'training':(p.is_completed?'done':''),bc=p.agent_type==='single_metric'?'b-single':'b-multi',tl=p.agent_type==='single_metric'?(p.primary_metric||'').toUpperCase():'MULTI',sb=p.is_training?'b-train':(p.is_completed?'b-done':''),sl=p.is_training?'TRAINING':(p.is_completed?'DONE':'WAIT');
return`<div class="card ${st}" id="c-${p.profile_id}"><div class="card-hd"><div class="card-ti"><span class="card-nm">${p.profile_id}</span><span class="badge ${bc}">${tl}</span></div><span class="badge ${sb}">${sl}</span></div>
<div class="prog"><div class="prog-bar"><div class="prog-fill" id="pf-${p.profile_id}" style="width:${p.progress_pct}%"></div></div><div class="prog-info"><span>${num(p.progress_pct,1)}% · ${num(p.current_timesteps)} steps</span><span>ETA ${fmt(p.eta_seconds)} · ${num(p.steps_per_second,1)} st/s</span></div></div>
<div class="met"><div class="met-grid">
<div class="met-i"><div class="lbl">TTFT</div><div class="val ttft" id="m-ttft-${p.profile_id}">${fmtM(p.current?.ttft,'ttft')}</div><div class="bst">best: ${fmtM(p.best?.ttft,'ttft')}</div></div>
<div class="met-i"><div class="lbl">Carbon</div><div class="val carbon" id="m-carbon-${p.profile_id}">${fmtM(p.current?.carbon)}</div><div class="bst">best: ${fmtM(p.best?.carbon)}</div></div>
<div class="met-i"><div class="lbl">Water</div><div class="val water" id="m-water-${p.profile_id}">${fmtM(p.current?.water)}</div><div class="bst">best: ${fmtM(p.best?.water)}</div></div>
<div class="met-i"><div class="lbl">Cost</div><div class="val cost" id="m-cost-${p.profile_id}">${fmtM(p.current?.cost,'cost')}</div><div class="bst">best: ${fmtM(p.best?.cost,'cost')}</div></div></div>
<div class="ch-row"><div class="ch-box"><div class="ch-ti">Reward (smoothed)</div><div class="ch-wrap"><canvas id="ch-r-${p.profile_id}"></canvas></div></div>
<div class="ch-box"><div class="ch-ti">${p.primary_metric||'Metric'}</div><div class="ch-wrap"><canvas id="ch-m-${p.profile_id}"></canvas></div></div></div>
<div class="pl-row"><div class="pl-box"><div class="pl-ti">Power</div><div class="pw-grid" id="pw-${p.profile_id}">${[...Array(12)].map((_,i)=>`<div class="pw-c off"><span class="dc">DC${i}</span><span>-</span></div>`).join('')}</div></div>
<div class="pl-box"><div class="pl-ti">Routing</div><div class="rt-bar" id="rt-${p.profile_id}">${[...Array(12)].map((_,i)=>`<div class="rt-seg" style="width:0;background:${DC_COL[i]}"></div>`).join('')}</div></div></div></div>
<div class="card-ft"><div class="st-grp"><div class="mini"><span>μ:</span><span>${num(p.stats?.reward_mean,3)}</span></div><div class="mini"><span>σ:</span><span>${num(p.stats?.reward_std,3)}</span></div><div class="mini"><span>↓</span><span>${num(p.stats?.reward_min,2)}</span></div><div class="mini"><span>↑</span><span>${num(p.stats?.reward_max,2)}</span></div></div>
<button class="btn-stop" onclick="stop('${p.profile_id}')" ${!p.is_training?'disabled':''}>Stop</button></div></div>`}
function mkCh(id,col){const ctx=document.getElementById(id)?.getContext('2d');if(!ctx)return null;return new Chart(ctx,{type:'line',data:{labels:[],datasets:[{data:[],borderColor:col,borderWidth:1.5,fill:false,tension:.3,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{grid:{color:'rgba(48,54,61,.5)'},ticks:{font:{size:7},color:'#8b949e',maxTicksLimit:3}}}}})}
function updPw(id,pl){const g=document.getElementById('pw-'+id);if(!g||!pl)return;g.querySelectorAll('.pw-c').forEach((c,i)=>{const d=pl[i]||pl[String(i)];if(d?.unit){const on=Object.values(d.unit).filter(v=>v==='ON').length,t=Object.keys(d.unit).length,pct=t?(on/t)*100:0;c.className='pw-c '+(pct>=70?'high':pct>=30?'med':pct>0?'low':'off');c.innerHTML=`<span class="dc">DC${i}</span><span>${on}/${t}</span>`}})}
function updRt(id,dist){const b=document.getElementById('rt-'+id);if(!b||!dist?.length)return;const segs=b.querySelectorAll('.rt-seg'),tot=dist.reduce((a,b)=>a+b,0)||1;dist.forEach((v,i)=>{if(segs[i]){const pct=(v/tot)*100;segs[i].style.width=pct+'%';segs[i].textContent=pct>10?i:''}})}
async function stop(id){if(confirm('Stop '+id+'?'))await fetch('/api/stop/'+id,{method:'POST'})}
async function refresh(){try{const r=await fetch('/api/data'),d=await r.json(),ps=Object.values(d.profiles||{});
document.getElementById('dot').className='dot '+(d.training_active?'':'idle');document.getElementById('st').textContent=d.training_active?'Training':'Idle';
document.getElementById('act').textContent=d.summary?.active||0;document.getElementById('dn').textContent=d.summary?.completed||0;document.getElementById('tm').textContent=fmt(d.global_elapsed);
let totS=0,totSPS=0,rSum=0,rCnt=0;ps.forEach(p=>{totS+=p.current_timesteps||0;totSPS+=p.steps_per_second||0;if(p.stats?.reward_mean){rSum+=p.stats.reward_mean;rCnt++}});
document.getElementById('s-tot').textContent=ps.length;document.getElementById('s-stp').textContent=num(totS);document.getElementById('s-spd').textContent=num(totSPS,1)+'/s';document.getElementById('s-rwd').textContent=rCnt?num(rSum/rCnt,3):'-';
const g=document.getElementById('grid');if(!ps.length){g.innerHTML='<div class="empty"><h2>Waiting</h2><p>Start training</p></div>';return}
ps.forEach(p=>{let c=document.getElementById('c-'+p.profile_id);if(!c){g.querySelector('.empty')?.remove();const t=document.createElement('div');t.innerHTML=mkCard(p);g.appendChild(t.firstElementChild);
charts['r-'+p.profile_id]=mkCh('ch-r-'+p.profile_id,'#58a6ff');const mc=p.primary_metric==='ttft'?'#58a6ff':p.primary_metric==='carbon'?'#3fb950':p.primary_metric==='water'?'#39c5cf':'#d29922';charts['m-'+p.profile_id]=mkCh('ch-m-'+p.profile_id,mc);c=document.getElementById('c-'+p.profile_id)}
c.className='card '+(p.is_training?'training':p.is_completed?'done':'');const pf=document.getElementById('pf-'+p.profile_id);if(pf)pf.style.width=p.progress_pct+'%';
['ttft','carbon','water','cost'].forEach(m=>{const e=document.getElementById('m-'+m+'-'+p.profile_id);if(e)e.textContent=fmtM(p.current?.[m],m)});
const rc=charts['r-'+p.profile_id];if(rc&&p.timesteps?.length){rc.data.labels=p.timesteps;rc.data.datasets[0].data=p.rewards_smoothed||p.rewards;rc.update('none')}
const mc=charts['m-'+p.profile_id];if(mc&&p.timesteps?.length){const md=p.primary_metric==='ttft'?p.ttft:p.primary_metric==='carbon'?p.carbon:p.primary_metric==='water'?p.water:p.cost;if(md?.length){mc.data.labels=p.timesteps.slice(-md.length);mc.data.datasets[0].data=md;mc.update('none')}}
updPw(p.profile_id,p.power_plan);updRt(p.profile_id,p.routing_dist);const btn=c.querySelector('.btn-stop');if(btn)btn.disabled=!p.is_training})}catch(e){console.error(e)}}
refresh();setInterval(refresh,1000);
</script></body></html>'''


@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)


@app.route('/api/data')
def get_data(): return jsonify(dashboard.get_all_data())


@app.route('/api/stop/<pid>', methods=['POST'])
def stop_profile(pid):
    dashboard.request_stop(pid)
    return jsonify({"status": "ok"})


_server = None


def run_dashboard_server(port=5000):
    global _server
    if _server and _server.is_alive(): return _server

    def run():
        import logging;
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)

    _server = threading.Thread(target=run, daemon=True)
    _server.start()
    time.sleep(0.3)
    print(f"\n{'=' * 50}\n  MARL Dashboard: http://localhost:{port}\n{'=' * 50}\n")
    return _server


if __name__ == "__main__":
    run_dashboard_server(5000)
    print("Running. Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Done.")