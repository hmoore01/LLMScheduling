#!/usr/bin/env python3
"""
autopush_results.py
===================
Periodically commits and pushes THIS machine's experiment results to GitHub.
Linux only. Built for MANY machines pushing to one repo at the same time.

How the multi-machine case is kept conflict-free
------------------------------------------------
  * Each machine commits its OWN uniquely-named file:
        experiment_results/LAHyper_Results_<machine>.csv
    where <machine> is the hostname (override with the MACHINE_ID env var).
    Machines never edit the same file, so there are never content conflicts.
  * Two machines can still race on `git push` (non-fast-forward). That's
    resolved with `pull --rebase --autostash` and retried -- clean, because the
    files are disjoint, so the rebase just replays this machine's commit on top.

Throttle (don't spam GitHub)
----------------------------
  * --min-interval N : push AT MOST once every N seconds (default 1800 = 30 min).
    Enforced via a small state file, so it holds even across restarts / --once.
  * The loop wakes every --check-interval seconds just to see if the cooldown
    has elapsed AND there's new data; if either isn't true, it does nothing.

Run standalone (recommended -- its own process):
    python autopush_results.py --min-interval 1800

Or drive it from the experiment as a background thread:
    import autopush_results
    autopush_results.start_background_pusher("/path/to/repo", min_interval=1800)

Auth (do ONCE, as the user this runs as):
    git config --global credential.helper store    # then one manual push w/ a PAT
  or an SSH key with no passphrase. GIT_TERMINAL_PROMPT=0 is forced below, so an
  un-cached credential FAILS FAST instead of hanging the process forever.
"""

import argparse
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime

# -- Config --------------------------------------------------------------------
REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.abspath(__file__)))

# What the experiment runner writes (shared name -- never committed directly).
SOURCE_REL = "experiment_results/LAHyper_Results.csv"


# Per-machine identity used in the committed filename.
def _machine_id():
    raw = os.environ.get("MACHINE_ID") or socket.gethostname() or "machine"
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)


MACHINE_ID    = _machine_id()
COMMITTED_REL = f"experiment_results/LAHyper_Results_{MACHINE_ID}.csv"
SNAPSHOT_DIR  = "experiment_results/snapshots"

DEFAULT_MIN_INTERVAL_SEC   = 1800   # push at most this often
DEFAULT_CHECK_INTERVAL_SEC = 60     # how often the loop checks the cooldown


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run_git(args, timeout=120):
    """Run `git -C REPO_DIR <args>` with prompts disabled and a hard timeout.
    A timeout returns a synthetic non-zero result so callers treat it as a
    failure and simply retry next cycle -- it can never hang the process."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        return subprocess.run(["git", "-C", REPO_DIR, *args],
                              capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"git {' '.join(args)} timed out after {timeout}s")
        return subprocess.CompletedProcess(args, 124, "", "timeout")


# -- Throttle state (survives restarts / --once) -------------------------------
def _state_path():
    return os.path.join(REPO_DIR, "experiment_results", ".autopush_state")


def _last_push_ts():
    try:
        with open(_state_path()) as f:
            return float(f.read().strip())
    except Exception:
        return 0.0


def _set_last_push_ts(ts):
    try:
        os.makedirs(os.path.dirname(_state_path()), exist_ok=True)
        with open(_state_path(), "w") as f:
            f.write(str(ts))
    except Exception as e:
        log(f"(couldn't write throttle state: {e})")


def _try_push(max_attempts=4):
    """Push, resolving non-fast-forward races from other machines via rebase."""
    push = run_git(["push"])
    if push.returncode == 0:
        return True
    for attempt in range(1, max_attempts + 1):
        log(f"push rejected (another machine likely pushed) -- rebase+retry {attempt}/{max_attempts}")
        rb = run_git(["pull", "--rebase", "--autostash"])
        if rb.returncode != 0:
            log(f"rebase failed, aborting to keep tree clean: {rb.stderr.strip()[:300]}")
            run_git(["rebase", "--abort"])
            return False
        push = run_git(["push"])
        if push.returncode == 0:
            return True
    log("push still failing after retries -- will try again next cycle")
    return False


def push_once(min_interval=DEFAULT_MIN_INTERVAL_SEC, snapshots=False):
    """Copy this machine's results to its unique committed file, then commit +
    push -- but only if (a) the cooldown has elapsed and (b) data actually
    changed. Returns True only when a push happened."""
    now = time.time()
    if now - _last_push_ts() < min_interval:
        return False  # throttled -- too soon since last push

    src = os.path.join(REPO_DIR, SOURCE_REL)
    if not os.path.exists(src):
        return False  # runner hasn't written anything yet

    targets = [COMMITTED_REL]
    shutil.copy2(src, os.path.join(REPO_DIR, COMMITTED_REL))

    # Optional: also keep timestamped point-in-time copies (repo grows over time).
    if snapshots:
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_rel = f"{SNAPSHOT_DIR}/LAHyper_Results_{MACHINE_ID}_{stamp}.csv"
        os.makedirs(os.path.join(REPO_DIR, SNAPSHOT_DIR), exist_ok=True)
        shutil.copy2(src, os.path.join(REPO_DIR, snap_rel))
        targets.append(snap_rel)

    add = run_git(["add", "--", *targets])
    if add.returncode != 0:
        log("git add failed (is the committed file matched by .gitignore? give "
            "it a per-host name that isn't ignored):\n      "
            + (add.stderr.strip() or "(no message)")[:400])
        return False
    if run_git(["diff", "--cached", "--quiet"]).returncode == 0:
        return False  # nothing new since last push -- don't burn the cooldown

    msg = f"results[{MACHINE_ID}]: {datetime.now():%Y-%m-%d %H:%M:%S}"
    commit = run_git(["commit", "-m", msg])
    if commit.returncode != 0:
        detail = (commit.stderr.strip() or commit.stdout.strip() or "(no message)")
        hint = ""
        if "identity" in detail.lower() or "who you are" in detail.lower():
            hint = ("\n    Set a commit identity once (email need not be real):\n"
                    "      git config --global user.name  \"Your Name\"\n"
                    "      git config --global user.email \"you@example.com\"")
        log("commit failed:\n      " + detail[:400] + hint)
        return False

    if _try_push():
        _set_last_push_ts(time.time())
        log(f"pushed {COMMITTED_REL}")
        return True
    return False


def preflight(repo_dir=None, remote="origin", timeout=30):
    """Verify this machine can push to GitHub BEFORE the run starts.

    Non-destructive: it lists the remote's refs (which exercises connectivity +
    authentication) but writes nothing. For GitHub the same credential that can
    read can push (a PAT with `repo` scope, or an SSH key), so a successful
    ls-remote is a strong signal that pushes will succeed too -- barring branch
    protection. Returns (ok: bool, message: str)."""
    global REPO_DIR
    if repo_dir:
        REPO_DIR = os.path.abspath(repo_dir)

    # 1. Are we inside a git work tree at all?
    if run_git(["rev-parse", "--is-inside-work-tree"]).returncode != 0:
        return False, f"{REPO_DIR} is not a git repository."

    # 2. Is a remote configured?
    url = run_git(["remote", "get-url", remote])
    if url.returncode != 0:
        return False, (f"no git remote '{remote}' configured here "
                       f"(add one: git remote add {remote} <url>).")
    remote_url = url.stdout.strip()

    # 2b. Is a commit identity configured? Read-only checks below don't need it,
    #     but `git commit` does -- so verify now, or commits fail later while
    #     the rest looks healthy.
    name  = run_git(["config", "user.name"]).stdout.strip()
    email = run_git(["config", "user.email"]).stdout.strip()
    if not name or not email:
        return False, ("git commit identity not set -- pushes authenticate but "
                       "commits will fail. Set it once (email need not be real):\n"
                       "      git config --global user.name  \"Your Name\"\n"
                       "      git config --global user.email \"you@example.com\"")

    # 3. Can we reach AND authenticate to it, non-interactively?
    #    GIT_TERMINAL_PROMPT=0 (forced in run_git) makes a missing credential
    #    fail fast here instead of hanging a background thread later.
    ls = run_git(["ls-remote", "--heads", remote], timeout=timeout)
    if ls.returncode != 0:
        return False, (
            f"cannot reach/authenticate to {remote_url}:\n"
            f"      {ls.stderr.strip()[:400]}\n"
            f"    Fix (as the user this runs as): cache a credential once --\n"
            f"      git config --global credential.helper store\n"
            f"      git push        # paste a PAT (repo scope) as the password\n"
            f"    ...or use an SSH key with no passphrase."
        )

    return True, f"push OK  ->  {remote_url}  (as {MACHINE_ID})"


def check_can_push(repo_dir=None, remote="origin"):
    """Run preflight() and print a clear, prominent result. Returns the ok bool.
    Does NOT raise -- a misconfigured machine is reported loudly but the caller
    decides whether to proceed."""
    ok, msg = preflight(repo_dir, remote)
    if ok:
        log(f"preflight: {msg}")
    else:
        bar = "=" * 72
        log(bar)
        log("AUTOPUSH PREFLIGHT FAILED -- results will NOT upload to GitHub:")
        for line in msg.splitlines():
            log(f"  {line}")
        log(bar)
    return ok


def start_background_pusher(repo_dir, min_interval=DEFAULT_MIN_INTERVAL_SEC,
                            check_interval=DEFAULT_CHECK_INTERVAL_SEC, snapshots=False):
    """Importable entry point: run the pusher as a daemon thread inside another
    process (e.g. the experiment runner). Returns the Thread."""
    global REPO_DIR
    REPO_DIR = os.path.abspath(repo_dir)

    def loop():
        while True:
            try:
                push_once(min_interval=min_interval, snapshots=snapshots)
            except Exception as e:
                log(f"(push thread error, continuing: {e})")
            time.sleep(check_interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def main():
    global REPO_DIR
    p = argparse.ArgumentParser(description="Auto-push experiment results to GitHub (Linux, multi-machine).")
    p.add_argument("--repo", default=REPO_DIR, help="Path to the git repo root.")
    p.add_argument("--min-interval", type=int, default=DEFAULT_MIN_INTERVAL_SEC,
                   help="Minimum seconds between pushes (default 1800).")
    p.add_argument("--check-interval", type=int, default=DEFAULT_CHECK_INTERVAL_SEC,
                   help="Seconds between cooldown checks (default 60).")
    p.add_argument("--snapshots", action="store_true",
                   help="Also keep timestamped per-machine copies in snapshots/.")
    p.add_argument("--once", action="store_true",
                   help="Attempt a single push and exit (respects the throttle).")
    args = p.parse_args()
    REPO_DIR = os.path.abspath(args.repo)

    if run_git(["rev-parse", "--is-inside-work-tree"]).returncode != 0:
        log(f"ERROR: {REPO_DIR} is not a git repository.")
        raise SystemExit(1)

    # Loud preflight so a credential/remote problem is obvious at launch. We
    # don't exit on failure -- the network/creds may come good later and the
    # loop self-heals -- but you'll see the warning immediately.
    check_can_push()

    log(f"machine={MACHINE_ID}  repo={REPO_DIR}")
    log(f"committing: {COMMITTED_REL}")
    log(f"min-interval={args.min_interval}s  check-interval={args.check_interval}s  "
        f"snapshots={args.snapshots}  once={args.once}")

    if args.once:
        push_once(min_interval=args.min_interval, snapshots=args.snapshots)
        return

    while True:
        try:
            push_once(min_interval=args.min_interval, snapshots=args.snapshots)
        except Exception as e:
            log(f"unexpected error (continuing): {e}")
        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
