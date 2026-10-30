"""Balance-patch heuristic loop: improve the champion, then search a counter to it, repeat (CPU, Python sim).

Every generation's candidates are split over the hosts by their free CPU (gpu2 / wsl / this Mac, checked before each
generation), played with examples/econ_stats.py against the evaluation pool (post-patch heuristics only), pulled back
to reports/bp/<gen>/ and ranked by the paired margin vs the champion on the same (opponent, seed).

  improve: CEM around the champion (+ the newest counter) vs the whole pool -> top-k confirmed on fresh seeds ->
           adopted as the new champion when the paired gain over the pool is clear.
  counter: wide CEM (structural knobs on) vs the champion alone -> best confirmed -> joins the pool if it beats it.

State lives in reports/bp/state.json (resumable).  Run detached:  nohup .venv/bin/python scripts/bp_loop.py &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from blackout_env.heuristics.params import resolve_descriptor, sample_overrides, with_params  # noqa: E402

OUT = REPO / "reports" / "bp"
STATE = OUT / "state.json"
THREADS = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=."
# host -> (cores we may fill, reserve kept free, remote repo or None for local)
HOSTS = {"poscat_gpu2": (28, 6, "~/blackout-env-bp"), "wsl_gpu": (12, 2, "~/blackout-env-bp"), "mac": (18, 2, None),
         "poscat_gpu1": (28, 2, "~/blackout-env-bp")}
MAX_WORKERS = {"poscat_gpu1": 16,  # user (2026-10-31): gpu1 CPU up to ~16 cores, never disturbing others
               "poscat_gpu2": 6}   # user (10-31 05:15): kyungdain trains for the same goal on gpu2 -> leave it room


def max_workers(h, default):
    # user (10-31 05:10): gpu1 may use 20 cores until 08:00 KST that morning
    if h == "poscat_gpu1" and time.time() < time.mktime((2026, 10, 31, 8, 0, 0, 0, 0, -1)):
        return 20
    return MAX_WORKERS.get(h, default)
CAP_S = 300          # match cap (s); matches are decided early, most end by 100 points
SCREEN_SEEDS = 24    # x both sides per opponent
CONFIRM_SEEDS = 64


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (OUT / "loop.log").open("a") as fh:
        fh.write(line + "\n")


def sh(cmd, host=None, timeout=None):
    if host and host != "mac":
        cmd = ["ssh", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=60", host, cmd]
    else:
        cmd = ["bash", "-c", cmd]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------------------------------- hosts

FOREIGN_CPU = ("ps -eo uid=,pcpu=,args= | awk -v me=\"$(id -u)\" "
               "'!($1==me && $0 ~ /econ_stats|multiprocessing/) {s+=$2} END {printf \"%d\", s/100}'")


def free_workers() -> dict[str, int]:
    out = {}
    for h, (cores, reserve, _) in HOSTS.items():
        try:
            r = sh(FOREIGN_CPU, h, timeout=30)
            busy = int(r.stdout.strip() or 999) if r.returncode == 0 else 999
        except Exception:  # unreachable (the laptop sleeps)
            busy = 999
        out[h] = min(max_workers(h, cores), max(0, cores - reserve - busy))
    return out


def sync_code():
    br = subprocess.run(["git", "-C", str(REPO), "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
    for h, (_, _, repo) in HOSTS.items():
        if repo is None:
            continue
        subprocess.run(["git", "-C", str(REPO), "push", "-q", "-f", h, f"{br}:refs/heads/bp-incoming"], capture_output=True)
        sh(f"cd {repo} && git fetch -q origin bp-incoming && git checkout -q -B feat/balance-patch FETCH_HEAD", h, timeout=60)


def run_on(host, cands, opps, seeds, seed0, gen, workers):
    """Play ``cands`` x ``opps`` on ``host``; results land in OUT/<gen>/<host>/matches.jsonl locally."""
    repo = HOSTS[host][2] or str(REPO)
    rel = f"reports/bp/{gen}"
    (OUT / gen).mkdir(parents=True, exist_ok=True)
    cf = OUT / gen / "cands.txt"
    cf.write_text(";".join(cands))
    of = OUT / gen / "opps.txt"
    of.write_text(";".join(opps))
    if host != "mac":
        sh(f"mkdir -p {repo}/{rel}", host)
        subprocess.run(["scp", "-q", str(cf), str(of), f"{host}:{repo}/{rel}/"], check=True)
    cmd = (f"cd {repo} && {THREADS} nice -n 19 .venv/bin/python examples/econ_stats.py play --ours \"$(cat {rel}/cands.txt)\" "
           f"--opponent \"$(cat {rel}/opps.txt)\" --seeds {seeds} --seed0 {seed0} --cap-s {CAP_S} --workers {workers} "
           f"--out {rel} > {rel}/play.log 2>&1")
    for attempt in range(3):
        r = sh(cmd, host)
        if host != "mac":
            subprocess.run(["rsync", "-a", f"{host}:{repo}/{rel}/", str(OUT / gen) + "/"], capture_output=True)
        if r.returncode == 0:
            return True
        log(f"  {host} failed (attempt {attempt}): {r.stderr[-300:]}")
        time.sleep(60)
    return False


def play(gen, cands, opps, seeds, seed0, must_host_first=()):
    """Play ``cands`` x ``opps`` over all hosts with free CPU: a shared queue, each host pulls small batches
    (so the slowest host never holds the generation back).  Blocking."""
    done = OUT / gen / "DONE"
    if done.exists():
        return load(gen)
    sync_code()
    free = {h: w for h, w in free_workers().items() if w >= 2}
    if not free:
        log("no free CPU anywhere; waiting 10 min")
        time.sleep(600)
        return play(gen, cands, opps, seeds, seed0, must_host_first)
    have = {r["ours"] for r in load(gen)}
    queue = [c for c in cands if c not in have]
    lock = threading.Lock()
    log(f"{gen}: {len(cands)} cands ({len(queue)} to play) x {len(opps)} opps x {seeds} seeds; "
        + ", ".join(f"{h}:{w}w" for h, w in free.items()))

    def worker(h):
        k = 0
        # batch ~ enough chunks (16 games each) to fill the host's workers once
        per = max(1, round(free[h] * 16 / (len(opps) * seeds * 2 * 1.0)))
        while True:
            with lock:
                batch = queue[:per]
                del queue[:per]
            if not batch:
                return
            ok = run_on(h, batch, opps, seeds, seed0, f"{gen}/{h}_{k}", free[h])
            k += 1
            if not ok:
                with lock:
                    queue.extend(batch)
                return

    threads = [threading.Thread(target=worker, args=(h,)) for h in free]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = load(gen)
    missing = [c for c in cands if c not in {r["ours"] for r in rows}]
    if missing:
        log(f"{gen}: {len(missing)} candidates missing, replaying them")
        return play(gen, cands, opps, seeds, seed0)
    done.write_text("ok")
    return rows


def load(gen):
    rows = []
    for f in (OUT / gen).glob("**/matches.jsonl"):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return rows


# ---------------------------------------------------------------------------------------------------- ranking

def table(rows):
    """{cand: {opp: {(seed): mean margin over sides}}}"""
    t = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        t[r["ours"]][r["opponent"]][r["seed"]].append(r["margin"])
    return {c: {o: {s: float(np.mean(v)) for s, v in d.items()} for o, d in od.items()} for c, od in t.items()}


def paired(t, cand, base, opps):
    """Mean over opponents of the paired (seed) margin diff cand - base, its SE, and cand's raw mean margin."""
    per_opp, se2, raw = [], [], []
    for o in opps:
        a, b = t.get(cand, {}).get(o, {}), t.get(base, {}).get(o, {})
        ks = [k for k in a if k in b]
        if len(ks) < 4:
            continue
        d = np.array([a[k] - b[k] for k in ks])
        per_opp.append(d.mean())
        se2.append(d.var(ddof=1) / len(d))
        raw.append(np.mean(list(a.values())))
    if not per_opp:
        return np.nan, np.nan, np.nan
    return float(np.mean(per_opp)), float(np.sqrt(np.sum(se2)) / len(per_opp)), float(np.mean(raw))


def raw_margin(t, cand, opps):
    v = [np.mean(list(t[cand][o].values())) for o in opps if o in t.get(cand, {})]
    return float(np.mean(v)) if v else np.nan


# ---------------------------------------------------------------------------------------------------- loop

def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    seed = resolve_descriptor("V34a")
    return {"gen": 0, "pool": {"B0": seed}, "champion": "B0", "counter": None, "phase": "improve", "stale": 0,
            "history": []}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1))


def cem(bases, n, seed, scale, structural):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        base = bases[len(out) % len(bases)]
        ov = sample_overrides(base, rng, scale, structural=structural)
        if not ov:
            continue
        d = with_params(base, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ov.items()})
        if d not in out and d not in bases:
            out.append(d)
    return out


def name_of(st, desc):
    for k, v in st["pool"].items():
        if v == desc:
            return k
    return None


def step(st):
    g = st["gen"]
    champ = st["pool"][st["champion"]]
    pool = list(st["pool"].values())
    if st["phase"] == "improve":
        bases = [champ] + ([st["pool"][st["counter"]]] if st["counter"] else [])
        cands = cem(bases, 40, 1000 + g, 1.0 if st["stale"] < 2 else 0.6, structural=True)
        rows = play(f"g{g:03d}_imp", [champ] + cands, pool, SCREEN_SEEDS, 10_000_000 + 1000 * g)
        t = table(rows)
        scored = sorted(((paired(t, c, champ, pool), c) for c in cands if c in t), key=lambda x: -np.nan_to_num(x[0][0], nan=-1e9))
        top = [c for _, c in scored[:6]]
        log(f"g{g} improve screen top: " + "; ".join(f"{s[0]:+.1f}±{s[1]:.1f}" for s, _ in scored[:6]))
        rows = play(f"g{g:03d}_impc", [champ] + top, pool, CONFIRM_SEEDS, 20_000_000 + 1000 * g)
        t = table(rows)
        conf = sorted(((paired(t, c, champ, pool), c) for c in top), key=lambda x: -np.nan_to_num(x[0][0], nan=-1e9))
        (d, se, raw), best = conf[0]
        log(f"g{g} improve confirm: " + "; ".join(f"{s[0]:+.1f}±{s[1]:.1f}" for s, _ in conf))
        if d > 1.5 and d > 2 * se:
            k = f"B{sum(1 for x in st['pool'] if x.startswith('B'))}"
            st["pool"][k] = best
            st["champion"] = k
            st["stale"] = 0
            log(f"g{g}: NEW CHAMPION {k} (+{d:.1f}±{se:.1f} over pool vs old)")
            st["history"].append({"gen": g, "event": "champion", "name": k, "gain": d, "se": se})
        else:
            st["stale"] += 1
            if st["stale"] >= 2:
                st["phase"] = "counter"
                st["stale"] = 0
    else:  # counter: beat the champion
        bases = [champ] + [v for k, v in st["pool"].items() if k != st["champion"]][-2:]
        cands = cem(bases, 40, 5000 + g, 2.0, structural=True)
        rows = play(f"g{g:03d}_ctr", cands, [champ], SCREEN_SEEDS * 2, 30_000_000 + 1000 * g)
        t = table(rows)
        scored = sorted(((raw_margin(t, c, [champ]), c) for c in cands if c in t), key=lambda x: -np.nan_to_num(x[0], nan=-1e9))
        top = [c for _, c in scored[:6]]
        log(f"g{g} counter screen top: " + "; ".join(f"{s:+.1f}" for s, _ in scored[:6]))
        rows = play(f"g{g:03d}_ctrc", top, [champ], CONFIRM_SEEDS, 40_000_000 + 1000 * g)
        t = table(rows)
        conf = sorted(((raw_margin(t, c, [champ]), c) for c in top), key=lambda x: -np.nan_to_num(x[0], nan=-1e9))
        m, best = conf[0]
        log(f"g{g} counter confirm: " + "; ".join(f"{s:+.1f}" for s, _ in conf))
        if m > 4.0:
            k = f"C{sum(1 for x in st['pool'] if x.startswith('C'))}"
            st["pool"][k] = best
            st["counter"] = k
            log(f"g{g}: COUNTER {k} beats {st['champion']} by {m:+.1f} -> joins the pool")
            st["history"].append({"gen": g, "event": "counter", "name": k, "margin": m})
        else:
            log(f"g{g}: no counter found (best {m:+.1f})")
            st["history"].append({"gen": g, "event": "no_counter", "margin": m})
        st["phase"] = "improve"
    st["gen"] = g + 1
    save_state(st)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = load_state()
    save_state(st)
    while not (OUT / "STOP").exists():
        st = load_state()  # re-read: state.json may be edited by hand between generations
        step(st)
    log("STOP file found, exiting")


if __name__ == "__main__":
    main()
