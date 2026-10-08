"""Elo-24: Bradley-Terry Elo + win-rate / margin heatmaps of a policy pool, distributed over several hosts.

Simulator only, full matches through ``examples/v27_eval.py``'s ``play_chunk`` (cap 425 s, ends at an absorption
that leaves no battery in play -- the gate setting of docs §35+).  Policies, seeds, stopping rules and hosts live in
``examples/elo24_config.json`` (heuristic descriptors, ``net:<path>`` students).  A game is (a, b, seed, side of
a) with a before b in config order; score 1 / 0.5 / 0 for a win / draw (margin 0) / loss; a *seed* always has both
side-swapped games.

Stages
------
1. every pair plays the fixed seed block ``stage1.seed0 + 0..seeds-1`` from both sides (276 pairs x 20 games).
2. rounds of extra seeds (``stage2.seed0 + k``, the k-th extra seed of that pair) until both criteria hold:
   (A) every pair's win-rate stopping SE <= ``pair_se`` (5 %) -- paired-seed SE with one won + one lost pseudo-seed
       (blackout_env/rating.py), topped up first, clamp(needed - n, min_add, max_add) seeds per round;
   (B) every Elo SE <= ``elo_se`` (12) -- greedy A-optimal picks (Sherman-Morrison, as elo_active.py), only once
       (A) holds.

Distribution
------------
The coordinator (``run``, on the Mac) owns ``<out>/plan.jsonl`` (planned chunks) and ``<out>/results.jsonl`` (one row
per game, append-only, dedup on the game key; a restart re-plans nothing and only queues the games still missing).
One thread per host opens a *session*: ``ssh host 'cd <dir> && nice -n N env ...THREADS=1 CUDA_VISIBLE_DEVICES=
python examples/elo24.py worker --workers W'`` (the Mac runs it locally), feeds chunk jobs over stdin as earlier ones
finish (dynamic load balancing), and reads ``ROW`` / ``DONE`` / ``FAIL`` lines back.  W is re-sized from the host's
load at every session start (sessions end after ``--session-min`` minutes); the Mac only works on AC power (checked
before each chunk) and the coordinator holds ``caffeinate -i`` while it runs.  A worker whose stdin closes without
``END`` (coordinator killed, ssh dropped) kills its own process group, so nothing is orphaned; its unfinished chunks
are re-queued.

    ED=examples/elo24.py; PY=../blackout-env/.venv/bin/python   # from this checkout
    PYTHONPATH=. $PY $ED hosts                      # AC / load / workers each host would get now
    PYTHONPATH=. $PY $ED setup [--dry-run]          # code (tracked files) + needed checkpoints -> ~/blackout-env-elo24
    caffeinate -i nohup env PYTHONPATH=. $PY $ED run --stage 1 > reports/elo24/run1.log 2>&1 &
    PYTHONPATH=. $PY $ED status                     # progress, both stopping criteria
    caffeinate -i nohup env PYTHONPATH=. $PY $ED run --stage 2 > reports/elo24/run2.log 2>&1 &
    PYTHONPATH=. $PY $ED report                     # ratings, heatmaps (PNG + CSV)
    PYTHONPATH=. $PY $ED simulate                   # game count of the stopping rules on synthetic ratings

Stop a run with ``kill <coordinator pid>`` (SIGTERM; a backgrounded job ignores Ctrl-C/SIGINT): every worker
sees its stdin close and kills its process group; rerun ``run`` to resume (finished games are kept).

Remote clones vs scripts/remote_sync.sh
---------------------------------------
``setup`` copies the *committed* tracked files of this checkout (commit first; ``git ls-files``) and the checkpoints
the config needs into ``~/blackout-env-elo24`` on every enabled ssh host and runs it with the shared venv
``~/blackout-env/.venv`` (PYTHONPATH=.).  It never touches the server mirror ``~/blackout-env``.
``scripts/remote_sync.sh push <host>`` is the other route but it git-pushes the *current branch* into
``~/blackout-env`` and checks that branch out there (``git checkout -- .`` first), which changes the code under any
run started from the mirror -- do not use it for this tool while other experiments run from the mirror.
``scripts/remote_sync.sh status|procs|kill-orphans|clean`` are safe (load, my processes, orphan cleanup).
Results never live on the servers: rows stream back over ssh into ``<out>/results.jsonl`` on the Mac.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

from blackout_env.rating import ELO, fit_bt, focus_se_elo, pair_seed_stats, plan_round, rating_se_elo  # noqa: E402

THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
THREAD_ENV["CUDA_VISIBLE_DEVICES"] = ""
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=10"]


# ---------------------------------------------------------------- config / files

def load_cfg(path) -> dict:
    cfg = json.loads(Path(path).read_text())
    names = [p["name"] for p in cfg["policies"]]
    assert len(set(names)) == len(names), "duplicate policy names"
    return cfg


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a torn last line after a crash
            pass
    return out


class Store:
    """plan.jsonl + results.jsonl of one run, restricted to the configured (name, spec) policies."""

    def __init__(self, cfg: dict, out: Path):
        self.cfg, self.out = cfg, out
        out.mkdir(parents=True, exist_ok=True)
        self.names = [p["name"] for p in cfg["policies"]]
        self.spec = {p["name"]: p["spec"] for p in cfg["policies"]}
        self.index = {n: i for i, n in enumerate(self.names)}
        self.lock = threading.Lock()
        self.games: dict[tuple, dict] = {}
        for r in read_jsonl(out / "results.jsonl"):
            if self._valid(r):
                self.games.setdefault(self.key(r), r)
        self.plan = [c for c in read_jsonl(out / "plan.jsonl") if self._valid(c)]

    def _valid(self, r) -> bool:
        return (r["a"] in self.spec and r["b"] in self.spec and self.spec[r["a"]] == r["spec_a"]
                and self.spec[r["b"]] == r["spec_b"] and self.index[r["a"]] < self.index[r["b"]])

    @staticmethod
    def key(r) -> tuple:
        return r["a"], r["b"], int(r["seed"]), int(r["side"])

    def add_rows(self, rows) -> int:
        new = 0
        with self.lock, (self.out / "results.jsonl").open("a") as fh:
            for r in rows:
                if self._valid(r) and self.key(r) not in self.games:
                    self.games[self.key(r)] = r
                    fh.write(json.dumps(r) + "\n")
                    new += 1
        return new

    def add_plan(self, chunks) -> None:
        with self.lock, (self.out / "plan.jsonl").open("a") as fh:
            for c in chunks:
                self.plan.append(c)
                fh.write(json.dumps(c) + "\n")

    def missing(self, chunk) -> list[tuple[int, int]]:
        return [(s, sd) for s in chunk["seeds"] for sd in (0, 1)
                if (chunk["a"], chunk["b"], s, sd) not in self.games]

    def pending(self) -> list[dict]:
        out = []
        for c in self.plan:
            m = self.missing(c)
            if m:
                out.append({**c, "games": m})
        return out

    def planned_seeds(self, a, b, stage) -> list[int]:
        return [s for c in self.plan if c["a"] == a and c["b"] == b and c["stage"] == stage for s in c["seeds"]]

    def pairs(self):
        k = len(self.names)
        return [(self.names[i], self.names[j]) for i in range(k) for j in range(i + 1, k)]

    def seed_scores(self) -> dict:
        """{(i, j): [score of i per complete seed]} for every pair."""
        by = defaultdict(dict)
        for (a, b, s, sd), r in self.games.items():
            by[(a, b)].setdefault(s, {})[sd] = score(r)
        out = {}
        for a, b in self.pairs():
            d = by.get((a, b), {})
            out[(self.index[a], self.index[b])] = [(v[0] + v[1]) / 2 for v in d.values() if len(v) == 2]
        return out

    def fit(self):
        g = list(self.games.values())
        return fit_bt(len(self.names), [self.index[r["a"]] for r in g], [self.index[r["b"]] for r in g],
                      [score(r) for r in g], prior_sd_elo=self.cfg.get("prior_sd_elo", 400.0))


def score(r) -> float:
    return 1.0 if r["margin"] > 0 else 0.5 if r["margin"] == 0 else 0.0


def chunk_of(store: Store, a, b, seeds, stage, batch, reason) -> dict:
    return {"a": a, "b": b, "spec_a": store.spec[a], "spec_b": store.spec[b], "seeds": [int(s) for s in seeds],
            "stage": stage, "batch": batch, "reason": reason}


def plan_stage1(store: Store) -> list[dict]:
    s1 = store.cfg["stage1"]
    want = range(s1["seed0"], s1["seed0"] + s1["seeds"])
    chunks = []
    for a, b in store.pairs():
        have = set(store.planned_seeds(a, b, 1))
        todo = [s for s in want if s not in have]
        if todo:
            chunks.append(chunk_of(store, a, b, todo, 1, 0, "stage1"))
    return chunks


def focus_idx(cfg: dict, r) -> list[int] | None:
    """stage2.focus_top = n > 0: the Elo criterion covers only the current top-n ratings (relative to their mean)."""
    n = int(cfg["stage2"].get("focus_top") or 0)
    return sorted(np.argsort(-np.asarray(r))[:n].tolist()) if n else None


def plan_stage2(store: Store) -> tuple[list[dict], str]:
    s2 = store.cfg["stage2"]
    r, cov, _, _ = store.fit()
    plan, reason = plan_round(store.seed_scores(), r, cov, pair_se=s2["pair_se"], elo_se=s2["elo_se"],
                              min_add=s2["min_add"], max_add=s2["max_add"], elo_pairs=s2["elo_pairs"],
                              per_policy=s2["per_policy"], elo_seeds=s2["elo_seeds"], focus=focus_idx(store.cfg, r))
    batch = 1 + max([c["batch"] for c in store.plan] + [0])
    chunks = []
    for (i, j), n in sorted(plan.items()):
        a, b = store.names[i], store.names[j]
        k0 = len(store.planned_seeds(a, b, 2))
        seeds = [s2["seed0"] + k0 + t for t in range(n)]
        for c in range(0, n, s2["chunk_seeds"]):
            chunks.append(chunk_of(store, a, b, seeds[c:c + s2["chunk_seeds"]], 2, batch, reason))
    return chunks, reason


# ---------------------------------------------------------------- worker (runs on every host)

_OUT_LOCK = threading.Lock()


def _emit(line: str) -> None:
    with _OUT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def run_job(job: dict) -> list[dict]:
    from v27_eval import play_chunk

    seeds = [s for s, _ in job["games"]]
    sides = [sd for _, sd in job["games"]]
    t0 = time.time()
    rows = play_chunk((job["spec_a"], job["spec_b"], seeds, sides, job["cap_s"], job["stop_exhausted"]))
    out = []
    for r in rows:
        out.append({"a": job["a"], "b": job["b"], "spec_a": job["spec_a"], "spec_b": job["spec_b"],
                    "seed": r["seed"], "side": r["side"], "margin": r["margin"], "score_a": r["score_ours"],
                    "score_b": r["score_opp"], "over": r["over"], "exhausted": r["exhausted"], "ticks": r["ticks"],
                    "first_a": r["first_ours"], "first_b": r["first_opp"], "stage": job["stage"],
                    "batch": job["batch"], "host": job.get("host", ""), "chunk_wall_s": round(time.time() - t0, 2),
                    "chunk_n": len(rows)})
    return out


def cmd_worker(args) -> None:
    """Chunk jobs on stdin (one JSON per line, ``END`` = no more), ``ROW``/``DONE``/``FAIL`` lines on stdout.
    stdin closing without ``END`` = the coordinator is gone: kill the whole process group (pool included)."""
    import multiprocessing as mp

    try:
        os.setpgrp()
    except OSError:
        pass
    os.environ.update(THREAD_ENV)
    pool = mp.get_context("spawn").Pool(args.workers)
    inflight = [0]
    cv = threading.Condition()
    ended = [False]

    def finished(job_id, rows=None, err=None):
        if err is None:
            for r in rows:
                _emit("ROW " + json.dumps(r))
            _emit(f"DONE {job_id}")
        else:
            _emit(f"FAIL {job_id} {type(err).__name__}: {str(err)[:300]}")
        with cv:
            inflight[0] -= 1
            cv.notify_all()

    def reader():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "END":
                with cv:
                    ended[0] = True
                    cv.notify_all()
                return
            job = json.loads(line)
            with cv:
                inflight[0] += 1
            pool.apply_async(run_job, (job,), callback=lambda rows, i=job["id"]: finished(i, rows),
                             error_callback=lambda e, i=job["id"]: finished(i, err=e))
        sys.stderr.write("worker: stdin closed without END, aborting\n")
        sys.stderr.flush()
        try:
            os.killpg(0, signal.SIGKILL)
        finally:
            os._exit(3)

    threading.Thread(target=reader, daemon=True).start()
    _emit(f"READY {args.workers}")
    with cv:
        while not (ended[0] and inflight[0] == 0):
            cv.wait(5)
    pool.close()
    pool.join()
    _emit("BYE")


# ---------------------------------------------------------------- hosts

def sh(cmd: list[str], timeout=60) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout


def on_ac() -> bool:
    return "AC Power" in sh(["pmset", "-g", "batt"])


class Host:
    def __init__(self, spec: dict):
        self.__dict__.update(spec)
        self.ssh = spec.get("ssh")
        self.last_w, self.last_end = 0, 0.0
        self.failures = 0

    def probe(self) -> dict:
        """{"ok", "nproc", "load1", "ac", "workers"}: workers this host may run now."""
        if self.ssh is None:
            ac = on_ac() if self.__dict__.get("require_ac") else True
            load1 = os.getloadavg()[0]
            nproc = os.cpu_count() or 1
            other = max(0.0, load1 - self.last_w * math.exp(-(time.time() - self.last_end) / 60))
            w = min(self.max_workers, int(nproc - other - self.__dict__.get("min_free_cores", 2))) if ac else 0
            return {"ok": True, "nproc": nproc, "load1": load1, "ac": ac, "workers": max(0, w)}
        try:
            out = sh(["ssh", *SSH_OPTS, self.ssh, "nproc; cut -d' ' -f1 /proc/loadavg"], timeout=40).split()
            nproc, load1 = int(out[0]), float(out[1])
        except Exception:  # noqa: BLE001
            return {"ok": False, "nproc": 0, "load1": float("nan"), "ac": True, "workers": 0}
        other = max(0.0, load1 - self.last_w * math.exp(-(time.time() - self.last_end) / 60))
        w = min(self.max_workers, int(nproc - other - self.__dict__.get("min_free_cores", 2)))
        return {"ok": True, "nproc": nproc, "load1": load1, "ac": True, "workers": max(0, w)}

    def worker_cmd(self, w: int) -> tuple[list[str], str | None]:
        env = [f"{k}={v}" for k, v in THREAD_ENV.items()] + ["PYTHONPATH=."]
        argv = ["nice", "-n", str(self.nice), "env", *env, self.python, "-u", "examples/elo24.py", "worker",
                "--workers", str(w)]
        if self.ssh is None:
            return argv, str(REPO)
        remote = " ".join(a if a.startswith("~/") else shlex.quote(a) for a in argv)  # keep ~ expandable
        return ["ssh", *SSH_OPTS, self.ssh, f"cd {shlex.quote(self.dir)} && exec {remote}"], None


# ---------------------------------------------------------------- coordinator

class Dispatcher:
    def __init__(self, store: Store, hosts: list[Host], session_s: float, log):
        self.store, self.hosts, self.session_s, self.log = store, hosts, session_s, log
        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.procs: set = set()
        self.tries: dict[int, int] = defaultdict(int)
        self.done_games = 0

    def run(self, chunks: list[dict]) -> None:
        m = self.store.cfg["match"]
        for n, c in enumerate(chunks):
            self.q.put({**c, "id": n, "cap_s": m["cap_s"], "stop_exhausted": m["stop_exhausted"]})
        self.total = sum(len(c["games"]) for c in chunks)
        self.t0 = time.time()
        threads = [threading.Thread(target=self.host_loop, args=(h,), daemon=True) for h in self.hosts]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(2)
        except KeyboardInterrupt:
            self.log("interrupted: stopping workers")
            self.stop.set()
            for p in list(self.procs):
                p.kill()
            raise

    def _requeue(self, job, why) -> None:
        self.tries[job["id"]] += 1
        if self.tries[job["id"]] > 3:
            self.log(f"dropping chunk {job['a']} vs {job['b']} after 3 tries ({why})")
            return
        missing = self.store.missing(job)
        if missing:
            self.q.put({**job, "games": missing})

    def host_loop(self, h: Host) -> None:
        while not self.stop.is_set():
            if self.q.empty():
                with self.lock:
                    busy = self.active
                if not busy:  # nothing queued and no session that could still re-queue a chunk
                    return
                self.stop.wait(30)
                continue
            pr = h.probe()
            if not pr["ok"] or pr["workers"] < 1:
                self.log(f"[{h.name}] not available ({pr}); retry in 120 s")
                if self.stop.wait(120):
                    return
                continue
            ok = self.session(h, pr["workers"])
            h.failures = 0 if ok else h.failures + 1
            if h.failures >= 3:
                self.log(f"[{h.name}] 3 failed sessions in a row: disabled for this batch")
                return
            if not ok and self.stop.wait(60):
                return

    def session(self, h: Host, w: int) -> bool:
        cmd, cwd = h.worker_cmd(w)
        self.log(f"[{h.name}] session: {w} workers")
        p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.procs.add(p)
        with self.lock:
            self.active += 1
        inflight: dict[int, dict] = {}
        t_start, ended = time.time(), False

        def feed() -> None:
            nonlocal ended
            if ended:
                return
            late = time.time() - t_start > self.session_s
            if not late and h.ssh is None and h.__dict__.get("require_ac") and not on_ac():
                self.log(f"[{h.name}] on battery: draining")
                late = True
            if not late:
                try:
                    job = self.q.get_nowait()
                except queue.Empty:
                    job = None
                if job is not None:
                    job = {**job, "host": h.name}
                    inflight[job["id"]] = job
                    p.stdin.write(json.dumps(job) + "\n")
                    p.stdin.flush()
                    return
            if not inflight:
                p.stdin.write("END\n")
                p.stdin.flush()
                ended = True

        ok = False
        try:
            for line in p.stdout:
                kind, _, rest = line.rstrip("\n").partition(" ")
                if kind == "READY":
                    for _ in range(w):
                        feed()
                elif kind == "ROW":
                    self.done_games += self.store.add_rows([json.loads(rest)])
                elif kind == "DONE":
                    job = inflight.pop(int(rest))
                    if self.store.missing(job):
                        self._requeue(job, "rows missing")
                    self.log(f"[{h.name}] {job['a']} vs {job['b']} done; {self.done_games}/{self.total} games, "
                             f"{(time.time() - self.t0) / 3600:.2f} h")
                    feed()
                elif kind == "FAIL":
                    jid, _, msg = rest.partition(" ")
                    job = inflight.pop(int(jid))
                    self.log(f"[{h.name}] FAIL {job['a']} vs {job['b']}: {msg}")
                    self._requeue(job, msg)
                    feed()
                elif kind == "BYE":
                    ok = True
            p.wait(timeout=60)
        except Exception as e:  # noqa: BLE001
            self.log(f"[{h.name}] session error {e!r}")
            p.kill()
        finally:
            self.procs.discard(p)
            for job in inflight.values():
                self._requeue(job, "session ended")
            with self.lock:
                self.active -= 1
            h.last_w, h.last_end = w, time.time()
        if not ok:
            self.log(f"[{h.name}] session ended abnormally (rc {p.returncode})")
        return ok


def make_logger(out: Path):
    fh = (out / "coordinator.log").open("a")

    def log(msg: str) -> None:
        line = time.strftime("%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


def cmd_run(args) -> None:
    cfg = load_cfg(args.config)
    store = Store(cfg, Path(args.out))
    log = make_logger(Path(args.out))
    hosts = [Host(h) for h in cfg["hosts"] if h.get("enabled", True) and (not args.hosts or h["name"] in args.hosts)]
    if any(h.ssh is None for h in hosts) and sys.platform == "darwin":
        caf = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])  # ends with us
        log(f"caffeinate pid {caf.pid}")
    rounds = 0
    while True:
        pending = store.pending()
        if not pending:
            s1 = plan_stage1(store)
            if s1:
                store.add_plan(s1)
                log(f"stage 1: planned {len(s1)} chunks")
                continue
            if args.stage == 1:
                log("stage 1 complete")
                break
            if rounds >= min(args.max_rounds, cfg["stage2"]["max_rounds"]):
                log(f"stopping after {rounds} stage-2 rounds (limit)")
                break
            chunks, reason = plan_stage2(store)
            if not chunks:
                log("stage 2 complete: every pair SE and every Elo SE within target")
                break
            store.add_plan(chunks)
            rounds += 1
            log(f"stage 2 round {rounds} ({reason}): {len(chunks)} chunks, "
                f"{sum(2 * len(c['seeds']) for c in chunks)} games")
            continue
        log(f"dispatching {len(pending)} chunks ({sum(len(c['games']) for c in pending)} games) "
            f"to {[h.name for h in hosts]}")
        Dispatcher(store, hosts, args.session_min * 60, log).run(pending)
        log(status_text(store))
        if store.pending():
            log("chunks still missing after dispatch (hosts unavailable?): stopping; rerun to resume")
            break
    (Path(args.out) / "status.txt").write_text(status_text(store) + "\n")


# ---------------------------------------------------------------- status / report

def summary(store: Store) -> dict:
    s2 = store.cfg["stage2"]
    r, cov, wins, counts = store.fit()
    se = rating_se_elo(cov)
    stats = {pair: pair_seed_stats(xs) for pair, xs in store.seed_scores().items()}
    fails = [p for p, st in stats.items() if st["n"] == 0 or st["se_stop"] > s2["pair_se"]]
    focus = focus_idx(store.cfg, r)
    fse = None if focus is None else focus_se_elo(cov, focus)
    crit = se if fse is None else fse
    return {"r": r, "cov": cov, "se": se, "wins": wins, "counts": counts, "stats": stats, "pair_fails": fails,
            "elo_fails": int((crit > s2["elo_se"]).sum()), "focus": focus, "focus_se": fse}


def status_text(store: Store) -> str:
    s2 = store.cfg["stage2"]
    s = summary(store)
    k = len(store.names)
    n_pairs = k * (k - 1) // 2
    pend = store.pending()
    hosts = defaultdict(int)
    walls = defaultdict(list)
    for g in store.games.values():
        hosts[g.get("host", "?")] += 1
        walls[g.get("host", "?")].append(g["chunk_wall_s"] / max(1, g["chunk_n"]))
    st = list(s["stats"].values())
    worst = max((x["se_stop"] for x in st if x["n"]), default=float("nan"))
    lines = [f"games {len(store.games)} (pending {sum(len(c['games']) for c in pend)} in {len(pend)} chunks), "
             f"pairs with games {sum(x['n'] > 0 for x in st)}/{n_pairs}",
             f"(A) pair win-rate stopping SE <= {s2['pair_se']:.0%}: {n_pairs - len(s['pair_fails'])}/{n_pairs} pass, "
             f"worst {worst:.1%}",
             (f"(B) Elo SE <= {s2['elo_se']:.0f}: {k - s['elo_fails']}/{k} pass, max {s['se'].max():.1f}, "
              f"mean {s['se'].mean():.1f}" if s["focus"] is None else
              f"(B) top-{len(s['focus'])} Elo SE (vs their mean) <= {s2['elo_se']:.0f}: "
              f"{len(s['focus']) - s['elo_fails']}/{len(s['focus'])} pass, max {s['focus_se'].max():.1f}, "
              f"mean {s['focus_se'].mean():.1f} [" + ", ".join(f"{store.names[i]} {v:.1f}" for i, v in
                                                             zip(s['focus'], s['focus_se'])) + "]; "
              f"all-24 SE max {s['se'].max():.1f}"),
             "games by host: " + ", ".join(f"{h} {n} ({np.mean(walls[h]):.1f} s/game in chunk)"
                                           for h, n in sorted(hosts.items()))]
    return "\n".join(lines)


def cmd_status(args) -> None:
    store = Store(load_cfg(args.config), Path(args.out))
    print(status_text(store))


def _heatmap(path: Path, labels: list[str], mat, games, se, title: str, margin: bool = False) -> None:
    """Same look as examples/tournament_adaptive.py's heatmap (RdYlGn, value / n / SE per cell, row mean)."""
    import matplotlib.pyplot as plt

    k = len(labels)
    scale = k / 19.0  # the earlier 19-heuristic figure was 15 x 12.5 in
    fig, axis = plt.subplots(figsize=(15.0 * scale, 12.5 * scale), constrained_layout=True)
    shown = np.where(np.eye(k, dtype=bool), np.nan, mat)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#fffde7")
    lim = float(np.nanmax(np.abs(shown))) if margin else 1.0
    image = axis.imshow(shown, vmin=-lim if margin else 0.0, vmax=lim, cmap=cmap)
    axis.set_xticks(range(k), labels, rotation=45, ha="right")
    axis.set_yticks(range(k), labels)
    axis.set_xlabel("column policy (opponent)")
    axis.set_ylabel("row policy")
    for i in range(k):
        for j in range(k):
            if i == j:
                axis.text(j, i, "—", ha="center", va="center", color="black", fontsize=9)
                continue
            value = mat[i, j]
            frac = (value + lim) / (2 * lim) if margin else value
            color = "white" if frac < 0.25 or frac > 0.75 else "black"
            axis.text(j, i - 0.21, f"{value:+.0f}" if margin else f"{value:.0%}", ha="center", va="center",
                      color=color, fontsize=9)
            axis.text(j, i + 0.12, f"n={games[i, j]}", ha="center", va="center", color=color, fontsize=6.2)
            if se is not None:
                axis.text(j, i + 0.32, f"SE {se[i, j]:.3f}".replace("0.", "."), ha="center", va="center",
                          color=color, fontsize=6.2)
    mean_rate = np.nanmean(shown, axis=1)
    for i in range(k):
        axis.text(k - 0.3, i, f"{mean_rate[i]:+.1f}" if margin else f"{mean_rate[i]:.0%}", ha="left",
                  va="center", fontsize=9)
    axis.text(k - 0.3, -0.85, "mean", ha="left", va="center", fontsize=8)
    axis.set_xlim(-0.5, k + 0.7)
    axis.set_title(title, fontsize=12)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("side-swapped mean margin" if margin else "side-swapped win rate")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _elo_plot(path: Path, names: list[str], elo, s: dict, order: list[int], games_total: int,
              target: float) -> None:
    """Same look as reports/elo_augment_20260918/elo_ratings.png: dots + 95% intervals, three colour tiers
    (green = the stage-2 focus group, blue = the rest of the distilled/teacher middle, red = the bottom group)."""
    import matplotlib.pyplot as plt

    k = len(names)
    focus = set(s["focus"] or [])
    fse = dict(zip(s["focus"] or [], s["focus_se"] if s["focus_se"] is not None else []))
    fig, axis = plt.subplots(figsize=(10.0, 0.37 * k + 1.6), constrained_layout=True)
    for row, i in enumerate(order):
        y = k - 1 - row
        tier = "#1b6e3c" if i in focus else ("#2c6fad" if row < k - 7 else "#b5382f")
        half = 1.96 * s["se"][i]
        axis.errorbar(elo[i], y, xerr=half, fmt="none", ecolor="gray", capsize=4, lw=1.5)
        axis.plot(elo[i], y, "o", color=tier, ms=9)
        note = f"  (top-group SE {fse[i]:.1f})" if i in fse else ""
        axis.text(elo[i] + half + 12, y, f"{elo[i]:.0f} ±{half:.0f}  (n={int(s['counts'][i].sum())}){note}",
                  va="center", fontsize=9, color="#444444")
    axis.set_yticks(range(k), [names[i] for i in order][::-1])
    axis.set_xlabel("Elo (Bradley–Terry, draws = 1/2)")
    axis.grid(axis="x", alpha=0.3)
    lo, hi = float(np.min(elo - 2.5 * s["se"])), float(np.max(elo + 2.5 * s["se"]))
    axis.set_xlim(lo - 50, hi + 0.22 * (hi - lo))
    worst = max(fse.values()) if fse else float(np.max(s["se"]))
    axis.set_title(f"Elo-24 ratings with 95% intervals (heuristics + distilled students)\n{games_total:,} games; "
                   f"top-{len(focus)} SE vs their own mean ≤ {target:.0f} Elo (max {worst:.1f}); "
                   "A-optimal game allocation", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def cmd_report(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store = Store(load_cfg(args.config), Path(args.out))
    out = Path(args.out)
    s = summary(store)
    names, k = store.names, len(store.names)
    elo = 1500 + s["r"] * ELO
    order = list(np.argsort(-elo))
    wr = np.full((k, k), np.nan)
    se = np.full((k, k), np.nan)
    se_stop = np.full((k, k), np.nan)
    ns = np.zeros((k, k), int)
    for (i, j), st in s["stats"].items():
        if st["n"]:
            wr[i, j], wr[j, i] = st["p"], 1 - st["p"]
            se[i, j] = se[j, i] = st["se"]
            se_stop[i, j] = se_stop[j, i] = st["se_stop"]
            ns[i, j] = ns[j, i] = 2 * st["n"]
    msum, mcnt = np.zeros((k, k)), np.zeros((k, k))
    for g in store.games.values():
        i, j = store.index[g["a"]], store.index[g["b"]]
        msum[i, j] += g["margin"]
        msum[j, i] -= g["margin"]
        mcnt[i, j] += 1
        mcnt[j, i] += 1
    margin = np.where(mcnt > 0, msum / np.maximum(mcnt, 1), np.nan)
    lab = [names[i] for i in order]

    def csv(name, mat, fmt):
        rows = ["," + ",".join(lab)]
        for i in order:
            rows.append(names[i] + "," + ",".join("" if not np.isfinite(mat[i, j]) or i == j else fmt % mat[i, j]
                                                  for j in order))
        (out / name).write_text("\n".join(rows) + "\n")

    csv("winrate.csv", wr, "%.4f")
    csv("winrate_se.csv", se, "%.4f")
    csv("winrate_se_stop.csv", se_stop, "%.4f")
    csv("margin.csv", margin, "%.2f")
    csv("games.csv", ns.astype(float), "%d")
    lines = ["rank,policy,elo,se,games"] + [f"{n + 1},{names[i]},{elo[i]:.1f},{s['se'][i]:.1f},"
                                            f"{int(s['counts'][i].sum())}" for n, i in enumerate(order)]
    (out / "ratings.csv").write_text("\n".join(lines) + "\n")
    (out / "ratings.json").write_text(json.dumps({names[i]: {"elo": float(elo[i]), "se": float(s["se"][i]),
                                                             "spec": store.spec[names[i]]} for i in order},
                                                 indent=1))
    games_total = len(store.games)
    _heatmap(out / "winrate.png", lab, wr[np.ix_(order, order)], ns[np.ix_(order, order)], se[np.ix_(order, order)],
             "Elo-24 round-robin win rate (row policy wins, draws = 1/2)\n"
             f"side-swapped, simulator, stopped once no battery remains; per-pair SE ≤ {store.cfg['stage2']['pair_se']:.2f}"
             f" (max {np.nanmax(se):.3f}), {games_total} games")
    _heatmap(out / "margin.png", lab, margin[np.ix_(order, order)], ns[np.ix_(order, order)], None,
             f"Elo-24 mean score margin (row − column), side-swapped, {games_total} games", margin=True)
    _elo_plot(out / "elo_ratings.png", names, elo, s, order, games_total, store.cfg["stage2"]["elo_se"])
    results = []
    for (i, j), st in s["stats"].items():
        w = d = l = 0
        for g in store.games.values():
            if {g["a"], g["b"]} == {names[i], names[j]}:
                m = g["margin"] if g["a"] == names[i] else -g["margin"]
                w, d, l = w + (m > 0), d + (m == 0), l + (m < 0)
        if w + d + l:
            results.append({"row": names[i], "column": names[j], "games": w + d + l, "wins": w, "draws": d,
                            "losses": l, "win_rate": (w + 0.5 * d) / (w + d + l)})
    (out / "tournament.json").write_text(json.dumps({"policy_ids": names, "results": results}, indent=1))
    diag = REPO / "examples/analyze_elo_diagnostics.py"
    subprocess.run([sys.executable, str(diag), str(out / "tournament.json"), "--output-dir",
                    str(out / "elo_diagnostics")], check=False, capture_output=True)
    resid = []
    for (i, j), st in s["stats"].items():
        if st["n"]:
            pred = 1 / (1 + math.exp(-(s["r"][i] - s["r"][j])))
            resid.append((abs(st["p"] - pred), names[i], names[j], st["p"], pred, st["n"]))
    resid.sort(reverse=True)
    txt = [status_text(store), "", "rank policy        Elo    SE"]
    txt += [f"{n + 1:4d} {names[i]:10s} {elo[i]:6.0f} {s['se'][i]:5.1f}" for n, i in enumerate(order)]
    txt += ["", "largest BT residuals (non-transitivity)"]
    txt += [f"  {a:>8s} vs {b:<8s} observed {o:.0%} predicted {p:.0%} ({n} seeds)" for _, a, b, o, p, n in resid[:10]]
    (out / "summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


# ---------------------------------------------------------------- simulate / hosts / setup

def cmd_simulate(args) -> None:
    """Synthetic Elo pool through the same planner: games and rounds the stopping rules need."""
    cfg = load_cfg(args.config)
    s1, s2 = cfg["stage1"], cfg["stage2"]
    k = len(cfg["policies"])
    rng = np.random.default_rng(args.seed)
    true = np.sort(rng.uniform(0, args.spread, k)) / ELO
    scores = {(i, j): [] for i in range(k) for j in range(i + 1, k)}
    games = []

    def play(i, j, n):
        p = 1 / (1 + math.exp(-(true[i] - true[j])))
        for _ in range(n):
            g = [0.5 if rng.random() < args.draw else float(rng.random() < p) for _ in (0, 1)]
            scores[(i, j)].append(sum(g) / 2)
            games.extend((i, j, x) for x in g)

    for (i, j) in scores:
        play(i, j, s1["seeds"])
    n1, rounds, why = len(games), 0, defaultdict(int)
    while rounds < s2["max_rounds"] and len(games) - n1 < s2["max_games"]:
        r, cov, _, _ = fit_bt(k, *zip(*games), prior_sd_elo=cfg.get("prior_sd_elo", 400.0))
        plan, reason = plan_round(scores, r, cov, pair_se=s2["pair_se"], elo_se=s2["elo_se"],
                                  min_add=s2["min_add"], max_add=s2["max_add"], elo_pairs=s2["elo_pairs"],
                                  per_policy=s2["per_policy"], elo_seeds=s2["elo_seeds"])
        if not plan:
            break
        before = len(games)
        for (i, j), n in plan.items():
            play(i, j, n)
        why[reason] += len(games) - before
        rounds += 1
    r, cov, _, _ = fit_bt(k, *zip(*games), prior_sd_elo=cfg.get("prior_sd_elo", 400.0))
    print(f"{k} policies, Elo spread {args.spread:.0f}, draw rate {args.draw:.2f}: stage 1 {n1} games, "
          f"stage 2 {len(games) - n1} games in {rounds} rounds ({dict(why)}); max Elo SE "
          f"{rating_se_elo(cov).max():.1f}; RMSE vs truth "
          f"{np.sqrt(np.mean(((r - (true - true.mean())) * ELO) ** 2)):.1f} Elo")


def cmd_hosts(args) -> None:
    cfg = load_cfg(args.config)
    for h in cfg["hosts"]:
        print(h["name"], "enabled" if h.get("enabled", True) else "disabled", Host(h).probe())


def needed_ckpts(cfg) -> list[str]:
    return sorted({p["spec"][4:] for p in cfg["policies"] if p["spec"].startswith("net:")})


def cmd_setup(args) -> None:
    """Tracked code (blackout_env/, examples/, tests/) + the configured checkpoints -> each remote host's run dir."""
    cfg = load_cfg(args.config)
    files = sh(["git", "-C", str(REPO), "ls-files", "blackout_env", "examples", "tests", "pyproject.toml",
                "semantic_map_config.json"])
    commit = sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).strip()
    srcs = [str(ckpt_src(c)) for c in needed_ckpts(cfg)]  # raises if one is missing locally
    for h in cfg["hosts"]:
        if not h.get("ssh") or not h.get("enabled", True) or (args.hosts and h["name"] not in args.hosts):
            continue
        dst = f"{h['ssh']}:{h['dir']}/"
        cmds = [["ssh", *SSH_OPTS, h["ssh"], f"mkdir -p {h['dir']}/checkpoints/distill/frozen"],
                ["rsync", "-a", "--files-from=-", str(REPO) + "/", dst],
                ["rsync", "-a", "--ignore-existing",
                 *srcs, f"{dst}checkpoints/distill/frozen/"],
                ["ssh", *SSH_OPTS, h["ssh"], f"cd {h['dir']} && echo {commit} > .elo24_commit && "
                 f"PYTHONPATH=. {h['python']} -c 'import blackout_env.rating, torch, numba; print(\"ok\")'"]]
        for c in cmds:
            print("$", " ".join(shlex.quote(x) for x in c))
            if not args.dry_run:
                subprocess.run(c, input=files if "--files-from=-" in c else None, text=True, check=True)


def ckpt_src(rel: str) -> Path:
    """Checkpoints are untracked: use this checkout's copy, else the main checkout's (worktree parents[2])."""
    for base in (REPO, REPO.parents[2]):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(rel)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO / "examples/elo24_config.json"))
    p.add_argument("--out", default=str(REPO / "reports/elo24"))
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run")
    q.add_argument("--stage", type=int, choices=(1, 2), default=1)
    q.add_argument("--hosts", nargs="*", help="subset of configured host names")
    q.add_argument("--session-min", type=float, default=60.0, help="re-size workers from load this often")
    q.add_argument("--max-rounds", type=int, default=10 ** 6)
    q = sub.add_parser("worker")
    q.add_argument("--workers", type=int, default=1)
    sub.add_parser("status")
    sub.add_parser("report")
    q = sub.add_parser("simulate")
    q.add_argument("--spread", type=float, default=1200.0, help="true Elo range (uniform)")
    q.add_argument("--draw", type=float, default=0.03)
    q.add_argument("--seed", type=int, default=0)
    sub.add_parser("hosts")
    q = sub.add_parser("setup")
    q.add_argument("--hosts", nargs="*")
    q.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    {"run": cmd_run, "worker": cmd_worker, "status": cmd_status, "report": cmd_report, "simulate": cmd_simulate,
     "hosts": cmd_hosts, "setup": cmd_setup}[args.cmd](args)


if __name__ == "__main__":
    main()
