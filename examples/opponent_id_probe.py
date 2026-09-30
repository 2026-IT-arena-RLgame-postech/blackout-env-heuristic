"""P2 (docs/policy_v2_design_ko.md section 6): can a memory identify the OPPONENT's policy from the
learner's observation history, and how fast?

Offline, no Unity.  Collection shards of past distillation runs are one match each (learner-side
rows in time order, ``meta.opponent_played`` = who was actually played).  A frozen student trunk
(default ``checkpoints/distill/d5n/latest.pt``) turns every frame into its 11 post-attention vector
tokens (10 units + team_state -- what the Q head reads, ``MyModel._trunk``), and probes predict the
opponent family at every tick:

  stateless   MLP on the current frame only (also scored with its per-frame log-probs averaged
              over the match so far: "stateless+avg", naive evidence accumulation, no training)
  gru         the same frame encoder, then a GRU over time
  kda         the same frame encoder, then one residual KDA layer (modules/kda.py, chunk form)
  tm          TemporalMemory (modules/temporal_memory.py, per-slot KDA over the 11 trunk tokens
              + slot attention) before the frame encoder -- the "b2" block itself as the probe

and the ablation ``raw_*``: the same stateless / gru / kda probes on hand-crafted features
(agent_states of all 10 units -- positions, team, items, classes -- plus team_state), to see if the
trunk features matter.

Time grid: old shards dropped ticks where no own unit had a label, so ticks are rebuilt from
team_state[2] (time left) and the match is resampled every ``--stride`` decisions (25/s) from its
start: each grid point takes the latest stored row at or before it (forward fill; the fraction of
filled points is reported).  The staleness is NOT a probe input -- which ticks are missing depends
on the game state and could leak the label.  A match contributes to a time only while it lasts.

Labels (``family``): ``ckpt:*`` -> checkpoint; v3, v4, v4_near, v6-v13 -> v4fam (one behavioural
family, docs/heuristic_policy_catalog_ko.md); every other strategic_vN on its own.  Matches are
balanced by capping each class at ``--per-class``; train/test are split by match SEED (the same
seed was replayed in several runs), so no test seed is ever seen in training.

Usage:
    python examples/opponent_id_probe.py                   # features (cached) + all probes
    python examples/opponent_id_probe.py --probes stateless gru --steps 800
    python examples/opponent_id_probe.py --rebuild          # recompute the feature cache

Outputs: ``--out`` (default reports/opponent_id_probe/): summary.md, results.json.
Cache: ``--cache`` (default runs/opponent_id_probe_cache/, gitignored; ~4 GB of fp16 tokens).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import blackout_env.model.modules.kda as kda_module  # noqa: E402
from blackout_env.model.modules.kda import KDA  # noqa: E402
from blackout_env.model.modules.temporal_memory import TemporalMemory  # noqa: E402

DECISIONS_PER_S = 25
MATCH_TICKS = 420 * DECISIONS_PER_S
N_VISION_TOKENS, N_VEC_TOKENS = 36, 11  # MyModel trunk layout: 36 vision, 10 units + team_state, SPR CLS
V4_FAMILY = {3, 4, 6, 7, 8, 9, 10, 11, 12, 13}
COARSE = {"v4fam": "v4fam", "v17": "v17", "v18": "v18", "v19": "v19", "checkpoint": "checkpoint"}  # rest -> other
REPORT_SECONDS = (2, 5, 10, 20, 30, 45, 60)
DEFAULT_DIRS = [f"checkpoints/distill/{r}/shards" for r in ("d2", "d3b", "d4", "d5", "d5n")]
ALL_PROBES = ("stateless", "gru", "kda", "tm", "raw_stateless", "raw_gru", "raw_kda")


# ----------------------------------------------------------------------------------------------
# labels and match selection

def family(opponent_played: str) -> str | None:
    """Opponent family of a shard's ``meta.opponent_played``; None = unknown/unusable."""
    if not opponent_played:
        return None
    if opponent_played.startswith("ckpt:"):
        return "checkpoint"
    name = opponent_played.split(":", 1)[1] if opponent_played.startswith("mixture:") else opponent_played
    m = re.fullmatch(r"strategic_v(\d+)(_near)?", name)
    if not m:
        return name  # a named heuristic outside the v-series: its own class
    n = int(m.group(1))
    return "v4fam" if n in V4_FAMILY else f"v{n}"


def coarse(fam: str) -> str:
    return COARSE.get(fam, "other")


def index_shards(dirs: list[Path]) -> list[dict]:
    out = []
    for d in dirs:
        for path in sorted(Path(d).glob("*.npz")):
            if path.name.endswith(".tmp.npz"):
                continue
            try:
                with np.load(path) as z:
                    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
            except Exception:  # noqa: BLE001 -- a truncated shard: skip
                continue
            if meta.get("tag", "collect") != "collect":
                continue
            fam = family(meta.get("opponent_played", ""))
            if fam is None:
                continue
            out.append({"path": str(path), "run": Path(d).parent.name, "seed": int(meta.get("seed", -1)),
                        "opponent": meta["opponent_played"], "family": fam})
    return out


def select_matches(index: list[dict], per_class: int, min_class: int, test_frac: float, rng_seed: int):
    """Balanced subset (<= per_class matches per family, classes with < min_class dropped) and a
    train/test split by seed."""
    rng = np.random.default_rng(rng_seed)
    by_fam: dict[str, list[dict]] = {}
    for rec in index:
        by_fam.setdefault(rec["family"], []).append(rec)
    classes = sorted(f for f, recs in by_fam.items() if len(recs) >= min_class)
    chosen = []
    for fam in classes:
        recs = by_fam[fam]
        order = rng.permutation(len(recs))[:per_class]
        chosen += [recs[i] for i in sorted(order)]

    def is_test(seed: int) -> bool:  # stable in the seed alone: same seed -> same side, every run
        h = int(hashlib.sha1(f"{rng_seed}:{seed}".encode()).hexdigest()[:8], 16)
        return h / 0xFFFFFFFF < test_frac

    for rec in chosen:
        rec["split"] = "test" if is_test(rec["seed"]) else "train"
        rec["label"] = classes.index(rec["family"])
    return chosen, classes


# ----------------------------------------------------------------------------------------------
# features

def load_grid(path: str, stride: int, max_ticks: int):
    """The shard's rows resampled to ticks stride, 2*stride, ... <= min(max_ticks, match end).
    Returns (graphic, team_state, agent_states) of the stored rows used, the row index of each
    grid point into those, and the number of forward-filled grid points."""
    with np.load(path) as z:
        ts_all = z["team_state"]
        ticks = np.rint((1.0 - ts_all[:, 2].astype(np.float64)) * MATCH_TICKS).astype(np.int64)
        if "tick" in z.files:
            ticks = z["tick"].astype(np.int64) + 1
        end = int(ticks.max())
        keep = np.flatnonzero(ticks <= max_ticks + stride)
        # strictly increasing (fp rounding late in a match can repeat a tick): keep the first
        run_max = np.maximum.accumulate(np.concatenate(([-1], ticks[keep][:-1])))
        keep = keep[ticks[keep] > run_max]
        rows_t = ticks[keep]
        grid = np.arange(stride, min(max_ticks, end) + 1, stride)
        pos = np.searchsorted(rows_t, grid, side="right") - 1
        ok = pos >= 0
        grid, pos = grid[ok], pos[ok]
        used = np.unique(pos)
        sel = keep[used]
        g, ts, a = z["graphic"][: sel.max() + 1][sel] if len(sel) else z["graphic"][:0], ts_all[sel], z["agent_states"][sel]
    remap = np.searchsorted(used, pos)
    filled = int((rows_t[pos] != grid).sum())
    return g, ts, a, remap, filled


def build_features(chosen: list[dict], ckpt: Path, cache: Path, stride: int, max_seconds: float,
                   device: str, batch: int = 1024):
    from blackout_env.train.distill.inference import load_net

    cache.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = load_net(state["policy_state"]).to(device).eval()
    D = int(net.hidden_size)
    max_ticks = int(round(max_seconds * DECISIONS_PER_S))
    n_grid = max_ticks // stride
    M = len(chosen)
    tokens = np.lib.format.open_memmap(cache / "tokens.npy", mode="w+", dtype=np.float16,
                                       shape=(M * n_grid, N_VEC_TOKENS, D))
    raw = np.zeros((M * n_grid, 10 * 12 + 4), dtype=np.float32)
    lengths = np.zeros(M, dtype=np.int32)
    filled = np.zeros(M, dtype=np.int32)
    t0 = time.time()

    def work(rec):
        return load_grid(rec["path"], stride, max_ticks)

    with ThreadPoolExecutor(4) as pool, torch.no_grad():
        for m, (g, ts, a, remap, nfill) in enumerate(pool.map(work, chosen)):
            outs = []
            for s in range(0, len(g), batch):
                gg = torch.from_numpy(np.ascontiguousarray(g[s:s + batch])).to(device).permute(0, 3, 1, 2).float() / 15.0
                tt = torch.from_numpy(ts[s:s + batch]).to(device)
                aa = torch.from_numpy(a[s:s + batch]).to(device)
                out = net._trunk(gg, tt, aa)[:, N_VISION_TOKENS:N_VISION_TOKENS + N_VEC_TOKENS]
                outs.append(out.half().cpu().numpy())
            tok = np.concatenate(outs) if outs else np.zeros((0, N_VEC_TOKENS, D), np.float16)
            L = len(remap)
            base = m * n_grid
            tokens[base:base + L] = tok[remap]
            raw[base:base + L] = np.concatenate((a.reshape(len(a), -1), ts), axis=1)[remap]
            lengths[m], filled[m] = L, nfill
            if m % 200 == 0:
                print(f"  features {m}/{M}  {time.time() - t0:.0f}s", flush=True)
    tokens.flush()
    np.save(cache / "raw.npy", raw)
    np.savez(cache / "meta.npz", lengths=lengths, filled=filled)
    return D


# ----------------------------------------------------------------------------------------------
# probes

class FrameEnc(nn.Module):
    """[B, T, 11, D] trunk tokens -> [B, T, H]: per-slot LayerNorm + shared projection, flatten, MLP."""

    def __init__(self, D: int, H: int, slot_dim: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.proj = nn.Linear(D, slot_dim)
        self.slot_emb = nn.Parameter(torch.zeros(N_VEC_TOKENS, slot_dim))
        self.mlp = nn.Sequential(nn.Linear(N_VEC_TOKENS * slot_dim, H), nn.GELU(), nn.Linear(H, H), nn.GELU())

    def forward(self, x):
        y = self.proj(self.norm(x)) + self.slot_emb
        return self.mlp(y.flatten(-2))


class RawEnc(nn.Module):
    def __init__(self, D_in: int, H: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(D_in, H), nn.GELU(), nn.Linear(H, H), nn.GELU())

    def forward(self, x):
        return self.mlp(x)


class Probe(nn.Module):
    def __init__(self, kind: str, D: int, n_classes: int, H: int = 128):
        super().__init__()
        self.kind = kind
        base = kind.removeprefix("raw_")
        self.base = base
        self.tm = TemporalMemory(D, N_VEC_TOKENS, n_heads=4, head_dim=64, chunk_size=64) if base == "tm" else None
        if self.tm is not None:  # a probe must be able to use the memory from step 0
            nn.init.normal_(self.tm.out_proj.weight, std=0.02)
        self.enc = RawEnc(D, H) if kind.startswith("raw_") else FrameEnc(D, H)
        if base == "gru":
            self.rnn = nn.GRU(H, H, batch_first=True)
        elif base == "kda":
            self.kda_norm = nn.RMSNorm(H)
            self.kda = KDA(H, n_heads=2, head_dim=64, chunk_size=64)
            self.kda_mlp = nn.Sequential(nn.RMSNorm(H), nn.Linear(H, H), nn.GELU())
        self.head = nn.Linear(H, n_classes)

    def forward(self, x):
        if self.tm is not None:
            x, _ = self.tm(x)
        h = self.enc(x)
        if self.base == "gru":
            h, _ = self.rnn(h)
        elif self.base == "kda":
            y, _ = self.kda(self.kda_norm(h))
            h = h + y
            h = h + self.kda_mlp(h)
        return self.head(h)


def batches(ids, feats, lengths, labels, n_grid, B, rng, shuffle=True, max_T=None):
    order = rng.permutation(len(ids)) if shuffle else np.arange(len(ids))
    for s in range(0, len(ids), B):
        mids = ids[order[s:s + B]]
        T = int(lengths[mids].max()) if max_T is None else max_T
        x = np.zeros((len(mids), T, *feats.shape[1:]), dtype=np.float32)
        mask = np.zeros((len(mids), T), dtype=bool)
        for j, m in enumerate(mids):
            L = min(int(lengths[m]), T)
            x[j, :L] = feats[m * n_grid:m * n_grid + L]
            mask[j, :L] = True
        yield mids, torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(labels[mids]).long()


def train_probe(kind, feats, lengths, labels, train_ids, test_ids, n_grid, n_classes, args, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    D = feats.shape[-1]
    net = Probe(kind, D, n_classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    B = args.batch_tm if kind == "tm" else args.batch
    steps = args.steps_tm if kind == "tm" else args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    counts = np.bincount(labels[train_ids], minlength=n_classes).astype(np.float64)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1) / n_classes, dtype=torch.float32, device=device)
    step, t0, losses = 0, time.time(), []
    net.train()
    while step < steps:
        for _, x, mask, y in batches(train_ids, feats, lengths, labels, n_grid, B, rng):
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = net(x)
            yy = y[:, None].expand(-1, logits.shape[1])
            loss = F.cross_entropy(logits[mask], yy[mask], weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(f"    [{kind}] step {step}/{steps} loss {np.mean(losses[-100:]):.3f}  {time.time() - t0:.0f}s", flush=True)
            if step >= steps:
                break
    net.eval()
    probs = np.zeros((len(test_ids), n_grid, n_classes), dtype=np.float32)
    with torch.no_grad():
        for j0, (_, x, mask, _) in enumerate(batches(test_ids, feats, lengths, labels, n_grid, B, rng,
                                                     shuffle=False, max_T=n_grid)):
            p = torch.softmax(net(x.to(device)).float(), -1).cpu().numpy()
            probs[j0 * B:j0 * B + len(p)] = p
    return probs, {"train_loss_last100": float(np.mean(losses[-100:])), "train_s": time.time() - t0, "steps": steps}


# ----------------------------------------------------------------------------------------------
# metrics

def macro_f1(y, p, n):
    f1s = []
    for c in range(n):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        if tp + fn == 0:
            continue
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def score_curve(probs, y, lengths, stride, classes, seconds):
    """Accuracy / macro-F1 / coarse accuracy at each time, over test matches still running."""
    n = len(classes)
    coarse_of = np.array([coarse(c) for c in classes])
    out = {}
    for s in seconds:
        k = int(round(s * DECISIONS_PER_S / stride)) - 1
        alive = lengths > k
        if not alive.any():
            continue
        pred = probs[alive, k].argmax(-1)
        yt = y[alive]
        out[str(s)] = {"n": int(alive.sum()), "acc": float((pred == yt).mean()),
                       "macro_f1": macro_f1(yt, pred, n),
                       "coarse_acc": float((coarse_of[pred] == coarse_of[yt]).mean()),
                       "majority_acc": float(np.bincount(yt, minlength=n).max() / len(yt))}
    return out


def confusion(probs, y, lengths, stride, n, s):
    k = int(round(s * DECISIONS_PER_S / stride)) - 1
    alive = lengths > k
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y[alive], probs[alive, k].argmax(-1)), 1)
    return cm


def cumulative_average(probs, lengths):
    """Stateless per-frame log-probs averaged over the match so far (naive evidence pooling)."""
    lp = np.log(np.clip(probs, 1e-6, 1.0))
    T = probs.shape[1]
    mask = (np.arange(T)[None, :] < lengths[:, None])[..., None]
    cs = np.cumsum(lp * mask, axis=1) / np.arange(1, T + 1)[None, :, None]
    return np.exp(cs - cs.max(-1, keepdims=True))


# ----------------------------------------------------------------------------------------------

def use_kda_solver(solve: str) -> None:
    """Make every KDA layer in this process call ``kda_chunk(..., solve=solve)``.

    Why: the default on MPS (``auto`` -> Neumann ``doubling``) is numerically wrong in fp32 when
    the keys inside a chunk are nearly identical and beta >~ 0.5 -- the partial products of
    (I - A)(I + A^2)(I + A^4)... grow like binomial coefficients before cancelling.  Constant
    input (a match's zero padding, or a frame that does not change) does exactly that: the TM
    probe went NaN within ~12 steps, and a synthetic chunk of one repeated key with beta = 0.9 is
    off by ~4e8 (finite, just wrong).  ``solve`` (torch.linalg.solve_triangular) is exact."""
    if solve == "auto":
        return
    orig = kda_module.kda_chunk

    def patched(*a, **k):
        k.setdefault("solve", solve)
        return orig(*a, **k)

    kda_module.kda_chunk = patched  # KDA.forward looks the name up in its module at call time


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard-dirs", nargs="+", default=DEFAULT_DIRS)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/distill/d5n/latest.pt"))
    p.add_argument("--per-class", type=int, default=300, help="max matches per opponent family")
    p.add_argument("--min-class", type=int, default=60, help="drop families with fewer matches")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--stride", type=int, default=4, help="grid step in decisions (25/s)")
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--probes", nargs="+", default=list(ALL_PROBES), choices=ALL_PROBES)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--steps-tm", type=int, default=800)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--batch-tm", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cache", type=Path, default=Path("runs/opponent_id_probe_cache"))
    p.add_argument("--out", type=Path, default=Path("reports/opponent_id_probe"))
    p.add_argument("--rebuild", action="store_true", help="recompute the feature cache")
    p.add_argument("--kda-solve", default="solve", choices=("solve", "doubling", "auto"),
                   help="KDA chunk solver (see use_kda_solver: MPS's default 'doubling' breaks here)")
    args = p.parse_args(argv)
    use_kda_solver(args.kda_solve)
    torch.set_num_threads(args.threads)
    t_start = time.time()

    print("indexing shards ...", flush=True)
    index = index_shards([Path(d) for d in args.shard_dirs])
    all_counts: dict[str, int] = {}
    for rec in index:
        all_counts[rec["family"]] = all_counts.get(rec["family"], 0) + 1
    chosen, classes = select_matches(index, args.per_class, args.min_class, args.test_frac, args.seed)
    n = len(classes)
    labels = np.array([r["label"] for r in chosen], dtype=np.int64)
    split = np.array([r["split"] for r in chosen])
    print(f"  {len(index)} shards; {len(chosen)} chosen, {n} classes: {classes}", flush=True)

    stride, n_grid = args.stride, int(round(args.max_seconds * DECISIONS_PER_S)) // args.stride
    key = {"paths": [r["path"] for r in chosen], "ckpt": str(args.checkpoint), "stride": stride,
           "max_seconds": args.max_seconds}
    key_hash = hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12]
    stamp = args.cache / "key.txt"
    if args.rebuild or not stamp.exists() or stamp.read_text() != key_hash:
        print("building features ...", flush=True)
        build_features(chosen, args.checkpoint, args.cache, stride, args.max_seconds, args.device)
        stamp.write_text(key_hash)
    t_feat = time.time() - t_start
    tokens = np.load(args.cache / "tokens.npy", mmap_mode="r")
    tokens = np.ascontiguousarray(tokens)  # into RAM: random access per batch
    raw = np.load(args.cache / "raw.npy")
    meta = np.load(args.cache / "meta.npz")
    lengths, filled = meta["lengths"], meta["filled"]
    usable = lengths >= 2
    train_ids = np.flatnonzero((split == "train") & usable)
    test_ids = np.flatnonzero((split == "test") & usable)
    y_test = labels[test_ids]
    L_test = lengths[test_ids]
    print(f"  train {len(train_ids)} / test {len(test_ids)} matches; forward-filled grid points "
          f"{filled.sum() / max(1, lengths.sum()):.1%}", flush=True)

    curve_s = sorted(set(REPORT_SECONDS) | set(range(1, int(args.max_seconds) + 1)))
    results = {"classes": classes, "config": vars(args) | {"cache": str(args.cache), "out": str(args.out),
                                                           "checkpoint": str(args.checkpoint)},
               "counts": {"all_shards": all_counts,
                          "chosen": {c: int((labels == i).sum()) for i, c in enumerate(classes)},
                          "train": {c: int((labels[train_ids] == i).sum()) for i, c in enumerate(classes)},
                          "test": {c: int((y_test == i).sum()) for i, c in enumerate(classes)}},
               "forward_filled_frac": float(filled.sum() / max(1, lengths.sum())),
               "test_matches_alive": {str(s): int((L_test > int(round(s * DECISIONS_PER_S / stride)) - 1).sum())
                                      for s in REPORT_SECONDS},
               "probes": {}, "curves": {}, "confusion": {}}
    probs_all = {}
    for kind in args.probes:
        feats = raw if kind.startswith("raw_") else tokens
        print(f"probe {kind} ...", flush=True)
        probs, info = train_probe(kind, feats, lengths, labels, train_ids, test_ids, n_grid, n,
                                  args, args.device)
        probs_all[kind] = probs
        results["probes"][kind] = info | {"at": score_curve(probs, y_test, L_test, stride, classes, REPORT_SECONDS)}
        results["curves"][kind] = {s: v["acc"] for s, v in score_curve(probs, y_test, L_test, stride, classes, curve_s).items()}
        print("   ", {s: round(v["acc"], 3) for s, v in results["probes"][kind]["at"].items()}, flush=True)
        if kind in ("stateless", "raw_stateless"):
            avg = cumulative_average(probs, L_test)
            name = kind + "+avg"
            probs_all[name] = avg
            results["probes"][name] = {"at": score_curve(avg, y_test, L_test, stride, classes, REPORT_SECONDS)}
            results["curves"][name] = {s: v["acc"] for s, v in score_curve(avg, y_test, L_test, stride, classes, curve_s).items()}
    for kind, probs in probs_all.items():
        results["confusion"][kind] = {str(s): confusion(probs, y_test, L_test, stride, n, s).tolist() for s in (5, 10, 30)}
    results["runtime_s"] = {"features": t_feat, "total": time.time() - t_start}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    (args.out / "summary.md").write_text(render_summary(results))
    print(f"wrote {args.out}/summary.md, results.json  ({time.time() - t_start:.0f}s)")


def render_summary(r: dict) -> str:
    classes = r["classes"]
    secs = [s for s in map(str, REPORT_SECONDS) if any(s in v["at"] for v in r["probes"].values())]
    lines = ["# P2 opponent-identification probe", "",
             f"Checkpoint `{r['config']['checkpoint']}` (frozen trunk); grid every {r['config']['stride']} decisions "
             f"({25 / r['config']['stride']:.2f} Hz) up to {r['config']['max_seconds']:.0f} s; "
             f"forward-filled grid points {r['forward_filled_frac']:.1%}. Split by match seed.", "",
             "## Classes (matches)", "", "| family | all shards | chosen | train | test |", "|---|---|---|---|---|"]
    for c in classes:
        lines.append(f"| {c} | {r['counts']['all_shards'].get(c, 0)} | {r['counts']['chosen'][c]} | "
                     f"{r['counts']['train'][c]} | {r['counts']['test'][c]} |")
    lines += ["", f"Chance = 1/{len(classes)} = {1 / len(classes):.3f}. Test matches still running: "
              + ", ".join(f"{s}s: {n}" for s, n in r["test_matches_alive"].items()), ""]
    for metric, title in (("acc", "Accuracy"), ("macro_f1", "Macro-F1"), ("coarse_acc", "Coarse accuracy (v4fam/v17/v18/v19/checkpoint/other)")):
        lines += [f"## {title} by elapsed match time", "", "| probe | " + " | ".join(f"{s} s" for s in secs) + " |",
                  "|---|" + "---|" * len(secs)]
        for kind, v in r["probes"].items():
            lines.append(f"| {kind} | " + " | ".join(f"{v['at'][s][metric]:.3f}" if s in v["at"] else "-" for s in secs) + " |")
        if metric == "acc":
            any_probe = next(iter(r["probes"].values()))
            lines.append("| majority | " + " | ".join(f"{any_probe['at'][s]['majority_acc']:.3f}" if s in any_probe["at"] else "-" for s in secs) + " |")
        lines.append("")
    lines += ["## Confusion (rows = true, cols = predicted), 10 s", ""]
    for kind in [k for k in ("kda", "gru", "stateless") if k in r["confusion"]][:2]:
        cm = np.array(r["confusion"][kind]["10"])
        lines += [f"**{kind}**", "", "| true \\ pred | " + " | ".join(classes) + " | recall |", "|---|" + "---|" * (len(classes) + 1)]
        for i, c in enumerate(classes):
            rec = cm[i, i] / max(1, cm[i].sum())
            lines.append(f"| {c} | " + " | ".join(str(x) for x in cm[i]) + f" | {rec:.2f} |")
        lines.append("")
    lines += [f"Runtime: features {r['runtime_s']['features']:.0f} s, total {r['runtime_s']['total']:.0f} s.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
