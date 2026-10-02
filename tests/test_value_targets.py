"""S1 value targets (blackout_env/value): horizons, truncation masks, the learner frame, splits,
and a smoke train over a tiny cache."""

import argparse
import json

import numpy as np
import pytest
import torch

from blackout_env.train.distill.collect import MatchTask, play_labelled_match
from blackout_env.train.distill.dataset import save_shard
from blackout_env.value.data import assign_split, build_cache, open_cache, unseen_group
from blackout_env.value.targets import DECISIONS_PER_S, lead_at, match_targets
from tests.test_distill import _FakeEnv

TICK_S = 1.0 / (420 * DECISIONS_PER_S)


def _team_state(ticks, own, opp):
    ts = np.zeros((len(ticks), 4), np.float32)
    ts[:, 0], ts[:, 1] = own, opp
    ts[:, 2] = 1.0 - np.asarray(ticks) * TICK_S
    return ts


def _meta(learner, opponent, result, elapsed_s):
    return {"learner_score": learner, "opponent_score": opponent, "result": result, "elapsed_s": elapsed_s}


def test_horizons_read_the_row_exactly_n_seconds_later_and_freeze_past_the_end():
    ticks = np.arange(300)  # 12 s of play
    own = np.where(ticks >= 100, 0.05, 0.0)  # +5 points at t = 4 s
    ts = _team_state(ticks, own, 0.0)
    t = match_targets(ts, ticks, _meta(5.0, 0.0, 1, 299 / 25), horizons_s=(2, 5, 20))
    assert t["delta_mask"].all()  # contiguous rows, and past the end the lead is the final lead
    assert t["delta"][0, 0] == 0 and t["delta"][50, 0] == pytest.approx(0.05) and t["delta"][49, 0] == 0
    assert t["delta"][0, 1] == pytest.approx(0.05) and t["delta"][100, 1] == 0
    assert t["past_end"][:, 2].all() and not t["past_end"][:174, 1].any() and t["past_end"][175:, 1].all()
    assert t["delta"][0, 2] == pytest.approx(0.05)  # 20 s is past the end: final lead 5
    assert np.allclose(t["final"], 0.05) and (t["result"] == 1).all()
    assert t["elapsed_s"][100] == pytest.approx(4.0, abs=1e-3)


def test_final_lead_counts_past_the_last_row_up_to_the_recorded_match_end():
    ticks = np.arange(100)
    ts = _team_state(ticks, 0.1, 0.0)
    # the match ran to 6 s although the shard stops at 4 s; the lead then became 0.2
    t = match_targets(ts, ticks, _meta(20.0, 0.0, 1, 6.0), horizons_s=(1, 5))
    assert not t["delta_mask"][99, 0]  # 4 s + 1 s: inside the unrecorded tail, unknown
    assert t["delta_mask"][99, 1] and t["delta"][99, 1] == pytest.approx(0.1)  # 9 s > 6 s: final
    # the same tail with an unchanged lead is known
    t = match_targets(ts, ticks, _meta(10.0, 0.0, 1, 6.0), horizons_s=(1,))
    assert t["delta_mask"].all() and (t["delta"] == 0).all()


def test_gaps_in_old_shards_are_known_only_when_the_lead_did_not_change_across_them():
    ticks = np.array([0, 1, 2, 60, 61, 62, 63])  # rows 3..59 were dropped (unlabelled)
    own = np.array([0, 0, 0, 0, 0, 0.03, 0.03])
    ts = _team_state(ticks, own, 0.0)
    t = match_targets(ts, ticks, _meta(3.0, 0.0, 1, 63 / 25), horizons_s=(1,))  # 25 ticks later
    assert t["delta_mask"][0] and t["delta"][0, 0] == 0  # tick 25 falls in a gap with lead 0 on both sides
    v, known, past = lead_at(ticks, own.astype(np.float32), np.array([30, 61, 62, 100]), 63, 0.03)
    assert known.tolist() == [True, True, True, True] and past.tolist() == [False, False, False, True]
    own2 = np.array([0, 0, 0, 0.02, 0.02, 0.03, 0.03])  # the lead changed inside the gap
    v, known, _ = lead_at(ticks, own2.astype(np.float32), np.array([30]), 63, 0.03)
    assert not known[0]


def test_unknown_ticks_mask_every_horizon_but_keep_the_outcome():
    ts = _team_state(np.arange(5), 0.0, 0.1)
    t = match_targets(ts, np.full(5, -1), _meta(0.0, 10.0, -1, 1.0))
    assert not t["delta_mask"].any() and np.allclose(t["final"], -0.1) and (t["result"] == -1).all()


@pytest.mark.parametrize("swap", [False, True])
def test_shard_targets_are_in_the_learner_frame(tmp_path, swap):
    """_FakeEnv: team A wins 30-10.  The learner's final lead is +20 points as team A, -20 as B."""
    task = MatchTask(seed=1, swap=swap, opponent="strategic_v4", beta=1.0, teacher="v17-teach", student=None, shard=None)
    rows, summary = play_labelled_match(_FakeEnv(), task, None)
    path = tmp_path / "checkpoints/distill/x/shards/a.npz"
    path.parent.mkdir(parents=True)
    save_shard(path, rows, meta=summary)
    entry = {**summary, "path": "checkpoints/distill/x/shards/a.npz", "split": "train"}
    n = build_cache(tmp_path, [entry], tmp_path / "cache", rows_per_match=8, workers=1, log=lambda m: None)
    cache = open_cache(tmp_path / "cache")
    assert n == len(rows["labels"])
    expect = -0.2 if swap else 0.2
    assert np.allclose(cache["final"], expect) and (cache["result"] == (-1 if swap else 1)).all()
    assert (cache["agent_states"][:, :5, 2] > 0).all()  # own units first: canonical frame


def test_unseen_split_catches_every_listed_policy_and_the_ood_suite():
    base = {"student": "checkpoints/distill/d4/latest.pt", "opponent": "mixture", "teacher": "v21-teach",
            "opponent_played": None}
    cases = {
        "checkpoints/distill/p5a/shards/1.npz": {},
        "checkpoints/distill/d6/shards/2.npz": {"opponent": "ckpt:checkpoints/distill/frozen/p5a_v22.pt"},
        "checkpoints/distill/d6/shards/3.npz": {"opponent": "perturb:eps=0.05,lag=2:ckpt:checkpoints/distill/frozen/p5a_v22.pt"},
        "reports/distill_heldout/shards/d9_latest__iqn__strategic_v17__1_A.npz": {"student": "checkpoints/distill/d9/latest.pt"},
        "reports/p4_heldout/shards/x.npz": {"student": "checkpoints/distill/p4a/latest.pt"},
        "reports/distill_ood/shards/y.npz": {},
        "checkpoints/distill/d2/shards/4.npz": {"opponent": "ckpt:checkpoints/offline/20260919-010456_42178/final.pt"},
    }
    for path, override in cases.items():
        assert assign_split({**base, "path": path, **override}) == "unseen", path
    assert unseen_group({**base, "path": "reports/distill_ood/shards/y.npz"}) == "ood"
    assert unseen_group({**base, "path": "checkpoints/distill/p5a/shards/1.npz"}) == "policy"
    seen = [assign_split({**base, "path": f"checkpoints/distill/p4s/shards/{i}.npz"}) for i in range(2000)]
    assert set(seen) == {"train", "heldout"} and 0.03 < seen.count("heldout") / len(seen) < 0.07


def _fake_shard(repo, rel, rng, n=60, result=1):
    ticks = np.arange(n)
    own = np.cumsum(rng.random(n) < 0.05) / 100
    ts = _team_state(ticks, own, 0.0)
    rows = {"graphic": rng.integers(0, 2, (n, 24, 24, 13)).astype(np.uint8), "team_state": ts,
            "agent_states": rng.normal(size=(n, 10, 12)).astype(np.float32).clip(-1, 1),
            "labels": np.zeros((n, 5), np.uint8), "kinds": np.zeros((n, 5), np.uint8), "tick": ticks.astype(np.int32)}
    meta = {"learner_score": float(own[-1] * 100), "opponent_score": 0.0, "result": result, "elapsed_s": n / 25,
            "rows": n, "student": None, "opponent": "mixture", "teacher": "v21-teach"}
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    save_shard(path, rows, meta=meta)
    return {**meta, "path": rel}


def test_smoke_train_and_eval(tmp_path):
    from blackout_env.value.model import load_value_net
    from blackout_env.value.train import evaluate_checkpoint, train

    rng = np.random.default_rng(0)
    entries = []
    for i, (split, rel) in enumerate([("train", "checkpoints/distill/a/shards/{}.npz")] * 4
                                     + [("heldout", "checkpoints/distill/b/shards/{}.npz")]
                                     + [("unseen", "checkpoints/distill/p5a/shards/{}.npz")]
                                     + [("unseen", "reports/distill_ood/shards/{}.npz")]):
        entry = _fake_shard(tmp_path, rel.format(i), rng, result=1 if i % 2 else -1)
        entries.append({**entry, "split": split})
    build_cache(tmp_path, entries, tmp_path / "cache", rows_per_match=16, workers=1, log=lambda m: None)
    args = argparse.Namespace(cache=tmp_path / "cache", out_dir=tmp_path / "run", init=None, steps=3, batch_size=8,
                              lr=1e-3, trunk_lr_scale=1.0, weight_decay=0.0, warmup=1, device="cpu", amp="none",
                              compile=False, eval_every=2, eval_rows=16, seed=0, num_threads=1)
    train(args)
    log = [json.loads(line) for line in (tmp_path / "run/log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in log] == [0, 2, 3] and np.isfinite(log[-1]["train"]["loss"])
    net = load_value_net(tmp_path / "run/best.pt")
    g = torch.zeros(2, 24, 24, 13, dtype=torch.uint8)
    pred = net.predict(g, torch.zeros(2, 4), torch.zeros(2, 10, 12))
    assert pred["delta"].shape == (2, 4) and np.allclose(pred["p"].sum(1), 1, atol=1e-5)
    report = evaluate_checkpoint(argparse.Namespace(device="cpu", ckpt=tmp_path / "run/best.pt", cache=tmp_path / "cache",
                                                    max_rows=None, batch_size=32, name="smoke"))
    assert set(report["smoke"]) == {"heldout", "unseen", "unseen_policy", "unseen_ood"}
