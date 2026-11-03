"""PPO league arm B (blackout_env/rl/league.py + PPOTrainer flags): centering, gate, ramp, snapshots, resume."""
import os

import numpy as np
import pytest

from blackout_env.rl import league as lg

CKPT = "checkpoints/distill/frozen/p24c_v321.pt"
VALUE = "checkpoints/value/s2v_f3xd2/iter2.pt"
needs_ckpt = pytest.mark.skipif(not os.path.exists(CKPT), reason="needs the P24c checkpoint")


# ------------------------------------------------------------------ pure logic
def test_center_by_opponent_math():
    adv = np.array([1.0, 3.0, 10.0, 20.0, -4.0, 6.0])
    opp = np.array([0, 0, 1, 1, 2, 2])
    out, means = lg.center_by_opponent(adv, opp, np.array([10, 10, 9]), min_games=10)
    assert np.allclose(means, [2.0, 15.0, 1.0])
    assert np.allclose(out[:4], [-1.0, 1.0, -5.0, 5.0])    # opponents 0 and 1 (>= 10 games) centred
    assert np.allclose(out[4:], adv[4:])                   # opponent 2 has 9 games: untouched
    # an opponent with no rows: nan mean, nothing else changes
    out2, m2 = lg.center_by_opponent(adv, np.zeros(6, int), np.array([12, 12]), 10)
    assert np.isnan(m2[1]) and abs(out2.mean()) < 1e-12 and np.allclose(out2, adv - adv.mean())


def test_gate_logic():
    w = np.array([1.0, 3.0])
    s = np.array([0.9, 0.5])
    assert lg.fixed_gate(s, w) == pytest.approx(0.25 * 0.9 + 0.75 * 0.5)
    kw = dict(thr=0.85, min_gap=100, min_start=150, max_snaps=4)
    assert lg.gate_open(0.85, 150, 0, 0, **kw)
    assert not lg.gate_open(0.849, 150, 0, 0, **kw)         # score
    assert not lg.gate_open(0.9, 149, 0, 0, **kw)           # since start
    assert not lg.gate_open(0.9, 249, 150, 1, **kw)         # since last snapshot
    assert lg.gate_open(0.9, 250, 150, 1, **kw)
    assert not lg.gate_open(0.9, 900, 800, 4, **kw)         # cap


def test_shares_and_weights():
    steps = [0.05, 0.10, 0.20]
    assert np.allclose(lg.snap_shares([0, 1], steps, 0.4), [0.05, 0.10])
    s = lg.snap_shares([2, 2, 2, 2], steps, 0.4)            # 0.8 > 0.6: scaled so the anchors keep 40 %
    assert s.sum() == pytest.approx(0.6) and np.allclose(s, 0.15)
    w = lg.league_weights(np.array([2.0, 2.0]), np.array([0.1, 0.1]))
    assert w.sum() == pytest.approx(1.0) and np.allclose(w, [0.4, 0.4, 0.1, 0.1])
    assert lg.league_weights(np.array([1.0]), np.zeros(0)).tolist() == [1.0]


def _fin(R, V, n=100):
    """finished-game sums for games with final reward R[i] and a constant critic value V[i] on each of n rows"""
    f = np.zeros(6)
    for r, v in zip(R, V):
        f += (1, n, n * r, n * r * r, n * (r - v), n * (r - v) ** 2)
    return f


def test_ev_mc():
    R = np.array([1.0, -1.0, 1.0, -1.0])
    assert lg.ev_mc(_fin(R, R)) == pytest.approx(1.0)
    assert lg.ev_mc(_fin(R, np.zeros(4))) == pytest.approx(0.0)
    assert np.isnan(lg.ev_mc(np.zeros(6))) and np.isnan(lg.ev_mc(_fin([1.0, 1.0], [0.5, 0.5])))


def test_ramp_logic():
    R = np.array([1.0, -1.0] * 60)
    good, bad = _fin(R, R * 0.8), _fin(R, -R)
    sw = [(100.0, 0.0, 100.0)] * 5              # std 1.0
    fw = [(100.0, 0.0, 100.0)] * 5
    kw = dict(it=100, last_step_iter=80, level=0, n_levels=3, every=20, min_games=100, max_ratio=1.5, min_ev=0.4)
    ok, info = lg.ramp_ok(fin=good, snap_adv_win=sw, fixed_adv_win=fw, **kw)
    assert ok and info["ratio"] == pytest.approx(1.0) and info["ev_mc"] > 0.9
    assert not lg.ramp_ok(fin=bad, snap_adv_win=sw, fixed_adv_win=fw, **kw)[0]                    # EV_mc < 0.4
    assert not lg.ramp_ok(fin=good, snap_adv_win=[(100.0, 0.0, 400.0)] * 5, fixed_adv_win=fw, **kw)[0]   # ratio 2 > 1.5
    assert lg.ramp_ok(fin=good, snap_adv_win=[(100.0, 0.0, 200.0)] * 5, fixed_adv_win=fw, **kw)[0]       # ratio 1.41
    assert not lg.ramp_ok(fin=good, snap_adv_win=sw, fixed_adv_win=fw, **{**kw, "last_step_iter": 81})[0]   # < 20 its
    assert not lg.ramp_ok(fin=good, snap_adv_win=sw, fixed_adv_win=fw, **{**kw, "level": 2})[0]            # last level
    assert not lg.ramp_ok(fin=good, snap_adv_win=sw, fixed_adv_win=fw, **{**kw, "min_games": 121})[0]      # too few games
    assert not lg.ramp_ok(fin=good, snap_adv_win=[], fixed_adv_win=fw, **kw)[0]                              # no rows


# ------------------------------------------------------------------ trainer
def _cfg(**kw):
    from blackout_env.rl.ppo import PPOConfig

    base = dict(init=CKPT, value=VALUE, device="cpu", games=4, segment=6, minibatch=8, epochs=1, critic_warmup=0, seed=5,
                opponents=[f"net:{CKPT}", "random"], opp_weights=[0.5, 1.0], pfsp="hard", heur_workers=0)
    base.update(kw)
    return PPOConfig(**base)


@needs_ckpt
def test_flags_off_identity():
    """Default flags: the batch advantage is exactly raw GAE / its std; league=True with an unreachable gate is bit-identical."""
    from blackout_env.rl.ppo import PPOTrainer

    outs = []
    for kw in ({}, {"league": True, "league_gate": 2.0}):
        tr = PPOTrainer(_cfg(**kw))
        cap = {}
        tr.diag_hook = lambda d, cap=cap: cap.update(d)
        batch, stats = tr.collect()
        assert np.allclose(batch["adv"], (cap["adv"] / (cap["adv"].std() + 1e-8)).astype(np.float32), atol=0, rtol=0)
        outs.append((batch["adv"].copy(), stats["adv_std"], batch["act"].copy()))
        tr2 = PPOTrainer(_cfg(**kw))
        s = tr2.step()
        assert np.isfinite(s["pg"]) and tr2.snaps == []
    assert np.array_equal(outs[0][0], outs[1][0]) and outs[0][1] == outs[1][1] and np.array_equal(outs[0][2], outs[1][2])


@needs_ckpt
def test_centering_in_trainer():
    from blackout_env.rl.ppo import PPOTrainer

    tr = PPOTrainer(_cfg(adv_center_opp=True, adv_center_min_games=1, games=6, segment=8))
    cap = {}
    tr.diag_hook = lambda d: cap.update(d)
    tr.trunk_opp[:] = [0, 0, 0, 1, 1, 1]
    batch, stats = tr.collect()
    row_opp = cap["opp_of"][cap["game"]]
    for i in (0, 1):
        assert abs(batch["adv"][row_opp == i].mean()) < 1e-6
    assert stats["opp_share"] == [0.5, 0.5]
    # not centred when below the games threshold
    tr = PPOTrainer(_cfg(adv_center_opp=True, adv_center_min_games=10, games=6, segment=8))
    tr.diag_hook = lambda d: cap.update(d)
    tr.trunk_opp[:] = [0, 0, 0, 1, 1, 1]
    batch, _ = tr.collect()
    assert np.allclose(batch["adv"], (cap["adv"] / (cap["adv"].std() + 1e-8)).astype(np.float32))


@needs_ckpt
def test_snapshot_gate_ramp_warmup_cpu(tmp_path):
    from blackout_env.rl.ppo import PPOTrainer

    cfg = _cfg(league=True, league_dir=str(tmp_path), league_gate=0.0, league_min_start=2, league_min_gap=2, league_max=2,
               league_warmup=2, league_ramp_every=2, league_ramp_min_games=0, league_adv_ratio=1e9, league_ev_min=-1e9,
               critic_warmup=0)
    tr = PPOTrainer(cfg)
    log = []
    for _ in range(7):
        tr.trunk_opp[:] = np.arange(4) % len(tr.opps)           # every opponent (incl. snapshots) has a game in each batch
        log.append(tr.step())
    ev = [(o["iter"], e) for o in log for e in o["league_events"]]
    snaps = [(i, e) for i, e in ev if "snapshot" in e]
    assert [i for i, _ in snaps] == [2, 4]                      # gate: it>=2, gap 2, cap 2 snapshots
    assert len(tr.snaps) == 2 and len(tr.opps) == 4 and os.path.exists(tr.snaps[0]["path"])
    # critic-only warm-up of 2 its after each snapshot: snapshot at iter 2 -> steps 3,4; at iter 4 -> steps 5,6
    assert [("pg" in o) for o in log] == [True, True, False, False, False, False, True]
    # ramp: snapshot 1 steps 5 % -> 10 % -> 20 % (every 2 its), snapshot 2 after 2 its from its creation
    ramps = [(i, e["ramp_step"], e["level"]) for i, e in ev if "ramp_step" in e]
    assert ramps[0] == (4, 1, 1) and (6, 1, 2) in ramps
    w = tr.opp_w
    assert w.sum() == pytest.approx(1.0) and w[2] == pytest.approx(0.20) and w[3] <= 0.10 + 1e-9
    assert w[:2].sum() >= 0.4 - 1e-9
    assert len(log[-1]["opp_score"]) == 4 and len(log[-1]["opp_share"]) == 4 and log[-1]["n_snaps"] == 2


@needs_ckpt
def test_snapshot_resume_cpu(tmp_path):
    from blackout_env.rl.ppo import PPOTrainer

    kw = dict(league=True, league_dir=str(tmp_path), league_gate=0.0, league_min_start=1, league_min_gap=1, league_max=2,
              league_ramp_every=1, league_ramp_min_games=0, league_adv_ratio=1e9, league_ev_min=-1e9, critic_warmup=0)
    tr = PPOTrainer(_cfg(**kw))
    for _ in range(3):
        tr.step()
    assert len(tr.snaps) == 2
    path = str(tmp_path / "latest.pt")
    tr.save_resume(path, {"decisions": 123, "wall_h": 0.5})
    assert not os.path.exists(path + ".tmp")
    tr2 = PPOTrainer(_cfg(**kw))
    ex = tr2.load_resume(path)
    assert ex == {"decisions": 123, "wall_h": 0.5}
    assert tr2.iter == tr.iter == 3 and len(tr2.opps) == len(tr.opps) == 4
    assert [s["level"] for s in tr2.snaps] == [s["level"] for s in tr.snaps]
    assert [s["last_step"] for s in tr2.snaps] == [s["last_step"] for s in tr.snaps]
    assert [list(s["win"]) for s in tr2.snaps] == [list(s["win"]) for s in tr.snaps]
    assert np.allclose(tr2.opp_score, tr.opp_score) and np.allclose(tr2.fin, tr.fin) and np.allclose(tr2.opp_w, tr.opp_w)
    assert tr2.last_snap_iter == tr.last_snap_iter and tr2.warm_until == tr.warm_until
    for k, v in tr.net.state_dict().items():
        assert np.array_equal(v.numpy(), tr2.net.state_dict()[k].numpy())
    s = tr2.step()                                                # keeps running after the resume
    assert s["iter"] == 4 and np.isfinite(s["vf_loss"])


@needs_ckpt
def test_update_micro_accumulation_matches_full_minibatch_cpu():
    """Gradient accumulation over micro-chunks = the full-minibatch update: same gradients (fp32 CPU, up to rounding)."""
    import torch

    from blackout_env.rl.ppo import PPOTrainer

    res = []
    for micro in (0, 4):
        torch.manual_seed(0)
        tr = PPOTrainer(_cfg(games=6, segment=6, minibatch=12, update_micro=micro, epochs=1, critic_warmup=0))
        batch, _ = tr.collect()
        grads = {"pol": [], "crit": []}
        tr.opt.step = lambda *a, g=grads, t=tr, **k: g["pol"].append([None if q.grad is None else q.grad.clone() for q in t.net.parameters()])
        tr.copt.step = lambda *a, g=grads, t=tr, **k: g["crit"].append([None if q.grad is None else q.grad.clone() for q in t.critic.parameters()])
        torch.manual_seed(1)
        st = tr.update(batch)
        res.append((grads, st))
    for key in ("pol", "crit"):
        assert len(res[0][0][key]) == len(res[1][0][key]) == 3        # 36 rows / minibatch 12
        for b0, b1 in zip(res[0][0][key], res[1][0][key]):
            for x, y in zip(b0, b1):
                assert (x is None) == (y is None)
                assert x is None or torch.allclose(x, y, atol=1e-7, rtol=1e-3)
        assert max(float(x.abs().max()) for x in res[0][0][key][0] if x is not None) > 0
    assert res[0][1]["vf_loss"] == pytest.approx(res[1][1]["vf_loss"], rel=1e-4)
    assert res[0][1]["pg"] == pytest.approx(res[1][1]["pg"], rel=1e-3, abs=1e-8)


@needs_ckpt
def test_league_ref_reset_cpu(tmp_path):
    """league_ref_reset: pi_ref <- the learner at each snapshot (off: pi_ref stays the init); kept across a resume."""
    import torch

    from blackout_env.rl.ppo import PPOTrainer

    kw = dict(league=True, league_dir=str(tmp_path), league_gate=0.0, league_min_start=2, league_min_gap=2, league_max=1,
              league_warmup=0, critic_warmup=0, lr=1e-3)
    init = {k: v.clone() for k, v in PPOTrainer(_cfg(**kw)).ref.net.state_dict().items()}
    same = lambda a, b: all(torch.equal(a[k], b[k]) for k in a)
    for reset in (False, True):
        tr = PPOTrainer(_cfg(**kw, league_ref_reset=reset))
        for _ in range(2):
            tr.step()
        assert len(tr.snaps) == 1
        ref = tr.ref.net.state_dict()
        assert same(ref, init) != reset
        if reset:
            snap = torch.load(tr.snaps[0]["path"], weights_only=True)["policy_state"]
            assert same(ref, snap)
            tr.step()                                             # learner moves on, ref stays at the snapshot
            assert same(tr.ref.net.state_dict(), snap) and not same(tr.net.state_dict(), snap)
            path = str(tmp_path / "latest.pt")
            tr.save_resume(path)
            tr2 = PPOTrainer(_cfg(**kw, league_ref_reset=True))
            tr2.load_resume(path)
            assert same(tr2.ref.net.state_dict(), snap)


@needs_ckpt
def test_stop_exhausted_ends_game():
    """stop_exhausted: the rollout is asked to stop at exhaustion and an exhausted game counts as ended (terminal reward,
    no bootstrap, restarted from a fresh seed)."""
    from blackout_env.rl.ppo import PPOTrainer

    tr = PPOTrainer(_cfg(stop_exhausted=True, games=3, segment=6))
    orig = tr._rollout

    def fake(*a, **kw):
        assert kw.get("stop_when_exhausted") is True
        res = orig(*a, **kw)
        res.exhausted[:] = False
        res.exhausted[1] = True   # pretend game 1 just ran out of batteries
        return res

    tr._rollout = fake
    cap = {}
    tr.diag_hook = lambda d: cap.update(d)
    old = tr.trunk[1]
    _, stats = tr.collect()
    assert stats["trunk_restarts"] == 1
    assert cap["over"].tolist() == [False, True, False]
    assert cap["boot"][1] == 0.0 and cap["boot"][0] != 0.0
    assert tr.trunk[1] is not old
    rows = np.nonzero(cap["game"] == 1)[0]
    last = rows[-1]   # terminal step: advantage = final reward - V
    assert np.isclose(cap["adv"][last], cap["final_r"][1] - cap["v"][last])
