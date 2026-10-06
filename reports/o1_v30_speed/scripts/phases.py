import sys, glob, re, time, collections
sys.path.insert(0, "/private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1")
import replay
from blackout_env.heuristics.v30_solo_sentry import StrategicHeuristicV30 as C
from blackout_env.heuristics.v22_teacher import StrategicHeuristicV22Teach as V22T
from blackout_env.heuristics.v17_teach import StrategicHeuristicV17Teach as V17T
from blackout_env.heuristics.v20_anticamp import StrategicHeuristicV20Teach as V20T
from blackout_env.heuristics.v21_safe_transform import StrategicHeuristicV21Teach as V21T
from blackout_env.heuristics.any_angle import AnyAngleMixin
T = collections.Counter(); N = collections.Counter()
depth = [0]
def wrap(cls, name, tag):
    f = getattr(cls, name)
    def w(*a, **k):
        t = time.perf_counter(); r = f(*a, **k); T[tag] += time.perf_counter() - t; N[tag] += 1; return r
    setattr(cls, name, w)
wrap(C, "act", "act")
wrap(C, "_context", "context")
wrap(C, "_composition_goals", "composition")
wrap(C, "_economy_goals", "economy")
wrap(C, "_deposit_goals", "deposit")
wrap(C, "_hunter_goals", "hunters")
wrap(C, "_labels", "labels")
wrap(V17T, "_labels", "labels_v17base")
wrap(AnyAngleMixin, "_any_angle", "any_angle(all)")
files = sorted(f for f in glob.glob("/private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/corpus/*.npz") if re.search(sys.argv[1], f))
replay.run(files[0]); T.clear(); N.clear()
for f in files: replay.run(f)
for k, v in T.most_common(): print(f"{k:16s} {v:7.2f}s {v / T['act'] * 100:5.1f}% {v / N['act'] * 1e6:7.1f} us/tick calls/tick {N[k] / N['act']:.2f}")
