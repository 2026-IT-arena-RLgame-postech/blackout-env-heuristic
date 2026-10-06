import sys, glob, re, time, collections, importlib
sys.path.insert(0, "/private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1")
import replay
H = "blackout_env.heuristics."
spec = [l.split() for l in sys.argv[2].split(";")]  # "module Class method tag"
T = collections.Counter(); N = collections.Counter()
from blackout_env.heuristics.v30_solo_sentry import StrategicHeuristicV30 as C
def wrap(cls, name, tag):
    f = cls.__dict__[name]
    is_static = isinstance(f, staticmethod)
    fn = f.__func__ if is_static else f
    def w(*a, **k):
        t = time.perf_counter(); r = fn(*a, **k); T[tag] += time.perf_counter() - t; N[tag] += 1; return r
    setattr(cls, name, staticmethod(w) if is_static else w)
from blackout_env.heuristics.any_angle import AnyAngleMixin; wrap(AnyAngleMixin, "act", "act")
for mod, cls, meth, tag in spec:
    wrap(getattr(importlib.import_module(H + mod), cls), meth, tag)
files = sorted(f for f in glob.glob("/private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/corpus/*.npz") if re.search(sys.argv[1], f))
replay.run(files[0]); T.clear(); N.clear()
for f in files: replay.run(f)
for k, v in T.most_common(): print(f"{k:22s} {v:7.2f}s {v / T['act'] * 100:5.1f}% {v / N['act'] * 1e6:7.1f} us/tick calls/tick {N[k] / N['act']:.2f}")
