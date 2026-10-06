"""Signal-based sampling profiler (ITIMER_PROF) over a replay subset."""
import sys, signal, collections, glob, re, os
sys.path.insert(0, "/private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1")
import replay
files = sorted(f for f in glob.glob("/private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/corpus/*.npz") if re.search(sys.argv[1], f))
replay.run(files[0])
incl = collections.Counter(); lines = collections.Counter(); self_ = collections.Counter(); total = [0]
def handler(sig, frame):
    st = []
    f = frame
    while f is not None:
        st.append((os.path.basename(f.f_code.co_filename), f.f_code.co_name, f.f_lineno)); f = f.f_back
    if not any(n == "act" and fn == "any_angle.py" for fn, n, _ in st):
        return
    total[0] += 1
    self_[st[0][:2]] += 1
    seen = set()
    for fr in st:
        if fr[:2] not in seen:
            seen.add(fr[:2]); incl[fr[:2]] += 1
        lines[fr] += 1
signal.signal(signal.SIGPROF, handler)
signal.setitimer(signal.ITIMER_PROF, 0.0003, 0.0003)
for f in files: replay.run(f)
signal.setitimer(signal.ITIMER_PROF, 0, 0)
n = total[0]
print("samples", n)
print("== inclusive by function")
for k, v in incl.most_common(70): print(f"{v / n * 100:5.1f}% {k[0]}:{k[1]}")
print("== self by function")
for k, v in self_.most_common(40): print(f"{v / n * 100:5.1f}% {k[0]}:{k[1]}")
print("== inclusive by line")
for k, v in lines.most_common(300):
    if v / n > 0.006: print(f"{v / n * 100:5.1f}% {k[0]}:{k[2]} {k[1]}")
