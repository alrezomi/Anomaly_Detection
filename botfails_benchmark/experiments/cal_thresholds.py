"""Normal-data thresholds per question: the conformal lower bound of the check-level P(sub-goal holds).
usage: cal_thresholds.py <results.jsonl> <out.json> [alpha] [margin]"""
import collections, json, sys, math
rows = [json.loads(l) for l in open(sys.argv[1])]
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else .05
margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.
values = collections.defaultdict(list)
for r in rows:
    t = r['tag']
    p = [sum(v for o, v in zip(r['options'], d) if o in t['holds']) for d in r['distributions']]
    values[t['task'], t['question']].append(sum(p) / len(p))
out = collections.defaultdict(dict)
for (task, q), v in sorted(values.items()):
    v = sorted(v); n = len(v)
    k = max(0, math.floor(alpha * (n + 1)) - 1)
    thr = max(.0, v[k] - margin)
    out[task][q] = round(thr, 4)
    print(f'{task:36s}{q:20s} n={n:3d} min {v[0]:.2f} p5 {v[int(.05 * n)]:.2f} median {v[n // 2]:.2f} -> threshold {thr:.3f}')
json.dump(out, open(sys.argv[2], 'w'), indent=1)
