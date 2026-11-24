"""Normal-data quantile thresholds per question (not calibrated safety probabilities).

Repeated frames within episodes and alarm selection require separate validation;
this script alone does not guarantee deployment or episode false-stop rates.
usage: cal_thresholds.py <results.jsonl> <out.json> [alpha] [margin]"""
import collections, json, sys, math
rows = [json.loads(l) for l in open(sys.argv[1])]
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else .05
margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.
if not 0 < alpha < 1 or margin < 0:
    raise ValueError('Require 0 < alpha < 1 and a nonnegative margin.')
values = collections.defaultdict(list)
for r in rows:
    t = r['tag']
    p = [sum(v for o, v in zip(r['options'], d) if o in t['holds']) for d in r['distributions']]
    values[t['task'], t['question']].append(sum(p) / len(p))
out = collections.defaultdict(dict)
for (task, q), v in sorted(values.items()):
    v = sorted(v); n = len(v)
    rank = math.floor(alpha * (n + 1))
    # Below the first finite rank, zero is conservative for scores in [0, 1].
    thr = max(.0, v[rank - 1] - margin) if rank else 0.
    # Rounding up would reject the calibration point defining the boundary.
    out[task][q] = thr
    print(f'{task:36s}{q:20s} n={n:3d} min {v[0]:.2f} p5 {v[int(.05 * n)]:.2f} median {v[n // 2]:.2f} -> threshold {thr:.3f}')
json.dump(out, open(sys.argv[2], 'w'), indent=1)
