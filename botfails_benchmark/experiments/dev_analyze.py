"""Compare prompt variants on development frames: how well P(sub-goal holds) separates failure from normal frames.

AUROC: positives = frames after the failure began; negatives = frames before it, normal episodes, normal training frames.
fail@.5: share of positives / negatives called failure at the fixed 0.5 rule.
fail@cal: the same with a threshold set from NORMAL TRAINING frames only (their minimum minus 0.02).
"""
import collections
import json
import sys


def auroc(pos, neg):
    if not pos or not neg:
        return float('nan')
    wins = sum((p > n) + .5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


rows = []
for path in sys.argv[1:]:
    rows += [json.loads(line) for line in open(path)]
groups = collections.defaultdict(list)
for row in rows:
    tag = row['tag']
    p = sum(v for o, v in zip(row['options'], row['distributions'][0]) if o in tag['holds'])
    groups[tag['task'], tag['variant']].append((p, tag))
print(f"{'task':36s}{'variant':16s}{'AUROC':>6s} {'n+':>4s} {'n-':>4s} | {'mean P normal':>13s} {'mean P failure':>14s} | {'fail@.5 pos/neg':>16s} | {'fail@cal pos/neg':>17s}")
last = None
for (task, variant), values in sorted(groups.items()):
    pos = [p for p, t in values if t['label'] == 1]
    neg = [p for p, t in values if t['label'] == 0]
    train = [p for p, t in values if t['source'] == 'train']
    cal = min(train) - .02 if train else .5
    # A failure lowers P(holds), so the score is 1 - P.
    a = auroc([1 - p for p in pos], [1 - p for p in neg])
    if task != last:
        print()
        last = task
    print(f"{task:36s}{variant:16s}{a:6.2f} {len(pos):4d} {len(neg):4d} | {sum(neg) / max(len(neg), 1):13.2f} {sum(pos) / max(len(pos), 1):14.2f} | "
          f"{sum(p < .5 for p in pos) / max(len(pos), 1):7.2f}/{sum(p < .5 for p in neg) / max(len(neg), 1):.2f}       | "
          f"{sum(p < cal for p in pos) / max(len(pos), 1):7.2f}/{sum(p < cal for p in neg) / max(len(neg), 1):.2f} (thr {cal:.2f})")
