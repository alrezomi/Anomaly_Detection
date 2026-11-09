"""Mean P(sub-goal holds) per development episode (before/after the onset) for each question. usage: per_episode.py results.jsonl [task-substr]"""
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
sub = sys.argv[2] if len(sys.argv) > 2 else ''
g = collections.defaultdict(lambda: collections.defaultdict(list))
train = collections.defaultdict(list)
for r in rows:
    t = r['tag']
    if sub not in t['task']: continue
    p = sum(v for o, v in zip(r['options'], r['distributions'][0]) if o in t['holds'])
    if t['source'] == 'train': train[t['task'], t['variant']].append(p); continue
    g[t['task'], t['variant']][t['episode'], t['label']].append(p)
for (task, variant), eps in sorted(g.items()):
    tr = train[task, variant]
    print(f"{task.split('_')[-1][:14]:14s}{variant:18s} train {sum(tr)/len(tr):.2f} | " + ' '.join(
        f"ep{e}{'+' if l else ''}:{sum(v)/len(v):.2f}" for (e, l), v in sorted(eps.items())))
