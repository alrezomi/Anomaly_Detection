"""Summarize a prompt-check job: P(sub-goal holds) per task/check/phase and the false failures."""
import collections
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
groups = collections.defaultdict(list)
for row in rows:
    tag = row['tag']
    holds = sum(p for option, p in zip(row['options'], row['distributions'][0]) if option in tag['holds'])
    full = dict(zip(row['options'], [round(p, 2) for p in row['distributions'][0]]))
    groups[(tag['task'], tag['check'], tag['phase'])].append((holds, tag['episode'], tag['frame_index'], full))
print(f"{'task':36s} {'check':8s} {'phase':9s} n  false-fail  mean  min   answers(mean)")
for (task, check, phase), values in sorted(groups.items()):
    probs = [v[0] for v in values]
    answers = collections.defaultdict(float)
    for v in values:
        for k, p in v[3].items():
            answers[k] += p / len(values)
    print(f"{task:36s} {check:8s} {phase:9s} {len(values):2d} {sum(p < .5 for p in probs):5d}      "
          f"{sum(probs) / len(probs):.2f}  {min(probs):.2f}  {dict((k, round(v, 2)) for k, v in answers.items())}")
if len(sys.argv) > 2:
    print('\nfalse failures:')
    for (task, check, phase), values in sorted(groups.items()):
        for holds, episode, frame, full in sorted(values)[:3]:
            if holds < .5:
                print(f'  {task} {check} {phase} {episode} frame {frame}: P(holds)={holds:.2f} {full}')
