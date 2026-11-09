"""tasks.json -> questions.json for bf_nominal_job.py, using the benchmark's own template."""
import json, sys
sys.path.insert(0, '/home/steribot/projects/AD/anomaly_detection')
from botfails_benchmark.prompts import task_questions
config = json.load(open(sys.argv[1]))
out = {}
for task, cfg in config.items():
    if task.startswith('_'):
        continue
    entries = {}
    for phase in ('holding', 'released'):
        for q in task_questions(cfg, phase):
            entry = entries.setdefault(q['name'], dict(name=q['name'], phases=[], question=q['text'], options=list(q['options']), holds=list(q['holds'])))
            entry['phases'].append(phase)
    out[task] = list(entries.values())
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print({t: [(e['name'], e['phases']) for e in v] for t, v in out.items()})
