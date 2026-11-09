"""Experiment 1: the user's idea, a richer task description, compared with the current questions."""
import json, sys
sys.path.insert(0, '/home/steribot/projects/AD/anomaly_detection')
from botfails_benchmark.prompts import CAMERA, check_question
tasks = json.load(open('/home/steribot/projects/AD/anomaly_detection/botfails_benchmark/tasks.json'))
desc = json.load(open(sys.argv[1]))
CAM2 = 'This image comes from a camera that watches the robot at work.'
GENERIC = 'Is a mistake visible in this image, such as an object in a wrong place, something spilled, or something dropped?'
out = {}
for task, d in desc.items():
    v = {}
    for check in tasks[task]['checks']:
        q = check_question(check)
        v[f"base_{q['name']}"] = dict(question=q['text'], options=list(q['options']), holds=list(q['holds']))
        v[f"desc_{q['name']}"] = dict(question=f"{d}\n{q['text'].replace(CAMERA, CAM2)}", options=list(q['options']), holds=list(q['holds']))
    v['generic_short'] = dict(question=f"{CAMERA} The robot's task is: {tasks[task]['instruction']}. {GENERIC}\nAnswer with one word: yes or no.", options=['yes', 'no'], holds=['no'])
    v['generic_desc'] = dict(question=f"{d}\n{CAM2} {GENERIC}\nAnswer with one word: yes or no.", options=['yes', 'no'], holds=['no'])
    out[task] = v
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print({t: list(v) for t, v in out.items()})
