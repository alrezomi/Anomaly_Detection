"""tasks.json -> tasks_generic.json: one instruction-only question per task (ablation, FIDeL-like input)."""
import json, sys
config = json.load(open(sys.argv[1]))
out = {'_about': 'Ablation: the same cascade, but one generic question that only uses the task instruction '
                 '(no per-task objects), asked in every phase after the first grasp. Generated from tasks.json.'}
for task, cfg in config.items():
    if task.startswith('_'):
        continue
    out[task] = dict(instruction=cfg['instruction'], checks=[dict(
        name='generic', kind='choice', phases=['holding', 'released'],
        lead=f'The robot\'s task is: "{cfg["instruction"]}". Which describes the image?',
        A='Every object is where it should be at this point of the task',
        B='An object was dropped, spilled, or put in a wrong place',
        C='The image does not show the task', holds=['A', 'C'])])
json.dump(out, open(sys.argv[2], 'w'), indent=1)
