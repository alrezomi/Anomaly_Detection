"""Queue the normal-data calibration of the VLM questions: check-level statistic (mean over the frames at t, t-0.5, t-1)
at random times of NORMAL TRAINING episodes, in the phases where each question is asked.
usage: cal_job.py <tasks.json> <job> [checks per question]"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, '/home/steribot/projects/AD/anomaly_detection')
from botfails_benchmark.data import episodes
from botfails_benchmark.phases import episode_phases
from botfails_benchmark.prompts import task_questions
E = Path('/home/steribot/projects/AD/evaluation/FIDeL')
SCRATCH = Path('/tmp/claude-1000/-home-steribot-projects/f0e2de92-2e5a-401c-8559-5de8ed21cbd2/scratchpad')
tasks = {k: v for k, v in json.load(open(sys.argv[1])).items() if not k.startswith('_')}
name, per = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 50
cases = []
for task, cfg in tasks.items():
    camera = cfg.get('camera', 'logitech_1')
    pool = {'start': [], 'holding': [], 'released': []}
    for episode in episodes(E / 'dataset' / 'BotFails', E / 'frames', 'normal_train', task):
        table, phases = episode.frame_table(camera), episode_phases(episode)
        for k in range(2, len(table)):
            t = table[k][1]
            pool[phases.at(t)[1]].append((episode.index, t, [str(table[j][2]).replace(str(E), '/evaluation') for j in (k, k - 1, k - 2)], phases.at(t)[1]))
    seen = {}
    for phase in pool:
        for q in task_questions(cfg, phase):
            seen.setdefault(q['name'], (q, []))[1].append(phase)
    for qname, (q, phases_used) in seen.items():
        options = [c for phase in phases_used for c in pool[phase]]
        for index, t, images, phase in random.Random(0).sample(options, min(per, len(options))):
            crop = q.get('crop')
            suffix = '#' + ','.join(str(v) for v in crop) if crop else ''
            cases.append(dict(images=[i + suffix for i in images], question=q['text'], options=list(q['options']),
                              tag=dict(task=task, question=qname, holds=list(q['holds']), episode=index, t=t, phase=phase)))
(SCRATCH / 'bf_jobs' / f'{name}.json').write_text(json.dumps(dict(cases=cases)))
print(len(cases), 'checks,', 3 * len(cases), 'forward passes')
