"""Build a prompt-check job on NORMAL training frames only (false-failure control).

usage: bf_nominal_job.py <questions.json> <job name> [frames per task and phase]
questions.json: {task: [{"name", "phases": [...], "question", "options", "holds": [...]}, ...]}
Frames are drawn per gripper phase (holding / released) from the normal_train episodes.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, '/home/steribot/projects/AD/anomaly_detection')
from botfails_benchmark.data import episodes  # noqa: E402
from botfails_benchmark.phases import episode_phases  # noqa: E402

E = Path('/home/steribot/projects/AD/evaluation/FIDeL')
SCRATCH = Path('/tmp/claude-1000/-home-steribot-projects/f0e2de92-2e5a-401c-8559-5de8ed21cbd2/scratchpad')


def main():
    questions = json.loads(Path(sys.argv[1]).read_text())
    name, per = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12
    rng = random.Random(0)
    cases = []
    for task, checks in questions.items():
        pool = {'holding': [], 'released': [], 'start': []}
        for episode in episodes(E / 'dataset' / 'BotFails', E / 'frames', 'normal_train', task):
            phases = episode_phases(episode)
            for frame_index, t, path in episode.frame_table('logitech_1'):
                pool[phases.at(t)[1]].append((episode.name, frame_index, str(path)))
        for phase in ('holding', 'released'):
            picks = rng.sample(pool[phase], min(per, len(pool[phase])))
            for check in checks:
                if phase not in check['phases']:
                    continue
                for episode_name, frame_index, path in picks:
                    cases.append(dict(images=[path.replace(str(E), '/evaluation')], question=check['question'],
                                      options=check['options'],
                                      tag=dict(task=task, phase=phase, check=check['name'], holds=check['holds'],
                                               episode=episode_name, frame_index=frame_index)))
        print(task, {k: len(v) for k, v in pool.items()})
    (SCRATCH / 'bf_jobs').mkdir(exist_ok=True)
    (SCRATCH / 'bf_jobs' / f'{name}.json').write_text(json.dumps(dict(cases=cases)))
    print(len(cases), 'cases')


if __name__ == '__main__':
    main()
