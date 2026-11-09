"""Queue a prompt test on the DEVELOPMENT episodes only (improvement/dev_split.json).

usage: dev_job.py <variants.json> <job> [step_sec] [train_normal_per_task]
variants.json: {task: {variant: {"question": str, "options": [...], "holds": [...], "camera": "logitech_1"}}}
Frames every step_sec of each development test episode (label 1 = positive, label 0 = negative), plus random
normal training frames (negatives). Held-out episodes are never touched here.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, '/home/steribot/projects/AD/anomaly_detection')
from botfails_benchmark.data import Episode, episodes  # noqa: E402
from botfails_benchmark.phases import episode_phases  # noqa: E402

E = Path('/home/steribot/projects/AD/evaluation/FIDeL')
SCRATCH = Path('/tmp/claude-1000/-home-steribot-projects/f0e2de92-2e5a-401c-8559-5de8ed21cbd2/scratchpad')
DEV = json.loads((E / 'improvement' / 'dev_split.json').read_text())['dev']


def container(path):
    return str(path).replace(str(E), '/evaluation')


def main():
    variants = json.loads(Path(sys.argv[1]).read_text())
    name, step = sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3.
    train_n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    rng, cases = random.Random(0), []
    for task, task_variants in variants.items():
        for variant, spec in task_variants.items():
            camera = spec.get('camera', 'logitech_1')
            for index in DEV[task]:
                episode = Episode(E / 'dataset' / 'BotFails', E / 'frames', 'test', task, index)
                labels, phases = episode.labels(), episode_phases(episode)
                onset = float(labels.argmax() / 30) if labels.any() else None
                for frame, t, path in episode.frame_table(camera):
                    if abs(t / step - round(t / step)) > 1e-6:
                        continue
                    cases.append(dict(images=[container(path) + spec.get('crop', '')], question=spec['question'], options=spec['options'],
                                      tag=dict(task=task, variant=variant, holds=spec['holds'], episode=index, t=t,
                                               label=int(labels[frame]), onset=onset, phase=phases.at(t)[1], source='test')))
            pool = []
            for episode in episodes(E / 'dataset' / 'BotFails', E / 'frames', 'normal_train', task):
                phases = episode_phases(episode)
                pool += [(episode.index, t, path, phases.at(t)[1]) for _, t, path in episode.frame_table(camera)]
            for index, t, path, phase in random.Random(0).sample(pool, min(train_n, len(pool))):
                cases.append(dict(images=[container(path) + spec.get('crop', '')], question=spec['question'], options=spec['options'],
                                  tag=dict(task=task, variant=variant, holds=spec['holds'], episode=index, t=t, label=0,
                                           onset=None, phase=phase, source='train')))
    (SCRATCH / 'bf_jobs' / f'{name}.json').write_text(json.dumps(dict(cases=cases)))
    print(len(cases), 'cases;', len(variants), 'tasks')


if __name__ == '__main__':
    main()
