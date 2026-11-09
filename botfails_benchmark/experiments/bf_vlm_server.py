"""Persistent RynnBrain server for BotFails prompt checks.

Jobs: /scratch/bf_jobs/<name>.json = {"cases": [{"images": [path, ...] | "multi": [[label, path], ...],
"question": str, "options": [...], "tag": {...}}]}. "images" are asked one by one; "multi" is one forward
pass with several images. A path may be "path#x0,y0,x1,y1" (crop in fractions of the image, upscaled by the
processor). Results stream to /scratch/bf_results/<name>.jsonl.
"""
import json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, '/workspace/anomaly_detection')
JOBS, RESULTS = Path('/scratch/bf_jobs'), Path('/scratch/bf_results')


def load(path):
    from PIL import Image
    path, _, box = path.partition('#')
    with Image.open(path) as image:
        image = image.convert('RGB')
    if box:
        w, h = image.size
        x0, y0, x1, y1 = [float(v) for v in box.split(',')]
        image = image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        scale = 640 / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.BICUBIC)
    return image


def main():
    from pipeline_settings import load_pipeline_config
    from rynnbrain_vlm.model import RynnBrainModel
    config = load_pipeline_config(Path('/config/pipeline_config.json'))
    vlm = config['rynnbrain']
    model = RynnBrainModel({**vlm['model'], 'lora_adapter_path': None, **vlm['vlm_check'].get('model', {})})
    JOBS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    print('READY', flush=True)
    while True:
        pending = sorted(JOBS.glob('*.json'))
        if not pending:
            time.sleep(1); continue
        job = pending[0]; name = job.stem
        try:
            cases = json.loads(job.read_text())['cases']
            out = RESULTS / f'{name}.jsonl'; out.unlink(missing_ok=True)
            started = time.time()
            with out.open('a') as stream:
                for number, case in enumerate(cases):
                    if (JOBS / f'{name}.cancel').exists():
                        break
                    if 'multi' in case:
                        turn = dict(role='user', images=[(l, load(p)) for l, p in case['multi']], text=case['question'])
                        distributions = [model.answer_distribution(turn, tuple(case['options']))]
                    else:
                        distributions = [model.answer_distribution(dict(role='user', images=[('Camera image:', load(p))],
                                         text=case['question']), tuple(case['options'])) for p in case['images']]
                    stream.write(json.dumps(dict(case=number, tag=case.get('tag'), options=case['options'], distributions=distributions)) + '\n')
                    stream.flush()
            print(f'{name}: {len(cases)} cases in {time.time() - started:.0f}s', flush=True)
        except Exception:
            traceback.print_exc()
        finally:
            job.rename(job.with_suffix('.done'))
            (JOBS / f'{name}.cancel').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
