"""Queue a reference-comparison test on development episodes.

usage: ref_job.py <job> <dino_dir> <camera> <window> task [task ...]
Frames of development episodes (every 2 s): crop around DINO's anomaly peak + same crop of the DINO-matched nominal frame.
Variants: multi-image question with crops, and with full images.
"""
import json, sys
from pathlib import Path
import pandas as pd
E = '/home/steribot/projects/AD/evaluation/FIDeL'
SC = Path('/tmp/claude-1000/-home-steribot-projects/f0e2de92-2e5a-401c-8559-5de8ed21cbd2/scratchpad')
DEV = json.load(open(f'{E}/improvement/dev_split.json'))['dev']
job, dino, camera, window = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
QUESTIONS = {
 'ref_crop_diff': 'Image 1 shows a part of the workspace in a correct run of the task. Image 2 shows the same part of the workspace now. Ignore the red robot arm and small shifts of the camera image. Is an object in image 2 missing, added, moved, or different compared with image 1?\nAnswer with one word: yes or no.',
 'ref_full_diff': 'Image 1 shows a correct run of the task. Image 2 shows the same moment now. Ignore the red robot arm. Is an object in image 2 missing, added, moved, or different compared with image 1?\nAnswer with one word: yes or no.',
}
cases = []
for task in sys.argv[5:]:
    f = pd.read_csv(f'{dino}/{task}/test_scores.csv')
    f = f[f.episode.isin([f'episode_{i:06d}' for i in DEV[task]])]
    f = f[(f.timestamp_sec * 2).round().astype(int) % 4 == 0]  # every 2 s
    for r in f.itertuples():
        now = f'/evaluation/frames/test/{task}_anomaly/{r.episode}/{camera}/{r.frame_index:06d}.jpg'
        ref = f'/evaluation/frames/normal_train/{task}_expert/{r.matched_episode.split("/")[-1]}/{camera}/{r.matched_frame_index:06d}.jpg'
        x0 = max(0, min(r.peak_x - window / 2, 1 - window)); y0 = max(0, min(r.peak_y - window / 2, 1 - window))
        box = f'#{x0:.3f},{y0:.3f},{x0 + window:.3f},{y0 + window:.3f}'
        tag = dict(task=task, episode=int(r.episode.split('_')[1]), t=float(r.timestamp_sec), label=int(r.label), onset=None, phase='', source='test',
                   holds=['no'], ratio=float(r.pooled_score / r.base_threshold))
        cases.append(dict(multi=[['Image 1:', ref + box], ['Image 2:', now + box]], question=QUESTIONS['ref_crop_diff'], options=['yes', 'no'], tag=dict(tag, variant='ref_crop_diff')))
        cases.append(dict(multi=[['Image 1:', ref], ['Image 2:', now]], question=QUESTIONS['ref_full_diff'], options=['yes', 'no'], tag=dict(tag, variant='ref_full_diff')))
(SC / 'bf_jobs' / f'{job}.json').write_text(json.dumps(dict(cases=cases)))
print(len(cases), 'cases')
