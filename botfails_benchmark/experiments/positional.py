"""Experiment: position-aware patch memory over ALL nominal frames, on development episodes only.

For each query patch, the distance to the nearest patch of any nominal frame within the 3x3 neighbourhood of the
same image position. Frame score = mean of the top 1 % of the 3x3-pooled map (like our pooled score).
usage: positional.py <camera> <task> [every_nth_nominal] [radius]
"""
import glob
import json
import sys

import numpy as np
import torch

E = '/evaluation'
DEV = json.load(open(f'{E}/improvement/dev_split.json'))['dev']
camera, task = sys.argv[1], sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
radius = int(sys.argv[4]) if len(sys.argv) > 4 else 1
device = 'cuda'


def load(split, episode=None):
    pattern = f'{E}/feature_cache/{split}/{task}/{camera}/{episode or "episode_*"}.npz'
    return sorted(glob.glob(pattern))


memory = []
for path in load('normal_train'):
    with np.load(path) as arrays:
        memory.append(arrays['features'][::every])
memory = torch.from_numpy(np.concatenate(memory)).to(device)  # float16, unit-norm features
N, P, D = memory.shape
G = int(np.sqrt(P))
memory = memory.reshape(N, G, G, D)
print(f'{task} {camera}: nominal {N} frames, grid {G}', flush=True)
offsets = [(r, c) for r in range(-radius, radius + 1) for c in range(-radius, radius + 1)]


@torch.inference_mode()
def score_frame(query):
    query = query.reshape(G, G, D)
    best = torch.full((G, G), -1., device=device, dtype=torch.float16)
    for r, c in offsets:
        shifted = torch.roll(memory, shifts=(r, c), dims=(1, 2))
        similarity = (shifted * query).sum(-1).max(0).values
        best = torch.maximum(best, similarity)
    distance = (1 - best).clamp_min(0).float()
    pooled = torch.nn.functional.avg_pool2d(distance[None, None], 3, 1, 1, count_include_pad=False)[0, 0].ravel()
    k = max(1, int(np.ceil(.01 * pooled.numel())))
    return float(pooled.topk(k).values.mean())


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    order = s.argsort().argsort() + 1.
    # ties are rare with float scores
    n1 = y.sum(); n0 = len(y) - n1
    return (order[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float('nan')


labels, scores, episodes = [], [], []
for index in DEV[task]:
    name = f'episode_{index:06d}'
    path = load('test', name)
    if not path:
        print('no cached features yet for', name)
        continue
    with np.load(path[0]) as arrays:
        features = torch.from_numpy(arrays['features']).to(device)
    table = open(f'{E}/frames/test/{task}_anomaly/{name}/{camera}/frames.csv').read().splitlines()[1:]
    frames = [int(line.split(',')[0]) for line in table]
    y = np.loadtxt(f'{E}/dataset/BotFails/labels/{task}_anomaly/{name}_labels.csv', dtype=float, ndmin=1).astype(int)[frames]
    s = [score_frame(f) for f in features]
    labels += y.tolist(); scores += s; episodes += [index] * len(s)
    onset = (y.argmax() * .5) if y.any() else None
    print(f'  {name}: onset {onset}  mean normal {np.mean([v for v, l in zip(s, y) if not l]):.3f}  mean anomalous {np.mean([v for v, l in zip(s, y) if l]) if y.any() else float("nan"):.3f}', flush=True)
print(f'{task} {camera}: positional patch memory AUROC on development frames = {auroc(labels, scores):.3f}')
