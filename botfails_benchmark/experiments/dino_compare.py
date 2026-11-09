"""Frame AUROC of the DINO score on DEVELOPMENT episodes, for several saved runs. usage: dino_compare.py name=dir ... -- task ..."""
import sys, json
import numpy as np, pandas as pd
E = '/home/steribot/projects/AD/evaluation/FIDeL'
DEV = json.load(open(f'{E}/improvement/dev_split.json'))['dev']
def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    r = pd.Series(s).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float('nan')
args = sys.argv[1:]; cut = args.index('--'); runs = dict(a.split('=') for a in args[:cut]); tasks = args[cut + 1:]
def first(frame, scale=1.1):
    run = 0
    for row in frame.sort_values('timestamp_sec').itertuples():
        run = run + 1 if row.pooled_score > row.base_threshold * scale else 0
        if run >= 3: return row.timestamp_sec
for task in tasks:
    print('==', task)
    for name, d in runs.items():
        try: f = pd.read_csv(f'{d}/{task}/test_scores.csv')
        except FileNotFoundError: print(f'  {name}: missing'); continue
        f = f[f.episode.isin([f'episode_{i:06d}' for i in DEV[task]])]
        a, b = auroc(f.label, f.pooled_score), auroc(f.label, f.pooled_score / f.base_threshold)
        details = []
        for ep, part in f.groupby('episode'):
            onset = part[part.label == 1].timestamp_sec.min() if part.label.any() else None
            al = first(part)
            details.append(f"{ep[-2:]}:" + ('N' if onset is None else f'on{onset:.0f}') + f"/{'-' if al is None else f'{al:.0f}'}")
        print(f'  {name:10s} AUROC raw {a:.2f} ratio {b:.2f} | episode:onset/first alarm  ' + ' '.join(details))
