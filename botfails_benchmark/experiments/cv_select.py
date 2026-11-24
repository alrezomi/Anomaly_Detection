"""Choose per task between three decision rules by 5-fold cross-validation over the test episodes.

Rules: 'dino' (a DINO alarm stops), 'vlm' (the VLM verifies DINO alarms), 'vlm+release' (the VLM also checks after each
gripper release; the same as 'vlm' for tasks without release questions). Folds = episode index % 5. For each fold the
rule is chosen per task on the OTHER folds (highest mean of Acc and W.Acc over the frames of those folds) and applied to
this fold, so no episode influences its own rule. Questions, thresholds and settings were fixed earlier on development
episodes (folds 0 and 1).
Reported: all 144 episodes, and the held-out 86 (folds 2, 3, 4), which played no part in choosing the questions.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path('/home/steribot/projects/AD/evaluation/FIDeL')
BEST = E / 'improvement' / 'dino' / 'best'
RUNS = {'vlm': E / 'improvement/final/cv_vlm', 'vlm+release': E / 'improvement/final/cv_vlm_rel'}
TASKS = sorted(p.name for p in BEST.iterdir())
PREFERENCE = ('dino', 'vlm', 'vlm+release')  # On a tie, the simplest rule.
ALARM_RUN, SCALE = 3, 1.1


def dino_stop(frame):
    run = 0
    for row in frame.itertuples():
        run = run + 1 if row.pooled_score > row.base_threshold * SCALE else 0
        if run >= ALARM_RUN:
            return row.timestamp_sec
    return None


def rates(labels, predictions):
    labels, predictions = np.asarray(labels, bool), np.asarray(predictions, bool)
    pos, neg = labels.sum(), (~labels).sum()
    tpr = (predictions & labels).sum() / pos if pos else np.nan
    tnr = (~predictions & ~labels).sum() / neg if neg else np.nan
    beta = neg / len(labels)
    return dict(TPR=100 * tpr, TNR=100 * tnr, Acc=100 * np.nanmean([tpr, tnr]), WAcc=100 * (beta * tnr + (1 - beta) * tpr))


def load():
    data = {}
    for task in TASKS:
        scores = pd.read_csv(BEST / task / 'test_scores.csv').sort_values(['episode', 'timestamp_sec'])
        stops = {rule: {r['episode']: r['stop_sec'] for r in json.loads((path / f'{task}.json').read_text())}
                 for rule, path in RUNS.items()}
        episodes = {}
        for name, frame in scores.groupby('episode'):
            episodes[name] = dict(index=int(name.split('_')[1]), time=frame.timestamp_sec.to_numpy(),
                                  label=frame.label.to_numpy().astype(int), stop=dict(dino=dino_stop(frame), **{r: s[name] for r, s in stops.items()}))
        data[task] = episodes
    return data


def predict(episode, rule):
    stop = episode['stop'][rule]
    return np.zeros_like(episode['label'], bool) if stop is None else episode['time'] >= stop


def evaluate(episodes, rule):
    labels = np.concatenate([e['label'] for e in episodes])
    predictions = np.concatenate([predict(e, rule) for e in episodes])
    return rates(labels, predictions)


def main():
    data = load()
    folds = range(5)
    chosen, rows = {}, []
    predictions = {task: {} for task in TASKS}
    for task, episodes in data.items():
        for fold in folds:
            train = [e for e in episodes.values() if e['index'] % 5 != fold]
            score = {rule: (lambda m: (m['Acc'] + m['WAcc']) / 2)(evaluate(train, rule)) for rule in PREFERENCE}
            best = max(PREFERENCE, key=lambda rule: (round(score[rule], 6), -PREFERENCE.index(rule)))
            chosen[task, fold] = best
            for name, episode in episodes.items():
                if episode['index'] % 5 == fold:
                    predictions[task][name] = predict(episode, best)
    def summary(part):
        table = []
        for task, episodes in data.items():
            selected = [(n, e) for n, e in episodes.items() if part(e['index'])]
            labels = np.concatenate([e['label'] for _, e in selected])
            pred = np.concatenate([predictions[task][n] for n, _ in selected])
            table.append(dict(task=task, **rates(labels, pred)))
        table = pd.DataFrame(table)
        return table, table[['TPR', 'TNR', 'Acc', 'WAcc']].mean()
    for name, part in (('All 144 episodes', lambda i: True), ('Held-out 86 episodes (folds 2-4)', lambda i: i % 5 not in (0, 1))):
        table, mean = summary(part)
        print(f'--- {name}: TPR {mean.TPR:.1f} TNR {mean.TNR:.1f} Acc {mean.Acc:.1f} WAcc {mean.WAcc:.1f}')
        print(table.round(1).to_string(index=False))
    print()
    for task in TASKS:
        print(f'{task:38s}', [chosen[task, fold] for fold in folds])
    out = E / 'improvement' / 'final' / 'cv_select.json'
    table_all, mean_all = summary(lambda i: True)
    table_heldout, mean_heldout = summary(lambda i: i % 5 not in (0, 1))
    out.write_text(json.dumps(dict(chosen={f'{t}|{f}': r for (t, f), r in chosen.items()},
                                   all144=dict(mean=mean_all.to_dict(), per_task=table_all.to_dict('records')),
                                   heldout86=dict(mean=mean_heldout.to_dict(), per_task=table_heldout.to_dict('records'))), indent=1))


if __name__ == '__main__':
    raise SystemExit('Archived hybrid experiment: per-task DINO-only rule selection was removed. Use final.sh for the strict DINO -> VLM cascade.')
