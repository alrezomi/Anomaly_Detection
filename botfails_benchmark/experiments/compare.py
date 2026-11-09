"""Compare the first evaluation with the improved system on the held-out, development and all test episodes."""
import pandas as pd

E = '/home/steribot/projects/AD/evaluation/FIDeL'
FIDEL = {'Acc': 73.6, 'WAcc': 78.65, 'AUROC': 70.5, 'F1': 62.9}
CASCADE, DINO = 'DINO + VLM check (cascade)', 'DINO alone (first alarm stops)'


def load(path, system=CASCADE):
    frame = pd.read_csv(f'{path}/end_to_end_metrics.csv')
    return frame[(frame.system == system)].set_index('task')


def ad(path):
    return pd.read_csv(f'{path}/ad_metrics.csv').set_index('task')


def episodes(path, column):
    counts = pd.read_csv(f'{path}/episodes.csv')[column].value_counts().to_dict()
    return (f"detected {counts.get('detected', 0)}, missed {counts.get('missed', 0)}, "
            f"before onset {counts.get('stopped_before_onset', 0)}, false stops {counts.get('false_stop', 0)}, "
            f"normal kept {counts.get('correct_continue', 0)}")


M = 'mean (10 tasks)'
reports = [('First evaluation, all 144', f'{E}/results/report'), ('Improved, held-out 86', f'{E}/improvement/final/report_heldout'),
           ('Improved, development 58', f'{E}/improvement/final/report_dev'), ('Improved, all 144', f'{E}/improvement/final/report_all')]
print(f"FIDeL: AUROC {FIDEL['AUROC']} F1 {FIDEL['F1']} | Acc ~{FIDEL['Acc']} W.Acc {FIDEL['WAcc']}\n")
for name, path in reports:
    a = ad(path).loc[M]
    c, d = load(path).loc[M], load(path, DINO).loc[M]
    print(f"{name:28s} AD AUROC {a.AUROC_raw:5.1f} F1 {a.F1_opt_raw:5.1f} | DINO alone Acc {d.Acc:5.1f} WAcc {d.WAcc:5.1f} | "
          f"Cascade TPR {c.TPR:5.1f} TNR {c.TNR:5.1f} Acc {c.Acc:5.1f} WAcc {c.WAcc:5.1f}")
    print(f"{'':28s} episodes: {episodes(path, 'cascade_outcome')}")
print('\nPer task, held-out episodes (Acc / W.Acc), first evaluation on ALL episodes for reference:')
first, held = load(f'{E}/results/report'), load(f'{E}/improvement/final/report_heldout')
heldout_dino = load(f'{E}/improvement/final/report_heldout', DINO)
ad_first, ad_new = ad(f'{E}/results/report'), ad(f'{E}/improvement/final/report_all')
for task in held.index:
    if task.endswith('mean') or task.startswith('mean'):
        continue
    print(f"{task:38s} first {first.loc[task].Acc:5.1f}/{first.loc[task].WAcc:5.1f}  now held-out {held.loc[task].Acc:5.1f}/{held.loc[task].WAcc:5.1f} "
          f"(TPR {held.loc[task].TPR:5.1f} TNR {held.loc[task].TNR:5.1f}; DINO alone held-out {heldout_dino.loc[task].Acc:5.1f})  AUROC {ad_first.loc[task].AUROC_raw:5.1f} -> {ad_new.loc[task].AUROC_raw:5.1f}")
