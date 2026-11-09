import json, sys
import pandas as pd
d, name = sys.argv[1], sys.argv[2]
e = pd.read_csv(f'{d}/report/end_to_end_metrics.csv')
ep = pd.read_csv(f'{d}/report/episodes.csv')
def row(sys_):
    t = e[(e.system == sys_) & (e.task == 'mean (10 tasks)')].iloc[0]
    return f'TPR {t.TPR:5.1f} TNR {t.TNR:5.1f} Acc {t.Acc:5.1f} WAcc {t.WAcc:5.1f}'
print(f'--- {name}')
print(' DINO alone:', row('DINO alone (first alarm stops)'), ' episodes', ep.dino_outcome.value_counts().to_dict())
print(' DINO + VLM:', row('DINO + VLM check (cascade)'), ' episodes', ep.cascade_outcome.value_counts().to_dict())
c = e[(e.system == 'DINO + VLM check (cascade)') & e.task.str.contains('_')]
print('   per task Acc:', ' '.join(f"{t.task.split('_')[-1][:8]}={t.Acc:.0f}" for t in c.itertuples()))
