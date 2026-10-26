"""Select whole-demo decisions or offline, causal-prefix alarm decisions."""
import numpy as np
import pandas as pd


def configure_decision_mode(selection, vlm, *, skip_vlm=False):
    mode = selection.get('decision_mode', 'whole_demo')
    if not isinstance(mode, str) or mode not in {'whole_demo', 'timeline'}:
        raise ValueError('vlm_benchmark.decision_mode must be whole_demo or timeline.')
    if mode == 'timeline':
        classifier = vlm.get('cop_classifier', {})
        if skip_vlm or not classifier.get('enabled', False):
            raise ValueError('Timeline benchmark requires an enabled cop_classifier and VLM evaluation.')
        # Prefix scoring is required for this protocol, even if optional plots
        # were disabled for ordinary whole-demo evaluation.
        vlm['cop_classifier'] = {**classifier, 'plot_timeline': True}
    return mode


def timeline_benchmark_row(original, method):
    """Keep the final outcome for audit; select any observed prefix crossing.

    A crossing proves a positive even if another prefix failed. An incomplete
    no-alert sequence cannot prove a negative. ROC scores require a complete
    sequence, since a missing prefix could contain its maximum.
    """
    row = dict(original)
    fields = ('decision', 'decision_correct', 'failure_probability', 'failure_percent',
              'anomaly_score', 'threshold', 'model_path', 'error', 'score_kind')
    row.update({f'whole_demo_classifier_{key}': row.get(f'classifier_{key}') for key in fields})
    row.update({f'classifier_{key}': None for key in fields if key != 'score_kind'})
    row.update(benchmark_decision_mode='timeline', timeline_first_alarm_time_sec=None,
               timeline_peak_score=None, timeline_peak_score_ratio=None,
               timeline_valid_points=0, timeline_total_points=0,
               timeline_classifier_source=None, timeline_error=None)
    try:
        status = row.get(f'{method}_timeline_status')
        path = row.get(f'{method}_timeline_csv')
        if status not in {'created', 'partial'} or not path:
            raise ValueError(row.get(f'{method}_timeline_error') or f'Timeline is {status or "unavailable"}.')
        data = pd.read_csv(path)
        times = pd.to_numeric(data.timestamp_sec, errors='coerce').to_numpy(dtype=float)
        scores = pd.to_numeric(data.anomaly_score, errors='coerce').to_numpy(dtype=float)
        cutoffs = pd.to_numeric(data.threshold, errors='coerce').to_numpy(dtype=float)
        if not len(times) or not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) <= 0).any():
            raise ValueError('Timeline timestamps must be finite, nonnegative and strictly increasing.')
        valid = np.isfinite(scores) & (scores >= 0) & np.isfinite(cutoffs) & (cutoffs > 0)
        if method == 'logistic':
            valid &= (scores <= 1) & (cutoffs <= 1)
            if valid.any() and not np.allclose(cutoffs[valid], cutoffs[valid][0], rtol=0, atol=1e-12):
                raise ValueError('Logistic timeline must use one classifier threshold throughout the recording.')
        if 'error' in data:
            valid &= data.error.fillna('').astype(str).str.strip().eq('').to_numpy()
        row.update(timeline_total_points=len(times), timeline_valid_points=int(valid.sum()))
        if not valid.any():
            raise ValueError('No valid prefix scores are available.')
        for column, destination in [('threshold_source', 'timeline_classifier_source'),
                                    ('classifier_model_path', 'classifier_model_path')]:
            if column in data:
                values = data.loc[valid, column].dropna().astype(str).unique()
                if len(values) != 1:
                    raise ValueError(f'Timeline must use one {column}.')
                row[destination] = values[0]
        alerts = np.flatnonzero(valid & (scores >= cutoffs))
        complete = status == 'created' and valid.all()
        if not complete:
            row['timeline_error'] = 'Incomplete prefix scores; no-alert decisions and peak ROC scores are unavailable.'
        if len(alerts):
            row['timeline_first_alarm_time_sec'] = float(times[alerts[0]])
        elif not complete:
            raise ValueError(row['timeline_error'])
        row['classifier_decision'] = 'failure' if len(alerts) else 'success'
        expected = {'normal': 'success', 'fail': 'failure'}.get(row.get('ground_truth_label'))
        row['classifier_decision_correct'] = row['classifier_decision'] == expected if expected else None
        if complete:
            ratios = scores[valid]/cutoffs[valid]
            peak = int(np.argmax(ratios))
            row.update(timeline_peak_score=float(scores[valid][peak]),
                       timeline_peak_score_ratio=float(ratios[peak]))
            if method == 'logistic':
                # Maximum prefix probability is a bag ranking score, not a
                # calibrated probability of any failure over the whole bag.
                row.update(classifier_failure_probability=float(scores[valid].max()),
                           classifier_failure_percent=float(scores[valid].max()*100),
                           classifier_threshold=float(cutoffs[valid][0]))
            else:
                # kNN has a different nominal threshold at each prefix length.
                # Normalize before pooling so ROC agrees with the alarm rule.
                row.update(classifier_anomaly_score=float(ratios.max()), classifier_threshold=1.)
        else:
            row['classifier_threshold'] = float(cutoffs[valid][0]) if method == 'logistic' else 1.
    except (OSError, ValueError, AttributeError, KeyError, pd.errors.ParserError) as error:
        row.update(classifier_decision=None, classifier_decision_correct=None, classifier_error=str(error),
                   timeline_error=str(error))
    return row
