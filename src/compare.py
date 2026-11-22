"""Paired descriptive comparisons of CVAE with every training control."""

import pandas as pd


def interpretation(recall, false_alarms):
    if recall > 0:
        return 'Observed improvement' if false_alarms <= 0 else 'Trade-off: higher recall, more false alarms'
    if recall < 0:
        return 'Trade-off: lower recall, fewer false alarms' if false_alarms < 0 else 'Lower recall without fewer false alarms'
    if false_alarms < 0:
        return 'Equal recall, fewer false alarms'
    if false_alarms > 0:
        return 'Equal recall, more false alarms'
    return 'No change in recall or false-alarm rate'


def compare_methods(metrics, output):
    keys = ['detector', 'partition', 'fold', 'seed']
    measures = ['recall', 'false_alarm_rate', 'average_precision']
    if metrics.duplicated(keys + ['arm']).any():
        raise ValueError('Duplicate paired metric rows')
    cvae = metrics[metrics.arm == 'cvae'][keys + measures]
    pairs = metrics[metrics.arm != 'cvae'].merge(
        cvae, on=keys, how='left', validate='many_to_one', suffixes=('_control', '_cvae'))
    if pairs[[name + '_cvae' for name in measures]].isna().any().any():
        raise ValueError('A control has no matching CVAE evaluation')
    for name in measures:
        pairs[name + '_delta'] = pairs[name + '_cvae'] - pairs[name + '_control']
    columns = keys + ['arm'] + [name + '_delta' for name in measures]
    pairs[columns].to_csv(output / 'paired_differences.csv', index=False)
    rows = []
    for (detector, partition, control), group in pairs.groupby(['detector', 'partition', 'arm']):
        recall = float(group.recall_delta.mean())
        far = float(group.false_alarm_rate_delta.mean())
        # Treat machine-precision differences as zero for interpretation only.
        recall = 0.0 if abs(recall) < 1e-12 else recall
        far = 0.0 if abs(far) < 1e-12 else far
        rows.append(dict(
            detector=detector, partition=partition, control=control,
            paired_fits=len(group), mean_recall_delta=recall,
            mean_false_alarm_rate_delta=far,
            mean_average_precision_delta=float(group.average_precision_delta.mean()),
            pairs_with_higher_recall=int((group.recall_delta > 1e-12).sum()),
            pairs_with_no_far_increase=int((group.false_alarm_rate_delta <= 1e-12).sum()),
            interpretation=interpretation(recall, far)))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'comparisons.csv', index=False)
    return summary
