"""Apply one paired statistical analysis to both 100-seed experiments."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/'downstream100_protocol.json').read_text())
CIFAR = json.loads((HERE/'protocol.json').read_text())


def measure(x, indices):
    lower, upper = np.quantile(x[indices].mean(axis=1),[.025,.975])
    return {'mean':float(x.mean()),'sd':float(x.std(ddof=1)),
            'median':float(np.median(x)),'ci95':[float(lower),float(upper)]}


def summarize(raw, output):
    out = json.loads(raw.read_text())
    assert out['protocol']==CFG
    assert out['test_n']==10000
    n = CFG['seed_count']
    expected = list(range(CFG['seed_start'],CFG['seed_start']+n))
    bootstrap = np.random.default_rng(CFG['bootstrap_seed']).integers(0,n,(CFG['bootstrap_draws'],n))
    records, summary, differences = [], {}, {}
    for setup in ('legacy_calibrated','fixed10'):
        item = out['setups'][setup]
        runs = sorted(item['runs'],key=lambda v:v['seed'])
        assert [v['seed'] for v in runs]==expected
        assert all(v['net_fixes']==v['fixed']-v['damaged'] for v in runs)
        assert all(0<=v['auto_correct']<=out['test_n'] and 0<=v['none_correct']<=out['test_n'] for v in runs)
        assert all(v['audit_overlap']<=min(v['audit'],v['actions']) for v in runs)
        scale = 100/out['test_n']
        pairs = np.array([v['auto_correct']-v['none_correct'] for v in runs],dtype=np.int64)
        delta = pairs*scale
        differences[setup] = delta
        fit_labels = CIFAR['cifar_fit_n'] if setup=='fixed10' else 0
        reference_labels = CIFAR['cifar_ref_n'] if setup=='fixed10' else 0
        audits = np.array([v['audit'] for v in runs])
        if setup=='fixed10':
            assert item['n']+fit_labels+reference_labels==50000
            assert (audits==0).all()
        else:
            assert (audits==round(CFG['legacy_audit_fraction']*item['n'])).all()
        summary[setup] = dict(
            n_seeds=n,train_n=item['n'],test_n=out['test_n'],
            accuracy_none_pct=measure(np.array([v['none_correct'] for v in runs])*scale,bootstrap),
            accuracy_auto_pct=measure(np.array([v['auto_correct'] for v in runs])*scale,bootstrap),
            accuracy_delta_pp=measure(delta,bootstrap),
            positive=int((pairs>0).sum()),ties=int((pairs==0).sum()),negative=int((pairs<0).sum()),
            fixed=measure(np.array([v['fixed'] for v in runs]),bootstrap),
            damaged=measure(np.array([v['damaged'] for v in runs]),bootstrap),
            net_fixes=measure(np.array([v['net_fixes'] for v in runs]),bootstrap),
            net_fixes_pp=measure(np.array([v['net_fixes'] for v in runs])*100/item['n'],bootstrap),
            action_pct=measure(np.array([v['actions'] for v in runs])*100/item['n'],bootstrap),
            audit_pct=measure(audits*100/item['n'],bootstrap),
            clean_label_use={
                'setup_fit_labels':fit_labels,
                'setup_reference_labels':reference_labels,
                'audit_checks_per_deployment':measure(audits,bootstrap),
                'audit_action_overlap':measure(np.array([v['audit_overlap'] for v in runs]),bootstrap),
                'records_requiring_trusted_information_from_scratch':measure(
                    fit_labels+reference_labels+audits,bootstrap),
            })
        for v,d in zip(runs,delta):
            records.append(dict(setup=setup,train_n=item['n'],test_n=out['test_n'],**v,
                                setup_fit_labels=fit_labels,setup_reference_labels=reference_labels,
                                records_requiring_trusted_information_from_scratch=fit_labels+reference_labels+v['audit'],
                                accuracy_delta_pp=float(d)))
    summary['fixed_minus_legacy_effect_pp'] = measure(differences['fixed10']-differences['legacy_calibrated'],bootstrap)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'downstream100_per_seed.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (output/'downstream100_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for key in ('legacy_calibrated','fixed10'):
        z=summary[key]
        print(key,z['accuracy_delta_pp'],(z['positive'],z['ties'],z['negative']),z['net_fixes']['mean'],z['action_pct']['mean'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,default=HERE/'results')
    a=p.parse_args()
    summarize(a.raw,a.out)
