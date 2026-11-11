"""Bind manuscript claims to the paired summaries and raw result tables."""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE=Path(__file__).resolve().parent


def render(results, paper):
    s=json.loads((results/'paired_summary.json').read_text())
    b=pd.read_csv(results/'transfer_bounds.csv')
    def ci(row, places=2):
        return f"{row['mean']:.{places}f} [{row['lo']:.{places}f}, {row['hi']:.{places}f}]"
    rank_counts=[s[f'rank:{d}/NC-PCA/NC-RP/{r}']['any_inversion']
                 for d in ('hwdb','c10','c100') for r in (.1,.2,.4)]
    assert sum(rank_counts)==90
    da=s['regret:c100/NC-Aux/noisy_label/0.4']['review']
    ch=s['regret:c10/Chen-ID-hard/aggre_label/0.1']['review']
    old=s['downstream:archived/10seed_3conv/calib']
    au=s['downstream:new/auto']
    assert au['n']==10
    swaps=[-s[f'source_swap:{d}/0.1/NC/CNN']['mean'] for d in ('hwdb','c10','c100')]
    inter=[s[f'interaction:{d}/0.1/CNN-PCA'] for d in ('hwdb','c10','c100')]
    out={
    'AbstractResults':f"PCA and random projection reverse order in {sum(rank_counts)}/90 paired dataset--rate--seed comparisons when their nearest-class generators are exchanged. On CIFAR-100N, choosing under an independently generated nearest-class protocol loses {da['mean']:.2f} corrections per 100 records at 10\\% review, whereas the two AQuA mechanisms select the human-target winner in this comparison.",
    'RankResults':r"Table~\ref{tab:matrix} shows the full primary cross. Exchanging NC-PCA and NC-RP reverses the PCA--RP AUC order in all ten splits at each of the three rates on all three datasets. CNN retains first place in those particular swaps, so each has exactly one of three pairs inverted and Kendall's $\tau_b=1/3$. CIFAR-100 also has a top-rank change: RP leads under NC-CNN and NC-Aux by mean AUC, while CNN leads under Uniform and both AQuA mechanisms. The top rank on CASIA stays with CNN at the primary rate.",
    'SourceResults':"At the primary rate, the CNN--PCA interactions are " + ", ".join(ci(i,3) for i in inter) +
        f" on CASIA, CIFAR-10, and CIFAR-100, respectively. Switching CNN's own nearest-class generator to Aux raises CNN AUC by {swaps[0]:.3f}, {swaps[1]:.3f}, and {swaps[2]:.3f}. On CIFAR-100 it rises from 0.266 to 0.414, still below RP's 0.484 on the same labels. The independent nearest-class control therefore removes part of the loss while preserving a practical transfer problem.",
    'PublicResults':"Switching between the two AQuA source encoders retains CNN as the mean-AUC winner in both mechanisms on all datasets at the primary rate. For CIFAR-100 the ID source change raises CNN AUC by 0.018, compared with 0.148 for nearest class. A larger shift occurs when the same Chen destinations are applied to high-confusion records instead of the random mask. CNN AUC changes from 0.935 to 0.744 on CASIA, from 0.782 to 0.496 on CIFAR-10, and from 0.527 to 0.368 on CIFAR-100. On CIFAR-100, CNN moves behind RP by AUC, while CNN remains the best fixed-budget reviewer in this cell.",
    'RegretResults':r"Figure~\ref{fig:regret} shows where rank changes matter. At the primary rate and budget, every CASIA source picks the target review winner in all ten splits, despite the stable PCA--RP reversals. On CIFAR-100, NC-CNN selects RP in all ten splits. Switching to NC-RP on the target loses " +
         ci(s['regret:c100/NC-CNN/NC-RP/0.1']['review']) + " fixes per 100 records. Selecting on NC-Aux instead loses " +
         ci(s['regret:c100/NC-Aux/NC-RP/0.1']['review']) +
         " under review-based selection, and 3.42 under AUC-based selection. Under NC-Aux, review selection picks CNN, RP, and PCA in four, four, and two splits. The losses vary with the action cap and error prevalence. The full artifact retains every direction and budget.",
    'HumanResults':r"Among the three prototype margins, both NC-CNN and NC-Aux choose RP on CIFAR-100N in all ten prevalence-matched splits. Each loses " +
         ci(da) + r" fixes per 100 records relative to CNN, around 158 fewer fixes per 10,000 records at the same budget. Both AQuA generator families and Uniform choose CNN in all ten splits and have zero realized regret there. On CIFAR-10N aggregate, Chen hard selects RP in four splits and loses " +
         ci(ch) + r" fixes per 100 records. Chen mask and the AUC-based choice under Chen hard have zero regret there. At 10\% review all tested sources have zero regret on the three individual CIFAR-10N variants and worst. Table~\ref{tab:human} gives target detection and cleaning metrics.",
    'MechanismResults':r"Equation~\eqref{eq:decomp} reconstructs all crossed margin AUCs within $10^{-10}$. On CIFAR-100, replacing NC-CNN with NC-Aux reduces the zero-score share among errors from 71.6\% to 34.1\%. On CIFAR-10 the Chen hard mask reduces CNN true-nearest accuracy among errors from 65.3\% to 18.9\% and raises it among non-errors from 65.8\% to 71.0\%. The record-selection intervention changes both conditional distributions.",
    'CoverageResults':r"With a 10\% CIFAR-100N audit, mean precision among reviewing splits is 0.717, but only 47.5\% of all splits reach 0.70. The mean total cost is 21.3\% of the probe. On CIFAR-10N aggregate with a 50\% audit, follow-up executes in 47.0\% of splits and reaches 0.70 in 1.0\% of all splits.",
    'PolicyResults':r"At the 10\% cap (Table~\ref{tab:policy}), fixed review corrects more labels than calibrated review across the six variants. For aggregate the paired difference, calibrated minus fixed, is " +
         ci(s['policy:c10/aggre_label']) +
         r" fixes per 100 records. Fixed review averages 0.443 precision on aggregate and never meets the 0.70 target. Calibrated review attains 0.766 precision when it executes and meets the target in 70\% of all splits. The gains compare different precision objectives under a common cap. Automatic relabeling yields 7.16 net fixes per 100 on worst, loses 2.83 on aggregate, and is near zero on CIFAR-100N. Figure~\ref{fig:budget} shows how the three policies change with the cap.",
    'DownstreamScope':f"Selection regret in this study counts foregone label corrections. It gives no estimate of test-accuracy regret. A secondary CIFAR-10N worst intervention in the supplement records a test-accuracy change of {ci(au)} percentage points with automatic relabeling of a fixed 10\\% prefix. The archived calibrated run records {ci(old)} points, with an interval spanning zero. The runs differ in training pool, detector, and action rule. They provide no matched estimate of how varying the budget changes downstream accuracy.",
    'BoundResults':f"Across five destination-only pairs, three datasets, and ten splits, the AUC bound certifies {int(b.auc_certified.sum())}/{len(b)} pairwise orders and the prefix bound certifies {int(b.hits_certified.sum())}/{len(b)}. These are finite-pool checks using both sets of claims. The certificates are sufficient and can abstain even when the order transfers."
    }
    downstream100 = results/'downstream100_summary.json'
    if downstream100.exists():
        d = json.loads(downstream100.read_text())
        legacy, fixed = d['legacy_calibrated'], d['fixed10']
        audited = legacy['clean_label_use']['audit_checks_per_deployment']['mean']
        trusted = (fixed['clean_label_use']['setup_fit_labels']+
                   fixed['clean_label_use']['setup_reference_labels'])
        def pp(group):
            row=group['accuracy_delta_pp']
            lo,hi=row['ci95']
            return f"{row['mean']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        out['DownstreamScope'] = (
            "Selection regret counts missed corrections. The supplement separately reruns both CIFAR-10N worst "
            "downstream setups with 100 matched training seeds and the same paired analysis. Relative to unchanged "
            f"labels, calibrated automatic relabeling changes test accuracy by {pp(legacy)} percentage points, "
            f"and fixed 10\\% relabeling changes it by {pp(fixed)}. The intervals resample training seeds on a fixed "
            "test set. The calibrated setup uses "
            f"{audited:,.0f} audited records per deployment, while the fixed setup starts with {trusted:,} trusted training "
            "and reference labels. These are different supervision costs. The training pool, detector, "
            "and action rule also differ, so the contrast identifies neither a budget effect nor comparative cost efficiency."
        )
    dst=paper/'tables/findings.tex'
    dst.write_text('\n'.join('\\newcommand{\\'+key+'}{'+val+'}' for key,val in out.items())+'\n',encoding='ascii')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=HERE/'results')
    p.add_argument('--paper',type=Path,default=HERE/'paper')
    a=p.parse_args()
    render(a.results,a.paper)
