"""Plot the two completed tape+statistics continuations, using recorded pass totals."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache'/'matplotlib'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

OUT = ROOT/'reports'/'tape_statistics_new_cases_success_20261024'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (20261011, 20261012)
COLORS = ('#1575b5', '#dd711a')


def main():
    data = {'description': 'Changing-policy training on 256 formerly held-out cases; weights and statistics both adapt.',
            'passes': 512, 'rollouts_per_pass': 4096,
            'smoothing': 'Nonoverlapping 4-pass bins, pooled successes / pooled rollouts. Faint lines show each pass.',
            'seeds': {}}
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    for seed, color in zip(SEEDS, COLORS):
        folder = ROOT/'log'
        prefix = f'uct_inspired_1024_cases_16_cells_train_seed{seed}_stage02_new'
        source = folder/f'{prefix}.passes.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        summary = json.loads((folder/f'{prefix}.summary.json').read_text())
        assert summary['completed_updates'] == 131072
        assert len(rows) == 512 and [r['pass_index'] for r in rows] == list(range(1,513))
        assert all(r['original_cases'] is None and r['newly_added_cases'] == r['all_cases'] for r in rows)
        assert all(r['all_cases']['rollouts'] == 4096 for r in rows)
        successes = np.array([r['all_cases']['successes'] for r in rows])
        totals = np.array([r['all_cases']['rollouts'] for r in rows])
        rates = 100*successes/totals
        bins = 100*successes.reshape(-1,4).sum(axis=1)/totals.reshape(-1,4).sum(axis=1)
        final16 = float(100*successes[-16:].sum()/totals[-16:].sum())
        ax.plot(np.arange(1,513), rates, color=color, alpha=.22, lw=.9)
        ax.plot(np.arange(2.5,513,4), bins, color=color, lw=2.5,
                label=f'{seed}   |   final 16 passes: {final16:.2f}%')
        first_sustained90 = next((i+1 for i in range(len(rates)-7) if (rates[i:i+8]>=90).all()), None)
        first_nonzero = next((i+1 for i,v in enumerate(successes) if v), None)
        entry = dict(source=str(source.relative_to(ROOT)), final_16_pass_success_pct=final16,
                     full_stage_success_pct=float(100*successes.sum()/totals.sum()),
                     first_nonzero_success_pass=first_nonzero,
                     first_of_8_consecutive_passes_at_least_90pct=first_sustained90,
                     runtime_minutes=rows[-1]['elapsed_seconds']/60,
                     series=[dict(pass_index=r['pass_index'], successes=int(s),rollouts=int(n),
                                  success_pct=float(v),elapsed_seconds=r['elapsed_seconds'])
                             for r,s,n,v in zip(rows,successes,totals,rates)])
        data['seeds'][str(seed)] = entry
        print(json.dumps({k:v for k,v in entry.items() if k!='series'}), flush=True)
    ax.set_xlim(0,512);ax.set_ylim(0,102)
    ax.set_xlabel('Pass over the 256 new cases', fontsize=12, labelpad=10)
    ax.set_ylabel('Successful rollouts', fontsize=12, labelpad=10)
    ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    ax.xaxis.set_major_locator(MultipleLocator(64));ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(alpha=.19);ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.legend(loc='lower right', fontsize=11, framealpha=.95, borderpad=.8)
    fig.suptitle('Learning the new cases with tape + statistics input', fontsize=19,x=.085,ha='left',y=.975)
    fig.text(.085,.913,'Both seeds completed 512 passes Ã‚Â· 2,097,152 rollouts each Ã‚Â· weights and statistics keep updating',fontsize=11,color='#475569')
    fig.text(.085,.035,'Bold: pooled 4-pass success rate. Faint: individual passes. These are adaptation/training curves, not frozen-policy evaluation.',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.085,right=.98,bottom=.15,top=.855)
    for extension in ('png','pdf','svg'):
        fig.savefig(OUT/f'new_case_success.{extension}',dpi=180,facecolor='white')
    (OUT/'new_case_success_data.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    plt.close(fig)


if __name__=='__main__':
    main()
