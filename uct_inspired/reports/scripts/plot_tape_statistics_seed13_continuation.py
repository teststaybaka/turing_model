"""Plot seed 13's two completed new-case stages from recorded pass aggregates."""
import hashlib
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

BASE = ROOT/'log'
OUT = ROOT/'reports'/'tape_statistics_seed13_new_cases'
PREFIX = 'uct_inspired_1024_cases_16_cells_train_seed20261013'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    combined, protocols, sources = [], [], []
    elapsed_offset = 0.
    for stage in (1,2):
        prefix = f'{PREFIX}_stage{stage+1:02d}_new'
        protocol = json.loads((BASE/f'{prefix}.protocol.json').read_text())
        summary = json.loads((BASE/f'{prefix}.summary.json').read_text())
        rows = [json.loads(line) for line in (BASE/f'{prefix}.passes.jsonl').read_text().splitlines()]
        assert len(rows)==512 and [r['pass_index'] for r in rows]==list(range(1,513))
        assert summary['completed_updates']==131072 and summary['source_checkpoint_unchanged']
        assert all(r['original_cases'] is None and r['newly_added_cases']==r['all_cases'] for r in rows)
        assert all(r['all_cases']['rollouts']==4096 and r['all_cases']['updates']==256 for r in rows)
        for r in rows:
            a=r['all_cases']
            combined.append(dict(pass_index=r['pass_index']+512*(stage-1),stage=stage,
                                 stage_pass=r['pass_index'],successes=a['successes'],rollouts=a['rollouts'],
                                 success_pct=100*a['successes']/a['rollouts'],
                                 cumulative_training_seconds=elapsed_offset+r['elapsed_seconds']))
        elapsed_offset+=rows[-1]['elapsed_seconds']
        protocols.append(protocol);sources.append(str((BASE/f'{prefix}.passes.jsonl').relative_to(ROOT)))
    assert protocols[0]['config']==protocols[1]['config']
    assert protocols[0]['training_cases']==protocols[1]['training_cases']
    source=BASE/f'{PREFIX}_stage02_new.checkpoint.pt'
    # Paths embedded in historical protocols remain unchanged after flattening;
    # the checkpoint content hash still establishes exact lineage.
    with source.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==protocols[1]['source_checkpoint_sha256']
    successes=np.array([r['successes'] for r in combined]);totals=np.array([r['rollouts'] for r in combined])
    rates=100*successes/totals
    bins=100*successes.reshape(-1,4).sum(1)/totals.reshape(-1,4).sum(1)
    first_success=next(i+1 for i,v in enumerate(successes) if v)
    first90=next((i+1 for i in range(len(rates)-7) if (rates[i:i+8]>=90).all()),None)
    first_end=float(100*successes[496:512].sum()/totals[496:512].sum())
    final_end=float(100*successes[-16:].sum()/totals[-16:].sum())
    data=dict(seed=20261013,sources=sources,first_success_pass=first_success,
              first_of_8_consecutive_passes_at_least_90pct=first90,
              first_stage_final16_success_pct=first_end,second_stage_final16_success_pct=final_end,
              total_new_case_rollouts=int(totals.sum()),series=combined,
              notes=['Weights, optimizer, statistics, and RNG carried across the boundary.',
                     '256 formerly held-out cases only; both weights and statistics adapt.',
                     'Solid line pools nonoverlapping 4-pass windows; faint line shows individual passes.',
                     'The horizontal axis excludes the pause between stages.'])
    fig,ax=plt.subplots(figsize=(12.8,6.8))
    ax.axvspan(512.5,1024,color='#eaf3fb',alpha=.75,zorder=0)
    ax.plot(np.arange(1,1025),rates,color='#1474b5',alpha=.25,lw=.9)
    ax.plot(np.arange(2.5,1025,4),bins,color='#1474b5',lw=2.6)
    ax.axvline(512.5,color='#546477',ls='--',lw=1.3)
    ax.text(256,.97,'First 512 passes',ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=11,color='#475569')
    ax.text(768,.97,'Continued checkpoint: +512 passes',ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=11,color='#475569')
    ax.annotate(f'First success: pass {first_success}',xy=(first_success,rates[first_success-1]),xytext=(145,27),
                arrowprops=dict(arrowstyle='->',color='#64748b'),fontsize=11,color='#475569')
    ax.text(.98,.11,f'Final 16 passes: {final_end:.2f}%\nPrevious stage ended at {first_end:.2f}%',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=12,
            bbox=dict(boxstyle='round,pad=.6',facecolor='white',edgecolor='#d1d9e0',alpha=.98))
    ax.set_xlim(0,1024);ax.set_ylim(0,108)
    ax.xaxis.set_major_locator(MultipleLocator(128));ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    ax.set_xlabel('Cumulative passes over the 256 new cases',fontsize=12,labelpad=10)
    ax.set_ylabel('Successful rollouts',fontsize=12,labelpad=10)
    ax.tick_params(labelsize=11);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Seed 20261013: escaping the new-case plateau',x=.085,ha='left',fontsize=20,y=.975)
    fig.text(.085,.915,'Tape + statistics + budget input Ã‚Â· 1,024 passes Ã‚Â· 4,194,304 rollouts Ã‚Â· weights and statistics both adapt',fontsize=11,color='#475569')
    fig.text(.085,.035,'Bold: pooled 4-pass success rate. Faint: each pass. Dashed boundary: checkpoint continuation with no reset. Training curves.',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.085,right=.98,bottom=.15,top=.855)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'seed13_success.{ext}',dpi=180,facecolor='white')
    (OUT/'seed13_success_data.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    plt.close(fig)
    print(json.dumps({k:v for k,v in data.items() if k!='series'}))


if __name__=='__main__':main()
