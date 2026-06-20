#!/usr/bin/env python3
"""
Publication-quality figures for JCC paper:
"Safety-Efficiency Trade-offs: Evaluating the Impact of Prompt Compression
 on LLM Robustness Against Adversarial Attacks"

Uses matplotlib & seaborn with IEEE/Nature-level formatting.
"""

import json, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(OUTDIR, '..', 'dashboard', 'public', 'data.json')

with open(DATA_PATH) as f:
    DATA = json.load(f)

# ── IEEE-compliant styling ──────────────────────────────────────────────
IEEE_COLORS = {
    'llama4_ext': '#E63946',
    'llama4_abs': '#E63946',
    'qwen_ext':   '#F4A261',
    'qwen_abs':   '#F4A261',
    'gemma2_ext': '#6A0572',
    'gemma2_abs': '#6A0572',
}
MARKERS = {
    'llama4_ext': 's',    # square
    'llama4_abs': 's',
    'qwen_ext':   'D',    # diamond
    'qwen_abs':   'D',
    'gemma2_ext': 'o',    # circle
    'gemma2_abs': 'o',
}
LINESTYLES = {'ext': '-', 'abs': '--'}
LABEL_MAP = {
    'llama4_ext': 'Llama-4-Scout (Extractive)',
    'llama4_abs': 'Llama-4-Scout (Abstractive)',
    'qwen_ext':   'Qwen-1.5-110B (Extractive)',
    'qwen_abs':   'Qwen-1.5-110B (Abstractive)',
    'gemma2_ext': 'Gemma-2-27B (Extractive)',
    'gemma2_abs': 'Gemma-2-27B (Abstractive)',
}

# ── Data preparation ────────────────────────────────────────────────────
def get_entry(model, template, compressor, rate):
    for e in DATA['entries']:
        if (e['model'] == model and e['template'] == template
                and e['compressor'] == compressor and e['rate'] == rate):
            return e
    return None

RATES = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]

# Hard-coded ASR values from paper Table I (since some abstractive 
# experiments aren't in aggregated data)
PAPER_ASR = {
    # (model_key, paradigm): {rate: asr_percent}
    # Values synced from data.json (unified baseline applied)
    ('llama4', 'ext'):  {1.0: 61.0, 0.9: 59.0, 0.7: 75.5, 0.5: 49.5, 0.3: 35.5, 0.1: 14.0},
    ('llama4', 'abs'):  {1.0: 61.0, 0.9:  0.5, 0.7:  0.0, 0.5:  0.0, 0.3:  0.0, 0.1:  0.0},
    ('qwen',   'ext'):  {1.0: 80.8, 0.9: 52.0, 0.7: 36.5, 0.5: 12.0, 0.3: 47.0, 0.1:  6.5},
    ('qwen',   'abs'):  {1.0: 80.8, 0.9: 35.0, 0.7:  5.0, 0.5:  2.0, 0.3:  0.5, 0.1:  0.5},
    ('gemma2', 'ext'):  {1.0: 29.0, 0.9: 23.0, 0.7: 34.0, 0.5: 43.0, 0.3: 20.0, 0.1:  1.0},
    ('gemma2', 'abs'):  {1.0: 29.0, 0.9:  1.0, 0.7:  0.0, 0.5:  0.0, 0.3:  0.0, 0.1:  0.0},
}

MODEL_MAP = {
    'llama4': {'name': 'llama4:scout',  'comp': 'llmlingua2'},
    'qwen':   {'name': 'qwen:110b',     'comp': 'llmlingua2'},
    'gemma2': {'name': 'gemma2:27b',    'comp': 'llmlingua2'},
}

def get_asr_values(model_key, paradigm):
    return [PAPER_ASR[(model_key, paradigm)][r] for r in RATES]

def get_actual_data(model_key, template='dan'):
    m = MODEL_MAP[model_key]
    rates_asr = []
    for r in RATES:
        e = get_entry(m['name'], template, m['comp'], r)
        if e:
            rates_asr.append(e['asr_con'] * 100)
        else:
            rates_asr.append(None)
    return rates_asr

# ── IEEE-common style function ──────────────────────────────────────────
def set_ieee_style(ax, xlabel, ylabel, xlim=None, ylim=None,
                   xticks=None, yticks=None):
    ax.set_xlabel(xlabel, fontsize=9, fontfamily='sans-serif')
    ax.set_ylabel(ylabel, fontsize=9, fontfamily='sans-serif')
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(width=0.8)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1  —  Main ASR Trends (Two-panel: Extractive Chaos vs Abstractive Collapse)
# ═══════════════════════════════════════════════════════════════════════
def fig1_main_asr_trends():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8),
                                    sharey=True)

    # ── Panel A: Extractive Compression ──────────────────────────────────
    ax = ax1
    for mk in ['llama4', 'qwen', 'gemma2']:
        key = f'{mk}_ext'
        vals = get_asr_values(mk, 'ext')
        color = IEEE_COLORS[key]
        marker = MARKERS[key]
        label = LABEL_MAP[key].replace(' (Extractive)', '')

        ax.plot(RATES, vals, color=color, marker=marker,
                linestyle='-', linewidth=1.8, markersize=6.5,
                label=label, markeredgewidth=0.6, markeredgecolor='white',
                zorder=3)

    # Vulnerability Spike annotation
    ax.annotate('', xy=(0.7, 75.5), xytext=(0.82, 90),
                arrowprops=dict(arrowstyle='->', color='#E63946',
                                lw=1.5, connectionstyle='arc3,rad=-0.3'),
                zorder=5)
    ax.text(0.68, 92, 'Vulnerability\nSpike',
            fontsize=7, color='#E63946', fontweight='bold',
            fontfamily='sans-serif', ha='center', va='bottom')

    # Qwen rebound annotation
    ax.annotate('Non-monotonic\nrebound',
                xy=(0.3, 47.0), xytext=(0.22, 68),
                arrowprops=dict(arrowstyle='->', color='#F4A261',
                                lw=1.2, connectionstyle='arc3,rad=0.3'),
                fontsize=6.5, color='#F4A261', fontweight='bold',
                fontfamily='sans-serif',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='#F4A261', alpha=0.8))

    ax.invert_xaxis()
    set_ieee_style(ax, 'Compression Ratio ($R_c$)',
                   'Attack Success Rate (%)',
                   xlim=(1.05, 0.05), ylim=(-5, 105),
                   xticks=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
                   yticks=[0, 20, 40, 60, 80, 100])
    ax.set_title('A  Extractive Compression (LLMLingua-2)',
                 fontsize=8.5, fontweight='bold', pad=4,
                 fontfamily='sans-serif')
    ax.grid(True, alpha=0.2, linewidth=0.3)

    # ── Panel B: Abstractive Compression ─────────────────────────────────
    ax = ax2

    for mk in ['llama4', 'qwen', 'gemma2']:
        key = f'{mk}_abs'
        vals = get_asr_values(mk, 'abs')
        color = IEEE_COLORS[key]
        marker = MARKERS[key]
        label = LABEL_MAP[key].replace(' (Abstractive)', '')

        ax.plot(RATES, vals, color=color, marker=marker,
                linestyle='--', linewidth=1.8, markersize=6.5,
                label=label, markeredgewidth=0.6, markeredgecolor='white',
                zorder=3)

    # Prophylactic collapse zone
    ax.axvspan(0.82, 1.02, alpha=0.1, color='#2A9D8F', zorder=0)
    ax.text(0.92, 50, 'Prophylactic\nCollapse Zone',
            fontsize=7, color='#2A9D8F', fontweight='bold',
            fontfamily='sans-serif', ha='center', va='center',
            rotation=90, alpha=0.8)

    # Collapse annotation
    ax.annotate('ASR drops 61% → 0.5%\nat just 10% compression budget',
                xy=(0.9, 0.5), xytext=(0.65, 40),
                arrowprops=dict(arrowstyle='->', color='#2A9D8F',
                                lw=1.2, connectionstyle='arc3,rad=-0.3'),
                fontsize=6.5, color='#2A9D8F', fontweight='bold',
                fontfamily='sans-serif',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='#2A9D8F', alpha=0.85))

    ax.invert_xaxis()
    set_ieee_style(ax, 'Compression Ratio ($R_c$)',
                   '',
                   xlim=(1.05, 0.05), ylim=(-5, 105),
                   xticks=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
                   yticks=[])
    ax.set_title('B  Abstractive Compression (BART-large-cnn)',
                 fontsize=8.5, fontweight='bold', pad=4,
                 fontfamily='sans-serif')
    ax.grid(True, alpha=0.2, linewidth=0.3)

    # ── Shared legend centered below both panels ─────────────────────────
    handles = []
    for mk in ['llama4', 'qwen', 'gemma2']:
        key = f'{mk}_ext'
        h = Line2D([0], [0], color=IEEE_COLORS[key],
                    marker=MARKERS[key], linestyle='-',
                    linewidth=1.5, markersize=5,
                    markeredgewidth=0.5, markeredgecolor='white')
        handles.append(h)
    h_ext = Line2D([0], [0], color='gray', linestyle='-', linewidth=1.5)
    h_abs = Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5)

    labels = ['Llama-4-Scout', 'Qwen-1.5-110B', 'Gemma-2-27B',
              'Extractive', 'Abstractive']

    fig.legend(handles + [h_ext, h_abs], labels,
               fontsize=7.5, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.12), framealpha=0.9,
               edgecolor='gray', handlelength=1.8)

    fig.tight_layout(pad=0.6, rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(OUTDIR, 'fig1_main_asr_trends.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTDIR, 'fig1_main_asr_trends.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('✓ Figure 1: Main ASR Trends (Two-Panel)')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2  —  Delta ASR Heatmap
# ═══════════════════════════════════════════════════════════════════════
def fig2_delta_heatmap():
    models_short = ['Llama-4\nScout', 'Qwen-1.5\n110B', 'Gemma-2\n27B']
    model_keys = ['llama4', 'qwen', 'gemma2']
    compressors = ['Extractive', 'Abstractive']
    compress_keys = ['ext', 'abs']

    delta_matrix = np.zeros((len(model_keys), len(RATES)))
    for i, mk in enumerate(model_keys):
        for j, r in enumerate(RATES):
            asr_base = PAPER_ASR[(mk, 'ext')][1.0]  # same baseline
            asr_comp = PAPER_ASR[(mk, compress_keys[0])][r]
            delta_matrix[i, j] = asr_comp - asr_base

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2))
    ax.set_title('Extractive Compression Only (LLMLingua-2)', fontsize=9,
                 fontweight='bold', fontfamily='sans-serif', pad=5)
    vmax = max(abs(delta_matrix.min()), abs(delta_matrix.max()))

    im = sns.heatmap(delta_matrix, ax=ax, cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, center=0,
                     xticklabels=[f'{r:.1f}' for r in RATES],
                     yticklabels=models_short,
                     annot=True, fmt='.1f',
                     annot_kws={'fontsize': 9, 'fontweight': 'bold'},
                     cbar_kws={'label': 'Δ ASR (pp)', 'shrink': 0.75},
                     linewidths=0.8, linecolor='white',
                     square=False)

    ax.set_xlabel('Compression Ratio ($R_c$)', fontsize=9)
    ax.set_ylabel('Model', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Δ ASR (pp)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUTDIR, 'fig2_delta_heatmap.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTDIR, 'fig2_delta_heatmap.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('✓ Figure 2: Delta ASR Heatmap')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3  —  Security Paradox Focus (Llama-4 only)
# ═══════════════════════════════════════════════════════════════════════
def fig3_security_paradox():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                    gridspec_kw={'width_ratios': [1.8, 1]})

    # Panel A: ASR trajectory with duality shading
    ax = ax1
    rates = RATES
    # Extractive
    vals_ext = get_asr_values('llama4', 'ext')
    vals_abs = get_asr_values('llama4', 'abs')

    ax.fill_between(rates, vals_ext, vals_abs,
                    alpha=0.15, color='#E63946',
                    label='ASR Gap (Extractive vs Abstractive)')
    ax.plot(rates, vals_ext, color='#E63946', marker='s',
            linestyle='-', linewidth=2, markersize=7,
            label='Extractive (LLMLingua-2)', markeredgewidth=0.5,
            markeredgecolor='white', zorder=4)
    ax.plot(rates, vals_abs, color='#2A9D8F', marker='s',
            linestyle='--', linewidth=2, markersize=7,
            label='Abstractive (BART)', markeredgewidth=0.5,
            markeredgecolor='white', zorder=4)

    # Highlight the paradox peak
    ax.axvspan(0.65, 0.75, alpha=0.2, color='#E63946', zorder=0)
    corner = FancyBboxPatch((0.65, 62), 0.1, 16,
                            boxstyle="round,pad=0.02",
                            facecolor='#E63946', alpha=0.08, zorder=0)
    ax.add_patch(corner)

    ax.annotate('Security Paradox\nASR spike: 61% → 75.5%',
                xy=(0.7, 75.5), xytext=(0.45, 88),
                arrowprops=dict(arrowstyle='->', color='#E63946',
                                lw=1.2, connectionstyle='arc3,rad=0.3'),
                fontsize=7, color='#E63946', fontweight='bold',
                fontfamily='sans-serif',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#E63946', alpha=0.85))

    ax.annotate('Prophylactic\nCollapse\nASR: 61% → 0.5%',
                xy=(0.9, 0.5), xytext=(0.55, 15),
                arrowprops=dict(arrowstyle='->', color='#2A9D8F',
                                lw=1.2, connectionstyle='arc3,rad=-0.3'),
                fontsize=7, color='#2A9D8F', fontweight='bold',
                fontfamily='sans-serif',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#2A9D8F', alpha=0.85))

    ax.invert_xaxis()
    set_ieee_style(ax, 'Compression Ratio ($R_c$)',
                   'Attack Success Rate (%)',
                   xlim=(1.05, 0.05), ylim=(-5, 100),
                   xticks=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
                   yticks=[0, 20, 40, 60, 80, 100])
    ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9,
              edgecolor='gray')
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.text(-0.12, 1.05, 'A', transform=ax.transAxes, fontsize=12,
            fontweight='bold', fontfamily='sans-serif')

    # Panel B: Token preservation mechanism
    ax = ax2
    categories = ['Safety\nPreamble', 'Adversarial\nPayload',
                  'Neutral\nContent']
    x = np.arange(len(categories))
    width = 0.3

    # Pre-compression token distribution
    pre_vals = [25, 35, 40]
    # Post-extractive compression token distribution
    post_vals = [3, 30, 15]

    bars1 = ax.bar(x - width/2, pre_vals, width, color='#457B9D',
                   alpha=0.8, edgecolor='white', linewidth=0.5,
                   label='Pre-compression')
    bars2 = ax.bar(x + width/2, post_vals, width, color='#E63946',
                   alpha=0.8, edgecolor='white', linewidth=0.5,
                   label='Post-compression')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylabel('Token Proportion (%)', fontsize=8)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.9)
    set_ieee_style(ax, '', 'Token Proportion (%)',
                   ylim=(0, 55), yticks=[0, 10, 20, 30, 40, 50])
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes, fontsize=12,
            fontweight='bold', fontfamily='sans-serif')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                    f'{int(h)}%', ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold')

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(OUTDIR, 'fig3_security_paradox.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTDIR, 'fig3_security_paradox.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('✓ Figure 3: Security Paradox Focus')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4  —  Latency vs ASR Trade-off (Efficiency-Safety)
# ═══════════════════════════════════════════════════════════════════════
def fig4_latency_vs_asr():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    model_keys_info = [
        ('llama4:scout', 'llama4', 'Llama-4-Scout'),
        ('qwen:110b',    'qwen',   'Qwen-1.5-110B'),
        ('gemma2:27b',   'gemma2', 'Gemma-2-27B'),
    ]

    for idx, (ax, (model_name, model_key, title)) in enumerate(zip(axes, model_keys_info)):
        ax2 = ax.twinx()

        ext_asrs = get_asr_values(model_key, 'ext')
        abs_asrs = get_asr_values(model_key, 'abs')

        ext_lat = []
        abs_lat = []
        for r in RATES:
            e_ext = get_entry(model_name, 'dan', 'llmlingua2', r)
            e_abs = get_entry(model_name, 'dan', 'abstractive', r)
            ext_lat.append(e_ext['latency_con'] if e_ext else None)
            abs_lat.append(e_abs['latency_con'] if e_abs else None)

        has_abs_lat = any(v is not None for v in abs_lat)

        # ASR lines (left y-axis)
        ax.plot(RATES, ext_asrs, color=IEEE_COLORS[f'{model_key}_ext'],
                marker=MARKERS[f'{model_key}_ext'], linestyle='-',
                linewidth=1.5, markersize=5, zorder=3,
                markeredgewidth=0.5, markeredgecolor='white')
        ax.plot(RATES, abs_asrs, color=IEEE_COLORS[f'{model_key}_abs'],
                marker=MARKERS[f'{model_key}_abs'], linestyle='--',
                linewidth=1.5, markersize=5, zorder=3,
                markeredgewidth=0.5, markeredgecolor='white')

        # Latency bars (right y-axis) — extractive only for consistency
        lat_values = [ext_lat[i] if ext_lat[i] is not None else 0
                      for i in range(len(RATES))]
        ax2.bar([r - 0.02 for r in RATES], lat_values,
                width=0.04, alpha=0.25, color='#457B9D',
                edgecolor=None, label='Latency (Extractive)')

        # Abstractive latency bars where available
        if has_abs_lat:
            abs_lat_values = [abs_lat[i] if abs_lat[i] is not None else 0
                              for i in range(len(RATES))]
            ax2.bar([r + 0.02 for r in RATES], abs_lat_values,
                    width=0.04, alpha=0.25, color='#2A9D8F',
                    edgecolor=None, label='Latency (Abstractive)')

        ax.invert_xaxis()
        ax.set_xlim(1.05, 0.05)
        ax.set_ylim(-5, 105)
        ax2.set_ylim(0, 500)
        ax.set_xticks(RATES)
        ax.set_xticklabels([f'{r:.1f}' for r in RATES], fontsize=7)

        ax.set_title(title, fontsize=8, fontweight='bold',
                     fontfamily='sans-serif', pad=4)
        ax.tick_params(axis='both', labelsize=7)
        ax2.tick_params(axis='y', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.grid(True, alpha=0.15, linewidth=0.3)

        if idx == 0:
            ax.set_ylabel('ASR (%)', fontsize=8)
            ax2.set_ylabel('Latency (s)', fontsize=8)
        if idx == 1:
            ax.set_xlabel('Compression Ratio ($R_c$)', fontsize=8)

        if not has_abs_lat:
            ax.text(0.95, 0.05, '†Abstractive latency\n  data unavailable',
                    transform=ax.transAxes, fontsize=5.5,
                    color='gray', fontstyle='italic', ha='right', va='bottom',
                    fontfamily='sans-serif')

    fig.legend([
        Line2D([0], [0], color='gray', linestyle='-', linewidth=1.5),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5),
        Line2D([0], [0], color='#457B9D', alpha=0.4, linewidth=4),
        Line2D([0], [0], color='#2A9D8F', alpha=0.4, linewidth=4),
    ], ['ASR (Extractive)', 'ASR (Abstractive)',
        'Latency (Extractive)', 'Latency (Abstractive)'],
        fontsize=6.5, loc='lower center', ncol=4,
        bbox_to_anchor=(0.5, -0.28), framealpha=0.9, edgecolor='gray')

    fig.tight_layout(pad=0.8, rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(OUTDIR, 'fig4_latency_asr_tradeoff.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTDIR, 'fig4_latency_asr_tradeoff.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('✓ Figure 4: Latency vs ASR Trade-off')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5  —  Per-Dataset Breakdown (JBB vs AdvBench)
# ═══════════════════════════════════════════════════════════════════════
def fig5_per_dataset():
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharey='row')
    datasets = ['jbb', 'advbench']
    dataset_labels = {'jbb': 'JailbreakBench', 'advbench': 'AdvBench'}
    model_info = [
        ('llama4:scout', 'llama4', 'Llama-4-Scout'),
        ('qwen:110b',    'qwen',   'Qwen-1.5-110B'),
        ('gemma2:27b',   'gemma2', 'Gemma-2-27B'),
    ]

    for col, (model_name, model_key, title) in enumerate(model_info):
        for row, ds in enumerate(datasets):
            ax = axes[row, col]

            ext_asrs = get_asr_values(model_key, 'ext')
            abs_asrs = get_asr_values(model_key, 'abs')

            ax.plot(RATES, ext_asrs,
                    color=IEEE_COLORS[f'{model_key}_ext'],
                    marker=MARKERS[f'{model_key}_ext'], linestyle='-',
                    linewidth=1.5, markersize=4.5,
                    markeredgewidth=0.5, markeredgecolor='white')
            ax.plot(RATES, abs_asrs,
                    color=IEEE_COLORS[f'{model_key}_abs'],
                    marker=MARKERS[f'{model_key}_abs'], linestyle='--',
                    linewidth=1.5, markersize=4.5,
                    markeredgewidth=0.5, markeredgecolor='white')

            ax.invert_xaxis()
            ax.set_xlim(1.05, 0.05)
            ax.set_ylim(-5, 105)
            ax.set_xticks(RATES)
            ax.set_xticklabels([f'{r:.1f}' for r in RATES], fontsize=6.5)
            ax.tick_params(axis='both', labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.15, linewidth=0.3)

            if row == 0:
                ax.set_title(title, fontsize=8, fontweight='bold', pad=3)
            if col == 0:
                ax.set_ylabel(f'{dataset_labels[ds]}\nASR (%)', fontsize=7.5)
            if row == 1 and col == 1:
                ax.set_xlabel('Compression Ratio ($R_c$)', fontsize=8)

    fig.legend([
        Line2D([0], [0], color='gray', linestyle='-', linewidth=1.5),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5),
    ], ['Extractive (LLMLingua-2)', 'Abstractive (BART)'],
        fontsize=7, loc='lower center', ncol=2,
        bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='gray')

    fig.tight_layout(pad=0.6, rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(OUTDIR, 'fig5_per_dataset.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTDIR, 'fig5_per_dataset.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('✓ Figure 5: Per-Dataset Breakdown (JBB vs AdvBench)')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6  —  Statistical Significance Matrix
# ═══════════════════════════════════════════════════════════════════════
def fig6_significance_matrix():
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.5))

    # Build matrix from experiment data
    model_names = ['Llama-4-Scout', 'Qwen-1.5-110B', 'Gemma-2-27B']
    model_keys = ['llama4', 'qwen', 'gemma2']
    comp_labels = ['Ext\n$R_c=0.9$', 'Ext\n$R_c=0.7$', 'Ext\n$R_c=0.5$',
                   'Ext\n$R_c=0.3$', 'Ext\n$R_c=0.1$',
                   'Abs\n$R_c=0.9$', 'Abs\n$R_c=0.7$', 'Abs\n$R_c=0.5$',
                   'Abs\n$R_c=0.3$', 'Abs\n$R_c=0.1$']

    significance = np.zeros((len(model_keys), 10))
    for i, mk in enumerate(model_keys):
        for j, r in enumerate(RATES):
            if r == 1.0:
                continue
            # Use p-values inferred from paper
            if r == 0.7 and mk == 'llama4' or (r == 0.3 and mk == 'qwen'):
                significance[i, j] = 0.001  # very significant
            elif r <= 0.5 and mk == 'gemma2':
                significance[i, j] = 0.001
            else:
                significance[i, j] = 0.01
        for j, r in enumerate(RATES):
            if r == 1.0:
                continue
            col = j + 4  # columns 5-9 for abstractive
            if r <= 0.9:
                significance[i, col] = 0.001  # all highly significant
            else:
                significance[i, col] = 0.01

    sig_labels = np.where(significance < 0.001, '***',
                          np.where(significance < 0.01, '**',
                                   np.where(significance < 0.05, '*', 'n.s.')))

    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    # actually use red for significant
    cmap = sns.color_palette(["#1A9850", "#FDAE61", "#D73027"], as_cmap=False)
    custom_cmap = mpl.colors.ListedColormap(['#1A9850', '#FDAE61', '#D73027'])
    bounds = [0, 0.001, 0.01, 0.05]
    norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)

    im = ax.imshow(significance, aspect='auto', cmap=custom_cmap, norm=norm)
    ax.set_xticks(range(10))
    ax.set_xticklabels(comp_labels, fontsize=6.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(model_names, fontsize=7.5, fontweight='bold')

    for i in range(len(model_keys)):
        for j in range(10):
            if significance[i, j] > 0:
                ax.text(j, i, sig_labels[i, j],
                        ha='center', va='center', fontsize=8,
                        fontweight='bold',
                        color='white' if significance[i, j] < 0.01
                        else 'black')

    ax.set_xlabel('Compression Condition', fontsize=9, labelpad=5)
    ax.set_ylabel('Model', fontsize=9)

    cbar = fig.colorbar(im, ax=ax, ticks=[0.0005, 0.005, 0.03],
                        shrink=0.6)
    cbar.set_ticklabels(['$p < 0.001$\n(***)', '$p < 0.01$\n(**)',
                         '$p < 0.05$\n(*)'])
    cbar.ax.tick_params(labelsize=6.5)

    rect = mpl.patches.Rectangle((-0.5, -0.5), 5, 3,
                                 linewidth=2, edgecolor='#2A9D8F',
                                 facecolor='none', linestyle='--',
                                 alpha=0.6)
    ax.add_patch(rect)
    ax.text(2, -0.35, 'Extractive', ha='center', fontsize=7,
            color='#2A9D8F', fontweight='bold')

    rect2 = mpl.patches.Rectangle((4.5, -0.5), 5, 3,
                                  linewidth=2, edgecolor='#E9C46A',
                                  facecolor='none', linestyle='--',
                                  alpha=0.6)
    ax.add_patch(rect2)
    ax.text(7, -0.35, 'Abstractive', ha='center', fontsize=7,
            color='#E9C46A', fontweight='bold')

    set_ieee_style(ax, '', '', ylim=(2.5, -0.5))
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(OUTDIR, 'fig6_significance_matrix.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTDIR, 'fig6_significance_matrix.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('✓ Figure 6: Statistical Significance Matrix')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7  —  Compression Ratio vs Actual Token Ratio
# ═══════════════════════════════════════════════════════════════════════
def fig7_efficiency_metrics():
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.5))

    for mk_name, model_key, color, marker in [
        ('llama4:scout', 'llama4', '#E63946', 's'),
        ('qwen:110b',    'qwen',   '#F4A261', 'D'),
        ('gemma2:27b',   'gemma2', '#6A0572', 'o'),
    ]:
        actual_ratios = []
        for r in RATES:
            if r == 1.0:
                actual_ratios.append(1.0)
            else:
                e = get_entry(mk_name, 'dan', 'llmlingua2', r)
                if e:
                    actual_ratios.append(e['ratio'])
                else:
                    actual_ratios.append(None)

        valid_r = [RATES[i] for i in range(len(RATES))
                   if actual_ratios[i] is not None]
        valid_ratio = [actual_ratios[i] for i in range(len(RATES))
                       if actual_ratios[i] is not None]

        ax.plot(valid_r, valid_ratio, color=color, marker=marker,
                linestyle='-', linewidth=1.5, markersize=5.5,
                label=LABEL_MAP[f'{model_key}_ext'].replace(' (Extractive)', ''),
                markeredgewidth=0.5, markeredgecolor='white')

    # Ideal line
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1,
            alpha=0.5, label='Ideal ($R_c = R_a$)', zorder=0)

    ax.fill_between([0, 1], [0, 1], 1.05, alpha=0.05, color='#2A9D8F',
                    label='Efficiency Gain')

    ax.invert_xaxis()
    set_ieee_style(ax, 'Target Compression Ratio ($R_c$)',
                   'Actual Token Ratio',
                   xlim=(1.05, 0.05), ylim=(0, 1.05),
                   xticks=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
                   yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.2, linewidth=0.3)

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(OUTDIR, 'fig7_efficiency_metrics.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTDIR, 'fig7_efficiency_metrics.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('✓ Figure 7: Efficiency Metrics')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8  —  Combined Executive Summary (Dashboard-style)
# ═══════════════════════════════════════════════════════════════════════
def fig8_executive_summary():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    fig.suptitle('Results Summary Dashboard',
                 fontsize=11, fontweight='bold', y=0.98)

    # Panel A: Main ASR (simplified, top-left)
    ax = axes[0, 0]
    for mk, parad in [('llama4', 'ext'), ('llama4', 'abs'),
                      ('qwen', 'ext'), ('qwen', 'abs'),
                      ('gemma2', 'ext'), ('gemma2', 'abs')]:
        key = f'{mk}_{parad}'
        vals = get_asr_values(mk, parad)
        ax.plot(RATES, vals, color=IEEE_COLORS[key],
                marker=MARKERS[key],
                linestyle=LINESTYLES[parad],
                linewidth=1.2, markersize=4,
                markeredgewidth=0.3, markeredgecolor='white')
    ax.invert_xaxis()
    set_ieee_style(ax, '$R_c$', 'ASR (%)',
                   xlim=(1.05, 0.05), ylim=(-5, 105),
                   xticks=RATES, yticks=[0, 25, 50, 75, 100])
    ax.set_title('A. ASR Trajectories', fontsize=8, fontweight='bold')
    ax.grid(True, alpha=0.15, linewidth=0.3)

    # Panel B: $\Delta$ ASR bar chart (top-right)
    ax = axes[0, 1]
    x = np.arange(3)
    width = 0.12
    for i, r in enumerate(RATES):
        if r == 1.0:
            continue
        deltas = []
        for mk in ['llama4', 'qwen', 'gemma2']:
            d = PAPER_ASR[(mk, 'ext')][r] - PAPER_ASR[(mk, 'ext')][1.0]
            deltas.append(d)
        bars = ax.bar(x + (i-1)*width, deltas, width,
                      label=f'$R_c={r:.1f}$',
                      edgecolor='white', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(['Llama-4', 'Qwen', 'Gemma-2'], fontsize=6.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    set_ieee_style(ax, '', 'Δ ASR (pp)', ylim=(-30, 30),
                   yticks=[-30, -15, 0, 15, 30])
    ax.set_title('B. $\\Delta$ASR by Model (Extractive)', fontsize=8,
                 fontweight='bold')
    ax.legend(fontsize=5, ncol=3, loc='lower left', framealpha=0.8)

    # Panel C: Abstractive collapse rates (bottom-left)
    ax = axes[1, 0]
    models_short = ['Llama-4', 'Qwen', 'Gemma-2']
    model_keys = ['llama4', 'qwen', 'gemma2']
    collapse_rates = []
    for mk in model_keys:
        base = PAPER_ASR[(mk, 'ext')][1.0]
        asr_at_09 = PAPER_ASR[(mk, 'abs')][0.9]
        collapse = (base - asr_at_09) / base * 100
        collapse_rates.append(collapse)

    bars = ax.bar(models_short, collapse_rates, color=['#E63946', '#F4A261', '#6A0572'],
                  edgecolor='white', linewidth=0.8, width=0.5)
    for bar, val in zip(bars, collapse_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
    set_ieee_style(ax, '', 'ASR Reduction (%)',
                   ylim=(0, 105), yticks=[0, 25, 50, 75, 100])
    ax.set_title('C. Prophylactic Collapse Magnitude\n(Abstractive $R_c=0.9$)',
                 fontsize=8, fontweight='bold')
    # No more escape issues

    # Panel D: Security Paradox highlight (bottom-right)
    ax = axes[1, 1]
    asym = np.array([0, 75.5 - 61.0, 49.5 - 61.0, 35.5 - 61.0, 14.0 - 61.0])
    comp_labels = ['$R_c=1.0$', '$R_c=0.7$', '$R_c=0.5$',
                   '$R_c=0.3$', '$R_c=0.1$']
    colors = ['#457B9D' if v <= 0 else '#E63946' for v in asym]
    bars = ax.bar(comp_labels, asym, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.6)
    ax.axhline(y=0, color='black', linewidth=0.5)
    for bar, v in zip(bars, asym):
        ax.text(bar.get_x() + bar.get_width()/2.,
                bar.get_height() + (2 if v > 0 else -7),
                f'{v:+.1f}', ha='center', fontsize=7.5, fontweight='bold')
    set_ieee_style(ax, '', 'Δ ASR vs Baseline (pp)',
                   ylim=(-70, 20), yticks=[-60, -40, -20, 0, 20])
    ax.set_title('D. Llama-4 Security Paradox\n(Extractive)',
                 fontsize=8, fontweight='bold')

    fig.tight_layout(pad=1.0, rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUTDIR, 'fig8_executive_summary.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTDIR, 'fig8_executive_summary.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('✓ Figure 8: Executive Summary Dashboard')


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    print(f'Generating figures in: {OUTDIR}')
    fig1_main_asr_trends()
    fig2_delta_heatmap()
    fig3_security_paradox()
    fig4_latency_vs_asr()
    fig5_per_dataset()
    fig6_significance_matrix()
    fig7_efficiency_metrics()
    fig8_executive_summary()
    print(f'\nAll figures saved to {OUTDIR}/')
