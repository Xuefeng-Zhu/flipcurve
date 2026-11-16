"""Render Chinese labels on the released report data without new model calls."""
from pathlib import Path
import tempfile

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fontTools.ttLib import TTCollection

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'report' / 'figures_zh'
OUT.mkdir(exist_ok=True)
font_path = Path(tempfile.gettempdir()) / 'flipcurve-HeitiSC.ttf'
collection = TTCollection('/System/Library/Fonts/STHeiti Light.ttc')
next(f for f in collection.fonts if f['name'].getDebugName(1) == 'Heiti SC').save(font_path)
font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({'font.family': 'Heiti SC', 'axes.unicode_minus': False})

COLORS = {'gemma-4-31b-it':'#D55E00', 'qwen3.8-27b':'#0072B2',
          'qwen3.5-397b':'#56B4E9', 'glm-5.3-flash':'#009E73',
          'glm-5.2':'#CC79A7', 'deepseek-v4-flash':'#E69F00'}
SHORT = {'gemma-4-31b-it':'Gemma-4-31B', 'qwen3.8-27b':'Qwen3.8-27B',
         'qwen3.5-397b':'Qwen3.5-397B', 'glm-5.3-flash':'GLM-5.3-Flash',
         'glm-5.2':'GLM-5.2', 'deepseek-v4-flash':'DeepSeek-V4-Flash*'}
data = pd.read_csv(ROOT / 'flip_curves_all.csv')
data = data[data.subset == 'all']
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
for model, group in data.groupby('model'):
    for ax, metric, count in zip(axes, ['W', 'C'], ['n_wrong', 'n_right']):
        ax.errorbar(group.k, group[metric],
                    yerr=[group[metric]-group[metric+'_lo'], group[metric+'_hi']-group[metric]],
                    marker='o', capsize=3, color=COLORS[model],
                    label=f'{SHORT[model]} (n={group[count].iloc[-1]})')
axes[0].set_title('W(k)：正确建议采纳率（初始答错）', fontsize=11)
axes[1].set_title('C(k)：错误建议屈从率（初始答对）', fontsize=11)
axes[0].set_ylabel('当前回答等于目标选项的比例')
for ax in axes:
    ax.set_xlabel('施压层级'); ax.set_xticks([1, 2, 3]); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / 'flip_curves_all_zh.png', dpi=240)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.8))
positions = {'gemma-4-31b-it':(15,84), 'deepseek-v4-flash':(8,90),
             'qwen3.8-27b':(15,72), 'qwen3.5-397b':(15,46),
             'glm-5.2':(15,60), 'glm-5.3-flash':(4,28)}
for row in data[data.k == 3].itertuples():
    color = COLORS[row.model]
    ax.errorbar(100*row.C, 100*row.W, color=color,
                xerr=[[100*(row.C-row.C_lo)], [100*(row.C_hi-row.C)]],
                yerr=[[100*(row.W-row.W_lo)], [100*(row.W_hi-row.W)]],
                fmt='o', capsize=3)
    ax.annotate(SHORT[row.model], (100*row.C,100*row.W), xytext=positions[row.model],
                fontsize=10, color=color,
                arrowprops=dict(arrowstyle='-', color=color, lw=.6, alpha=.7))
ax.set_xlabel('错误建议屈从率 C(3)，%（越低越好）')
ax.set_ylabel('正确建议采纳率 W(3)，%（越高越好）')
ax.set_xlim(0,25); ax.set_ylim(20,100); ax.grid(alpha=.3)
ax.set_title('完整施压流程结束时的 W(3) 与 C(3)')
fig.tight_layout()
fig.savefig(OUT / 'fig_wc_zh.png', dpi=240)
plt.close(fig)
