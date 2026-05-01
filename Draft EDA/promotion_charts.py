"""
Promotion Analysis — 2 Figures
  Figure 1 (subarg1): Budget vs Margin Mismatch  →  3 panels
  Figure 2 (subarg2): Seasonal Misalignment       →  3 panels
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────
try:
    INPUT_DIR
except NameError:
    INPUT_DIR = Path('input')

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

C_COLORS = {
    'Outdoor':    '#1D9E75',
    'Streetwear': '#E24B4A',
    'Casual':     '#2E86AB',
    'GenZ':       '#EF9F27',
}
CAT_ORDER = ['Streetwear', 'Outdoor', 'Casual', 'GenZ']

FONT_TITLE  = dict(fontsize=13, fontweight='bold', color='#1a1a1a')
FONT_SUP    = dict(fontsize=15, fontweight='bold', color='#1a1a1a')
FONT_LABEL  = dict(fontsize=10, color='#444')
FONT_TICK   = dict(fontsize=9)
SPINE_COLOR = '#cccccc'

def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.tick_params(colors='#555', labelsize=9)
    ax.grid(axis='y', color='#ebebeb', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# ── Load data ───────────────────────────────────────────────────────────────
orders      = pd.read_csv(INPUT_DIR / 'orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv(INPUT_DIR / 'order_items.csv')
products    = pd.read_csv(INPUT_DIR / 'products.csv')

orders_d = orders[orders['order_status'] == 'delivered'].copy()
orders_d['year']  = orders_d['order_date'].dt.year
orders_d['month'] = orders_d['order_date'].dt.month

# ── Base join ────────────────────────────────────────────────────────────────
items = (
    order_items
    .merge(orders_d[['order_id', 'year', 'month', 'order_date']], on='order_id', how='inner')
    .merge(products[['product_id', 'category', 'cogs']], on='product_id', how='left')
)
items['gross_item']    = items['unit_price'] * items['quantity']
items['discount_amt']  = items['discount_amount'].fillna(0)
items['net_revenue']   = items['gross_item'] - items['discount_amt']
items['total_cogs']    = items['cogs'] * items['quantity']
items['has_promo']     = ((items['discount_amt'] > 0) | items['promo_id'].notna()).astype(int)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  SUBARG 1: Budget vs Margin Mismatch
# ════════════════════════════════════════════════════════════════════════════

# ── Panel A data: Budget share by category (2019–2022) ──────────────────────
p2_items = items[items['year'].between(2019, 2022)].copy()

budget_cat = (
    p2_items[p2_items['has_promo'] == 1]
    .groupby('category')['discount_amt']
    .sum()
    .reset_index(name='total_discount')
)
budget_cat['budget_share'] = budget_cat['total_discount'] / budget_cat['total_discount'].sum() * 100
budget_cat = budget_cat.set_index('category').reindex(CAT_ORDER).reset_index()

# ── Panel B data: Gross Margin by category (all years) ──────────────────────
margin_cat = (
    items.groupby('category')
    .agg(total_net_rev=('net_revenue', 'sum'),
         total_cogs_val=('total_cogs', 'sum'))
    .reset_index()
)
margin_cat['gross_margin_pct'] = (
    (margin_cat['total_net_rev'] - margin_cat['total_cogs_val'])
    / margin_cat['total_net_rev'] * 100
)
margin_cat = margin_cat.set_index('category').reindex(CAT_ORDER).reset_index()

# ── Panel C data: Gross profit per promo VND by category ────────────────────
promo_items = p2_items[p2_items['has_promo'] == 1].copy()
promo_items['gross_profit'] = promo_items['net_revenue'] - promo_items['total_cogs']

gp_per_promo = (
    promo_items.groupby('category')
    .agg(total_gp=('gross_profit', 'sum'),
         total_disc=('discount_amt', 'sum'))
    .reset_index()
)
gp_per_promo['gp_per_promo_vnd'] = gp_per_promo['total_gp'] / gp_per_promo['total_disc']
gp_per_promo = gp_per_promo.set_index('category').reindex(CAT_ORDER).reset_index()

# ── Draw Figure 1 ───────────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(17, 6.5), facecolor='white')
fig1.suptitle(
    'Subargument 1 — Phân bổ Ngân sách Khuyến mãi Sai lệch với Biên Lợi nhuận',
    **FONT_SUP, y=1.01
)
gs = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.38)

bar_colors = [C_COLORS.get(c, '#888') for c in CAT_ORDER]

# Panel A — Budget share
ax1 = fig1.add_subplot(gs[0])
bars = ax1.bar(CAT_ORDER, budget_cat['budget_share'], color=bar_colors, width=0.55,
               alpha=0.92, zorder=3, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, budget_cat['budget_share']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#222')
ax1.set_title('(A) Tỷ trọng Ngân sách KM\nthực chi theo Category (2019–2022)', **FONT_TITLE)
ax1.set_ylabel('% Tổng discount amount', **FONT_LABEL)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax1.set_ylim(0, budget_cat['budget_share'].max() * 1.22)
ax1.set_xticklabels(CAT_ORDER, **FONT_TICK)

# Annotation arrow Streetwear vs GenZ
sw_val = budget_cat.loc[budget_cat['category'] == 'Streetwear', 'budget_share'].values[0]
gz_val = budget_cat.loc[budget_cat['category'] == 'GenZ', 'budget_share'].values[0]
ax1.annotate('', xy=(3, gz_val + 1), xytext=(0, sw_val + 1),
             arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.5))
ax1.text(1.5, (sw_val + gz_val)/2 + 3,
         f'×{sw_val/gz_val:.0f} chênh lệch', ha='center', fontsize=8.5,
         color='#c0392b', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.25', fc='#fdecea', alpha=0.85))
style_ax(ax1)

# Panel B — Dual-axis: Budget share vs Gross Margin
ax2 = fig1.add_subplot(gs[1])
x = np.arange(len(CAT_ORDER))
width = 0.38

bars2 = ax2.bar(x - width/2, budget_cat['budget_share'], width, color=bar_colors,
                alpha=0.75, zorder=3, label='Budget share (%)', edgecolor='white')
ax2.set_ylabel('% Ngân sách KM', **FONT_LABEL)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax2.set_ylim(0, budget_cat['budget_share'].max() * 1.35)

ax2b = ax2.twinx()
bars3 = ax2b.bar(x + width/2, margin_cat['gross_margin_pct'], width, color='#6c5ce7',
                 alpha=0.75, zorder=3, label='Gross Margin (%)', edgecolor='white')
ax2b.set_ylabel('Gross Margin (%)', fontsize=10, color='#6c5ce7')
ax2b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
ax2b.tick_params(axis='y', colors='#6c5ce7')
ax2b.set_ylim(0, margin_cat['gross_margin_pct'].max() * 1.55)
for spine in ax2b.spines.values():
    spine.set_color(SPINE_COLOR)

for bar, val in zip(bars2, budget_cat['budget_share']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color='#444')
for bar, val in zip(bars3, margin_cat['gross_margin_pct']):
    ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
              f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color='#6c5ce7')

ax2.set_title('(B) Nghịch lý: Budget cao → Margin thấp\nBudget share vs Gross Margin theo Category', **FONT_TITLE)
ax2.set_xticks(x)
ax2.set_xticklabels(CAT_ORDER, **FONT_TICK)
h1 = mpatches.Patch(color=bar_colors[0], alpha=0.75, label='Budget share (%)')
h2 = mpatches.Patch(color='#6c5ce7', alpha=0.75, label='Gross Margin (%)')
ax2.legend(handles=[h1, h2], fontsize=8.5, loc='upper right')
style_ax(ax2)
ax2.grid(visible=False)

# Panel C — GP per promo VND (horizontal bar)
ax3 = fig1.add_subplot(gs[2])
vals = gp_per_promo['gp_per_promo_vnd'].values
baseline = vals[CAT_ORDER.index('Streetwear')]  # Streetwear = baseline

norm_vals = vals / baseline  # ratio vs Streetwear
colors_c = ['#e74c3c' if v < 1.0 else '#27ae60' for v in norm_vals]

hbars = ax3.barh(CAT_ORDER[::-1], norm_vals[::-1], color=colors_c[::-1],
                 alpha=0.88, zorder=3, edgecolor='white', height=0.5)
ax3.axvline(x=1.0, color='#e74c3c', linestyle='--', linewidth=1.4, alpha=0.8, zorder=4)
ax3.text(1.02, -0.55, 'Streetwear\n(baseline)', fontsize=8, color='#e74c3c', va='top')

for bar, val, raw in zip(hbars, norm_vals[::-1], vals[::-1]):
    ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}× ({raw:,.0f} VND)',
             va='center', ha='left', fontsize=9, color='#222')

ax3.set_title('(C) Hiệu suất sinh lời của KM\nGross Profit / đồng Promo (vs Streetwear baseline)', **FONT_TITLE)
ax3.set_xlabel('Ratio vs Streetwear (1.0 = baseline)', **FONT_LABEL)
ax3.set_xlim(0, norm_vals.max() * 1.45)
style_ax(ax3)
ax3.grid(axis='x', color='#ebebeb', linewidth=0.8)
ax3.grid(axis='y', visible=False)

plt.tight_layout()
fig1.savefig(OUTPUT_DIR / 'fig1_subarg1_budget_vs_margin.png', dpi=150,
             bbox_inches='tight', facecolor='white')
plt.show()
print('✅ Saved: fig1_subarg1_budget_vs_margin.png')


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  SUBARG 2: Seasonal Misalignment
# ════════════════════════════════════════════════════════════════════════════

# ── Panel A data: Avg volume share by month across all years ─────────────────
# Tính tỷ trọng volume tháng so với tổng năm (normalize trong từng năm)
vol_month_yr = (
    items.groupby(['year', 'month'])['quantity']
    .sum()
    .reset_index(name='qty')
)
yr_total = vol_month_yr.groupby('year')['qty'].sum().reset_index(name='yr_qty')
vol_month_yr = vol_month_yr.merge(yr_total, on='year')
vol_month_yr['month_share_pct'] = vol_month_yr['qty'] / vol_month_yr['yr_qty'] * 100

avg_month_vol = (
    vol_month_yr.groupby('month')['month_share_pct']
    .mean()
    .reset_index()
)

# ── Panel B data: Promo spend share by month (2019–2022) ─────────────────────
promo_month_yr = (
    p2_items[p2_items['has_promo'] == 1]
    .groupby(['year', 'month'])['discount_amt']
    .sum()
    .reset_index()
)
yr_promo_total = promo_month_yr.groupby('year')['discount_amt'].sum().reset_index(name='yr_disc')
promo_month_yr = promo_month_yr.merge(yr_promo_total, on='year')
promo_month_yr['promo_share_pct'] = promo_month_yr['discount_amt'] / promo_month_yr['yr_disc'] * 100

avg_month_promo = (
    promo_month_yr.groupby('month')['promo_share_pct']
    .mean()
    .reset_index()
)

# ── Panel C data: Promo efficiency (volume per promo VND) by month ───────────
eff_month = (
    p2_items.groupby('month')
    .agg(total_qty=('quantity', 'sum'),
         total_disc=('discount_amt', 'sum'))
    .reset_index()
)
eff_month['total_disc'] = eff_month['total_disc'].replace(0, np.nan)
eff_month['vol_per_promo_vnd'] = eff_month['total_qty'] / eff_month['total_disc']
eff_month['vol_per_promo_norm'] = eff_month['vol_per_promo_vnd'] / eff_month['vol_per_promo_vnd'].mean()

MONTHS = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']
PEAK = [4, 5, 6]
PEAK2 = [12]

# ── Draw Figure 2 ───────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(17, 6.5), facecolor='white')
fig2.suptitle(
    'Subargument 2 — Promotion không được căn theo Mùa vụ: Rải đều vào vùng nhu cầu thấp',
    **FONT_SUP, y=1.01
)
gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.38)

def shade_peaks(ax, ymax_factor=1.0):
    ylo, yhi = ax.get_ylim()
    for m in PEAK:
        ax.axvspan(m - 1.5, m - 0.5, color='#27ae60', alpha=0.10, zorder=0)
    for m in PEAK2:
        ax.axvspan(m - 1.5, m - 0.5, color='#2E86AB', alpha=0.10, zorder=0)

# Panel A — Avg monthly volume share
ax4 = fig2.add_subplot(gs2[0])
vol_vals = avg_month_vol['month_share_pct'].values
bar_vol_colors = ['#27ae60' if m in PEAK else ('#2E86AB' if m in PEAK2 else '#b0bec5')
                  for m in avg_month_vol['month']]
bars4 = ax4.bar(avg_month_vol['month'], vol_vals, color=bar_vol_colors,
                alpha=0.85, zorder=3, width=0.65, edgecolor='white')

for bar, val in zip(bars4, vol_vals):
    if val > 0:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5, color='#333')

ax4.set_title('(A) Sức bán tự nhiên theo Tháng\n(Avg % volume trong năm — toàn bộ 10 năm)', **FONT_TITLE)
ax4.set_ylabel('% Volume trong năm', **FONT_LABEL)
ax4.set_xticks(range(1, 13))
ax4.set_xticklabels(MONTHS, **FONT_TICK)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
style_ax(ax4)
shade_peaks(ax4)

# Legend peaks
p_green = mpatches.Patch(color='#27ae60', alpha=0.3, label='Peak T4–T6')
p_blue  = mpatches.Patch(color='#2E86AB', alpha=0.3, label='Peak T12')
ax4.legend(handles=[p_green, p_blue], fontsize=8, loc='upper left')

# Panel B — Promo spend share vs volume share overlay
ax5 = fig2.add_subplot(gs2[1])
promo_vals = avg_month_promo['promo_share_pct'].values

x = np.arange(1, 13)
width = 0.38

ax5.bar(x - width/2, vol_vals, width, color='#636e72', alpha=0.55, label='Volume share (%)', zorder=3, edgecolor='white')
ax5.bar(x + width/2, promo_vals, width, color='#e17055', alpha=0.78, label='Promo spend share (%)', zorder=3, edgecolor='white')

# Highlight gap in peak months
for m in PEAK:
    v = vol_vals[m-1]
    p = promo_vals[m-1]
    gap = p - v
    color = '#27ae60' if gap > 0 else '#c0392b'
    ax5.annotate('', xy=(m + width/2, max(v, p) + 0.15),
                 xytext=(m - width/2, max(v, p) + 0.15),
                 arrowprops=dict(arrowstyle='<->', color=color, lw=1.2))
    ax5.text(m, max(v, p) + 0.5, f'{gap:+.1f}pp',
             ha='center', fontsize=7.5, color=color, fontweight='bold')

ax5.set_title('(B) Promo Spend vs Volume Share theo Tháng\n(P2: 2019–2022) — Promotion có đi đúng vào đỉnh mùa?', **FONT_TITLE)
ax5.set_ylabel('%', **FONT_LABEL)
ax5.set_xticks(x)
ax5.set_xticklabels(MONTHS, **FONT_TICK)
ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
ax5.legend(fontsize=8.5, loc='upper right')
style_ax(ax5)
shade_peaks(ax5)

# Panel C — Promo efficiency by month (vol per promo VND)
ax6 = fig2.add_subplot(gs2[2])
eff_colors = ['#27ae60' if m in PEAK else ('#2E86AB' if m in PEAK2 else '#b0bec5')
              for m in eff_month['month']]
bars6 = ax6.bar(eff_month['month'], eff_month['vol_per_promo_norm'],
                color=eff_colors, alpha=0.85, zorder=3, width=0.65, edgecolor='white')

ax6.axhline(y=1.0, color='#636e72', linestyle='--', linewidth=1.3, alpha=0.8, zorder=4)
ax6.text(12.5, 1.01, 'Avg', fontsize=8, color='#636e72', va='bottom', ha='right')

for bar, val in zip(bars6, eff_month['vol_per_promo_norm']):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f'{val:.2f}×', ha='center', va='bottom', fontsize=8, color='#333')

ax6.set_title('(C) Hiệu suất KM theo Tháng\nVol bán được / đồng Promo (1.0 = trung bình năm)', **FONT_TITLE)
ax6.set_ylabel('Normalized efficiency (×avg)', **FONT_LABEL)
ax6.set_xticks(range(1, 13))
ax6.set_xticklabels(MONTHS, **FONT_TICK)
style_ax(ax6)
shade_peaks(ax6)
ax6.legend(handles=[p_green, p_blue], fontsize=8, loc='upper right')

plt.tight_layout()
fig2.savefig(OUTPUT_DIR / 'fig2_subarg2_seasonal_misalignment.png', dpi=150,
             bbox_inches='tight', facecolor='white')
plt.show()
print('✅ Saved: fig2_subarg2_seasonal_misalignment.png')
