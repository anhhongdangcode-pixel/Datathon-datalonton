"""
RETENTION EDA — 4 góc phân tích
1. Danh mục sản phẩm mua đầu tiên   (Mua gì thì dễ ở lại?)
2. Nhạy cảm khuyến mãi              (Khách săn sale đơn đầu có trung thành không?)
3. Tác động vận hành                (Giao chậm có làm khách bỏ đi luôn không?)
4. Kênh thu hút khách hàng          (Kênh nào mang lại khách "xịn" nhất?)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.2f}'.format)

INPUT_DIR = Path('input')

# ─── Palette ────────────────────────────────────────────────────────────────
C_BLUE   = '#2E86AB'
C_GREEN  = '#1D9E75'
C_RED    = '#E24B4A'
C_AMBER  = '#EF9F27'
C_GREY   = '#888780'
C_PURPLE = '#7B5EA7'

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 130, 'figure.facecolor': 'white'})

# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
print("📂 Loading data …")

orders      = pd.read_csv(INPUT_DIR / 'orders.csv',      parse_dates=['order_date'])
order_items = pd.read_csv(INPUT_DIR / 'order_items.csv')
products    = pd.read_csv(INPUT_DIR / 'products.csv')
shipments   = pd.read_csv(INPUT_DIR / 'shipments.csv',   parse_dates=['ship_date', 'delivery_date'])
payments    = pd.read_csv(INPUT_DIR / 'payments.csv')

print(f"  orders      : {orders.shape}")
print(f"  order_items : {order_items.shape}")
print(f"  products    : {products.shape}")
print(f"  shipments   : {shipments.shape}")
print(f"  payments    : {payments.shape}")

# ════════════════════════════════════════════════════════════════════════════
# BẢNG BASE — chỉ giữ đơn hợp lệ (bỏ cancelled)
# Thêm first_order_date đúng chuẩn
# ════════════════════════════════════════════════════════════════════════════
valid_orders = orders[orders['order_status'] != 'cancelled'].copy()

# Bỏ installments = 2 (rác)
bad_order_ids = payments[payments['installments'] == 2]['order_id']
valid_orders  = valid_orders[~valid_orders['order_id'].isin(bad_order_ids)]

# first_order_date per customer
first_order = (
    valid_orders.groupby('customer_id')['order_date']
    .min()
    .rename('first_order_date')
    .reset_index()
)
valid_orders = valid_orders.merge(first_order, on='customer_id', how='left')

# Cohort month (từ first_order_date)
valid_orders['cohort_month'] = valid_orders['first_order_date'].dt.to_period('M')
valid_orders['order_month']  = valid_orders['order_date'].dt.to_period('M')
valid_orders['month_number'] = (
    (valid_orders['order_month'] - valid_orders['cohort_month']).apply(lambda x: x.n)
)

# Customers có ≥2 đơn (để tính retention thực)
repeat_customers = (
    valid_orders.groupby('customer_id')['order_id'].nunique()
    .pipe(lambda s: s[s >= 2].index)
)

print(f"\n✅ valid_orders: {len(valid_orders):,} rows")
print(f"   Customers có ≥2 đơn: {len(repeat_customers):,}")


# ════════════════════════════════════════════════════════════════════════════
# HÀM TIỆN ÍCH — Tính Retention Rate theo nhóm
# ════════════════════════════════════════════════════════════════════════════
def calc_retention(df, group_col, max_month=12):
    """
    Trả về DataFrame: group | month_number | retention_rate
    Chỉ tính Month 0 → max_month.
    """
    records = []
    for grp, sub in df.groupby(group_col):
        cohort_customers = sub[sub['month_number'] == 0]['customer_id'].unique()
        n_cohort = len(cohort_customers)
        if n_cohort == 0:
            continue
        for m in range(0, max_month + 1):
            active = sub[(sub['month_number'] == m) &
                         (sub['customer_id'].isin(cohort_customers))]['customer_id'].nunique()
            records.append({group_col: grp, 'month_number': m,
                            'retention_rate': active / n_cohort * 100,
                            'n_cohort': n_cohort})
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# 1. DANH MỤC SẢN PHẨM MUA ĐẦU TIÊN
# ════════════════════════════════════════════════════════════════════════════
print("\n── [1] Danh mục sản phẩm mua đầu tiên ──")

# Lấy đơn đầu tiên của mỗi khách
first_orders = valid_orders[valid_orders['month_number'] == 0][['order_id', 'customer_id']].drop_duplicates()

# Join với order_items → products để lấy category
first_items = (
    first_orders
    .merge(order_items[['order_id', 'product_id']], on='order_id')
    .merge(products[['product_id', 'category']], on='product_id')
    .drop_duplicates(subset='customer_id')   # mỗi khách 1 dòng (dùng item đầu tiên)
)

# Merge category vào valid_orders
orders_with_cat = valid_orders.merge(
    first_items[['customer_id', 'category']].rename(columns={'category': 'first_category'}),
    on='customer_id', how='left'
)

ret_cat = calc_retention(orders_with_cat.dropna(subset=['first_category']),
                         'first_category', max_month=12)

# Vẽ
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Góc 1 — Danh mục sản phẩm mua đầu tiên & Retention', fontsize=14, fontweight='bold')

# 1A: Line chart retention theo tháng
ax = axes[0]
palette = sns.color_palette('tab10', n_colors=ret_cat['first_category'].nunique())
for i, (cat, sub) in enumerate(ret_cat.groupby('first_category')):
    ax.plot(sub['month_number'], sub['retention_rate'],
            marker='o', markersize=4, label=f"{cat} (n={sub['n_cohort'].iloc[0]:,})",
            color=palette[i])
ax.set_xlabel('Tháng kể từ đơn đầu')
ax.set_ylabel('Retention Rate (%)')
ax.set_title('Retention theo danh mục mua đầu (Month 0–12)')
ax.legend(fontsize=8, loc='upper right')
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xticks(range(0, 13))

# 1B: Bar chart Month-3 retention (tóm gọn)
ax2 = axes[1]
m3 = ret_cat[ret_cat['month_number'] == 3].sort_values('retention_rate', ascending=True)
colors_bar = [C_GREEN if v >= m3['retention_rate'].median() else C_RED
              for v in m3['retention_rate']]
ax2.barh(m3['first_category'], m3['retention_rate'], color=colors_bar)
ax2.axvline(m3['retention_rate'].median(), color=C_GREY, linestyle='--', label='Median')
ax2.set_xlabel('Retention Rate tháng 3 (%)')
ax2.set_title('So sánh Retention tháng 3 theo danh mục')
ax2.xaxis.set_major_formatter(mticker.PercentFormatter())
ax2.legend()

plt.tight_layout()
plt.savefig('retention_1_category.png', bbox_inches='tight')
plt.show()
print("  ✅ Saved retention_1_category.png")

# In số liệu tóm tắt
summary_cat = (
    ret_cat[ret_cat['month_number'].isin([1, 3, 6, 12])]
    .pivot_table(index='first_category', columns='month_number', values='retention_rate')
    .round(1)
)
summary_cat.columns = [f'Month {c}' for c in summary_cat.columns]
print("\n  Retention (%) theo danh mục:")
print(summary_cat.to_string())


# ════════════════════════════════════════════════════════════════════════════
# 2. NHẠY CẢM KHUYẾN MÃI — đơn đầu có discount không?
# ════════════════════════════════════════════════════════════════════════════
print("\n── [2] Nhạy cảm khuyến mãi ──")

# Đơn đầu tiên của mỗi khách: có promo hay không?
first_order_promo = (
    first_orders
    .merge(order_items[['order_id', 'promo_id', 'discount_amount']], on='order_id')
    .groupby('customer_id')
    .agg(
        has_promo  = ('promo_id', lambda x: x.notna().any()),
        total_disc = ('discount_amount', 'sum')
    )
    .reset_index()
)
first_order_promo['first_order_type'] = np.where(
    first_order_promo['has_promo'], 'Có khuyến mãi', 'Không KM'
)

orders_with_promo = valid_orders.merge(
    first_order_promo[['customer_id', 'first_order_type']],
    on='customer_id', how='left'
)

ret_promo = calc_retention(orders_with_promo.dropna(subset=['first_order_type']),
                           'first_order_type', max_month=12)

# Vẽ
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Góc 2 — Nhạy cảm khuyến mãi đơn đầu & Retention', fontsize=14, fontweight='bold')

# 2A: Line chart
ax = axes[0]
colors_promo = {'Có khuyến mãi': C_AMBER, 'Không KM': C_BLUE}
for grp, sub in ret_promo.groupby('first_order_type'):
    ax.plot(sub['month_number'], sub['retention_rate'],
            marker='o', markersize=5, label=f"{grp} (n={sub['n_cohort'].iloc[0]:,})",
            color=colors_promo.get(grp, C_GREY), linewidth=2)
ax.set_xlabel('Tháng kể từ đơn đầu')
ax.set_ylabel('Retention Rate (%)')
ax.set_title('Retention: Có KM vs Không KM ở đơn đầu')
ax.legend()
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xticks(range(0, 13))

# 2B: % Repeat purchase theo nhóm
ax2 = axes[1]
repeat_flag = first_order_promo.copy()
repeat_flag['is_repeat'] = repeat_flag['customer_id'].isin(repeat_customers)
repeat_by_type = (
    repeat_flag.groupby('first_order_type')['is_repeat']
    .agg(['sum', 'count'])
    .assign(repeat_rate=lambda d: d['sum'] / d['count'] * 100)
    .reset_index()
)
bars = ax2.bar(repeat_by_type['first_order_type'], repeat_by_type['repeat_rate'],
               color=[colors_promo.get(t, C_GREY) for t in repeat_by_type['first_order_type']],
               width=0.4)
for bar, (_, row) in zip(bars, repeat_by_type.iterrows()):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{row['repeat_rate']:.1f}%\n(n={row['count']:,})",
             ha='center', va='bottom', fontsize=10)
ax2.set_ylabel('Tỷ lệ mua lại (%)')
ax2.set_title('Tỷ lệ khách có ≥2 đơn theo nhóm KM')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylim(0, repeat_by_type['repeat_rate'].max() * 1.25)

plt.tight_layout()
plt.savefig('retention_2_promo.png', bbox_inches='tight')
plt.show()
print("  ✅ Saved retention_2_promo.png")


# ════════════════════════════════════════════════════════════════════════════
# 3. TÁC ĐỘNG VẬN HÀNH — giao chậm → bỏ đi không?
# ════════════════════════════════════════════════════════════════════════════
print("\n── [3] Tác động vận hành ──")

# Tính days_to_deliver cho đơn đầu tiên
ship_valid = shipments.dropna(subset=['ship_date', 'delivery_date']).copy()
ship_valid['days_to_deliver'] = (ship_valid['delivery_date'] - ship_valid['ship_date']).dt.days

first_order_ship = (
    first_orders
    .merge(ship_valid[['order_id', 'days_to_deliver']], on='order_id', how='left')
    .groupby('customer_id')['days_to_deliver']
    .first()
    .reset_index()
)

# Bin delivery speed
q33 = first_order_ship['days_to_deliver'].quantile(0.33)
q66 = first_order_ship['days_to_deliver'].quantile(0.66)
print(f"  Delivery quantiles: 33%={q33:.0f}d, 66%={q66:.0f}d")

def bin_delivery(d):
    if pd.isna(d):
        return 'Không có ship data'
    elif d <= q33:
        return f'Nhanh (≤{q33:.0f}d)'
    elif d <= q66:
        return f'Trung bình ({q33:.0f}–{q66:.0f}d)'
    else:
        return f'Chậm (>{q66:.0f}d)'

first_order_ship['delivery_group'] = first_order_ship['days_to_deliver'].apply(bin_delivery)

orders_with_ship = valid_orders.merge(
    first_order_ship[['customer_id', 'delivery_group']],
    on='customer_id', how='left'
)

# Chỉ lấy 3 nhóm có ship data
orders_ship_clean = orders_with_ship[
    orders_with_ship['delivery_group'] != 'Không có ship data'
]

ret_ship = calc_retention(orders_ship_clean, 'delivery_group', max_month=12)

# Vẽ
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Góc 3 — Tốc độ giao hàng đơn đầu & Retention', fontsize=14, fontweight='bold')

ax = axes[0]
colors_ship = {}
for grp in ret_ship['delivery_group'].unique():
    if 'Nhanh' in grp:
        colors_ship[grp] = C_GREEN
    elif 'Trung' in grp:
        colors_ship[grp] = C_AMBER
    else:
        colors_ship[grp] = C_RED

for grp, sub in ret_ship.groupby('delivery_group'):
    ax.plot(sub['month_number'], sub['retention_rate'],
            marker='o', markersize=5, label=f"{grp} (n={sub['n_cohort'].iloc[0]:,})",
            color=colors_ship.get(grp, C_GREY), linewidth=2)
ax.set_xlabel('Tháng kể từ đơn đầu')
ax.set_ylabel('Retention Rate (%)')
ax.set_title('Retention theo tốc độ giao đơn đầu (Month 0–12)')
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xticks(range(0, 13))

# 3B: Distribution days_to_deliver + repeat rate
ax2 = axes[1]
repeat_ship = first_order_ship[first_order_ship['delivery_group'] != 'Không có ship data'].copy()
repeat_ship['is_repeat'] = repeat_ship['customer_id'].isin(repeat_customers)
rr_ship = (
    repeat_ship.groupby('delivery_group')['is_repeat']
    .agg(['sum', 'count'])
    .assign(repeat_rate=lambda d: d['sum'] / d['count'] * 100)
    .reset_index()
)
order_map = {g: i for i, g in enumerate(sorted(rr_ship['delivery_group']))}
rr_ship['_ord'] = rr_ship['delivery_group'].map(order_map)
rr_ship = rr_ship.sort_values('_ord')

bars = ax2.bar(rr_ship['delivery_group'], rr_ship['repeat_rate'],
               color=[colors_ship.get(g, C_GREY) for g in rr_ship['delivery_group']],
               width=0.4)
for bar, (_, row) in zip(bars, rr_ship.iterrows()):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{row['repeat_rate']:.1f}%", ha='center', va='bottom', fontsize=10)
ax2.set_ylabel('Tỷ lệ mua lại (%)')
ax2.set_title('Tỷ lệ mua lại theo tốc độ giao đơn đầu')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylim(0, rr_ship['repeat_rate'].max() * 1.25)
plt.xticks(rotation=10)

plt.tight_layout()
plt.savefig('retention_3_shipping.png', bbox_inches='tight')
plt.show()
print("  ✅ Saved retention_3_shipping.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. KÊNH THU HÚT KHÁCH HÀNG — order_source đơn đầu
# ════════════════════════════════════════════════════════════════════════════
print("\n── [4] Kênh thu hút khách hàng ──")

first_order_source = (
    first_orders
    .merge(valid_orders[['order_id', 'order_source']].drop_duplicates(), on='order_id', how='left')
    [['customer_id', 'order_source']]
    .rename(columns={'order_source': 'first_source'})
)

orders_with_source = valid_orders.merge(
    first_order_source, on='customer_id', how='left'
)

ret_source = calc_retention(orders_with_source.dropna(subset=['first_source']),
                            'first_source', max_month=12)

# Vẽ
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Góc 4 — Kênh thu hút (order_source) & Retention', fontsize=14, fontweight='bold')

ax = axes[0]
palette_src = sns.color_palette('Set2', n_colors=ret_source['first_source'].nunique())
for i, (src, sub) in enumerate(ret_source.groupby('first_source')):
    ax.plot(sub['month_number'], sub['retention_rate'],
            marker='o', markersize=4, label=f"{src} (n={sub['n_cohort'].iloc[0]:,})",
            color=palette_src[i], linewidth=2)
ax.set_xlabel('Tháng kể từ đơn đầu')
ax.set_ylabel('Retention Rate (%)')
ax.set_title('Retention theo kênh thu hút (Month 0–12)')
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xticks(range(0, 13))

# 4B: Heatmap Month 1,3,6,12 theo kênh
ax2 = axes[1]
pivot_src = (
    ret_source[ret_source['month_number'].isin([1, 3, 6, 12])]
    .pivot_table(index='first_source', columns='month_number', values='retention_rate')
    .round(1)
)
pivot_src.columns = [f'M{c}' for c in pivot_src.columns]
sns.heatmap(pivot_src, annot=True, fmt='.1f', cmap='RdYlGn',
            linewidths=0.5, ax=ax2, cbar_kws={'label': 'Retention %'})
ax2.set_title('Heatmap Retention (%) theo kênh & tháng')
ax2.set_ylabel('')

plt.tight_layout()
plt.savefig('retention_4_channel.png', bbox_inches='tight')
plt.show()
print("  ✅ Saved retention_4_channel.png")

# ════════════════════════════════════════════════════════════════════════════
# TỔNG HỢP — LTV proxy: avg orders per customer theo từng phân nhóm
# ════════════════════════════════════════════════════════════════════════════
print("\n── [Summary] LTV proxy — số đơn TB per customer ──")

def avg_orders(base_df, group_col):
    return (
        base_df[base_df['month_number'] > 0]
        .groupby(['customer_id', group_col])['order_id'].nunique()
        .reset_index()
        .groupby(group_col)['order_id'].mean()
        .rename('avg_repeat_orders')
        .round(2)
    )

print("\n  Theo danh mục:")
print(avg_orders(orders_with_cat.dropna(subset=['first_category']), 'first_category').to_string())

print("\n  Theo promo:")
print(avg_orders(orders_with_promo.dropna(subset=['first_order_type']), 'first_order_type').to_string())

print("\n  Theo shipping speed:")
print(avg_orders(orders_ship_clean, 'delivery_group').to_string())

print("\n  Theo kênh:")
print(avg_orders(orders_with_source.dropna(subset=['first_source']), 'first_source').to_string())

print("\n✅ Done! Files saved: retention_1–4.png")
