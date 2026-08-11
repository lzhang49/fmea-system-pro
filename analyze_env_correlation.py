# -*- coding: utf-8 -*-
"""
2022-2023年有效月份深度分析
- 哪些月份有信号？哪些月份赚钱？
- 和指数位置的关系？（指数在高位/低位/反弹期？）
- 是指数趋势决定的，还是情绪周期决定的？
"""
import pandas as pd
import numpy as np

T = pd.read_csv('trades_lht_main.csv', dtype={'code': str})
T['date'] = T['date'].astype(str)
T['year'] = T['year'].astype(int)
T['month'] = T['date'].str[:7]

# 加载创业板指
idx = pd.read_csv('idx_CYB.csv')
idx['date'] = idx.iloc[:, 0].astype(str)
idx['close'] = idx['close'].astype(float)
idx = idx.sort_values('date').reset_index(drop=True)
idx['ma20'] = idx['close'].rolling(20).mean()
idx['ma60'] = idx['close'].rolling(60).mean()

# 计算指数250日分位（相对位置）
def pct_rank(series, window=250):
    """滚动计算当前值在过去window天内的百分位"""
    result = []
    for i in range(len(series)):
        if i < window:
            result.append(np.nan)
        else:
            window_vals = series.iloc[i-window:i]
            rank = (window_vals < series.iloc[i]).sum() / window
            result.append(rank)
    return result

idx['pos_250'] = pct_rank(idx['close'], 250)

# 月度聚合
monthly_idx = idx.groupby(idx['date'].str[:7]).agg(
    start_close=('close', 'first'),
    end_close=('close', 'last'),
    avg_close=('close', 'mean'),
    month_ret=('close', lambda x: (x.iloc[-1]/x.iloc[0]-1)*100),
    avg_pos=('pos_250', 'mean'),
    end_pos=('pos_250', 'last'),
    avg_ma20=('ma20', 'mean'),
).reset_index()
monthly_idx.columns = ['month', 'start_close', 'end_close', 'avg_close', 'month_ret', 'avg_pos', 'end_pos', 'avg_ma20']

# 月度交易统计
monthly_trades = T.groupby('month').agg(
    n=('raw', 'size'),
    avg_ret=('raw', lambda x: x.mean()*100),
    win_rate=('raw', lambda x: (x>0).mean()*100),
    best=('raw', lambda x: x.max()*100),
    worst=('raw', lambda x: x.min()*100),
).reset_index()

# 合并
merged = monthly_trades.merge(monthly_idx, on='month', how='left')

# 只看2022-2023
m_22_23 = merged[merged['month'].str.startswith(('2022', '2023'))].sort_values('month')

print('='*90)
print('  【2022-2023年 有信号月份全景】')
print('='*90)
print(f'  {"月份":<8s} {"笔数":>4s} {"均收益%":>8s} {"胜率%":>7s} {"当月指数%":>10s} {"指数位置":>9s} {"趋势":>8s} {"赚钱?":>6s}')
print(f'  {"-"*80}')

for _, r in m_22_23.iterrows():
    pos_label = f'{r["end_pos"]*100:.0f}%分位' if not np.isnan(r['end_pos']) else 'N/A'
    # 趋势：收盘价 vs MA20
    trend = '↑站上' if r['end_close'] > r['avg_ma20'] else '↓跌破'
    profit = '✅赚' if r['avg_ret'] > 0 else '❌亏'
    print(f'  {r["month"]:<8s} {r["n"]:>4d} {r["avg_ret"]:>+7.2f}% {r["win_rate"]:>6.1f}% '
          f'{r["month_ret"]:>+9.1f}% {pos_label:>8s} {trend:>7s} {profit:>6s}')

print(f'\n  解读：指数位置 = 收盘价在过去250日的百分位（越高=越贵）')
print(f'        趋势 = 月末收盘价 vs 20日均线（站上=短期强势）')

print(f'\n{"="*90}')
print(f'  【赚钱月 vs 亏钱月 对比】')
print(f'{"="*90}')

win_months = m_22_23[m_22_23['avg_ret'] > 0]
lose_months = m_22_23[m_22_23['avg_ret'] <= 0]

print(f'  赚钱月份: {len(win_months)}个 (", ".join(win_months["month"].tolist()))')
print(f'  亏钱月份: {len(lose_months)}个 (", ".join(lose_months["month"].tolist()))')
print()
print(f'  {"指标":<18s} {"赚钱月均值":>12s} {"亏钱月均值":>12s} {"差异":>10s}')
print(f'  {"-"*60}')
for col, label in [
    ('n', '信号数量(笔)'),
    ('month_ret', '当月指数涨跌(%)'),
    ('end_pos', '月末指数分位(%)'),
    ('avg_pos', '月均指数分位(%)'),
]:
    w = win_months[col].mean() * (100 if 'pos' in col or 'ret' in col else 1)
    l = lose_months[col].mean() * (100 if 'pos' in col or 'ret' in col else 1)
    print(f'  {label:<18s} {w:>12.2f} {l:>12.2f} {w-l:>+10.2f}')

print(f'\n  结论：赚钱月信号更多、指数当月涨、指数位置偏高？')

print(f'\n{"="*90}')
print(f'  【信号出现时，指数处于什么位置？】  所有年份统计')
print(f'{"="*90}')
print()

# 每笔交易对应买入日的指数位置
idx_map = dict(zip(idx['date'], zip(idx['close'], idx['pos_250'], idx['ma20'])))

positions = []
for _, r in T.iterrows():
    d = r['date']
    if d in idx_map:
        close, pos, ma20 = idx_map[d]
        above_ma20 = close > ma20 if not np.isnan(ma20) else None
        positions.append({
            'date': d, 'year': r['year'], 'raw': r['raw']*100,
            'idx_close': close, 'idx_pos': pos*100 if not np.isnan(pos) else None,
            'above_ma20': above_ma20,
        })

pos_df = pd.DataFrame(positions)
print(f'  有指数数据的交易: {len(pos_df)} / {len(T)}')
print()

# 按指数位置分箱
pos_df['pos_bin'] = pd.cut(pos_df['idx_pos'], bins=[0, 20, 40, 60, 80, 100], 
                           labels=['0-20%极低位', '20-40%低位', '40-60%中位', '60-80%高位', '80-100%极高位'])

bin_stat = pos_df.groupby('pos_bin', observed=True).agg(
    n=('raw', 'size'),
    avg_ret=('raw', 'mean'),
    win_rate=('raw', lambda x: (x>0).mean()*100),
    median=('raw', 'median'),
)
print(f'  {"指数位置":<14s} {"n":>4s} {"均收益%":>8s} {"胜率%":>7s} {"中位%":>7s}')
print(f'  {"-"*48}')
for b in bin_stat.index:
    r = bin_stat.loc[b]
    print(f'  {b:<14s} {r["n"]:>4.0f} {r["avg_ret"]:>+7.2f}% {r["win_rate"]:>6.1f}% {r["median"]:>+6.2f}%')

print(f'\n  看MA20趋势:')
ma_stat = pos_df.groupby('above_ma20').agg(
    n=('raw', 'size'),
    avg_ret=('raw', 'mean'),
    win_rate=('raw', lambda x: (x>0).mean()*100),
)
for abv in [True, False]:
    if abv in ma_stat.index:
        r = ma_stat.loc[abv]
        label = '站上MA20(多头)' if abv else '跌破MA20(空头)'
        print(f'    {label}: n={r["n"]:.0f}, 均收益{r["avg_ret"]:+.2f}%, 胜率{r["win_rate"]:.1f}%')

print(f'\n{"="*90}')
print(f'  【2022-2023 逐笔交易 + 指数环境】')
print(f'{"="*90}')
print()

t_22 = pos_df[pos_df['year'].isin([2022, 2023])].sort_values('date')
print(f'  {"日期":<12s} {"代码":<8s} {"收益%":>7s} {"指数点位":>8s} {"250日分位":>9s} {"MA20":>6s} {"盈亏":>6s}')
print(f'  {"-"*70}')

# 需要code，从T里拿
t_22_full = T[T['year'].isin([2022, 2023])][['code','date','raw']].merge(pos_df[['date','idx_close','idx_pos','above_ma20']], on='date')
for _, r in t_22_full.iterrows():
    ma = '↑' if r['above_ma20'] else '↓'
    pnl = '🟢' if r['raw'] > 0 else '🔴'
    print(f'  {r["date"]:<12s} {r["code"]:<8s} {r["raw"]*100:>+6.2f}% '
          f'{r["idx_close"]:>8.0f} {r["idx_pos"]:>8.0f}%   {ma:>4s}  {pnl:>4s}')

print(f'\n{"="*90}')
print(f'  【相关性分析】  策略收益 vs 指数因素')
print(f'{"="*90}')
print()

# 计算各因素和策略收益的相关系数
corr_data = pos_df.dropna(subset=['idx_pos', 'above_ma20'])
corr_pos = corr_data['raw'].corr(corr_data['idx_pos'])
corr_ma = corr_data['raw'].corr(corr_data['above_ma20'].astype(int))

# 月频相关性
monthly_both = merged.dropna(subset=['avg_ret', 'month_ret', 'end_pos'])
corr_month_ret = monthly_both['avg_ret'].corr(monthly_both['month_ret'])
corr_month_pos = monthly_both['avg_ret'].corr(monthly_both['end_pos'])

print(f'  逐笔维度:')
print(f'    策略收益 vs 指数250日分位: {corr_pos:+.3f}')
print(f'    策略收益 vs 是否站上MA20: {corr_ma:+.3f}')
print()
print(f'  月度维度:')
print(f'    策略月均收益 vs 指数当月涨跌: {corr_month_ret:+.3f}')
print(f'    策略月均收益 vs 月末指数位置: {corr_month_pos:+.3f}')

print(f'\n  解读:')
print(f'    正相关越大 → 指数越强/越高，策略越赚钱')
print(f'    负相关 → 指数越弱/越低，策略越赚钱（逆向）')
print(f'    0附近 → 关系不大')

# 保存结果
import json
result = {
    'monthly_22_23': m_22_23[['month','n','avg_ret','win_rate','month_ret','end_pos']].to_dict('records'),
    'pos_bins': {str(b): {'n': int(bin_stat.loc[b, 'n']), 'avg_ret': round(bin_stat.loc[b, 'avg_ret'], 2),
                          'win_rate': round(bin_stat.loc[b, 'win_rate'], 1)} for b in bin_stat.index},
    'ma20_stat': {
        'above': {'n': int(ma_stat.loc[True, 'n']) if True in ma_stat.index else 0,
                  'avg_ret': round(ma_stat.loc[True, 'avg_ret'], 2) if True in ma_stat.index else 0,
                  'win_rate': round(ma_stat.loc[True, 'win_rate'], 1) if True in ma_stat.index else 0},
        'below': {'n': int(ma_stat.loc[False, 'n']) if False in ma_stat.index else 0,
                  'avg_ret': round(ma_stat.loc[False, 'avg_ret'], 2) if False in ma_stat.index else 0,
                  'win_rate': round(ma_stat.loc[False, 'win_rate'], 1) if False in ma_stat.index else 0},
    },
    'correlations': {
        'trade_vs_pos': round(corr_pos, 3),
        'trade_vs_ma20': round(corr_ma, 3),
        'month_vs_idxret': round(corr_month_ret, 3),
        'month_vs_pos': round(corr_month_pos, 3),
    },
    'trades_22_23': t_22_full.to_dict('records'),
}

with open('env_correlation.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
print(f'\n分析数据已保存: env_correlation.json')
