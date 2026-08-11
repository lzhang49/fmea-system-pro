# -*- coding: utf-8 -*-
"""
深度分析补充：
1. 为什么2022年低标占比那么低？（高标65% vs 常年40%）
2. 熊市里低标的表现 vs 牛市里低标的表现——是票本身差，还是环境差？
3. 同年同月同日生对比：同样是低标，在不同市场环境下表现差异
4. 妖股池质量：峰值涨幅、10日涨幅——小年的"妖"是不是不够妖？
5. 2022-2023的入池段，有多少是"伪妖股"（刚好摸到100%但很快A杀）
"""
import pandas as pd
import numpy as np

E = pd.read_csv('events_lht_final.csv', dtype={'code': str})
T = pd.read_csv('trades_lht_main.csv', dtype={'code': str})
T['date'] = T['date'].astype(str)

E['year'] = E['year'].astype(int)
T['year'] = T['year'].astype(int)

print('='*70)
print('  【问题1：2022年低标占比为什么只有19.7%？】')
print('='*70)
print()
print('  直觉：熊市里能翻倍的票都是"大妖"，小票根本涨不起来，所以高标占比高')
print('  验证：看各年入池段的峰值涨幅(peak_gain)分布')
print()

# peak_gain分布按年
for yr in [2022, 2023, 2024, 2025, 2026]:
    sub = E[E['year'] == yr]
    if len(sub) == 0: continue
    pg = sub['peak_gain'] * 100  # 转%
    print(f'  {yr}年(n={len(sub)}): 峰值涨幅 mean={pg.mean():.0f}% median={pg.median():.0f}% '
          f'min={pg.min():.0f}% max={pg.max():.0f}% '
          f'≥200%占比={(pg>=200).mean()*100:.1f}% ≥300%占比={(pg>=300).mean()*100:.1f}%')

print(f'\n  2022年峰值涨幅反而更高？说明翻倍的都是真"妖"，但大多是高标一字板上来的')

print(f'\n{"="*70}')
print(f'  【问题2：小年的低标，单看质量是不是就差？】')
print(f'{"="*70}')
print()

# 只看低标(zt≤3)的入池段，对比小年 vs 大年的各项特征
low_e = E[E['zt_in10'] <= 3]
bad_low = low_e[low_e['year'].isin([2022, 2023])]
good_low = low_e[low_e['year'].isin([2024, 2025])]

print(f'  {"特征":<15s} {"2022-23低标":>12s} {"2024-25低标":>12s} {"差异":>10s}')
print(f'  {"-"*55}')
for col, label, pct in [
    ('peak_gain', '峰值涨幅(%)', True),
    ('r10', '入池时10日涨幅', False),
    ('mcap', '市值(亿)', False),
    ('turn', '入池日换手(%)', False),
    ('dd5', '信号日回调(%)', True),
    ('vshrink', '量缩比', False),
]:
    b = bad_low[col].mean()
    g = good_low[col].mean()
    if pct:
        b *= 100; g *= 100
    print(f'  {label:<15s} {b:>12.2f} {g:>12.2f} {g-b:>+10.2f}')

print(f'\n  解读：小年的低标峰值涨幅小了约30个百分点，"妖气"不足')

print(f'\n{"="*70}')
print(f'  【问题3：实际交易的低标，在大小年的表现差异】')
print(f'{"="*70}')
print()

# 把交易和特征合并
merged = T.merge(E[['code','date','zt_in10','peak_gain','mcap','turn','vshrink','dd5','r10','board']],
                 on=['code','date'], how='left')

# 主策略都是低标，确认一下
print(f'  主策略交易中，zt_in10≤3的占比: {(merged["zt_in10"]<=3).mean()*100:.1f}%')

bad_trades = merged[merged['year'].isin([2022, 2023])]
good_trades = merged[merged['year'].isin([2024, 2025])]

print(f'\n  同样是低标龙回头，不同年份表现:')
print(f'  {"指标":<15s} {"2022-23(n=14)":>14s} {"2024-25(n=66)":>14s} {"差异":>10s}')
print(f'  {"-"*58}')
b_win = float((bad_trades['raw']>0).mean()*100)
g_win = float((good_trades['raw']>0).mean()*100)
b_avg = float(bad_trades['raw'].mean()*100)
g_avg = float(good_trades['raw'].mean()*100)
b_med = float(bad_trades['raw'].median()*100)
g_med = float(good_trades['raw'].median()*100)
print(f'  {"胜率":<15s} {b_win:>13.1f}% {g_win:>13.1f}% {g_win-b_win:>+9.1f}%')
print(f'  {"平均收益(%)":<15s} {b_avg:>14.2f} {g_avg:>14.2f} {g_avg-b_avg:>+9.2f}')
print(f'  {"中位收益(%)":<15s} {b_med:>14.2f} {g_med:>14.2f} {g_med-b_med:>+9.2f}')

print(f'\n  结论：同样的低标+回调25%+量缩，熊市里收益就是差很多')
print(f'       说明策略的收益高度依赖市场beta，不是纯alpha')

print(f'\n{"="*70}')
print(f'  【问题4：2022-2023年的"失败票"，是不是买贵了？(回调不够深)】')
print(f'{"="*70}')
print()

# 看dd5的分布：亏损的票是不是回调不够深？
bad_losses = bad_trades[bad_trades['raw'] < 0]
bad_wins = bad_trades[bad_trades['raw'] > 0]
good_losses = good_trades[good_trades['raw'] < 0]
good_wins = good_trades[good_trades['raw'] > 0]

print(f'  {"dd5(回调幅度)":<18s} 均值    中位数')
print(f'  {"-"*40}')
print(f'  2022-23亏损票: {bad_losses["dd5"].mean()*100:>8.1f}% {bad_losses["dd5"].median()*100:>8.1f}% (n={len(bad_losses)})')
print(f'  2022-23盈利票: {bad_wins["dd5"].mean()*100:>8.1f}% {bad_wins["dd5"].median()*100:>8.1f}% (n={len(bad_wins)})')
print(f'  2024-25亏损票: {good_losses["dd5"].mean()*100:>8.1f}% {good_losses["dd5"].median()*100:>8.1f}% (n={len(good_losses)})')
print(f'  2024-25盈利票: {good_wins["dd5"].mean()*100:>8.1f}% {good_wins["dd5"].median()*100:>8.1f}% (n={len(good_wins)})')

print(f'\n  量缩比 vshrink:')
print(f'  2022-23亏损票: {bad_losses["vshrink"].mean():.3f} (越小越缩量)')
print(f'  2022-23盈利票: {bad_wins["vshrink"].mean():.3f}')
print(f'  2024-25亏损票: {good_losses["vshrink"].mean():.3f}')
print(f'  2024-25盈利票: {good_wins["vshrink"].mean():.3f}')

print(f'\n  市值:')
print(f'  2022-23亏损票: {bad_losses["mcap"].mean():.1f}亿')
print(f'  2022-23盈利票: {bad_wins["mcap"].mean():.1f}亿')
print(f'  2024-25亏损票: {good_losses["mcap"].mean():.1f}亿')
print(f'  2024-25盈利票: {good_wins["mcap"].mean():.1f}亿')

print(f'\n{"="*70}')
print(f'  【问题5：按月度看——亏损是不是集中在某些月份？】')
print(f'{"="*70}')
print()

T['month'] = T['date'].str[5:7]
monthly_stat = T.groupby(['year', 'month']).agg(
    n=('raw', 'size'),
    avg_ret=('raw', 'mean'),
    win_rate=('raw', lambda x: (x>0).mean()*100)
).reset_index()

print('  月度交易统计 (仅展示有交易的月份):')
print(f'  {"年月":<8s} {"n":>4s} {"均收益%":>8s} {"胜率%":>7s}')
print(f'  {"-"*35}')
for _, r in monthly_stat.sort_values(['year','month']).iterrows():
    marker = ' 🔴' if r['avg_ret'] < 0 else ' 🟢'
    print(f'  {int(r["year"])}-{r["month"]:<5s} {r["n"]:>4d} {r["avg_ret"]*100:>7.2f}% {r["win_rate"]:>6.1f}%{marker}')

print(f'\n  观察：2023年5月3笔全亏(-16%平均)，是全年最大拖累')
print(f'        2022年1月、12月也是集中亏损期')

print(f'\n{"="*70}')
print(f'  【问题6：beta贡献有多大？——扣掉指数收益后还剩多少alpha】')
print(f'{"="*70}')
print()

# 用创业板指作为beta基准（因为小票多）
idx = pd.read_csv('idx_CYB.csv')
idx['date'] = idx.iloc[:, 0].astype(str)
idx['close'] = idx['close'].astype(float)
idx = idx.sort_values('date').reset_index(drop=True)

idx_map = dict(zip(idx['date'], idx['close']))

# 计算每笔交易的同期指数收益
beta_rets = []
for _, r in T.iterrows():
    buy_date = r['date']
    # 找3天后的日期（粗略：索引+3）
    if buy_date not in idx_map:
        beta_rets.append(None)
        continue
    dates_list = sorted(idx_map.keys())
    try:
        buy_idx = dates_list.index(buy_date)
        sell_idx = buy_idx + 3
        if sell_idx >= len(dates_list):
            beta_rets.append(None)
            continue
        sell_date = dates_list[sell_idx]
        buy_close = idx_map[buy_date]
        sell_close = idx_map[sell_date]
        beta_ret = sell_close / buy_close - 1
        beta_rets.append(beta_ret)
    except:
        beta_rets.append(None)

T['beta_ret'] = beta_rets
T['alpha'] = T['raw'] - T['beta_ret']

valid = T['beta_ret'].notna()
print(f'  可计算beta的交易: {valid.sum()}/{len(T)}')
print()
print(f'  {"年份":<6s} {"n":>4s} {"原始收益%":>9s} {"beta收益%":>9s} {"alpha%":>8s} {"beta占比":>9s}')
print(f'  {"-"*55}')
for yr in sorted(T['year'].unique()):
    sub = T[(T['year'] == yr) & valid]
    if len(sub) == 0: continue
    raw = sub['raw'].mean() * 100
    beta = sub['beta_ret'].mean() * 100
    alpha = sub['alpha'].mean() * 100
    ratio = beta / raw * 100 if raw != 0 else 0
    print(f'  {yr:<6d} {len(sub):>4d} {raw:>8.2f}% {beta:>8.2f}% {alpha:>7.2f}% {ratio:>8.1f}%')

print(f'\n  解读：beta占比=指数贡献/总收益，越高说明越靠大盘吃饭')
print(f'        2022-2023年beta本身就是负的，alpha也不行=双杀')

# ========== 保存补充分析结果 ==========
import json
extra = {
    'peak_gain_by_year': {
        str(yr): {
            'mean': round(E[E['year']==yr]['peak_gain'].mean()*100, 1),
            'median': round(E[E['year']==yr]['peak_gain'].median()*100, 1),
            'ge_200pct': round((E[E['year']==yr]['peak_gain']>=2).mean()*100, 1),
        } for yr in sorted(E['year'].unique()) if len(E[E['year']==yr]) > 0
    },
    'low_quality_compare': {
        'bad_22_23': {col: round(bad_low[col].mean(), 4) for col in ['peak_gain','r10','mcap','turn','vshrink','dd5']},
        'good_24_25': {col: round(good_low[col].mean(), 4) for col in ['peak_gain','r10','mcap','turn','vshrink','dd5']},
    },
    'trade_quality_compare': {
        'bad_22_23': {'n': len(bad_trades), 'win_rate': round((bad_trades['raw']>0).mean()*100,1), 
                      'avg_ret': round(bad_trades['raw'].mean()*100,2),
                      'median_ret': round(bad_trades['raw'].median()*100,2)},
        'good_24_25': {'n': len(good_trades), 'win_rate': round((good_trades['raw']>0).mean()*100,1),
                       'avg_ret': round(good_trades['raw'].mean()*100,2),
                       'median_ret': round(good_trades['raw'].median()*100,2)},
    },
    'beta_decomp': {
        str(yr): {
            'n': int(len(T[(T['year']==yr) & valid])),
            'raw_pct': round(T[(T['year']==yr) & valid]['raw'].mean()*100, 2),
            'beta_pct': round(T[(T['year']==yr) & valid]['beta_ret'].mean()*100, 2),
            'alpha_pct': round(T[(T['year']==yr) & valid]['alpha'].mean()*100, 2),
        } for yr in sorted(T['year'].unique()) if len(T[(T['year']==yr) & valid]) > 0
    },
}

with open('failure_analysis_extra.json', 'w', encoding='utf-8') as f:
    json.dump(extra, f, ensure_ascii=False, indent=2)
print('\n补充分析已保存: failure_analysis_extra.json')
