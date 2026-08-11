# -*- coding: utf-8 -*-
"""
研究2022-2023年龙回头策略失效原因
对比维度：
1. 信号数量（妖股池大小 = 情绪热度代理）
2. 连板高度分布（低标/中标/高标比例变化）
3. 收益分布（是不是亏钱的亏更多，还是赢的赚更少）
4. 板块分布（是否集中在某些弱板块）
5. 失败案例共性（回撤深度不够？量没缩？市值偏大/小？）
6. 市场beta（大盘环境对比）
"""
import pandas as pd
import numpy as np
import os, gzip, pickle
from collections import defaultdict

T = pd.read_csv('trades_lht_main.csv', dtype={'code': str})
T['date'] = T['date'].astype(str)
T = T.sort_values('date').reset_index(drop=True)

E = pd.read_csv('events_lht_final.csv', dtype={'code': str})
E['date'] = E['date'].astype(str)

print('='*70)
print('  【信号数量 / 情绪热度 对比】')
print('='*70)

# 各年入池段数（妖股池大小）
E['year'] = E['year'].astype(int)
yearly_pool = E.groupby('year').size()
yearly_trades = T.groupby('year').size()

print(f'  {"年份":<6s} {"入池段数":>8s} {"主规则交易":>10s} {"信号/入池比":>10s}')
print(f'  {"-"*45}')
for y in sorted(yearly_pool.index):
    n_pool = yearly_pool[y]
    n_trade = yearly_trades.get(y, 0)
    ratio = n_trade / n_pool * 100 if n_pool > 0 else 0
    print(f'  {y:<6d} {n_pool:>8d} {n_trade:>10d} {ratio:>9.1f}%')

print(f'\n  解读：2022-2023年入池段少 = 妖股少 = 情绪冷，信号自然也少')

print(f'\n{"="*70}')
print(f'  【连板高度分布 对比】')
print(f'{"="*70}')

# 各年zt_in10分布
E['zt_layer'] = pd.cut(E['zt_in10'], bins=[-1,3,6,10], labels=['低标≤3','中标4~6','高标≥7'])
zt_by_year = E.groupby(['year', 'zt_layer']).size().unstack(fill_value=0)
zt_pct = zt_by_year.div(zt_by_year.sum(axis=1), axis=0) * 100

print(f'  {"年份":<6s} {"低标数":>8s} {"中标数":>8s} {"高标数":>8s}  低标占比  中标占比  高标占比')
print(f'  {"-"*65}')
for y in sorted(zt_by_year.index):
    row = zt_by_year.loc[y]
    pct = zt_pct.loc[y]
    print(f'  {y:<6d} {row["低标≤3"]:>8d} {row["中标4~6"]:>8d} {row["高标≥7"]:>8d}   {pct["低标≤3"]:>5.1f}%   {pct["中标4~6"]:>5.1f}%   {pct["高标≥7"]:>5.1f}%')

print(f'\n  解读：看低标占比——主策略做的就是低标，低标多才有肉吃')

print(f'\n{"="*70}')
print(f'  【主规则交易收益分布 对比】')
print(f'{"="*70}')

def year_stats(yr):
    sub = T[T['year'] == yr]
    if len(sub) == 0:
        return None
    wins = sub[sub['raw'] > 0]
    losses = sub[sub['raw'] < 0]
    return {
        'n': len(sub),
        'win_rate': (sub['raw'] > 0).mean() * 100,
        'avg_ret': sub['raw'].mean() * 100,
        'avg_win': wins['raw'].mean() * 100 if len(wins) > 0 else 0,
        'avg_loss': losses['raw'].mean() * 100 if len(losses) > 0 else 0,
        'pf': abs(wins['raw'].sum() / losses['raw'].sum()) if len(losses) > 0 and losses['raw'].sum() != 0 else 99,
        'best': sub['raw'].max() * 100,
        'worst': sub['raw'].min() * 100,
        'median': sub['raw'].median() * 100,
    }

print(f'  {"年份":<6s} {"n":>4s} {"胜率%":>7s} {"均收益%":>8s} {"中位%":>7s} {"平均盈%":>8s} {"平均亏%":>8s} {"盈亏比":>7s} {"最佳%":>7s} {"最差%":>7s}')
print(f'  {"-"*80}')
for y in sorted(T['year'].unique()):
    s = year_stats(y)
    if s is None: continue
    print(f'  {y:<6d} {s["n"]:>4d} {s["win_rate"]:>6.1f}% {s["avg_ret"]:>7.2f}% {s["median"]:>6.2f}% {s["avg_win"]:>7.2f}% {s["avg_loss"]:>7.2f}% {s["pf"]:>6.2f} {s["best"]:>6.2f}% {s["worst"]:>6.2f}%')

print(f'\n  关键看：2022-2023是胜率低？还是盈亏比差？还是都差？')

# ========== 失败案例深挖 ==========
print(f'\n{"="*70}')
print(f'  【2022-2023年亏损交易明细（按亏损排序）】')
print(f'{"="*70}')

loss_22_23 = T[(T['year'].isin([2022, 2023])) & (T['raw'] < 0)].sort_values('raw')
print(f'  亏损笔数: {len(loss_22_23)}')
print(loss_22_23[['code','date','year','raw','net','mae','mfe']].to_string(index=False))

# 加载final表匹配更多特征
loss_with_feat = loss_22_23.merge(E[['code','date','turn','zt_in10','mcap','vshrink','dd5','peak_gain','board','r10']], 
                                    on=['code','date'], how='left')
# 可能date对不上，因为final表的date是信号日，trades的date也是信号日，应该能对上
print(f'\n  匹配到特征的: {loss_with_feat["turn"].notna().sum()} / {len(loss_with_feat)}')

if loss_with_feat['turn'].notna().sum() > 0:
    matched = loss_with_feat.dropna(subset=['turn'])
    print(f'\n  亏损票特征平均值:')
    print(f'    平均换手率: {matched["turn"].mean():.1f}%')
    print(f'    平均10日涨停数: {matched["zt_in10"].mean():.1f}次')
    print(f'    平均市值: {matched["mcap"].mean():.1f}亿')
    print(f'    平均量缩比: {matched["vshrink"].mean():.2f}')
    print(f'    平均回调幅度(dd5): {matched["dd5"].mean()*100:.1f}%')
    print(f'    平均峰值涨幅: {matched["peak_gain"].mean()*100:.1f}%')
    print(f'    板块分布:')
    print(matched['board'].value_counts())

# ========== 2024-2025大年对比 ==========
print(f'\n{"="*70}')
print(f'  【2024-2025大年 vs 2022-2023小年 特征对比】')
print(f'{"="*70}')

# 先把trades和final表的特征都匹配上
all_with_feat = T.merge(E[['code','date','turn','zt_in10','mcap','vshrink','dd5','peak_gain','board','r10']],
                        on=['code','date'], how='left')

print(f'  总匹配: {all_with_feat["turn"].notna().sum()} / {len(all_with_feat)}')

# 分组比较
baddie_years = all_with_feat[all_with_feat['year'].isin([2022, 2023]) & all_with_feat['turn'].notna()]
goodie_years = all_with_feat[all_with_feat['year'].isin([2024, 2025]) & all_with_feat['turn'].notna()]

print(f'\n  {"特征":<15s} {"2022-23":>10s} {"2024-25":>10s} {"差异":>10s}')
print(f'  {"-"*50}')
for col, label, pct in [
    ('turn', '换手率(%)', False),
    ('zt_in10', '10日涨停数', False),
    ('mcap', '市值(亿)', False),
    ('vshrink', '量缩比', False),
    ('dd5', '回调幅度(%)', True),
    ('peak_gain', '峰值涨幅(%)', True),
    ('r10', '10日涨幅倍数', False),
]:
    b = baddie_years[col].mean()
    g = goodie_years[col].mean()
    if pct:
        b *= 100; g *= 100
    diff = g - b
    print(f'  {label:<15s} {b:>10.2f} {g:>10.2f} {diff:>+10.2f}')

# 板块分布对比
print(f'\n  板块分布:')
print(f'  {"板块":<10s} {"2022-23占比":>10s} {"2024-25占比":>10s}')
for brd in ['主板', '创业板', '科创板']:
    b = (baddie_years['board'] == brd).mean() * 100
    g = (goodie_years['board'] == brd).mean() * 100
    print(f'  {brd:<10s} {b:>9.1f}% {g:>9.1f}%')

# ========== 市场beta：指数表现 ==========
print(f'\n{"="*70}')
print(f'  【市场环境：指数年度涨跌幅】')
print(f'{"="*70}')

idx_files = {
    '上证指数': 'idx_SH.csv',  # 可能没有，先看看
}
# 用cache5y里的指数？
# 先看看有哪些指数文件
import glob
idx_files = glob.glob('idx_*.csv')
print(f'  可用指数文件: {idx_files}')

for fp in idx_files:
    try:
        idx = pd.read_csv(fp)
        name = fp.replace('idx_', '').replace('.csv', '')
        idx['date'] = idx.iloc[:, 0].astype(str)
        idx['close'] = idx['close'].astype(float) if 'close' in idx.columns else idx.iloc[:, 1].astype(float)
        for yr in [2022, 2023, 2024, 2025]:
            sub = idx[idx['date'].str.startswith(str(yr))]
            if len(sub) > 1:
                ret = (sub['close'].iloc[-1] / sub['close'].iloc[0] - 1) * 100
                print(f'  {name} {yr}: {ret:+.1f}%')
    except Exception as e:
        pass

# ========== 失败票的共性模式 ==========
# 看它们的MAE（最大浮亏）和MFE（最大浮盈）——是买了就一路跌，还是冲高回落？
print(f'\n{"="*70}')
print(f'  【亏损票的MAE/MFE分析】 买完后是"直接A杀"还是"冲高回落"？')
print(f'{"="*70}')

for yr_label, sub in [('2022-23亏损', baddie_years[baddie_years['raw'] < 0]), 
                      ('2024-25亏损', goodie_years[goodie_years['raw'] < 0]),
                      ('2022-23盈利', baddie_years[baddie_years['raw'] > 0]),
                      ('2024-25盈利', goodie_years[goodie_years['raw'] > 0])]:
    if len(sub) == 0: continue
    mae = sub['mae'].mean() * 100
    mfe = sub['mfe'].mean() * 100
    # mfe / |mae| 越大，说明冲高越多
    ratio = mfe / abs(mae) if mae != 0 else 99
    print(f'  {yr_label}: n={len(sub)}, 平均MAE={mae:.1f}%, 平均MFE={mfe:.1f}%, MFE/|MAE|={ratio:.2f}')

print(f'\n  解读：如果MFE大但最终亏损，说明"有机会赚但没走"，是卖点问题；')
print(f'        如果MAE很大且MFE很小，说明"买完就跌"，是买点/选股问题。')

# ========== 2023年为什么最差？ ==========
print(f'\n{"="*70}')
print(f'  【2023年深度：逐笔交易】')
print(f'{"="*70}')

t2023 = T[T['year'] == 2023].sort_values('date')
print(f'  2023年共{len(t2023)}笔交易:')
print(t2023[['code','date','raw','net','mae','mfe']].to_string(index=False))

# 计算2023年的净值走势
daily_2023 = t2023.groupby('date')['raw'].mean()
cum = (1 + daily_2023).cumprod()
print(f'\n  2023年累计收益: {(cum.iloc[-1]-1)*100:.1f}%')
print(f'  最大回撤(交易点): {(cum / cum.cummax() - 1).min()*100:.1f}%')

# ========== 保存分析数据 ==========
import json
analysis = {
    'yearly': {
        str(y): year_stats(y) for y in sorted(T['year'].unique()) if year_stats(y)
    },
    'pool_size': {str(y): int(yearly_pool[y]) for y in yearly_pool.index},
    'zt_distribution': {
        str(y): {
            'low': int(zt_by_year.loc[y, '低标≤3']) if y in zt_by_year.index else 0,
            'mid': int(zt_by_year.loc[y, '中标4~6']) if y in zt_by_year.index else 0,
            'high': int(zt_by_year.loc[y, '高标≥7']) if y in zt_by_year.index else 0,
        } for y in sorted(zt_by_year.index)
    },
    'feature_compare': {
        'bad_years': {col: round(baddie_years[col].mean(), 4) for col in ['turn','zt_in10','mcap','vshrink','dd5','peak_gain','r10']},
        'good_years': {col: round(goodie_years[col].mean(), 4) for col in ['turn','zt_in10','mcap','vshrink','dd5','peak_gain','r10']},
    },
}

with open('failure_analysis_22_23.json', 'w', encoding='utf-8') as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)
print(f'\n分析数据已保存: failure_analysis_22_23.json')
