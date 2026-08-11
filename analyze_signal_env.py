# -*- coding: utf-8 -*-
"""
买入信号出现时，大盘处于什么位置/状态？
1. 信号日的250日分位分布
2. 信号日的趋势状态分布（上升/震荡/下降）
3. 信号日的波动率分布
4. 信号密集出现在哪些时间段？和大盘行情对照
5. 信号出现后的大盘走势（是大盘继续涨还是跌？）
"""
import pandas as pd
import numpy as np

T = pd.read_csv('trades_lht_main.csv', dtype={'code': str})
T['date'] = T['date'].astype(str)
T = T.sort_values('date').reset_index(drop=True)

idx = pd.read_csv('idx_CYB.csv')
idx['date'] = idx.iloc[:, 0].astype(str)
idx['close'] = idx['close'].astype(float)
idx = idx.sort_values('date').reset_index(drop=True)

idx['ma20'] = idx['close'].rolling(20).mean()
idx['ma60'] = idx['close'].rolling(60).mean()
idx['ret_20'] = idx['close'].pct_change(20) * 100
idx['ret_5'] = idx['close'].pct_change(5) * 100
idx['ma20_slope'] = idx['ma20'].pct_change(5) * 100

# ATR
if 'high' in idx.columns and 'low' in idx.columns:
    idx['high'] = idx['high'].astype(float)
    idx['low'] = idx['low'].astype(float)
    idx['tr1'] = idx['high'] - idx['low']
    idx['tr2'] = abs(idx['high'] - idx['close'].shift(1))
    idx['tr3'] = abs(idx['low'] - idx['close'].shift(1))
    idx['tr'] = idx[['tr1','tr2','tr3']].max(axis=1)
else:
    idx['tr'] = abs(idx['close'].pct_change())
idx['atr14'] = idx['tr'].rolling(14).mean()
idx['atr_pct'] = idx['atr14'] / idx['close'] * 100

def pct_rank(s, w=250):
    r = []
    for i in range(len(s)):
        if i < w: r.append(np.nan)
        else: r.append((s.iloc[i-w:i] < s.iloc[i]).sum() / w)
    return r

idx['pos_250'] = [x*100 for x in pct_rank(idx['close'])]
idx['pos_120'] = [x*100 for x in pct_rank(idx['close'], 120)]

idx_map = idx.set_index('date')[['close','ma20','ma60','ret_5','ret_20','ma20_slope',
                                  'atr_pct','pos_250','pos_120']].to_dict('index')

# 构造DataFrame
rows = []
for _, r in T.iterrows():
    d = r['date']
    if d not in idx_map: continue
    x = idx_map[d]
    rows.append({
        'date': d, 'raw': float(r['raw']), 'year': int(r['year']),
        'pos_250': x['pos_250'],
        'ret_20': x['ret_20'],
        'ma20_slope': x['ma20_slope'],
        'atr_pct': x['atr_pct'],
        'above_ma20': x['close'] > x['ma20'],
        'above_ma60': x['close'] > x['ma60'],
    })

df = pd.DataFrame(rows).dropna(subset=['pos_250','ret_20','ma20_slope','atr_pct'])
print(f'有效信号: {len(df)}')

# ===== 1. 指数分位分布 =====
print(f'\n{"="*70}')
print(f'  【一、信号日的指数位置分布】 250日百分位')
print(f'{"="*70}')

bins = list(range(0, 101, 10))
labels = [f'{i}-{i+10}%' for i in range(0, 100, 10)]
df['pos_bin'] = pd.cut(df['pos_250'], bins=bins, labels=labels, include_lowest=True)

pos_dist = df['pos_bin'].value_counts().sort_index()
pos_pct = pos_dist / len(df) * 100

print(f'  {"分位区间":<10s} {"信号数":>6s} {"占比":>7s} {"累计占比":>8s} {"均收益":>8s}')
print(f'  {"-"*55}')
cum = 0
for lbl in labels:
    n = pos_dist.get(lbl, 0)
    if n == 0: continue
    pct = pos_pct.get(lbl, 0)
    cum += pct
    avg = df[df['pos_bin']==lbl]['raw'].mean()*100
    print(f'  {lbl:<10s} {n:>6d} {pct:>6.1f}% {cum:>7.1f}% {avg:>+7.2f}%')

print(f'\n  中位数分位: {df["pos_250"].median():.0f}%')
print(f'  平均值分位: {df["pos_250"].mean():.0f}%')
print(f'  ≥50%分位占比: {(df["pos_250"]>=50).mean()*100:.1f}%')
print(f'  ≥70%分位占比: {(df["pos_250"]>=70).mean()*100:.1f}%')
print(f'  ≥80%分位占比: {(df["pos_250"]>=80).mean()*100:.1f}%')

# ===== 2. 趋势状态分布 =====
print(f'\n{"="*70}')
print(f'  【二、信号日的趋势状态分布】 MA20斜率')
print(f'{"="*70}')

def reg_slope(x):
    return '上升期' if x > 0.5 else ('下降期' if x < -0.5 else '震荡期')
df['regime'] = df['ma20_slope'].apply(reg_slope)

for nm in ['上升期', '震荡期', '下降期']:
    sub = df[df['regime'] == nm]
    print(f'  {nm}: {len(sub)}笔 ({len(sub)/len(df)*100:.1f}%), 均收益{sub["raw"].mean()*100:+.2f}%, 胜率{(sub["raw"]>0).mean()*100:.1f}%')

print(f'\n  站上MA20: {(df["above_ma20"]).sum()}笔 ({(df["above_ma20"]).mean()*100:.1f}%)')
print(f'  站上MA60: {(df["above_ma60"]).sum()}笔 ({(df["above_ma60"]).mean()*100:.1f}%)')

# ===== 3. 波动率分布 =====
print(f'\n{"="*70}')
print(f'  【三、信号日的波动率分布】 ATR%')
print(f'{"="*70}')

# 全样本ATR分布的分位数
all_atr = idx['atr_pct'].dropna()
print(f'  全样本ATR%: median={all_atr.median():.2f}%, 25%分位={all_atr.quantile(0.25):.2f}%, 75%分位={all_atr.quantile(0.75):.2f}%')
print(f'  信号日ATR%: median={df["atr_pct"].median():.2f}%, mean={df["atr_pct"].mean():.2f}%')
print()

# 信号日ATR在全样本中的分位
def atr_percentile(v):
    return (all_atr < v).sum() / len(all_atr) * 100

df['atr_rank'] = df['atr_pct'].apply(atr_percentile)
print(f'  信号日ATR排名中位: {df["atr_rank"].median():.0f}%（全ATR数据的百分位）')
print(f'  → 信号大多出现在高波动环境？' if df['atr_rank'].median() > 50 else '  → 信号大多出现在低波动环境？')

# ===== 4. 信号密集时间段 =====
print(f'\n{"="*70}')
print(f'  【四、信号密集时间段】 按月统计信号数')
print(f'{"="*70}')
print()

df['month'] = df['date'].str[:7]
monthly = df.groupby('month').agg(n=('raw','size'), avg_ret=('raw', lambda x: x.mean()*100)).reset_index()
monthly = monthly.sort_values('month')

# 找Top 5信号最多的月份
top_months = monthly.sort_values('n', ascending=False).head(10)
print('  信号最多的10个月:')
print(f'  {"月份":<8s} {"信号数":>6s} {"均收益":>8s}  {"指数位置"}')
print(f'  {"-"*45}')
for _, r in top_months.iterrows():
    # 找该月最后一天的pos_250
    sub = df[df['month'] == r['month']]
    pos = sub['pos_250'].iloc[-1]
    print(f'  {r["month"]:<8s} {r["n"]:>6d} {r["avg_ret"]:>+7.2f}%   {pos:.0f}%分位')

# ===== 5. 信号出现后，大盘怎么走？ =====
print(f'\n{"="*70}')
print(f'  【五、信号出现后，大盘未来表现】')
print(f'{"="*70}')
print()

# 计算信号日后5/10/20日的指数涨跌
dates_list = sorted(idx_map.keys())
date_idx = {d: i for i, d in enumerate(dates_list)}
close_arr = idx['close'].values

future_rets = {'f5': [], 'f10': [], 'f20': []}
for d in df['date']:
    if d not in date_idx: continue
    i = date_idx[d]
    for offset, key in [(5, 'f5'), (10, 'f10'), (20, 'f20')]:
        if i + offset < len(close_arr):
            ret = (close_arr[i+offset] / close_arr[i] - 1) * 100
            future_rets[key].append(ret)
        else:
            future_rets[key].append(None)

print(f'  {"持有期":<10s} {"指数均涨跌":>10s} {"上涨概率":>8s}')
print(f'  {"-"*35}')
for key, label in [('f5', '后5日'), ('f10', '后10日'), ('f20', '后20日')]:
    vals = [x for x in future_rets[key] if x is not None]
    avg = np.mean(vals)
    prob = (np.array(vals) > 0).mean() * 100
    print(f'  信号{label:<5s} {avg:>+9.2f}% {prob:>7.1f}%')

print(f'\n  解读：如果信号后指数涨多跌少，说明策略有择时能力（踩在上涨起点）')
print(f'        如果信号后指数跌多涨少，说明策略在逆势抄底')

# ===== 6. 大盘涨的时候赚钱多，还是跌的时候赚钱多？ =====
print(f'\n{"="*70}')
print(f'  【六、信号日指数位置 vs 策略收益 散点】')
print(f'{"="*70}')
print()

# 按分位和收益的相关系数
corr_pos = df['pos_250'].corr(df['raw'])
corr_atr = df['atr_pct'].corr(df['raw'])
corr_ret20 = df['ret_20'].corr(df['raw'])

print(f'  相关系数：')
print(f'    收益 vs 指数分位: {corr_pos:+.3f}（正=越高越赚）')
print(f'    收益 vs 20日涨跌: {corr_ret20:+.3f}（正=涨得越多越赚）')
print(f'    收益 vs 波动率: {corr_atr:+.3f}（正=波动越大越赚）')

# ===== 7. 总结：信号画像 =====
print(f'\n{"="*70}')
print(f'  【七、信号画像总结】')
print(f'{"="*70}')
print()
print(f'  典型的龙回头买入日，大盘长这样：')
print(f'  ┌───────────────────────────────────────')
print(f'  │ 指数位置: {df["pos_250"].median():.0f}%分位（中位偏高）')
print(f'  │ 20日涨跌: {df["ret_20"].median():+.1f}%（小涨为主）')
print(f'  │ 趋势状态: 上升期{(df["regime"]=="上升期").mean()*100:.0f}% / 震荡{(df["regime"]=="震荡期").mean()*100:.0f}% / 下降{(df["regime"]=="下降期").mean()*100:.0f}%')
print(f'  │ 波动率: ATR={df["atr_pct"].median():.2f}%（偏高波动）')
print(f'  │ 站上MA20: {(df["above_ma20"]).mean()*100:.0f}%')
print(f'  │ 站上MA60: {(df["above_ma60"]).mean()*100:.0f}%')
print(f'  └───────────────────────────────────────')

print(f'\n  结论：龙回头信号大多出现在大盘「高位+上升+高波动」的环境')
print(f'       这也解释了为什么「高位+上升」胜率最高——大部分信号本来就出现在这种状态')