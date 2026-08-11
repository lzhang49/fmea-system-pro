# -*- coding: utf-8 -*-
"""
验证：震荡期 vs 上升期 vs 下降期，哪个成功率高？
三种划分方法 + 二维交叉 + 六状态
"""
import pandas as pd
import numpy as np

T = pd.read_csv('trades_lht_main.csv', dtype={'code': str})
T['date'] = T['date'].astype(str)
T = T.sort_values('date').reset_index(drop=True)

idx = pd.read_csv('idx_CYB.csv')
idx['date'] = idx.iloc[:, 0].astype(str)
idx['close'] = idx['close'].astype(float)
if 'high' in idx.columns: idx['high'] = idx['high'].astype(float)
if 'low' in idx.columns: idx['low'] = idx['low'].astype(float)
idx = idx.sort_values('date').reset_index(drop=True)

idx['ma20'] = idx['close'].rolling(20).mean()
idx['ret_20'] = idx['close'].pct_change(20) * 100
idx['ma20_slope'] = idx['ma20'].pct_change(5) * 100  # MA20的5日变化率

# ATR波动率
if 'high' in idx.columns and 'low' in idx.columns:
    idx['tr1'] = idx['high'] - idx['low']
    idx['tr2'] = abs(idx['high'] - idx['close'].shift(1))
    idx['tr3'] = abs(idx['low'] - idx['close'].shift(1))
    idx['tr'] = idx[['tr1','tr2','tr3']].max(axis=1)
else:
    idx['tr'] = abs(idx['close'].pct_change())
idx['atr14'] = idx['tr'].rolling(14).mean()
idx['atr_pct'] = idx['atr14'] / idx['close'] * 100

# 250日分位
def pct_rank(s, w=250):
    r = []
    for i in range(len(s)):
        if i < w: r.append(np.nan)
        else: r.append((s.iloc[i-w:i] < s.iloc[i]).sum() / w)
    return r
idx['pos_250'] = [x * 100 for x in pct_rank(idx['close'])]

idx_map = idx.set_index('date')[['close','ma20','ma20_slope','ret_20','atr_pct','pos_250']].to_dict('index')

envs = []
for _, r in T.iterrows():
    d = r['date']
    if d not in idx_map: continue
    x = idx_map[d]
    envs.append({
        'date': d, 'raw': float(r['raw']), 'year': int(r['year']),
        'ret_20': x['ret_20'], 'ma20_slope': x['ma20_slope'],
        'atr_pct': x['atr_pct'], 'pos_250': x['pos_250'],
    })

df = pd.DataFrame(envs).dropna(subset=['ret_20','ma20_slope','atr_pct','pos_250'])
print(f'有效交易: {len(df)}')

# ===== 方法1：20日涨跌 =====
print(f'\n{"="*75}')
print(f'  【方法1：20日涨跌幅】 上升(>+5%) / 震荡(-5%~+5%) / 下降(<-5%)')
print(f'{"="*75}')

def reg1(x):
    return '上升期' if x > 5 else ('下降期' if x < -5 else '震荡期')
df['r1'] = df['ret_20'].apply(reg1)

g = df.groupby('r1').agg(
    n=('raw','size'), avg=('raw', lambda x: x.mean()*100),
    win=('raw', lambda x: (x>0).mean()*100),
    med=('raw', lambda x: x.median()*100),
).reindex(['上升期','震荡期','下降期'])

print(f'  {"状态":<10s} {"n":>5s} {"占比":>7s} {"均收益":>8s} {"胜率":>7s} {"中位":>8s}')
print(f'  {"-"*55}')
for nm in ['上升期','震荡期','下降期']:
    r = g.loc[nm]
    print(f'  {nm:<10s} {r["n"]:>5.0f} {r["n"]/len(df)*100:>6.1f}% {r["avg"]:>+7.2f}% {r["win"]:>6.1f}% {r["med"]:>+7.2f}%')

# ===== 方法2：MA20斜率 =====
print(f'\n{"="*75}')
print(f'  【方法2：MA20斜率】 上升(>+0.5%) / 震荡(-0.5%~0.5%) / 下降(<-0.5%)')
print(f'{"="*75}')

def reg2(x):
    return '上升期' if x > 0.5 else ('下降期' if x < -0.5 else '震荡期')
df['r2'] = df['ma20_slope'].apply(reg2)

g2 = df.groupby('r2').agg(
    n=('raw','size'), avg=('raw', lambda x: x.mean()*100),
    win=('raw', lambda x: (x>0).mean()*100),
    med=('raw', lambda x: x.median()*100),
).reindex(['上升期','震荡期','下降期'])

print(f'  {"状态":<10s} {"n":>5s} {"占比":>7s} {"均收益":>8s} {"胜率":>7s} {"中位":>8s}')
print(f'  {"-"*55}')
for nm in ['上升期','震荡期','下降期']:
    r = g2.loc[nm]
    print(f'  {nm:<10s} {r["n"]:>5.0f} {r["n"]/len(df)*100:>6.1f}% {r["avg"]:>+7.2f}% {r["win"]:>6.1f}% {r["med"]:>+7.2f}%')

# ===== 方法3：波动率 =====
print(f'\n{"="*75}')
print(f'  【方法3：波动率(ATR%)】 高波动(震荡) vs 低波动(趋势)')
print(f'{"="*75}')

med = df['atr_pct'].median()
print(f'  ATR%中位值 = {med:.2f}%')
print()
df['high_vol'] = df['atr_pct'] > med

g3 = df.groupby('high_vol').agg(
    n=('raw','size'), avg=('raw', lambda x: x.mean()*100),
    win=('raw', lambda x: (x>0).mean()*100),
    med_ret=('raw', lambda x: x.median()*100),
)
print(f'  {"状态":<14s} {"n":>5s} {"均收益":>8s} {"胜率":>7s} {"中位":>8s}')
print(f'  {"-"*50}')
for is_hv, label in [(True, '高波动(震荡)'), (False, '低波动(趋势)')]:
    r = g3.loc[is_hv]
    print(f'  {label:<14s} {r["n"]:>5.0f} {r["avg"]:>+7.2f}% {r["win"]:>6.1f}% {r["med_ret"]:>+7.2f}%')

# ===== 方法4：二维矩阵 =====
print(f'\n{"="*75}')
print(f'  【方法4：二维矩阵】 趋势(MA20斜率) × 波动率')
print(f'{"="*75}')
print()

pvt_ret = df.pivot_table(index='r2', columns='high_vol', values='raw', aggfunc=lambda x: x.mean()*100)
pvt_win = df.pivot_table(index='r2', columns='high_vol', values='raw', aggfunc=lambda x: (x>0).mean()*100)
pvt_n = df.pivot_table(index='r2', columns='high_vol', values='raw', aggfunc='count')

print('  平均收益矩阵 (列: 低波动 / 高波动):')
print(pvt_ret.round(2))
print()
print('  胜率矩阵:')
print(pvt_win.round(1))
print()
print('  样本数矩阵:')
print(pvt_n.astype(int))

# ===== 方法5：6状态（位置+方向）=====
print(f'\n{"="*75}')
print(f'  【方法5：六状态】 指数位置高低 × MA20斜率方向')
print(f'{"="*75}')

def reg6(r):
    p, s = r['pos_250'], r['ma20_slope']
    if p >= 70 and s > 0: return '高位上升(主升)'
    if p >= 70 and s <= 0: return '高位震荡(筑顶)'
    if 30 <= p < 70 and s > 0: return '中位上升(反弹)'
    if 30 <= p < 70 and s <= 0: return '中位震荡(整理)'
    if p < 30 and s > 0: return '低位上升(见底)'
    return '低位下降(寻底)'

df['r6'] = df.apply(reg6, axis=1)
g6 = df.groupby('r6').agg(
    n=('raw','size'), avg=('raw', lambda x: x.mean()*100),
    win=('raw', lambda x: (x>0).mean()*100),
    med=('raw', lambda x: x.median()*100),
).sort_values('avg', ascending=False)

print(f'\n  {"状态":<16s} {"n":>4s} {"均收益":>8s} {"胜率":>7s} {"中位":>8s}')
print(f'  {"-"*55}')
for nm, r in g6.iterrows():
    m = ' ⭐' if r['win'] >= 80 else (' ⚠️' if r['win'] < 40 else '')
    print(f'  {nm:<16s} {r["n"]:>4.0f} {r["avg"]:>+7.2f}% {r["win"]:>6.1f}% {r["med"]:>+7.2f}%{m}')

# ===== 各状态累计收益 =====
print(f'\n{"="*75}')
print(f'  【各状态等权累计收益对比】')
print(f'{"="*75}')
print()

lists = {
    '全部': df,
    'MA20上升期': df[df['r2']=='上升期'],
    'MA20震荡期': df[df['r2']=='震荡期'],
    'MA20下降期': df[df['r2']=='下降期'],
    '高波动': df[df['high_vol']==True],
    '低波动': df[df['high_vol']==False],
    '高位上升': df[df['r6']=='高位上升(主升)'],
    '中位震荡': df[df['r6']=='中位震荡(整理)'],
    '低位下降': df[df['r6']=='低位下降(寻底)'],
}

print(f'  {"状态":<14s} {"n":>4s} {"累计收益":>10s} {"胜率":>7s} {"盈亏比":>7s}')
print(f'  {"-"*50}')
for nm, sub in lists.items():
    if len(sub) < 2: continue
    cum = (1 + sub['raw']).prod() - 1
    win = (sub['raw']>0).mean() * 100
    wins = sub[sub['raw']>0]['raw']
    losses = abs(sub[sub['raw']<0]['raw'])
    pf = wins.sum() / losses.sum() if len(losses) > 0 else 99
    print(f'  {nm:<14s} {len(sub):>4d} {cum*100:>+9.1f}% {win:>6.1f}% {pf:>6.2f}')

print(f'\n💡 结论: 上升期 > 震荡期 > 下降期 吗？ 看数据！')

import json
data = {
    'method1': {nm: {'n': int(g.loc[nm,'n']), 'avg': round(g.loc[nm,'avg'],2), 'win': round(g.loc[nm,'win'],1)}
                for nm in ['上升期','震荡期','下降期'] if nm in g.index},
    'method2': {nm: {'n': int(g2.loc[nm,'n']), 'avg': round(g2.loc[nm,'avg'],2), 'win': round(g2.loc[nm,'win'],1)}
                for nm in ['上升期','震荡期','下降期'] if nm in g2.index},
    'method3': {
        'high_vol': {'n': int(g3.loc[True,'n']), 'avg': round(g3.loc[True,'avg'],2), 'win': round(g3.loc[True,'win'],1)},
        'low_vol': {'n': int(g3.loc[False,'n']), 'avg': round(g3.loc[False,'avg'],2), 'win': round(g3.loc[False,'win'],1)},
    },
    'method6': [{
        'name': nm, 'n': int(r['n']), 'avg': round(r['avg'],2), 'win': round(r['win'],1),
    } for nm, r in g6.iterrows()],
}
with open('regime_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f'\n数据已保存: regime_data.json')