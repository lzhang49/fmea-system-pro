# -*- coding: utf-8 -*-
"""
指数区间增强分析
1. 10%分位精细分层
2. 多种指数过滤器效果对比
3. 过滤前后资金曲线对比
"""
import pandas as pd
import numpy as np
import os, gzip, pickle
from collections import defaultdict

T = pd.read_csv('trades_lht_main.csv', dtype={'code': str})
T['date'] = T['date'].astype(str)
T = T.sort_values('date').reset_index(drop=True)

idx = pd.read_csv('idx_CYB.csv')
idx['date'] = idx.iloc[:, 0].astype(str)
idx['close'] = idx['close'].astype(float)
idx = idx.sort_values('date').reset_index(drop=True)

idx['ma5'] = idx['close'].rolling(5).mean()
idx['ma10'] = idx['close'].rolling(10).mean()
idx['ma20'] = idx['close'].rolling(20).mean()
idx['ma60'] = idx['close'].rolling(60).mean()
idx['ma120'] = idx['close'].rolling(120).mean()

def pct_rank(series, window=250):
    result = []
    for i in range(len(series)):
        if i < window:
            result.append(np.nan)
        else:
            w = series.iloc[i-window:i]
            result.append((w < series.iloc[i]).sum() / window)
    return result

idx['pos_250'] = pct_rank(idx['close'], 250)
idx['pos_120'] = pct_rank(idx['close'], 120)
idx['ret_20'] = idx['close'].pct_change(20)
idx['ret_60'] = idx['close'].pct_change(60)

idx_map = idx.set_index('date')[['close','ma5','ma10','ma20','ma60','ma120',
                                  'pos_250','pos_120','ret_20','ret_60']].to_dict('index')

envs = []
for _, r in T.iterrows():
    d = r['date']
    if d not in idx_map:
        continue
    x = idx_map[d]
    envs.append({
        'date': d, 'year': int(r['year']), 'raw': float(r['raw']),
        'pos_250': x['pos_250']*100 if not np.isnan(x['pos_250']) else None,
        'pos_120': x['pos_120']*100 if not np.isnan(x['pos_120']) else None,
        'above_ma5': x['close'] > x['ma5'] if not np.isnan(x['ma5']) else None,
        'above_ma10': x['close'] > x['ma10'] if not np.isnan(x['ma10']) else None,
        'above_ma20': x['close'] > x['ma20'] if not np.isnan(x['ma20']) else None,
        'above_ma60': x['close'] > x['ma60'] if not np.isnan(x['ma60']) else None,
        'above_ma120': x['close'] > x['ma120'] if not np.isnan(x['ma120']) else None,
        'ret_20': x['ret_20']*100 if not np.isnan(x['ret_20']) else None,
        'ret_60': x['ret_60']*100 if not np.isnan(x['ret_60']) else None,
    })

df = pd.DataFrame(envs).dropna(subset=['pos_250'])
print(f'有效交易: {len(df)}')

# ===== 一、10%分位精细分层 =====
print(f'\n{"="*70}')
print(f'  【一、10%分位精细分层】 创业板指250日分位')
print(f'{"="*70}')

bins = list(range(0, 101, 10))
labels = [f'{i}-{i+10}%' for i in range(0, 100, 10)]
df['pos_bin'] = pd.cut(df['pos_250'], bins=bins, labels=labels, include_lowest=True)

print(f'  {"分位":<10s} {"n":>4s} {"均收益%":>9s} {"胜率%":>7s} {"中位%":>8s} {"盈亏":>7s}')
print(f'  {"-"*55}')
for lbl in labels:
    sub = df[df['pos_bin']==lbl]
    n = len(sub)
    if n == 0: continue
    avg = sub['raw'].mean()*100
    win = (sub['raw']>0).mean()*100
    med = sub['raw'].median()*100
    nw = int((sub['raw']>0).sum())
    nl = n - nw
    print(f'  {lbl:<10s} {n:>4d} {avg:>+8.2f}% {win:>6.1f}% {med:>+7.2f}%  {nw}/{nl}')

# ===== 二、MA均线过滤器 =====
print(f'\n{"="*70}')
print(f'  【二、MA均线过滤器】 站上均线才做')
print(f'{"="*70}')

filters = [
    ('不过滤', lambda d: [True]*len(d)),
    ('站上MA5', lambda d: d['above_ma5']==True),
    ('站上MA10', lambda d: d['above_ma10']==True),
    ('站上MA20', lambda d: d['above_ma20']==True),
    ('站上MA60', lambda d: d['above_ma60']==True),
    ('站上MA120', lambda d: d['above_ma120']==True),
]

print(f'  {"过滤器":<12s} {"n":>5s} {"通过率":>7s} {"均收益%":>9s} {"胜率%":>7s} {"提升":>7s}')
print(f'  {"-"*58}')
baseline = df['raw'].mean()*100
for name, fn in filters:
    sub = df[fn(df)]
    n = len(sub)
    if n < 2: continue
    avg = sub['raw'].mean()*100
    win = (sub['raw']>0).mean()*100
    rate = n/len(df)*100
    lift = avg - baseline
    print(f'  {name:<12s} {n:>5d} {rate:>6.1f}% {avg:>+8.2f}% {win:>6.1f}% {lift:>+6.2f}%')

# ===== 三、分位阈值 =====
print(f'\n{"="*70}')
print(f'  【三、分位阈值过滤】 ≥X%分位才做')
print(f'{"="*70}')

print(f'  {"阈值":<10s} {"n":>5s} {"通过率":>7s} {"均收益%":>9s} {"胜率%":>7s} {"提升":>7s}')
print(f'  {"-"*58}')
for th in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    sub = df[df['pos_250'] >= th]
    n = len(sub)
    if n < 2: continue
    avg = sub['raw'].mean()*100
    win = (sub['raw']>0).mean()*100
    rate = n/len(df)*100
    lift = avg - baseline
    print(f'  ≥{th}%分位    {n:>5d} {rate:>6.1f}% {avg:>+8.2f}% {win:>6.1f}% {lift:>+6.2f}%')

# ===== 四、20日涨跌动量 =====
print(f'\n{"="*70}')
print(f'  【四、20日涨跌动量分组】')
print(f'{"="*70}')

df['ret20_bin'] = pd.cut(df['ret_20'], bins=[-100,-10,-5,0,5,10,100],
                          labels=['<-10%','-10~-5%','-5~0%','0~5%','5~10%','>10%'])
print(f'  {"20日涨跌":<10s} {"n":>4s} {"均收益%":>9s} {"胜率%":>7s}')
print(f'  {"-"*35}')
for lbl in ['<-10%','-10~-5%','-5~0%','0~5%','5~10%','>10%']:
    sub = df[df['ret20_bin']==lbl]
    if len(sub) == 0: continue
    print(f'  {lbl:<10s} {len(sub):>4d} {sub["raw"].mean()*100:>+8.2f}% {(sub["raw"]>0).mean()*100:>6.1f}%')

# ===== 五、组合过滤器 =====
print(f'\n{"="*70}')
print(f'  【五、组合过滤器效果】')
print(f'{"="*70}')

combos = [
    ('不过滤', lambda d: [True]*len(d)),
    ('站上MA20', lambda d: d['above_ma20']==True),
    ('站上MA60', lambda d: d['above_ma60']==True),
    ('≥50%分位', lambda d: d['pos_250']>=50),
    ('MA20+≥30%', lambda d: (d['above_ma20']==True)&(d['pos_250']>=30)),
    ('MA20+≥50%', lambda d: (d['above_ma20']==True)&(d['pos_250']>=50)),
    ('MA60+≥40%', lambda d: (d['above_ma60']==True)&(d['pos_250']>=40)),
    ('MA20+ret20>0', lambda d: (d['above_ma20']==True)&(d['ret_20']>0)),
]

print(f'  {"组合":<16s} {"n":>5s} {"通过率":>7s} {"均收益%":>9s} {"胜率%":>7s} {"提升":>7s}')
print(f'  {"-"*62}')
for name, fn in combos:
    sub = df[fn(df)]
    n = len(sub)
    if n < 3: continue
    avg = sub['raw'].mean()*100
    win = (sub['raw']>0).mean()*100
    rate = n/len(df)*100
    lift = avg - baseline
    print(f'  {name:<16s} {n:>5d} {rate:>6.1f}% {avg:>+8.2f}% {win:>6.1f}% {lift:>+6.2f}%')

# ===== 六、过滤后资金模拟 =====
print(f'\n{"="*70}')
print(f'  【六、过滤后 · 10万本金资金曲线】')
print(f'{"="*70}')

sample_file = os.path.join('cache5y', '000001.pkl')
with gzip.open(sample_file, 'rb') as f:
    sample_df = pickle.load(f).reset_index(drop=True)
cal_dates = sample_df['date'].astype(str).tolist()
INIT_CAP = 100000
FEE = 0.0016

def simulate(filtered_df):
    daily_trades = defaultdict(list)
    for _, r in filtered_df.iterrows():
        daily_trades[r['date']].append(r['raw'])
    cash = INIT_CAP
    holdings = []
    eq = []
    for i, date in enumerate(cal_dates):
        new_h = []
        for h in holdings:
            if h[0] <= i: cash += h[1]
            else: new_h.append(h)
        holdings = new_h
        signals = daily_trades.get(date, [])
        if signals and cash > 0:
            per = cash / len(signals)
            for ret in signals:
                buy = per * (1-FEE/2)
                si = i + 3
                if si >= len(cal_dates): continue
                sell = buy * (1+ret) * (1-FEE/2)
                holdings.append((si, sell))
            cash = 0
        hv = sum(h[1] for h in holdings)
        eq.append(cash + hv)
    return eq

sims = {}
for name, fn in [
    ('原始策略', lambda d: [True]*len(df)),
    ('站上MA20', lambda d: d['above_ma20']==True),
    ('MA20+≥30%分位', lambda d: (d['above_ma20']==True)&(d['pos_250']>=30)),
    ('站上MA60', lambda d: d['above_ma60']==True),
    ('≥50%分位', lambda d: d['pos_250']>=50),
]:
    sub = df[fn(df)]
    eq = simulate(sub)
    sims[name] = eq

start_d = df['date'].min()
end_d = df['date'].max()
si = cal_dates.index(start_d) if start_d in cal_dates else 0
ei = cal_dates.index(end_d) if end_d in cal_dates else len(cal_dates)-1
dates_slice = cal_dates[si:ei+1]

print(f'  {"策略":<16s} {"期末":>10s} {"总收益":>9s} {"年化":>8s} {"最大回撤":>9s} {"卡玛":>6s}')
print(f'  {"-"*65}')
for name, eq in sims.items():
    eq_s = eq[si:ei+1]
    final = eq_s[-1]
    tot = (final/INIT_CAP-1)*100
    yrs = len(eq_s)/252
    ann = ((final/INIT_CAP)**(1/yrs)-1)*100 if yrs>0 else 0
    peak = pd.Series(eq_s).cummax()
    mdd = ((pd.Series(eq_s)/peak)-1).min()*100
    calm = ann/abs(mdd) if mdd != 0 else 0
    print(f'  {name:<16s} {final:>10,.0f} {tot:>+8.1f}% {ann:>+7.1f}% {mdd:>8.1f}% {calm:>6.2f}')

# 保存数据
import json
data = {
    'dates': dates_slice,
    'pos_bins': [
        {'bin': lbl, 'n': int(len(df[df['pos_bin']==lbl])),
         'avg_ret': round(df[df['pos_bin']==lbl]['raw'].mean()*100, 2),
         'win_rate': round((df[df['pos_bin']==lbl]['raw']>0).mean()*100, 1)}
        for lbl in labels if len(df[df['pos_bin']==lbl]) > 0
    ],
    'ma_filters': [
        {'name': name, 'n': int(len(df[fn(df)])),
         'pass_rate': round(len(df[fn(df)])/len(df)*100, 1),
         'avg_ret': round(df[fn(df)]['raw'].mean()*100, 2),
         'win_rate': round((df[fn(df)]['raw']>0).mean()*100, 1)}
        for name, fn in filters if len(df[fn(df)]) >= 2
    ],
    'equity': {name: sims[name][si:ei+1] for name in sims},
    'final_metrics': {
        name: {
            'final': round(sims[name][ei], 0),
            'total_ret': round((sims[name][ei]/INIT_CAP-1)*100, 1),
        } for name in sims
    },
}

with open('idx_filter_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
print(f'\n数据已保存: idx_filter_data.json')
