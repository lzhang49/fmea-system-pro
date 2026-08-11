# -*- coding: utf-8 -*-
"""
二板打板 — 正期望角落穷举
============================================================================
主回测显示打板 5 年全线为负。本脚本回答一个问题：
  "有没有任何一种打板细分场景是正的？"
穷举实战最常用的 6 个维度：
  A. T+1 开盘涨幅 (低开打板 vs 高开打板) —— 打板实战第一变量
  B. 二板当日封板强度 (一字 / 强封 / 弱封-盘中炸过)
  C. 市场情绪 (当日全市场首板数量分位)
  D. 二板日相对前高的位置 (首次上攻 vs 高位接力)
  E. 二板日成交额量级 (小票游资 vs 大资金)
  F. 组合式最优搜索: 所有 A×B×C 交叉格子, 找正期望格
"""
import os, glob, time
import numpy as np
import pandas as pd

CACHE = 'cache5y'
BT_START = '2021-08-01'
COST = 0.003
WARMUP = 60


def zt_thr(c):
    return 0.20 if c.startswith(('68', '30')) else 0.10


print("载入缓存并重建二板事件 (含扩展特征)...", flush=True)
paths = {}
EV = []
t0 = time.time()
for fp in sorted(glob.glob(os.path.join(CACHE, '*.pkl'))):
    code = os.path.basename(fp)[:-4]
    try:
        df = pd.read_pickle(fp, compression='gzip')
    except Exception:
        continue
    if len(df) < WARMUP + 20:
        continue
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64) if 'volume' in df else np.zeros(len(c))
    tov = df['turnover'].values.astype(np.float64)
    osh = df['outstanding_share'].values.astype(np.float64)
    d = df['date'].values.astype(str)
    n = len(c)
    paths[code] = dict(o=o, h=h, l=l, c=c, d=d)

    tz = zt_thr(code)
    prev = np.maximum(c[:-1], 1e-9)
    isz = np.zeros(n, dtype=bool)
    isz[1:] = (c[1:] / prev - 1.0) >= tz - 0.004
    s = pd.Series(isz.astype(int))
    run = s.groupby((s == 0).cumsum()).cumsum().values

    for i in np.where(isz & (run == 2))[0]:
        if i < WARMUP or d[i] < BT_START or i + 6 >= n:
            continue
        ztp_i = round(c[i - 1] * (1 + tz) + 1e-9, 2)
        # 二板日封板强度: 当日最低价距涨停的幅度 (0=一字, 越大越弱)
        weak = (ztp_i - l[i]) / max(ztp_i, 1e-9)
        # 相对前 60 日高点的位置
        pos = c[i] / max(np.max(h[max(0, i - 60):i]), 1e-9)
        # 成交额 (亿) 与 60日均量比
        amt = c[i] * v[i] / 1e8 if v[i] > 0 else np.nan
        vr = v[i] / max(np.mean(v[max(0, i - 60):i]), 1e-9) if v[i] > 0 else np.nan
        # T+1 开盘涨幅
        gap = o[i + 1] / c[i] - 1.0 if c[i] > 0 else np.nan
        EV.append((code, int(i), d[i], c[i] * osh[i] / 1e8, tov[i],
                   weak, pos, amt, vr, gap))

E2 = pd.DataFrame(EV, columns=['code', 'i', 'date', 'mcap', 'turn',
                               'weak', 'pos', 'amt', 'vr', 'gap'])
E2['year'] = E2['date'].str[:4]
print("二板事件 %d 个, 用时 %.0fs" % (len(E2), time.time() - t0))

# 市场情绪: 当日全市场二板数量
sent = E2.groupby('date').size().rename('n2b')
E2 = E2.merge(sent, left_on='date', right_index=True)
q33, q67 = E2['n2b'].quantile([0.33, 0.67])
print("当日二板数量分位: 33%%=%.0f  67%%=%.0f" % (q33, q67))


def daban(code, i, N=1):
    p = paths[code]
    o, h, l, c = p['o'], p['h'], p['l'], p['c']
    nn = len(c)
    k = i + 1
    if k + N >= nn or c[i] <= 0:
        return None
    tz = zt_thr(code)
    ztp = round(c[i] * (1 + tz) + 1e-9, 2)
    if h[k] < ztp - 0.005:
        return None
    if l[k] >= ztp - 0.005:
        return None
    for dd in range(k, min(k + N + 1, nn)):
        if c[dd - 1] > 0 and (c[dd] / c[dd - 1] - 1.0) < -(tz + 0.05):
            return None
    return c[k + N] / ztp - 1.0


def rets(sub, N=1):
    out = []
    for code, i in zip(sub['code'].values, sub['i'].values):
        r = daban(code, int(i), N)
        if r is not None:
            out.append(r)
    return np.array(out)


BASE = 0.0029   # 同池随机日基准 (与主脚本一致)


def line(lbl, r, minn=30):
    if len(r) < minn:
        print("%-30s%8d   样本不足" % (lbl, len(r)))
        return None
    win, los = r[r > 0], r[r <= 0]
    pl = win.mean() / abs(los.mean()) if len(los) and los.mean() < 0 else np.nan
    e = r.mean() - BASE
    t = e / (r.std(ddof=1) / np.sqrt(len(r))) if r.std(ddof=1) > 0 else 0
    flag = '  <== 正' if e - COST > 0 else ''
    print("%-30s%8d%8.1f%%%9.2f%%%9s%10.2f%%%8.2f%s" %
          (lbl, len(r), (r > 0).mean() * 100, r.mean() * 100,
           ('%.2f' % pl) if pl == pl else '--', (e - COST) * 100, t, flag))
    return e - COST


def hdr(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print("%-30s%8s%8s%9s%9s%10s%8s" % ('分组', 'n', '胜率', '期望', '盈亏比', '成本后超额', 't'))
    print("-" * 100)


# A. T+1 开盘涨幅
hdr("A. 按 T+1 开盘涨幅分层 (打板实战第一变量: 低开打 vs 高开打)")
for lbl, m in [('大幅低开 <-3%', E2['gap'] < -0.03),
               ('低开 -3%~-1%', (E2['gap'] >= -0.03) & (E2['gap'] < -0.01)),
               ('平开 -1%~+1%', (E2['gap'] >= -0.01) & (E2['gap'] < 0.01)),
               ('小高开 +1%~+3%', (E2['gap'] >= 0.01) & (E2['gap'] < 0.03)),
               ('中高开 +3%~+6%', (E2['gap'] >= 0.03) & (E2['gap'] < 0.06)),
               ('大高开 >+6%', E2['gap'] >= 0.06)]:
    line(lbl, rets(E2[m]))

# B. 二板日封板强度
hdr("B. 按二板当日封板强度 (weak = 当日最低价距涨停的幅度)")
for lbl, m in [('一字板 (weak<0.5%)', E2['weak'] < 0.005),
               ('强封 0.5%~3%', (E2['weak'] >= 0.005) & (E2['weak'] < 0.03)),
               ('中封 3%~7%', (E2['weak'] >= 0.03) & (E2['weak'] < 0.07)),
               ('弱封 >7% (盘中炸过)', E2['weak'] >= 0.07)]:
    line(lbl, rets(E2[m]))

# C. 市场情绪
hdr("C. 按市场情绪 (当日全市场二板数量)")
for lbl, m in [('冰点 (二板数<%.0f)' % q33, E2['n2b'] < q33),
               ('中性', (E2['n2b'] >= q33) & (E2['n2b'] < q67)),
               ('高潮 (二板数>=%.0f)' % q67, E2['n2b'] >= q67)]:
    line(lbl, rets(E2[m]))

# D. 相对前高位置
hdr("D. 按二板日相对前 60 日高点的位置")
for lbl, m in [('低位首攻 <0.85', E2['pos'] < 0.85),
               ('中位 0.85~1.0', (E2['pos'] >= 0.85) & (E2['pos'] < 1.0)),
               ('创新高 >=1.0', E2['pos'] >= 1.0)]:
    line(lbl, rets(E2[m]))

# E. 成交额与量比
hdr("E. 按二板日成交额 / 量比")
va = E2['amt'].dropna()
if len(va) > 100:
    a1, a2 = va.quantile([0.33, 0.67])
    for lbl, m in [('小额 <%.1f亿' % a1, E2['amt'] < a1),
                   ('中额', (E2['amt'] >= a1) & (E2['amt'] < a2)),
                   ('大额 >=%.1f亿' % a2, E2['amt'] >= a2)]:
        line(lbl, rets(E2[m]))
    for lbl, m in [('温和放量 量比<2', E2['vr'] < 2),
                   ('放量 2~5', (E2['vr'] >= 2) & (E2['vr'] < 5)),
                   ('巨量 >=5', E2['vr'] >= 5)]:
        line(lbl, rets(E2[m]))
else:
    print("成交量字段缺失, 跳过")

# F. 组合穷举
print("\n" + "=" * 100)
print("F. 组合穷举 — 开盘涨幅 × 封板强度 × 情绪 (共 %d 格), 列出成本后超额为正的格子" % (6 * 4 * 3))
print("=" * 100)
gaps = [('低开<-1%', E2['gap'] < -0.01),
        ('平开', (E2['gap'] >= -0.01) & (E2['gap'] < 0.01)),
        ('小高开1~3%', (E2['gap'] >= 0.01) & (E2['gap'] < 0.03)),
        ('中高开3~6%', (E2['gap'] >= 0.03) & (E2['gap'] < 0.06)),
        ('大高开>6%', E2['gap'] >= 0.06)]
weaks = [('一字/强封', E2['weak'] < 0.03),
         ('中封', (E2['weak'] >= 0.03) & (E2['weak'] < 0.07)),
         ('弱封', E2['weak'] >= 0.07)]
sents = [('冰点', E2['n2b'] < q33),
         ('中性', (E2['n2b'] >= q33) & (E2['n2b'] < q67)),
         ('高潮', E2['n2b'] >= q67)]
pos_cells, all_cells = [], 0
print("%-30s%8s%8s%9s%10s%8s" % ('组合', 'n', '胜率', '期望', '成本后超额', 't'))
print("-" * 100)
for gl, gm in gaps:
    for wl, wm in weaks:
        for sl, sm in sents:
            r = rets(E2[gm & wm & sm])
            all_cells += 1
            if len(r) < 30:
                continue
            e = r.mean() - BASE - COST
            t = (r.mean() - BASE) / (r.std(ddof=1) / np.sqrt(len(r))) if r.std(ddof=1) > 0 else 0
            if e > 0:
                pos_cells.append((gl, wl, sl, len(r), (r > 0).mean(), r.mean(), e, t))
                print("%-30s%8d%7.1f%%%9.2f%%%9.2f%%%8.2f" %
                      ('%s + %s + %s' % (gl, wl, sl), len(r), (r > 0).mean() * 100,
                       r.mean() * 100, e * 100, t))
if not pos_cells:
    print("(全部 %d 格中, 样本>=30 的格子无一为正)" % all_cells)
else:
    print("\n注意: 在 %d 个格子里挑出 %d 个正格, 多重比较下需要 t > 3 才可信"
          % (all_cells, len(pos_cells)))

# G. 最佳情形叠加因子再看
print("\n" + "=" * 100)
print("G. 把 A~E 各维度最优选项叠加, 看能否翻正")
print("=" * 100)
print("%-30s%8s%8s%9s%10s%8s" % ('分组', 'n', '胜率', '期望', '成本后超额', 't'))
print("-" * 100)
best = (E2['gap'] < 0.01) & (E2['weak'] < 0.03) & (E2['mcap'] >= 60) & (E2['turn'] < 0.052)
line('低开平开+强封+大市值+低换手', rets(E2[best]), minn=20)
best2 = (E2['gap'] < 0.01) & (E2['weak'] < 0.03) & (E2['mcap'] >= 100)
line('低开平开+强封+市值>100亿', rets(E2[best2]), minn=20)

print("\n用时 %.1f 分钟" % ((time.time() - t0) / 60))
