# -*- coding: utf-8 -*-
"""
归因诊断：5年回测(负超额) vs 前7轮回测(正超额) 的结论为何反转？
用【同一套代码】跑 4 种组合，把差异精确拆解到两个因素：
  A. 股票池：712只"2026涨停活跃股"(有前视偏差)  vs  全A 5189只(无偏)
  B. 一字板可成交性约束：关闭(前7轮做法)  vs  开启(真实约束)
若"旧池+关闭约束"能复现 v8 的 +2.99%，则证明新代码无 bug，反转来自偏差而非错误。
"""
import os, glob, time
import numpy as np
import pandas as pd

COST = 0.003
N_HOLD = 3
STOP = -0.10
np.random.seed(42)


def zt_thr(c):
    return 0.20 if c.startswith(('68', '30')) else 0.10


def load_pool(kind):
    """kind: 'old'=712只2026活跃股(csv) / 'all'=全A 5189只(pkl)"""
    paths = {}
    if kind == 'old':
        files = sorted(glob.glob('cache_daily/*.csv'))
        for fp in files:
            code = os.path.basename(fp)[:-4]
            df = pd.read_csv(fp)
            if len(df) < 30:
                continue
            paths[code] = df
    else:
        files = sorted(glob.glob('cache5y/*.pkl'))
        for fp in files:
            code = os.path.basename(fp)[:-4]
            try:
                df = pd.read_pickle(fp, compression='gzip')
            except Exception:
                continue
            if len(df) < 30:
                continue
            paths[code] = df
    out = {}
    for code, df in paths.items():
        out[code] = dict(
            o=df['open'].values.astype(float), h=df['high'].values.astype(float),
            l=df['low'].values.astype(float), c=df['close'].values.astype(float),
            tov=df['turnover'].values.astype(float),
            osh=df['outstanding_share'].values.astype(float),
            d=df['date'].values.astype(str))
    return out


def scan_events(paths, start, end, warmup):
    EV = []
    for code, p in paths.items():
        c, d, tov, osh = p['c'], p['d'], p['tov'], p['osh']
        n = len(c)
        if n < warmup + 10:
            continue
        tz = zt_thr(code)
        isz = np.zeros(n, dtype=bool)
        isz[1:] = (c[1:] / np.maximum(c[:-1], 1e-9) - 1.0) >= tz - 0.004
        s = pd.Series(isz.astype(int))
        run = s.groupby((s == 0).cumsum()).cumsum().values
        for i in np.where(isz & (run == 2))[0]:
            if i < warmup or not (start <= d[i] <= end) or i + 1 + N_HOLD >= n:
                continue
            EV.append((code, int(i), d[i], c[i] * osh[i] / 1e8, tov[i]))
    return pd.DataFrame(EV, columns=['code', 'i', 'date', 'mcap', 'turn'])


def one_ret(paths, code, i, N, stop, skip_yizi):
    p = paths[code]
    o, h, l, c = p['o'], p['h'], p['l'], p['c']
    nn = len(c)
    k = i + 1
    if k + N >= nn or o[k] <= 0 or c[i] <= 0:
        return None, 'short'
    tz = zt_thr(code)
    if skip_yizi and o[k] == h[k] == l[k] and (o[k] / c[i] - 1.0) >= tz - 0.004:
        return None, 'yizi'
    entry = o[k]
    if stop is None:
        return c[k + N] / entry - 1.0, 'ok'
    sp = entry * (1 + stop)
    exitp = None
    for dd in range(k + 1, k + N + 1):
        if l[dd] <= sp:
            exitp = min(o[dd], sp)
            break
    if exitp is None:
        exitp = c[k + N]
    return exitp / entry - 1.0, 'ok'


def sig_rets(paths, evdf, skip_yizi):
    out = []
    yizi = 0
    for code, i in zip(evdf['code'].values, evdf['i'].values):
        r, st = one_ret(paths, code, int(i), N_HOLD, STOP, skip_yizi)
        if st == 'yizi':
            yizi += 1
        if r is not None:
            out.append(r)
    return np.array(out), yizi


def base_rets(paths, codes, start, end, warmup):
    out = []
    for code in codes:
        p = paths[code]
        d, nn = p['d'], len(p['c'])
        for i in range(warmup, nn - N_HOLD - 3, 5):
            if not (start <= d[i] <= end):
                continue
            r, st = one_ret(paths, code, i, N_HOLD, STOP, skip_yizi=False)
            if r is not None:
                out.append(r)
    return np.array(out)


def summ(r, base, label):
    if len(r) < 20 or len(base) < 20:
        return "%-42s 样本不足 (n=%d)" % (label, len(r))
    e = r.mean() - base.mean()
    sd = r.std(ddof=1)
    t = e / (sd / np.sqrt(len(r))) if sd > 0 else 0.
    return ("%-42s n=%5d  胜率%5.1f%%  期望%+7.2f%%  基准%+6.2f%%  超额%+7.2f%%  成本后%+7.2f%%  t=%6.2f"
            % (label, len(r), (r > 0).mean() * 100, r.mean() * 100, base.mean() * 100,
               e * 100, (e - COST) * 100, t))


t0 = time.time()
print("=" * 118)
print("归因诊断：结论反转来自【股票池偏差】还是【代码错误】？")
print("=" * 118, flush=True)

P_OLD = load_pool('old')
P_ALL = load_pool('all')
print("旧池(2026涨停活跃股): %d 只   |   全A池: %d 只\n" % (len(P_OLD), len(P_ALL)), flush=True)

# 统一到 2026 年同一时间窗，唯一变量 = 股票池 & 一字板约束
START, END = '2026-01-05', '2026-08-07'

for pool_name, P, warmup in [('旧池 712只', P_OLD, 1), ('全A 5189只', P_ALL, 60)]:
    EV = scan_events(P, START, END, warmup)
    S1 = EV[(EV['turn'] < 0.052) & (EV['mcap'] >= 60)]
    B = base_rets(P, EV['code'].unique(), START, END, warmup)
    print("-" * 118)
    print("【%s】2026-01~08  二板事件 %d 个，其中 S1 %d 个" % (pool_name, len(EV), len(S1)))
    for yz, tag in [(False, '关闭一字板约束(前7轮做法)'), (True, '开启一字板约束(真实)')]:
        r_all, y1 = sig_rets(P, EV, yz)
        r_s1, y2 = sig_rets(P, S1, yz)
        print("  " + summ(r_all, B, '纯二板 / ' + tag))
        print("  " + summ(r_s1, B, 'S1三因子 / ' + tag) +
              ("   [一字板弃单%d]" % y2 if yz else ""))

# 补充：旧池的股票在全A池里同期表现（同一批票、同一代码、验证数据源一致性）
print("-" * 118)
common = sorted(set(P_OLD.keys()) & set(P_ALL.keys()))
print("两池共有股票 %d 只 —— 用全A缓存数据重跑这批票，验证数据源一致性" % len(common))
SUB = {k: P_ALL[k] for k in common}
EVs = scan_events(SUB, START, END, 60)
S1s = EVs[(EVs['turn'] < 0.052) & (EVs['mcap'] >= 60)]
Bs = base_rets(SUB, EVs['code'].unique(), START, END, 60)
r1, _ = sig_rets(SUB, EVs, False)
r2, _ = sig_rets(SUB, S1s, False)
print("  " + summ(r1, Bs, '纯二板 / 关闭一字板约束'))
print("  " + summ(r2, Bs, 'S1三因子 / 关闭一字板约束'))

print("\n用时 %.1f 分钟" % ((time.time() - t0) / 60))