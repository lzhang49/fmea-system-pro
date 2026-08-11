# -*- coding: utf-8 -*-
"""
二板"排板/打板"胜率回测 (2021-08 ~ 2026-08, 全A 5189 只)
============================================================================
与前面 v1~v8 的根本差异：成交机制不同
  之前(S1)  : 事件次日【开盘价】买入  → 一定能成交
  本次(打板): 事件次日【涨停价】排队  → 只有盘中触及涨停才成交，否则空仓不动

打板三个专属指标：
  触板率 = 二板次日盘中摸到涨停的比例   (决定有多少次能出手)
  封板率 = 二板次日收盘涨停的比例       (即"晋级三板成功率"= 排板核心胜率)
  炸板率 = 触板但收盘没封住的比例       (打板的主要亏损来源)

可成交性约束（不做则严重高估）：
  1. 一字板 (open==high==low==涨停) → 排不进队，弃单
  2. T字板/全天涨停价成交 (low==涨停) → 同样视为排不进，弃单
  3. 除权异常跳空 → 剔除
  注：盘中触板但封单过大导致排不进的情况，日线数据无法识别，见报告局限。

买入价 = 涨停价 (四舍五入到分)；A股 T+1，最早 T+2 才能卖。
成本 0.3% 双边。
"""
import os, glob, time
import numpy as np
import pandas as pd

CACHE = 'cache5y'
BT_START = '2021-08-01'
COST = 0.003
WARMUP = 60
np.random.seed(42)


def zt_thr(c):
    return 0.20 if c.startswith(('68', '30')) else 0.10


def board_of(c):
    if c.startswith('68'):
        return '科创'
    if c.startswith('30'):
        return '创业'
    return '主板'


print("=" * 104)
print("1. 载入全 A 日线并扫描全部连板事件 (首板~5板+)")
print("=" * 104, flush=True)

paths = {}
EV = []
t0 = time.time()
files = sorted(glob.glob(os.path.join(CACHE, '*.pkl')))
print("缓存股票数: %d" % len(files), flush=True)

for idx, fp in enumerate(files):
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

    hit = np.where(isz & (run >= 1) & (run <= 6))[0]
    for i in hit:
        if i < WARMUP or d[i] < BT_START or i + 6 >= n:
            continue
        EV.append((code, int(i), d[i], int(run[i]), c[i] * osh[i] / 1e8,
                   tov[i], board_of(code)))

    if (idx + 1) % 1500 == 0:
        print("  已处理 %d/%d  事件 %d  用时%.0fs"
              % (idx + 1, len(files), len(EV), time.time() - t0), flush=True)

EVDF = pd.DataFrame(EV, columns=['code', 'i', 'date', 'run', 'mcap', 'turn', 'board'])
EVDF['year'] = EVDF['date'].str[:4]
print("\n载入完成: %d 只股票, 连板事件 %d 个, 用时 %.0fs"
      % (len(paths), len(EVDF), time.time() - t0))
print("\n各连板高度事件数:")
print(EVDF.groupby('run').size().to_string())


# ======================= 2. 打板成交与收益模型 =======================
def daban(code, i, N=1, sell='close', stop_bomb=False):
    """
    在事件日 i 的【次日 k=i+1】以涨停价排板买入。
    返回 (状态, 收益)
      状态: 'nohit' 次日没摸到涨停(不出手) / 'yizi' 一字或T字排不进 /
            'xr' 除权异常 / 'short' 数据不足 / 'ok' 成交
      N     : 持有天数, 0 表示 T+2 开盘就卖
      sell  : 'close' -> 第 k+N 日收盘卖; 'open' -> 第 k+N 日开盘卖
      stop_bomb: True 表示"炸板即走"—— 若 k 日收盘未封住涨停, 当日收盘价卖出
    """
    p = paths[code]
    o, h, l, c = p['o'], p['h'], p['l'], p['c']
    nn = len(c)
    k = i + 1
    if k + max(N, 1) >= nn or c[i] <= 0 or o[k] <= 0:
        return 'short', None
    tz = zt_thr(code)
    ztp = round(c[i] * (1 + tz) + 1e-9, 2)
    # 次日没有摸到涨停 → 排板不成交, 不产生交易
    if h[k] < ztp - 0.005:
        return 'nohit', None
    # 一字板 / 全天封死在涨停价 → 排不进队
    if l[k] >= ztp - 0.005:
        return 'yizi', None
    # 除权异常
    for dd in range(k, min(k + N + 1, nn)):
        if c[dd - 1] > 0 and (c[dd] / c[dd - 1] - 1.0) < -(tz + 0.05):
            return 'xr', None
    entry = ztp
    sealed = c[k] >= ztp - 0.005          # 当日收盘是否封住
    if stop_bomb and not sealed:
        return 'ok', c[k] / entry - 1.0   # 炸板即走: 当日收盘价出
    if N == 0:
        return 'ok', o[k + 1] / entry - 1.0
    px = c[k + N] if sell == 'close' else o[k + N]
    return 'ok', px / entry - 1.0


def scan(evdf, N=1, sell='close', stop_bomb=False):
    rets, st = [], {'ok': 0, 'nohit': 0, 'yizi': 0, 'xr': 0, 'short': 0}
    sealed_n = 0
    for code, i in zip(evdf['code'].values, evdf['i'].values):
        s, r = daban(code, int(i), N, sell, stop_bomb)
        st[s] += 1
        if r is not None:
            rets.append(r)
    return np.array(rets), st


def seal_stats(evdf):
    """触板率 / 封板率 / 炸板率 / 一字率 (不涉及买卖, 纯统计)"""
    tot = hit = seal = yizi = 0
    for code, i in zip(evdf['code'].values, evdf['i'].values):
        p = paths[code]
        h, l, c = p['h'], p['l'], p['c']
        k = int(i) + 1
        if k >= len(c) or c[int(i)] <= 0:
            continue
        tz = zt_thr(code)
        ztp = round(c[int(i)] * (1 + tz) + 1e-9, 2)
        tot += 1
        if h[k] >= ztp - 0.005:
            hit += 1
            if l[k] >= ztp - 0.005:
                yizi += 1
            if c[k] >= ztp - 0.005:
                seal += 1
    if tot == 0:
        return None
    return dict(tot=tot, hit=hit / tot, seal=seal / tot, yizi=yizi / tot,
                bomb=(hit - seal) / max(1, hit))


def summ(r, base, label=''):
    if r is None or len(r) < 20:
        return None
    win = r[r > 0]
    los = r[r <= 0]
    pl = (win.mean() / abs(los.mean())) if len(los) > 0 and los.mean() < 0 else np.nan
    e = r.mean() - (base.mean() if base is not None and len(base) >= 20 else 0.0)
    sd = r.std(ddof=1)
    t = e / (sd / np.sqrt(len(r))) if sd > 0 else 0.0
    return dict(label=label, n=len(r), wr=(r > 0).mean(), exp=r.mean(),
                pl=pl, excess=e, cost=e - COST, t=t)


def show(rows, title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print("%-26s%8s%9s%10s%9s%10s%11s%8s" %
          ('分组', 'n', '胜率', '期望', '盈亏比', '超额', '成本后', 't'))
    print("-" * 104)
    for s in rows:
        if s is None:
            continue
        print("%-26s%8d%8.1f%%%9.2f%%%9s%9.2f%%%10.2f%%%8.2f" %
              (s['label'], s['n'], s['wr'] * 100, s['exp'] * 100,
               ('%.2f' % s['pl']) if s['pl'] == s['pl'] else '--',
               s['excess'] * 100, s['cost'] * 100, s['t']))


# ======================= 3. 基准: 同池随机日买入 =======================
print("\n构建基准组 (同池随机日 T+1 开盘买入)...", flush=True)
BASE_CACHE = {}
POOL = EVDF['code'].unique()


def base_all(N):
    if N in BASE_CACHE:
        return BASE_CACHE[N]
    rets, yrs = [], []
    for code in POOL:
        p = paths[code]
        o, c, d = p['o'], p['c'], p['d']
        nn = len(c)
        for i in range(WARMUP, nn - N - 3, 20):
            if d[i] < BT_START or o[i + 1] <= 0:
                continue
            rets.append(c[i + 1 + N] / o[i + 1] - 1.0)
            yrs.append(d[i][:4])
    BASE_CACHE[N] = (np.array(rets), np.array(yrs))
    return BASE_CACHE[N]


def base_rets(year=None, N=1):
    r, y = base_all(N)
    return r if year is None else r[y == year]


B1 = base_rets(N=1)
print("基准样本 %d 笔, 均值 %.2f%%" % (len(B1), B1.mean() * 100), flush=True)


# ======================= 4. 排板核心指标: 各高度晋级率 =======================
print("\n" + "=" * 104)
print("2. 排板核心指标 — 各连板高度的次日触板率 / 封板率(晋级率) / 炸板率")
print("=" * 104)
print("%-14s%10s%12s%14s%12s%12s" %
      ('连板高度', '事件数', '次日触板率', '次日封板率', '其中一字率', '触板后炸板率'))
print("-" * 104)
for rv in [1, 2, 3, 4, 5]:
    sub = EVDF[EVDF['run'] == rv]
    s = seal_stats(sub)
    if s is None:
        continue
    print("%-14s%10d%11.1f%%%13.1f%%%11.1f%%%11.1f%%" %
          ('%d 板' % rv, s['tot'], s['hit'] * 100, s['seal'] * 100,
           s['yizi'] * 100, s['bomb'] * 100))

print("\n二板晋级率逐年:")
print("%-14s%10s%12s%14s%12s" % ('年份', '事件数', '触板率', '封板率(晋级)', '炸板率'))
print("-" * 104)
E2 = EVDF[EVDF['run'] == 2]
for y in sorted(E2['year'].unique()):
    s = seal_stats(E2[E2['year'] == y])
    if s:
        print("%-14s%10d%11.1f%%%13.1f%%%11.1f%%" %
              (y, s['tot'], s['hit'] * 100, s['seal'] * 100, s['bomb'] * 100))


# ======================= 5. 二板打板收益: 逐年 =======================
R2, ST2 = scan(E2, N=1)
print("\n" + "=" * 104)
print("3. 二板打板可成交性统计 (共 %d 个二板事件)" % len(E2))
print("=" * 104)
tot = max(1, len(E2))
print("  成交(盘中触板可排队): %6d  (%.1f%%)" % (ST2['ok'], ST2['ok'] / tot * 100))
print("  次日没摸到涨停 不出手: %6d  (%.1f%%)" % (ST2['nohit'], ST2['nohit'] / tot * 100))
print("  一字/T字 排不进队    : %6d  (%.1f%%)" % (ST2['yizi'], ST2['yizi'] / tot * 100))
print("  除权剔除 / 数据不足  : %6d / %d" % (ST2['xr'], ST2['short']))

rows = [summ(R2, B1, '5年合计')]
for y in sorted(E2['year'].unique()):
    r, _ = scan(E2[E2['year'] == y], N=1)
    rows.append(summ(r, base_rets(year=y, N=1), y + ' 年'))
show(rows, "4. 二板打板 — 涨停价买入 / 持有到 T+2 收盘 / 无止损")


# ======================= 6. 各连板高度打板对比 =======================
rows = []
for rv in [1, 2, 3, 4, 5]:
    r, _ = scan(EVDF[EVDF['run'] == rv], N=1)
    rows.append(summ(r, B1, '打 %d 板' % rv))
show(rows, "5. 打板高度对比 (5年合计, 持有到 T+2 收盘)")


# ======================= 7. 卖出方式 × 持有期 =======================
rows = []
for N, sell, lbl in [(0, 'open', 'T+2 开盘卖 (隔日冲高)'),
                     (1, 'close', 'T+2 收盘卖'),
                     (2, 'close', 'T+3 收盘卖'),
                     (3, 'close', 'T+4 收盘卖'),
                     (5, 'close', 'T+6 收盘卖')]:
    r, _ = scan(E2, N=N, sell=sell)
    b = base_rets(N=max(N, 1))
    rows.append(summ(r, b, lbl))
r, _ = scan(E2, N=1, stop_bomb=True)
rows.append(summ(r, B1, '炸板即走(当日收盘出)'))
show(rows, "6. 二板打板 — 卖出方式扫描")


# ======================= 8. 因子分层 =======================
rows = []
for lbl, sub in [
    ('全部二板打板', E2),
    ('低换手 <5.2%', E2[E2['turn'] < 0.052]),
    ('高换手 >=5.2%', E2[E2['turn'] >= 0.052]),
    ('小盘 <60亿', E2[E2['mcap'] < 60]),
    ('中大盘 >=60亿', E2[E2['mcap'] >= 60]),
    ('S1 双因子', E2[(E2['turn'] < 0.052) & (E2['mcap'] >= 60)]),
    ('主板', E2[E2['board'] == '主板']),
    ('创业板', E2[E2['board'] == '创业']),
    ('科创板', E2[E2['board'] == '科创']),
]:
    r, _ = scan(sub, N=1)
    rows.append(summ(r, B1, lbl))
show(rows, "7. 二板打板 — 因子/板块分层 (5年合计)")


# ======================= 9. 换手 × 市值 敏感性 =======================
print("\n" + "=" * 104)
print("8. 二板打板 参数敏感性 — 换手率 × 流通市值 (成本后超额%)")
print("=" * 104)
turns = [0.03, 0.05, 0.08, 0.15, 9.99]
caps = [0, 30, 60, 100, 200]
print("%-12s" % '换手<', end='')
for cp in caps:
    print("%15s" % ('市值>=%d亿' % cp), end='')
print()
for tv in turns:
    print("%-12s" % ('%.0f%%' % (tv * 100) if tv < 1 else '不限'), end='')
    for cp in caps:
        sub = E2[(E2['turn'] < tv) & (E2['mcap'] >= cp)]
        r, _ = scan(sub, N=1)
        s = summ(r, B1)
        print("%15s" % ('%+.2f%%(n%d)' % (s['cost'] * 100, s['n']) if s else '样本不足'), end='')
    print()

E2.to_csv('events_daban_2b.csv', index=False, encoding='utf-8-sig')
print("\n二板事件表已导出: events_daban_2b.csv (%d 行)" % len(E2))
print("总用时 %.1f 分钟" % ((time.time() - t0) / 60))
