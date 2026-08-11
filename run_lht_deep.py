# -*- coding: utf-8 -*-
"""
龙回头深度检验：判定"回调第5日买入"是真机制还是相位挑选噪音
核心检验:
  A. 相位曲线 —— 峰值后第 k 日(k=0..15)逐日买入的超额, 看是否平滑
  B. 事件内自对照 —— 用同一事件其他相位日作基准, 剥离"同层基准相位偏差"
  C. 对照池 —— 10日涨幅 30~60% 的普通强势股是否也有第5日效应
  D. 逐年 + LOYO(留一年) + 止损扫描 + 交叉分层(修正换手率单位)
"""
import os, glob, time
import numpy as np
import pandas as pd

CACHE = 'cache5y'
BT_START = '2021-08-01'
COST = 0.003
WARMUP = 60
POOL_LOOK = 10
COOLDOWN = 60
PEAK_WIN = 40
DD_CONFIRM = 0.10
np.random.seed(42)


def zt_thr(c):
    return 0.20 if c.startswith(('68', '30')) else 0.10


def board_of(c):
    if c.startswith('68'):
        return '科创'
    if c.startswith('30'):
        return '创业'
    return '主板'


def tstat(r, base=0.0):
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return 0.0 if sd <= 0 else (r.mean() - base) / (sd / np.sqrt(len(r)))


paths = {}


def build(pool_lo, pool_hi):
    """按 10日涨幅区间构建入池事件, 返回 [(code, t_in, peak_i, confirm_i, meta...)]  """
    evs = []
    for fp in files:
        code = os.path.basename(fp)[:-4]
        p = paths.get(code)
        if p is None:
            continue
        c, h, o, l, d = p['c'], p['h'], p['o'], p['l'], p['d']
        tov, osh, run = p['tov'], p['osh'], p['run']
        n = len(c)
        r10 = np.full(n, np.nan)
        r10[POOL_LOOK:] = c[POOL_LOOK:] / np.maximum(c[:-POOL_LOOK], 1e-9) - 1.0
        cand = np.where((r10 >= pool_lo) & (r10 < pool_hi) & (np.arange(n) >= WARMUP))[0]
        last_in = -10 ** 9
        for t_in in cand:
            if t_in - last_in < COOLDOWN:
                continue
            if d[t_in] < BT_START or t_in + PEAK_WIN + 30 >= n:
                continue
            last_in = t_in
            peak_p, peak_i, confirm_i = -1.0, -1, -1
            for j in range(t_in, min(t_in + PEAK_WIN, n)):
                if h[j] > peak_p:
                    peak_p, peak_i = h[j], j
                if j > peak_i and peak_p > 0 and c[j] <= peak_p * (1 - DD_CONFIRM):
                    confirm_i = j
                    break
            if confirm_i < 0 or peak_i + 20 >= n:
                continue
            evs.append(dict(code=code, t_in=t_in, peak_i=peak_i, confirm_i=confirm_i,
                            peak_p=peak_p, date_in=d[t_in], year=d[t_in][:4],
                            run_max=int(run[max(t_in - POOL_LOOK, 0):t_in + 1].max()),
                            mcap=c[t_in] * osh[t_in] / 1e8,
                            turn=tov[t_in] * 100.0,
                            board=board_of(code), r10=r10[t_in]))
    return pd.DataFrame(evs)


def open_buy(code, i, N=3, stop=None):
    p = paths[code]
    o, h, l, c = p['o'], p['h'], p['l'], p['c']
    nn = len(c)
    k = i + 1
    if k + N >= nn or i < 0 or c[i] <= 0 or o[k] <= 0:
        return None
    tz = zt_thr(code)
    ztp = round(c[i] * (1 + tz) + 1e-9, 2)
    if o[k] >= ztp - 0.005 and l[k] >= ztp - 0.005:
        return None
    for dd in range(k, min(k + N + 1, nn)):
        if c[dd - 1] > 0 and (c[dd] / c[dd - 1] - 1.0) < -(tz + 0.05):
            return None
    ent = o[k]
    if stop is not None:
        for dd in range(k + 1, k + N + 1):
            if c[dd] / ent - 1.0 <= -stop:
                nxt = dd + 1
                if nxt >= nn:
                    return c[dd] / ent - 1.0
                tzl = round(c[dd] * (1 - tz) + 1e-9, 2)
                if h[nxt] <= tzl + 0.005:
                    continue
                return o[nxt] / ent - 1.0
    return c[k + N] / ent - 1.0


def phase_rets(ev, k, N=3, stop=None):
    """峰值后第 k 个交易日作为信号日, 次日开盘买"""
    out = []
    for code, pk in zip(ev['code'].values, ev['peak_i'].values):
        r = open_buy(code, int(pk) + k, N, stop)
        if r is not None:
            out.append(r)
    return np.array(out)


print("=" * 116)
print("0. 载入缓存")
print("=" * 116, flush=True)
t0 = time.time()
files = sorted(glob.glob(os.path.join(CACHE, '*.pkl')))
for fp in files:
    code = os.path.basename(fp)[:-4]
    try:
        df = pd.read_pickle(fp, compression='gzip')
    except Exception:
        continue
    if len(df) < WARMUP + 40:
        continue
    c = df['close'].values.astype(np.float64)
    n = len(c)
    tz = zt_thr(code)
    isz = np.zeros(n, dtype=bool)
    isz[1:] = (c[1:] / np.maximum(c[:-1], 1e-9) - 1.0) >= tz - 0.004
    s = pd.Series(isz.astype(int))
    paths[code] = dict(o=df['open'].values.astype(np.float64),
                       h=df['high'].values.astype(np.float64),
                       l=df['low'].values.astype(np.float64),
                       c=c, d=df['date'].values.astype(str),
                       tov=df['turnover'].values.astype(np.float64),
                       osh=df['outstanding_share'].values.astype(np.float64),
                       run=s.groupby((s == 0).cumsum()).cumsum().values)
print("载入 %d 只, 用时 %.0fs" % (len(paths), time.time() - t0), flush=True)

EV = build(1.00, 99.0)
print("【主池】10日涨幅>=100%%  入池事件 %d 个" % len(EV))
CTRL = build(0.30, 0.60)
print("【对照池】10日涨幅30~60%%  入池事件 %d 个" % len(CTRL), flush=True)


# ======================= A. 相位曲线（决定性检验） =======================
print("\n" + "=" * 116)
print("A. 相位曲线：峰值后第 k 日买入（T+1开盘，持有3日）—— 真机制应平滑，噪音则孤峰")
print("=" * 116)
print("%-8s%9s%9s%11s%11s%9s   %s" % ('相位k', '成交n', '胜率', '期望', '成本后', 't', '走势'))
print("-" * 116)
curve = []
for k in range(0, 16):
    r = phase_rets(EV, k, 3)
    if len(r) < 30:
        print("%-8d%9d   样本不足" % (k, len(r)))
        continue
    m = r.mean()
    bar = '#' * max(int(round((m * 100 + 4) * 2)), 0)
    print("%-8d%9d%8.1f%%%10.2f%%%10.2f%%%9.2f   %s"
          % (k, len(r), (r > 0).mean() * 100, m * 100, (m - COST) * 100, tstat(r), bar))
    curve.append((k, len(r), m, tstat(r)))

print("\n对照池（普通强势股 10日涨30~60%）同一相位曲线：")
print("%-8s%9s%9s%11s%9s" % ('相位k', '成交n', '胜率', '期望', 't'))
print("-" * 116)
ctrl_curve = []
for k in range(0, 16):
    r = phase_rets(CTRL, k, 3)
    if len(r) < 30:
        continue
    ctrl_curve.append((k, len(r), r.mean(), tstat(r)))
    print("%-8d%9d%8.1f%%%10.2f%%%9.2f" % (k, len(r), (r > 0).mean() * 100,
                                           r.mean() * 100, tstat(r)))


# ======================= B. 事件内自对照 =======================
print("\n" + "=" * 116)
print("B. 事件内自对照：第5日 vs 同一批事件的相位均值（剥离基准相位偏差）")
print("=" * 116)
K = 5
r5 = phase_rets(EV, K, 3)
others = np.concatenate([phase_rets(EV, k, 3) for k in range(0, 16) if k != K])
print("第5日: n=%d 期望%+.2f%%   其余相位合计: n=%d 期望%+.2f%%"
      % (len(r5), r5.mean() * 100, len(others), others.mean() * 100))
diff = r5.mean() - others.mean()
se = np.sqrt(r5.var(ddof=1) / len(r5) + others.var(ddof=1) / len(others))
print("差值 %+.2f%%   两样本 t = %.2f" % (diff * 100, diff / se if se > 0 else 0))

print("\n主池 vs 对照池（同为第5日相位）:")
c5 = phase_rets(CTRL, K, 3)
print("主池(翻倍股) n=%d 期望%+.2f%%    对照池(普通强势) n=%d 期望%+.2f%%"
      % (len(r5), r5.mean() * 100, len(c5), c5.mean() * 100))
se2 = np.sqrt(r5.var(ddof=1) / len(r5) + c5.var(ddof=1) / len(c5))
print("差值 %+.2f%%   两样本 t = %.2f"
      % ((r5.mean() - c5.mean()) * 100, (r5.mean() - c5.mean()) / se2 if se2 > 0 else 0))


# ======================= C. 逐年 + LOYO =======================
print("\n" + "=" * 116)
print("C. 第5日买点逐年稳定性（持有3日，对照同年对照池同相位）")
print("=" * 116)
print("%-10s%9s%9s%11s%12s%12s%9s" % ('年份', '成交n', '胜率', '期望', '对照池同相位', '差值', 't'))
print("-" * 116)
yr_rows = []
for y in sorted(EV['year'].unique()):
    sub = EV[EV['year'] == y]
    r = phase_rets(sub, K, 3)
    cb = phase_rets(CTRL[CTRL['year'] == y], K, 3)
    if len(r) < 15:
        print("%-10s%9d   样本不足" % (y, len(r)))
        continue
    base = cb.mean() if len(cb) >= 30 else 0.0
    print("%-10s%9d%8.1f%%%10.2f%%%11.2f%%%11.2f%%%9.2f"
          % (y, len(r), (r > 0).mean() * 100, r.mean() * 100, base * 100,
             (r.mean() - base - COST) * 100, tstat(r, base + COST)))
    yr_rows.append((y, len(r), r.mean() - base - COST))

print("\n留一年检验（LOYO：剔除某年后其余年份合计）:")
print("%-14s%9s%12s%9s" % ('剔除年份', 'n', '成本后期望', 't'))
print("-" * 116)
for y in sorted(EV['year'].unique()):
    sub = EV[EV['year'] != y]
    r = phase_rets(sub, K, 3)
    if len(r) < 50:
        continue
    print("%-14s%9d%11.2f%%%9.2f" % ('剔除' + y, len(r), (r.mean() - COST) * 100,
                                     tstat(r, COST)))


# ======================= D. 止损与持有期 =======================
print("\n" + "=" * 116)
print("D. 第5日买点：持有期 × 止损扫描（成本后期望）")
print("=" * 116)
hdr = "%-12s" % '止损'
for N in (1, 2, 3, 5, 8):
    hdr += "%14s" % ('持%d日' % N)
print(hdr)
print("-" * 116)
for stop in [None, 0.03, 0.05, 0.07, 0.10]:
    row = "%-12s" % ('无' if stop is None else '-%.0f%%' % (stop * 100))
    for N in (1, 2, 3, 5, 8):
        r = phase_rets(EV, K, N, stop)
        if len(r) < 30:
            row += "%14s" % '-'
            continue
        row += "%13s" % ('%+.2f%%(%.1f)' % ((r.mean() - COST) * 100, tstat(r, COST)))
    print(row, flush=True)


# ======================= E. 分层（修正换手率单位） =======================
print("\n" + "=" * 116)
print("E. 第5日买点分层（持有3日，换手率单位已修正为%）")
print("=" * 116)


def lay(sub, label, minn=25):
    r = phase_rets(sub, K, 3)
    if len(r) < minn:
        print("%-22s%8d   样本不足" % (label, len(r)))
        return None
    print("%-22s%8d%8.1f%%%10.2f%%%11.2f%%%9.2f"
          % (label, len(r), (r > 0).mean() * 100, r.mean() * 100,
             (r.mean() - COST) * 100, tstat(r, COST)))
    return (label, len(r), r.mean() - COST, tstat(r, COST))


cands = []
print("\n[连板高度]")
print("%-22s%8s%9s%11s%12s%9s" % ('分层', 'n', '胜率', '期望', '成本后', 't'))
print("-" * 116)
for lo, hi, nm in [(0, 1, '无连板/单板'), (2, 3, '二~三板'), (4, 6, '四~六板'), (7, 99, '七板以上')]:
    x = lay(EV[(EV['run_max'] >= lo) & (EV['run_max'] <= hi)], nm)
    if x:
        cands.append(x)

print("\n[市值]")
print("%-22s%8s%9s%11s%12s%9s" % ('分层', 'n', '胜率', '期望', '成本后', 't'))
print("-" * 116)
for lo, hi, nm in [(0, 30, '<30亿'), (30, 60, '30~60亿'), (60, 150, '60~150亿'), (150, 1e9, '>150亿')]:
    x = lay(EV[(EV['mcap'] >= lo) & (EV['mcap'] < hi)], nm)
    if x:
        cands.append(x)

print("\n[入池日换手率]")
print("%-22s%8s%9s%11s%12s%9s" % ('分层', 'n', '胜率', '期望', '成本后', 't'))
print("-" * 116)
for lo, hi, nm in [(0, 5, '<5%'), (5, 10, '5~10%'), (10, 20, '10~20%'), (20, 1e9, '>20%')]:
    x = lay(EV[(EV['turn'] >= lo) & (EV['turn'] < hi)], nm)
    if x:
        cands.append(x)

print("\n[板块]")
print("%-22s%8s%9s%11s%12s%9s" % ('分层', 'n', '胜率', '期望', '成本后', 't'))
print("-" * 116)
for bd in ['主板', '创业', '科创']:
    x = lay(EV[EV['board'] == bd], bd)
    if x:
        cands.append(x)

print("\n[回调深度: 第5日收盘距峰值]")
print("%-22s%8s%9s%11s%12s%9s" % ('分层', 'n', '胜率', '期望', '成本后', 't'))
print("-" * 116)
dd5 = []
for code, pk, pp in zip(EV['code'].values, EV['peak_i'].values, EV['peak_p'].values):
    j = int(pk) + K
    p = paths[code]
    dd5.append(p['c'][j] / pp - 1.0 if j < len(p['c']) and pp > 0 else np.nan)
EV['dd5'] = dd5
for lo, hi, nm in [(-0.15, 0, '浅调<15%'), (-0.25, -0.15, '中调15~25%'),
                   (-0.35, -0.25, '深调25~35%'), (-1, -0.35, '重挫>35%')]:
    x = lay(EV[(EV['dd5'] > lo) & (EV['dd5'] <= hi)], nm)
    if x:
        cands.append(x)

if cands:
    cands.sort(key=lambda x: -x[2])
    print("\nTop3 分层: ")
    for lb, nn, e, t in cands[:3]:
        print("  %-20s n=%-6d 成本后%+.2f%%  t=%.2f" % (lb, nn, e * 100, t))
    thr = 2.81 if len(cands) > 20 else 2.5
    print("Bonferroni 门槛(k=%d): |t|>%.2f -> 最优 %s"
          % (len(cands), thr, '通过' if abs(cands[0][3]) > thr else '未通过'))

EV.to_csv('events_lht_phase.csv', index=False, encoding='utf-8-sig')
print("\n完成，总用时 %.0fs" % (time.time() - t0))
