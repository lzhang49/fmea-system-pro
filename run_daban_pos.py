# -*- coding: utf-8 -*-
"""
二板【日线位置】研究 + 分位置排板胜率回测  (2021-08 ~ 2026-08, 全A 5189 只)
============================================================================
核心问题：二板在日线上处于什么位置时，打板（涨停价排队）胜率更高？
  实战常见说法：
    "低位首次启动的二板才能打，高位加速的二板是接盘"
    "突破年内新高的二板空间打开"
    "均线粘合后启动的二板最好"
  本脚本用全 A 五年数据逐条检验。

【关键设计：位置必须在"启动前"测量】
  二板日自带两个涨停(+20%~+40%)，若用二板日收盘算位置，任何二板都会显示"高位"，
  这是同义反复。因此位置基准日 j = i - run（首板前一日），
  所有位置指标只用 j 日及之前的数据 → 严格无未来信息，且反映"从哪起涨"。

位置指标（全部在 j 日计算）：
  pos120     : 收盘在过去120日 [最低,最高] 区间中的百分位  0=地板 1=天花板
  d_high250  : 距过去250日最高价的距离 (负数, 越接近0越贴近年内高点)
  ma20dev    : 相对20日均线偏离
  ma60dev    : 相对60日均线偏离
  ret20/ret60: 启动前 20/60 日涨幅（已经涨了多少才启动）
  zt60       : 过去60日涨停次数（0=首次启动, 多=反复炒作）
  amp20      : 过去20日平均振幅（活跃度）
  bull       : MA5>MA20>MA60 多头排列
  newhigh    : 二板日收盘是否创250日新高（含二板本身，作"突破"标签）

成交模型与 run_daban_5y.py 完全一致：涨停价排板、一字/T字弃单、成本0.3%。
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


print("=" * 112)
print("1. 载入全 A 日线，扫描连板事件并计算【启动前】日线位置")
print("=" * 112, flush=True)

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

    # ---------- 滚动位置序列（当日及之前，无未来信息） ----------
    cs = pd.Series(c)
    hs = pd.Series(h)
    ls = pd.Series(l)
    hi120 = hs.rolling(120, min_periods=60).max().values
    lo120 = ls.rolling(120, min_periods=60).min().values
    hi250 = hs.rolling(250, min_periods=100).max().values
    ma5 = cs.rolling(5, min_periods=5).mean().values
    ma20 = cs.rolling(20, min_periods=20).mean().values
    ma60 = cs.rolling(60, min_periods=40).mean().values
    amp = pd.Series((h - l) / np.maximum(c, 1e-9)).rolling(20, min_periods=15).mean().values
    zt_cnt = pd.Series(isz.astype(float)).rolling(60, min_periods=30).sum().values

    hit = np.where(isz & (run >= 1) & (run <= 6))[0]
    for i in hit:
        if i < WARMUP or d[i] < BT_START or i + 6 >= n:
            continue
        j = i - int(run[i])                       # 启动前一日
        if j < 60:
            continue
        rng = hi120[j] - lo120[j]
        if not np.isfinite(rng) or rng <= 0 or not np.isfinite(hi250[j]) or c[j] <= 0:
            continue
        pos120 = (c[j] - lo120[j]) / rng
        d_high = c[j] / hi250[j] - 1.0
        m20 = c[j] / ma20[j] - 1.0 if np.isfinite(ma20[j]) and ma20[j] > 0 else np.nan
        m60 = c[j] / ma60[j] - 1.0 if np.isfinite(ma60[j]) and ma60[j] > 0 else np.nan
        r20 = c[j] / c[j - 20] - 1.0 if j >= 20 and c[j - 20] > 0 else np.nan
        r60 = c[j] / c[j - 60] - 1.0 if j >= 60 and c[j - 60] > 0 else np.nan
        bull = (np.isfinite(ma5[j]) and np.isfinite(ma20[j]) and np.isfinite(ma60[j])
                and ma5[j] > ma20[j] > ma60[j])
        nh = c[i] >= hi250[i - 1] - 1e-9 if np.isfinite(hi250[i - 1]) else False
        EV.append((code, int(i), d[i], int(run[i]), c[i] * osh[i] / 1e8, tov[i],
                   board_of(code), pos120, d_high, m20, m60, r20, r60,
                   zt_cnt[j] if np.isfinite(zt_cnt[j]) else np.nan,
                   amp[j] if np.isfinite(amp[j]) else np.nan,
                   bool(bull), bool(nh)))

    if (idx + 1) % 1500 == 0:
        print("  已处理 %d/%d  事件 %d  用时%.0fs"
              % (idx + 1, len(files), len(EV), time.time() - t0), flush=True)

EVDF = pd.DataFrame(EV, columns=['code', 'i', 'date', 'run', 'mcap', 'turn', 'board',
                                 'pos120', 'd_high', 'ma20dev', 'ma60dev',
                                 'ret20', 'ret60', 'zt60', 'amp20', 'bull', 'newhigh'])
EVDF['year'] = EVDF['date'].str[:4]
print("\n载入完成: %d 只股票, 连板事件 %d 个, 用时 %.0fs"
      % (len(paths), len(EVDF), time.time() - t0))

E2 = EVDF[EVDF['run'] == 2].copy()
print("二板事件: %d 个" % len(E2))


# ======================= 2. 打板成交与收益模型 (与 run_daban_5y 一致) =======================
def daban(code, i, N=1, sell='close', stop_bomb=False):
    p = paths[code]
    o, h, l, c = p['o'], p['h'], p['l'], p['c']
    nn = len(c)
    k = i + 1
    if k + max(N, 1) >= nn or c[i] <= 0 or o[k] <= 0:
        return 'short', None
    tz = zt_thr(code)
    ztp = round(c[i] * (1 + tz) + 1e-9, 2)
    if h[k] < ztp - 0.005:
        return 'nohit', None
    if l[k] >= ztp - 0.005:
        return 'yizi', None
    for dd in range(k, min(k + N + 1, nn)):
        if c[dd - 1] > 0 and (c[dd] / c[dd - 1] - 1.0) < -(tz + 0.05):
            return 'xr', None
    entry = ztp
    sealed = c[k] >= ztp - 0.005
    if stop_bomb and not sealed:
        return 'ok', c[k] / entry - 1.0
    if N == 0:
        return 'ok', o[k + 1] / entry - 1.0
    px = c[k + N] if sell == 'close' else o[k + N]
    return 'ok', px / entry - 1.0


def scan(evdf, N=1, sell='close', stop_bomb=False):
    rets = []
    for code, i in zip(evdf['code'].values, evdf['i'].values):
        s, r = daban(code, int(i), N, sell, stop_bomb)
        if r is not None:
            rets.append(r)
    return np.array(rets)


def open_buy(code, i, N=3):
    """对照口径：次日开盘买、持有 N 日收盘卖（跳过一字板）"""
    p = paths[code]
    o, h, l, c = p['o'], p['h'], p['l'], p['c']
    nn = len(c)
    k = i + 1
    if k + N >= nn or c[i] <= 0 or o[k] <= 0:
        return None
    tz = zt_thr(code)
    ztp = round(c[i] * (1 + tz) + 1e-9, 2)
    if o[k] >= ztp - 0.005 and l[k] >= ztp - 0.005:
        return None                      # 一字板买不到
    return c[k + N] / o[k] - 1.0


def scan_open(evdf, N=3):
    out = []
    for code, i in zip(evdf['code'].values, evdf['i'].values):
        r = open_buy(code, int(i), N)
        if r is not None:
            out.append(r)
    return np.array(out)


def seal_stats(evdf):
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


# ======================= 3. 基准 =======================
print("\n构建基准组...", flush=True)
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
B3 = base_rets(N=3)
print("基准 N=1 %d 笔 均值%.2f%% | N=3 %d 笔 均值%.2f%%"
      % (len(B1), B1.mean() * 100, len(B3), B3.mean() * 100), flush=True)


# ======================= 4. 二板位置画像 =======================
print("\n" + "=" * 112)
print("2. 二板【启动前】日线位置画像 — 这些票到底从哪起涨的")
print("=" * 112)
q = E2[['pos120', 'd_high', 'ma20dev', 'ma60dev', 'ret20', 'ret60', 'zt60', 'amp20']].describe(
    percentiles=[.1, .25, .5, .75, .9])
print(q.to_string(float_format=lambda x: '%.3f' % x))
print("\n多头排列占比: %.1f%%   二板创250日新高占比: %.1f%%"
      % (E2['bull'].mean() * 100, E2['newhigh'].mean() * 100))


# ======================= 5. 位置分层：晋级率 + 打板收益 =======================
def layer_table(groups, title, N=1):
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)
    print("%-24s%8s%9s%9s%9s%9s%9s%9s%10s%8s" %
          ('分组', '事件n', '触板率', '封板率', '炸板率', '成交n', '胜率', '期望',
           '成本后超额', 't'))
    print("-" * 112)
    out = []
    for lbl, sub in groups:
        if len(sub) < 30:
            print("%-24s%8d   样本不足" % (lbl, len(sub)))
            continue
        ss = seal_stats(sub)
        r = scan(sub, N=N)
        s = summ(r, B1 if N == 1 else base_rets(N=N), lbl)
        if s is None:
            print("%-24s%8d  触板成交样本不足" % (lbl, len(sub)))
            continue
        out.append((lbl, s))
        print("%-24s%8d%8.1f%%%8.1f%%%8.1f%%%9d%8.1f%%%8.2f%%%9.2f%%%8.2f" %
              (lbl, len(sub), ss['hit'] * 100, ss['seal'] * 100, ss['bomb'] * 100,
               s['n'], s['wr'] * 100, s['exp'] * 100, s['cost'] * 100, s['t']))
    return out


# --- 5.1 区间位置 pos120 ---
g = []
for lo, hi, lbl in [(-9, .2, 'A 地板区 pos<20%'), (.2, .4, 'B 低位 20~40%'),
                    (.4, .6, 'C 中位 40~60%'), (.6, .8, 'D 高位 60~80%'),
                    (.8, 9, 'E 天花板 pos>80%')]:
    g.append((lbl, E2[(E2['pos120'] >= lo) & (E2['pos120'] < hi)]))
L1 = layer_table(g, "3. 位置分层① — 启动前收盘在120日区间的百分位（0=地板 1=天花板）")

# --- 5.2 距年内高点 ---
g = []
for lo, hi, lbl in [(-9, -.5, 'A 深跌 距高点<-50%'), (-.5, -.3, 'B -50~-30%'),
                    (-.3, -.15, 'C -30~-15%'), (-.15, -.05, 'D -15~-5%'),
                    (-.05, 9, 'E 贴年高 >-5%')]:
    g.append((lbl, E2[(E2['d_high'] >= lo) & (E2['d_high'] < hi)]))
L2 = layer_table(g, "4. 位置分层② — 启动前距过去250日最高价的距离")

# --- 5.3 均线偏离 ---
g = []
for lo, hi, lbl in [(-9, -.1, 'A 深跌破MA60 <-10%'), (-.1, -.02, 'B -10~-2%'),
                    (-.02, .05, 'C 贴MA60 -2~5%'), (.05, .2, 'D 上方 5~20%'),
                    (.2, 9, 'E 远离MA60 >20%')]:
    g.append((lbl, E2[(E2['ma60dev'] >= lo) & (E2['ma60dev'] < hi)]))
L3 = layer_table(g, "5. 位置分层③ — 启动前相对 60 日均线偏离")

# --- 5.4 启动前涨幅（是否已经涨过一波） ---
g = []
for lo, hi, lbl in [(-9, -.2, 'A 前60日跌>20%'), (-.2, 0, 'B 跌0~20%'),
                    (0, .2, 'C 涨0~20%'), (.2, .5, 'D 涨20~50%'),
                    (.5, 9, 'E 已涨>50%')]:
    g.append((lbl, E2[(E2['ret60'] >= lo) & (E2['ret60'] < hi)]))
L4 = layer_table(g, "6. 位置分层④ — 启动前 60 日累计涨幅（高位加速 vs 低位启动）")

# --- 5.5 近60日涨停次数（首次启动 vs 反复炒作） ---
g = []
for lo, hi, lbl in [(-1, 1, 'A 0次 首次启动'), (1, 3, 'B 1~2次'),
                    (3, 6, 'C 3~5次'), (6, 99, 'D 6次以上 反复炒作')]:
    g.append((lbl, E2[(E2['zt60'] >= lo) & (E2['zt60'] < hi)]))
L5 = layer_table(g, "7. 位置分层⑤ — 启动前 60 日内涨停次数")

# --- 5.6 形态标签 ---
g = [('多头排列', E2[E2['bull']]), ('非多头排列', E2[~E2['bull']]),
     ('二板创250日新高', E2[E2['newhigh']]), ('未创新高', E2[~E2['newhigh']]),
     ('低波动 amp<3%', E2[E2['amp20'] < .03]), ('高波动 amp>=5%', E2[E2['amp20'] >= .05])]
L6 = layer_table(g, "8. 位置分层⑥ — 形态标签")


# ======================= 6. 位置 × 位置 交叉矩阵 =======================
print("\n" + "=" * 112)
print("9. 交叉矩阵 — 区间位置 × 启动前60日涨幅  (格内: 成本后超额% / 封板率% / n)")
print("=" * 112)
pos_bins = [(-9, .3, '低位<30%'), (.3, .6, '中位30~60%'), (.6, .85, '高位60~85%'), (.85, 9, '顶部>85%')]
ret_bins = [(-9, 0, '前跌'), (0, .2, '涨0~20%'), (.2, .5, '涨20~50%'), (.5, 9, '涨>50%')]
print("%-16s" % '', end='')
for _, _, rl in ret_bins:
    print("%22s" % rl, end='')
print()
for plo, phi, pl_ in pos_bins:
    print("%-16s" % pl_, end='')
    for rlo, rhi, _ in ret_bins:
        sub = E2[(E2['pos120'] >= plo) & (E2['pos120'] < phi) &
                 (E2['ret60'] >= rlo) & (E2['ret60'] < rhi)]
        if len(sub) < 30:
            print("%22s" % ('n%d 不足' % len(sub)), end='')
            continue
        ss = seal_stats(sub)
        s = summ(scan(sub, N=1), B1)
        if s is None:
            print("%22s" % '成交不足', end='')
            continue
        print("%22s" % ('%+.2f%%/%.0f%%/n%d' % (s['cost'] * 100, ss['seal'] * 100, s['n'])), end='')
    print()


# ======================= 7. 最优/最差层的分年度稳定性 =======================
print("\n" + "=" * 112)
print("10. 位置分层的逐年稳定性 (成本后超额%，检验是否只在某年有效)")
print("=" * 112)
cands = [('低位 pos<30%', E2[E2['pos120'] < .3]),
         ('顶部 pos>85%', E2[E2['pos120'] > .85]),
         ('深跌 距高点<-50%', E2[E2['d_high'] < -.5]),
         ('贴年高 >-5%', E2[E2['d_high'] > -.05]),
         ('首次启动 zt60=0', E2[E2['zt60'] < 1]),
         ('反复炒作 zt60>=6', E2[E2['zt60'] >= 6])]
years = sorted(E2['year'].unique())
print("%-22s" % '分组', end='')
for y in years:
    print("%12s" % y, end='')
print("%12s" % '全期')
print("-" * 112)
for lbl, sub in cands:
    print("%-22s" % lbl, end='')
    for y in years:
        s = summ(scan(sub[sub['year'] == y], N=1), base_rets(year=y, N=1))
        print("%12s" % ('%+.2f%%' % (s['cost'] * 100) if s else '--'), end='')
    s = summ(scan(sub, N=1), B1)
    print("%12s" % ('%+.2f%%' % (s['cost'] * 100) if s else '--'))


# ======================= 8. 对照：同样的位置分层，改用"次日开盘买" =======================
print("\n" + "=" * 112)
print("11. 对照口径 — 同样位置分层，改成【次日开盘买 / 持有3日】(排除打板机制影响)")
print("=" * 112)
print("%-24s%9s%9s%9s%11s%8s" % ('分组', '成交n', '胜率', '期望', '成本后超额', 't'))
print("-" * 112)
for lbl, sub in [('低位 pos<30%', E2[E2['pos120'] < .3]),
                 ('中位 30~60%', E2[(E2['pos120'] >= .3) & (E2['pos120'] < .6)]),
                 ('高位 60~85%', E2[(E2['pos120'] >= .6) & (E2['pos120'] < .85)]),
                 ('顶部 >85%', E2[E2['pos120'] >= .85]),
                 ('首次启动 zt60=0', E2[E2['zt60'] < 1]),
                 ('反复炒作 zt60>=6', E2[E2['zt60'] >= 6]),
                 ('二板创250日新高', E2[E2['newhigh']]),
                 ('未创新高', E2[~E2['newhigh']])]:
    r = scan_open(sub, N=3)
    s = summ(r, B3, lbl)
    if s is None:
        print("%-24s  样本不足" % lbl)
        continue
    print("%-24s%9d%8.1f%%%8.2f%%%10.2f%%%8.2f" %
          (lbl, s['n'], s['wr'] * 100, s['exp'] * 100, s['cost'] * 100, s['t']))


# ======================= 9. 位置 + 已验证因子 组合 =======================
print("\n" + "=" * 112)
print("12. 位置 + 换手/市值 组合筛选（打板口径）")
print("=" * 112)
print("%-40s%9s%9s%9s%11s%8s" % ('组合', '成交n', '胜率', '期望', '成本后超额', 't'))
print("-" * 112)
combos = [
    ('低位<30% + 低换手<5.2%', E2[(E2['pos120'] < .3) & (E2['turn'] < .052)]),
    ('低位<30% + 大市值>=60亿', E2[(E2['pos120'] < .3) & (E2['mcap'] >= 60)]),
    ('低位<30% + 首次启动', E2[(E2['pos120'] < .3) & (E2['zt60'] < 1)]),
    ('低位<30% + 首次启动 + 大市值', E2[(E2['pos120'] < .3) & (E2['zt60'] < 1) & (E2['mcap'] >= 60)]),
    ('贴年高 + 大市值>=60亿', E2[(E2['d_high'] > -.05) & (E2['mcap'] >= 60)]),
    ('创新高 + 低换手', E2[E2['newhigh'] & (E2['turn'] < .052)]),
    ('多头排列 + 低位<40%', E2[E2['bull'] & (E2['pos120'] < .4)]),
    ('深跌<-50% + 首次启动', E2[(E2['d_high'] < -.5) & (E2['zt60'] < 1)]),
]
for lbl, sub in combos:
    s = summ(scan(sub, N=1), B1, lbl)
    if s is None:
        print("%-40s  样本不足 (事件%d)" % (lbl, len(sub)))
        continue
    print("%-40s%9d%8.1f%%%8.2f%%%10.2f%%%8.2f" %
          (lbl, s['n'], s['wr'] * 100, s['exp'] * 100, s['cost'] * 100, s['t']))

n_tests = len(L1) + len(L2) + len(L3) + len(L4) + len(L5) + len(L6) + len(combos)
print("\n[多重比较提示] 本轮共检验约 %d 个分层，Bonferroni 校正后需 |t| > %.2f 才算显著"
      % (n_tests, 2.807 if n_tests > 20 else 2.5))

E2.to_csv('events_pos_2b.csv', index=False, encoding='utf-8-sig')
print("\n带位置字段的二板事件表已导出: events_pos_2b.csv (%d 行)" % len(E2))
print("总用时 %.1f 分钟" % ((time.time() - t0) / 60))
