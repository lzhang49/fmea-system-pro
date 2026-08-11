"""
v6b —— 务实版「严重异动」全景 (v6 里 DEV10>=50% 仅17例导致全样本不足, 此处放宽到可统计口径)

复用 v6 扫描逻辑, 但把「严重异动」主口径设为 DEV10>=30% (n=224, 可统计), 并补:
  - 买入节点全矩阵 (T1/T2/T3/T5_OPEN × 持有1/2/3/5/8/10日)
  - 因子叠加 (低换手<5.2% + 非小盘>60亿)
  - 分时段稳定性 (H1 1-4月 / H2 5-8月)
同时保留 DEV10>=40% (n=64) 作为「更严格但仍可统计」的对照。

结论目标: 在可统计样本下, 看清「严重异动」到底有没有独立于连板的 alpha, 以及最优买点。
"""
import pandas as pd, numpy as np, os, glob

CACHE = "cache_daily"
im = pd.read_csv("industry_map.csv", dtype={'code': str})
name_map = dict(zip(im['code'], im['name']))

IDX = {}
for k in ['SH_MAIN', 'SZ_MAIN', 'CYB', 'KCB']:
    d = pd.read_csv(f"idx_{k}.csv")
    d['date'] = d['date'].astype(str).str[:10]
    d = d[d['date'] >= '2025-12-01'].reset_index(drop=True)
    d['r'] = d['close'].astype(float).pct_change()
    IDX[k] = dict(zip(d['date'], d['r']))


def board_of(c):
    if c.startswith('68'): return 'KCB'
    if c.startswith('30'): return 'CYB'
    if c.startswith('6'): return 'SH_MAIN'
    return 'SZ_MAIN'


def l1_thr(c, n):
    if c.startswith(('68', '30')): return 0.30
    if 'ST' in str(n).upper() or '*' in str(n): return 0.15
    return 0.20


def zt_thr(c, n):
    if 'ST' in str(n).upper() or '*' in str(n): return 0.05
    if c.startswith(('68', '30')): return 0.20
    return 0.10


print("扫描异动...", flush=True)
paths, evs, base_pts = {}, [], []
for fp in glob.glob(os.path.join(CACHE, "*.csv")):
    code = os.path.basename(fp)[:-4]
    try:
        df = pd.read_csv(fp)
    except Exception:
        continue
    if df is None or len(df) < 40:
        continue
    df['date'] = df['date'].astype(str).str[:10]
    o = df['open'].astype(float).values; c = df['close'].astype(float).values
    h = df['high'].astype(float).values; l = df['low'].astype(float).values
    ds = list(df['date'].values); n = len(df)
    nm = name_map.get(code, '')
    osh = df['outstanding_share'].astype(float).values
    tov = df['turnover'].astype(float).values
    paths[code] = dict(o=o, c=c, h=h, l=l, d=ds)

    ir = IDX[board_of(code)]
    dev = np.zeros(n)
    for i in range(1, n):
        if c[i-1] > 0:
            dev[i] = (c[i]/c[i-1]-1) - ir.get(ds[i], 0.)

    t1 = l1_thr(code, nm); tz = zt_thr(code, nm)
    run_arr = np.zeros(n, dtype=int); run = 0
    for i in range(1, n):
        if c[i-1] <= 0:
            run = 0
        else:
            run = run+1 if (c[i]/c[i-1]-1) >= tz-0.004 else 0
        run_arr[i] = run

    l1_all, last = [], -999
    for i in range(3, n):
        d3 = dev[i-2:i+1].sum()
        if d3 < t1:
            continue
        l1_all.append(i)
        if i - last < 10:
            continue
        last = i
        d10 = dev[max(0, i-9):i+1].sum()
        hits = sum(1 for x in l1_all if i-9 <= x <= i)
        evs.append(dict(code=code, name=nm, i=i, date=ds[i],
                        dev3=d3, dev10=d10, hits=hits,
                        run=int(run_arr[i]), run_max=int(run_arr[max(0, i-4):i+1].max()),
                        mcap=c[i]*osh[i]/1e8, turn=tov[i]))
    for i in range(20, n-25, 5):
        base_pts.append(dict(code=code, i=i, turn=tov[i], mcap=c[i]*osh[i]/1e8))

EV = pd.DataFrame(evs)
print(f"异动事件 {len(EV)}\n")


def rets(evdf, mode='T1_OPEN', N=3):
    out = []
    for _, r in evdf.iterrows():
        p = paths[r['code']]; o, c = p['o'], p['c']; nn = len(c)
        i = int(r['i'])
        k = {'T1_OPEN': i+1, 'T2_OPEN': i+2, 'T3_OPEN': i+3, 'T5_OPEN': i+5}.get(mode, i+1)
        if k >= nn or o[k] <= 0 or k+N >= nn:
            continue
        out.append(c[k+N]/o[k]-1.0)
    return np.array(out)


def summ(r):
    if len(r) < 20:
        return None
    w, ls = r[r > 0], r[r <= 0]
    wr = len(w)/len(r)
    aw = float(w.mean()) if len(w) else 0.
    al = float(ls.mean()) if len(ls) else 0.
    sd = r.std(ddof=1)
    return dict(n=len(r), wr=wr, aw=aw, al=al,
                pl=(aw/abs(al)) if al else np.inf,
                exp=float(r.mean()), med=float(np.median(r)),
                t=float(r.mean()/(sd/np.sqrt(len(r)))) if sd > 0 else 0.)


# 基准: 全池随机日 T+1开盘
bl = []
for code, p in paths.items():
    o, c = p['o'], p['c']; nn = len(c)
    for i in range(20, nn-25, 5):
        k = i+1
        if k+3 >= nn or o[k] <= 0:
            continue
        bl.append(c[k+3]/o[k]-1.0)
BASE = np.array(bl); bexp = BASE.mean()
print(f"基准(全池随机日 T+1开盘 持有3日): n={len(BASE)} 期望={bexp*100:+.2f}%\n")


def matrix(sev, label):
    print(f"\n=== {label}  (n={len(sev)}, 涉及{sev['code'].nunique()}只) ===")
    hdr = f"{'持有':>4} | " + " ".join(f"{m:>9}" for m in ['T1_OPEN', 'T2_OPEN', 'T3_OPEN', 'T5_OPEN'])
    print(hdr)
    for N in [1, 2, 3, 5, 8, 10]:
        row = f"{N:>3}日 | "
        for mode in ['T1_OPEN', 'T2_OPEN', 'T3_OPEN', 'T5_OPEN']:
            s = summ(rets(sev, mode, N))
            if not s:
                row += f"{'不足':>9} "
            else:
                e = s['exp']-bexp-0.003
                row += f"{e*100:+6.2f}%".ljust(9) + " "
        print(row)
    # 明细: 各节点最优(成本后超额最大)
    best = []
    for mode in ['T1_OPEN', 'T2_OPEN', 'T3_OPEN', 'T5_OPEN']:
        for N in [1, 2, 3, 5, 8, 10]:
            s = summ(rets(sev, mode, N))
            if s:
                best.append((s['exp']-bexp-0.003, mode, N, s['wr'], s['pl'], s['t'], s['n']))
    best.sort(reverse=True)
    print("Top3 节点(成本后超额):")
    for e, mode, N, wr, pl, t, n in best[:3]:
        print(f"   {mode} 持有{N}日: 超额{e*100:+.2f}%  胜率{wr*100:.1f}%  盈亏比{pl:.2f}  t={t:.2f}  n={n}")


print("#"*100)
print("# D. 买点全矩阵 (务实严重口径)")
print("#"*100)
SEV30 = EV[EV['dev10'] >= 0.30]
SEV40 = EV[EV['dev10'] >= 0.40]
matrix(SEV30, "严重异动 DEV10>=30% (主口径, 可统计)")
matrix(SEV40, "严重异动 DEV10>=40% (更严格)")

print("\n" + "#"*100)
print("# F. 严重异动 + v3因子 (DEV10>=30%)")
print("#"*100)
print(f"{'口径':<34}{'样本':>7}{'胜率':>8}{'盈亏比':>8}{'期望':>8}{'中位数':>9}{'超额':>8}{'成本后':>8}{'t值':>7}")
for lab, sub in [
    ('严重异动 DEV10>=30% 无过滤',     SEV30),
    ('+ 低换手<5.2%',                  SEV30[SEV30['turn'] < 0.052]),
    ('+ 非小盘>60亿',                  SEV30[SEV30['mcap'] >= 60]),
    ('+ 低换手+非小盘',                SEV30[(SEV30['turn'] < 0.052) & (SEV30['mcap'] >= 60)]),
    ('+ 连板(run>=2)+低换手+非小盘',    SEV30[(SEV30['run_max'] >= 2) & (SEV30['turn'] < 0.052) & (SEV30['mcap'] >= 60)]),
]:
    s = summ(rets(sub))
    if not s:
        print(f"{lab:<34}{len(sub):>7}   样本不足(<20)"); continue
    e = s['exp']-bexp
    print(f"{lab:<34}{s['n']:>7}{s['wr']*100:>7.1f}%{s['pl']:>8.2f}{s['exp']*100:>7.2f}%"
          f"{s['med']*100:>8.2f}%{e*100:>+7.2f}%{(e-0.003)*100:>+7.2f}%{s['t']:>7.2f}")

print("\n" + "#"*100)
print("# G. 分时段稳定性 (DEV10>=30%)")
print("#"*100)
SEV30 = SEV30.copy(); SEV30['half'] = np.where(SEV30['date'] < '2026-05-01', 'H1', 'H2')
for h in ['H1', 'H2']:
    s = summ(rets(SEV30[SEV30['half'] == h]))
    if not s:
        print(f"{h}: 样本不足({len(SEV30[SEV30['half']==h])})"); continue
    print(f"{h}  样本{s['n']:>4}  胜率{s['wr']*100:>5.1f}%  期望{s['exp']*100:>6.2f}%  "
          f"中位{s['med']*100:>6.2f}%  超额{(s['exp']-bexp)*100:>+6.2f}%  t={s['t']:.2f}")

print("\n完成。明细见 events_v6.csv (v6已生成)")
