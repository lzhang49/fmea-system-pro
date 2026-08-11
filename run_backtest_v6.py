"""
v6 —— 聚焦「严重异动」: 严重度梯度扫描 + 买点 + 独立alpha检验

v5 里 L2(交易所严重异常波动标准) 只有 11 例, 无法统计。
本版把「严重度」做成连续梯度, 找到样本量与信号强度的平衡点, 并回答:
  Q1 严重异动是继续冲高(动量) 还是见顶回落(反转)?
  Q2 严重程度提高, 收益是单调改善还是恶化?
  Q3 剔除连板属性后, 严重异动是否仍有独立 alpha?

严重度口径(均为「上涨方向」):
  DEV10_x  : 连续10日累计涨幅偏离值 >= x%     (交易所严重标准 = 100%)
  HITS_k   : 连续10日内触发 k 次 L1 异常波动  (交易所严重标准 = 3次)
  DEV3_x   : 单次3日累计偏离 >= x% (超强单波)
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


print("扫描异动 + 计算严重度...", flush=True)
paths, evs, base_pts = {}, [], []
for fp in glob.glob(os.path.join(CACHE, "*.csv")):
    code = os.path.basename(fp)[:-4]
    try: df = pd.read_csv(fp)
    except Exception: continue
    if df is None or len(df) < 40: continue
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
        if c[i-1] > 0: dev[i] = (c[i]/c[i-1]-1) - ir.get(ds[i], 0.)

    t1 = l1_thr(code, nm); tz = zt_thr(code, nm)
    # 连板状态
    run_arr = np.zeros(n, dtype=int); run = 0
    for i in range(1, n):
        if c[i-1] <= 0: run = 0
        else:
            run = run+1 if (c[i]/c[i-1]-1) >= tz-0.004 else 0
        run_arr[i] = run

    l1_all, last = [], -999
    for i in range(3, n):
        d3 = dev[i-2:i+1].sum()
        if d3 < t1: continue
        l1_all.append(i)
        if i - last < 10: continue
        last = i
        d10 = dev[max(0, i-9):i+1].sum()
        hits = sum(1 for x in l1_all if i-9 <= x <= i)
        evs.append(dict(code=code, name=nm, i=i, date=ds[i],
                        dev3=d3, dev10=d10, hits=hits,
                        run=int(run_arr[i]), run_max=int(run_arr[max(0,i-4):i+1].max()),
                        mcap=c[i]*osh[i]/1e8, turn=tov[i]))
    for i in range(20, n-25, 5):
        base_pts.append(dict(code=code, i=i, turn=tov[i], mcap=c[i]*osh[i]/1e8))

EV = pd.DataFrame(evs); BS = pd.DataFrame(base_pts)
print(f"异动事件 {len(EV)} / 基准点 {len(BS)}\n")

print("#"*112)
print("# A. 严重度梯度 —— 样本量分布")
print("#"*112)
print(f"{'严重度口径':<26}{'事件数':>8}{'占比':>8}{'涉及个股':>9}")
GRADES = []
for x in [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]:
    m = EV['dev10'] >= x
    GRADES.append((f'DEV10>={x*100:.0f}%', m))
    print(f"{f'10日累计偏离 >= {x*100:.0f}%':<26}{m.sum():>8}{m.mean()*100:>7.1f}%{EV[m]['code'].nunique():>9}")
for k in [1, 2, 3]:
    m = EV['hits'] >= k
    GRADES.append((f'HITS>={k}', m))
    print(f"{f'10日内 L1 触发 >= {k} 次':<26}{m.sum():>8}{m.mean()*100:>7.1f}%{EV[m]['code'].nunique():>9}")
for x in [0.25, 0.30, 0.35, 0.40]:
    m = EV['dev3'] >= x
    GRADES.append((f'DEV3>={x*100:.0f}%', m))
    print(f"{f'单波3日偏离 >= {x*100:.0f}%':<26}{m.sum():>8}{m.mean()*100:>7.1f}%{EV[m]['code'].nunique():>9}")
m_off = (EV['dev10'] >= 1.00) | (EV['hits'] >= 3)
print(f"{'【交易所严重标准】':<26}{m_off.sum():>8}{m_off.mean()*100:>7.1f}%{EV[m_off]['code'].nunique():>9}")


def rets(evdf, mode='T1_OPEN', N=3):
    out = []
    for _, r in evdf.iterrows():
        p = paths[r['code']]; o, c, h, l = p['o'], p['c'], p['h'], p['l']; nn = len(c)
        i = int(r['i'])
        if mode == 'T1_OPEN': k = i+1
        elif mode == 'T2_OPEN': k = i+2
        elif mode == 'T3_OPEN': k = i+3
        elif mode == 'T5_OPEN': k = i+5
        else: k = i+1
        if k >= nn or o[k] <= 0 or k+N >= nn: continue
        out.append(c[k+N]/o[k]-1.0)
    return np.array(out)


def summ(r):
    if len(r) < 20: return None
    w, ls = r[r > 0], r[r <= 0]
    wr = len(w)/len(r)
    aw = float(w.mean()) if len(w) else 0.; al = float(ls.mean()) if len(ls) else 0.
    sd = r.std(ddof=1)
    return dict(n=len(r), wr=wr, aw=aw, al=al, pl=(aw/abs(al)) if al else np.inf,
                exp=float(r.mean()), med=float(np.median(r)),
                t=float(r.mean()/(sd/np.sqrt(len(r)))) if sd > 0 else 0.)


BASE3 = rets(BS.rename(columns={'i': 'i'}).assign(code=BS.index.map(lambda x: None)) if False else BS.assign(code=[None]*len(BS)), N=3) if False else None
# 基准: 直接用 base_pts 计算
bret = []
for _, r in BS.iterrows(): pass
bl = []
for code, p in paths.items():
    o, c = p['o'], p['c']; nn = len(c)
    for i in range(20, nn-25, 5):
        k = i+1
        if k+3 >= nn or o[k] <= 0: continue
        bl.append(c[k+3]/o[k]-1.0)
BASE = np.array(bl); bexp = BASE.mean()
print(f"\n基准(全池随机日 T+1开盘买 持有3日): n={len(BASE)}  期望={bexp*100:+.2f}%  中位={np.median(BASE)*100:+.2f}%")

print("\n" + "#"*112)
print("# B. 严重度 vs 收益 (T+1开盘买, 持有3日, 成本双边0.3%)")
print("#"*112)
print(f"{'严重度口径':<20}{'样本':>7}{'胜率':>8}{'均盈':>8}{'均亏':>8}{'盈亏比':>7}"
      f"{'期望':>8}{'中位数':>9}{'超额':>8}{'成本后':>8}{'t值':>7}")
for lab, m in GRADES:
    s = summ(rets(EV[m]))
    if not s: 
        print(f"{lab:<20}{int(m.sum()):>7}   样本不足(<20)"); continue
    e = s['exp']-bexp
    print(f"{lab:<20}{s['n']:>7}{s['wr']*100:>7.1f}%{s['aw']*100:>7.2f}%{s['al']*100:>7.2f}%"
          f"{s['pl']:>7.2f}{s['exp']*100:>7.2f}%{s['med']*100:>8.2f}%{e*100:>+7.2f}%"
          f"{(e-0.003)*100:>+7.2f}%{s['t']:>7.2f}")
s = summ(rets(EV[m_off]))
if s:
    e = s['exp']-bexp
    print(f"{'【交易所严重标准】':<20}{s['n']:>7}{s['wr']*100:>7.1f}%{s['aw']*100:>7.2f}%{s['al']*100:>7.2f}%"
          f"{s['pl']:>7.2f}{s['exp']*100:>7.2f}%{s['med']*100:>8.2f}%{e*100:>+7.2f}%"
          f"{(e-0.003)*100:>+7.2f}%{s['t']:>7.2f}")

print("\n" + "#"*112)
print("# C. 严重档 vs 温和档 直接对比 (分位切分, 各档互斥)")
print("#"*112)
print(f"{'10日累计偏离区间':<22}{'样本':>7}{'胜率':>8}{'盈亏比':>8}{'期望':>8}{'中位数':>9}{'超额':>8}{'成本后':>8}{'t值':>7}")
BINS = [(0.0, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.70), (0.70, 9.99)]
for lo, hi in BINS:
    m = (EV['dev10'] >= lo) & (EV['dev10'] < hi)
    s = summ(rets(EV[m]))
    if not s:
        print(f"{f'{lo*100:.0f}% ~ {hi*100:.0f}%':<22}{int(m.sum()):>7}   样本不足"); continue
    e = s['exp']-bexp
    lab = f'{lo*100:.0f}% ~ {hi*100:.0f}%' if hi < 9 else f'>= {lo*100:.0f}%'
    print(f"{lab:<22}{s['n']:>7}{s['wr']*100:>7.1f}%{s['pl']:>8.2f}{s['exp']*100:>7.2f}%"
          f"{s['med']*100:>8.2f}%{e*100:>+7.2f}%{(e-0.003)*100:>+7.2f}%{s['t']:>7.2f}")

print("\n" + "#"*112)
print("# D. 严重异动的买入节点 (口径: DEV10>=50%, 样本量与显著性的平衡点)")
print("#"*112)
SEV = EV[EV['dev10'] >= 0.50]
print(f"严重异动样本: {len(SEV)} 事件, {SEV['code'].nunique()} 只")
for N in [1, 2, 3, 5, 8, 10]:
    bl2 = []
    for code, p in paths.items():
        o, c = p['o'], p['c']; nn = len(c)
        for i in range(20, nn-25, 5):
            k = i+1
            if k+N >= nn or o[k] <= 0: continue
            bl2.append(c[k+N]/o[k]-1.0)
    b2 = np.mean(bl2)
    line = f"持有{N:>2}日 | "
    for mode in ['T1_OPEN', 'T2_OPEN', 'T3_OPEN', 'T5_OPEN']:
        s = summ(rets(SEV, mode, N))
        if not s: line += f"{mode}: -    "; continue
        line += f"{mode}:{(s['exp']-b2-0.003)*100:+6.2f}%(胜{s['wr']*100:.0f}%) "
    print(line)

print("\n" + "#"*112)
print("# E. ★独立alpha检验: 严重异动 中 有无连板属性的差异")
print("#"*112)
print(f"{'子集':<28}{'样本':>7}{'胜率':>8}{'盈亏比':>8}{'期望':>8}{'中位数':>9}{'超额':>8}{'成本后':>8}{'t值':>7}")
for lab, sub in [
    ('严重异动 ∩ 当日连板(run>=2)', SEV[SEV['run'] >= 2]),
    ('严重异动 ∩ 近5日有连板',      SEV[(SEV['run_max'] >= 2) & (SEV['run'] < 2)]),
    ('严重异动 ∩ 无连板',          SEV[SEV['run_max'] < 2]),
    ('—— 对照: 全部异动 ∩ 无连板', EV[EV['run_max'] < 2]),
    ('—— 对照: 全部异动 ∩ 连板',   EV[EV['run_max'] >= 2]),
]:
    s = summ(rets(sub))
    if not s:
        print(f"{lab:<28}{len(sub):>7}   样本不足(<20)"); continue
    e = s['exp']-bexp
    print(f"{lab:<28}{s['n']:>7}{s['wr']*100:>7.1f}%{s['pl']:>8.2f}{s['exp']*100:>7.2f}%"
          f"{s['med']*100:>8.2f}%{e*100:>+7.2f}%{(e-0.003)*100:>+7.2f}%{s['t']:>7.2f}")

print("\n" + "#"*112)
print("# F. 严重异动 + v3因子 (低换手<5.2% + 非小盘>60亿)")
print("#"*112)
print(f"{'口径':<30}{'样本':>7}{'胜率':>8}{'盈亏比':>8}{'期望':>8}{'中位数':>9}{'超额':>8}{'成本后':>8}{'t值':>7}")
for lab, sub in [
    ('严重异动(DEV10>=50%) 无过滤', SEV),
    ('严重异动 + 低换手',          SEV[SEV['turn'] < 0.052]),
    ('严重异动 + 非小盘',          SEV[SEV['mcap'] >= 60]),
    ('严重异动 + 低换手 + 非小盘',   SEV[(SEV['turn'] < 0.052) & (SEV['mcap'] >= 60)]),
    ('严重异动 + 连板 + 低换手非小盘', SEV[(SEV['run_max'] >= 2) & (SEV['turn'] < 0.052) & (SEV['mcap'] >= 60)]),
]:
    s = summ(rets(sub))
    if not s:
        print(f"{lab:<30}{len(sub):>7}   样本不足(<20)"); continue
    e = s['exp']-bexp
    print(f"{lab:<30}{s['n']:>7}{s['wr']*100:>7.1f}%{s['pl']:>8.2f}{s['exp']*100:>7.2f}%"
          f"{s['med']*100:>8.2f}%{e*100:>+7.2f}%{(e-0.003)*100:>+7.2f}%{s['t']:>7.2f}")

print("\n" + "#"*112)
print("# G. 分时段稳定性 (严重异动 DEV10>=50%, T+1开盘, 持有3日)")
print("#"*112)
SEV2 = SEV.copy(); SEV2['half'] = np.where(SEV2['date'] < '2026-05-01', 'H1', 'H2')
for h in ['H1', 'H2']:
    s = summ(rets(SEV2[SEV2['half'] == h]))
    if not s: print(f"{h}: 样本不足({len(SEV2[SEV2['half']==h])})"); continue
    print(f"{h}  样本{s['n']:>4}  胜率{s['wr']*100:>5.1f}%  期望{s['exp']*100:>6.2f}%  "
          f"中位{s['med']*100:>6.2f}%  超额{(s['exp']-bexp)*100:>+6.2f}%  t={s['t']:.2f}")

EV.to_csv("events_v6.csv", index=False, encoding='utf-8-sig')
print("\n明细: events_v6.csv")
