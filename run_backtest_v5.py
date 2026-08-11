"""
v5 —— 交易所「异常波动」监管股票池 + 多买入节点回测

监管池定义(严格按沪深交易所《股票交易异常波动》规则, 只取上涨方向):
  偏离值 = 个股当日涨跌幅 - 对标指数当日涨跌幅
  对标: 沪主板→上证综指 / 深主板中小板→深证成指 / 创业板(30)→创业板指 / 科创板(68)→科创50

  L1 异常波动: 连续3个交易日收盘价涨幅偏离值累计 >= +20% (ST股 +15%)
               创业板/科创板 >= +30%
  L2 严重异常波动: 连续10个交易日内累计涨幅偏离值 >= +100%
               或 连续10个交易日内发生3次及以上 L1
  L3 叠加换手/振幅: L1 且 触发日换手率 > 30日均换手 3倍 (监管重点关注特征)

真实性约束:
  - 异动公告在触发日收盘后/次日早间披露 → 最早可买 = T+1 开盘, 绝不允许 T 日买
  - 冷却期: 同一只股票 L1 触发后 10 个交易日内不重复计入(避免同一波行情反复计数)

买入节点:
  T1_OPEN  次日开盘 (公告后第一时间)
  T1_CLOSE 次日收盘 (看一天再决定)
  T2_OPEN  第二日开盘 (等情绪释放一天)
  T3_OPEN  第三日开盘
  T5_OPEN  第五日开盘 (等回踩)
  MA5_BACK 异动后10日内首次触及5日线的次日开盘 (龙回头)
"""
import akshare as ak
import pandas as pd, numpy as np, os, glob, json

CACHE = "cache_daily"
NS = [3, 5, 10]
COOLDOWN = 10

im = pd.read_csv("industry_map.csv", dtype={'code': str})
name_map = dict(zip(im['code'], im['name']))

# ---------- 指数 ----------
IDX_SPEC = {'SH_MAIN': 'sh000001', 'SZ_MAIN': 'sz399001',
            'CYB': 'sz399006', 'KCB': 'sh000688'}
idx_ret = {}
for k, s in IDX_SPEC.items():
    fp = f"idx_{k}.csv"
    if os.path.exists(fp):
        d = pd.read_csv(fp)
    else:
        d = ak.stock_zh_index_daily(symbol=s); d.to_csv(fp, index=False)
    d['date'] = d['date'].astype(str).str[:10]
    d = d[d['date'] >= '2025-12-01'].reset_index(drop=True)
    d['r'] = d['close'].astype(float).pct_change()
    idx_ret[k] = dict(zip(d['date'], d['r']))
print(f"指数基准载入: {list(idx_ret.keys())}", flush=True)


def board_of(code):
    if code.startswith('68'): return 'KCB'
    if code.startswith('30'): return 'CYB'
    if code.startswith('6'):  return 'SH_MAIN'
    return 'SZ_MAIN'


def l1_threshold(code, name):
    if code.startswith(('68', '30')): return 0.30
    if 'ST' in str(name).upper() or '*' in str(name): return 0.15
    return 0.20


def zt_thr(code, name):
    if 'ST' in str(name).upper() or '*' in str(name): return 0.05
    if code.startswith(('68', '30')): return 0.20
    return 0.10


# ---------- 加载个股 + 识别异动 ----------
print("扫描交易所异常波动触发...", flush=True)
paths, events = {}, []
for fp in glob.glob(os.path.join(CACHE, "*.csv")):
    code = os.path.basename(fp)[:-4]
    try: df = pd.read_csv(fp)
    except Exception: continue
    if df is None or len(df) < 40: continue
    df['date'] = df['date'].astype(str).str[:10]
    o = df['open'].astype(float).values; h = df['high'].astype(float).values
    l = df['low'].astype(float).values;  c = df['close'].astype(float).values
    ds = list(df['date'].values); n = len(df)
    nm = name_map.get(code, ''); bd = board_of(code)
    paths[code] = dict(o=o, h=h, l=l, c=c, d=ds, thr=zt_thr(code, nm))

    osh = df['outstanding_share'].astype(float).values
    tov = df['turnover'].astype(float).values
    ma5 = pd.Series(c).rolling(5).mean().values
    tov_ma30 = pd.Series(tov).rolling(30).mean().values

    ir = idx_ret[bd]
    dev = np.zeros(n)                      # 每日偏离值
    for i in range(1, n):
        if c[i-1] <= 0: continue
        dev[i] = (c[i]/c[i-1]-1) - ir.get(ds[i], 0.0)

    thr1 = l1_threshold(code, nm)
    last_hit = -999
    l1_hits = []
    for i in range(3, n):
        d3 = dev[i-2:i+1].sum()            # 连续3日累计偏离
        if d3 < thr1: continue
        l1_hits.append(i)
        if i - last_hit < COOLDOWN: continue
        last_hit = i
        d10 = dev[max(0, i-9):i+1].sum()
        cnt10 = sum(1 for x in l1_hits if i-9 <= x <= i)
        lvl = 'L2' if (d10 >= 1.00 or cnt10 >= 3) else 'L1'
        turn_spike = (tov_ma30[i] > 0 and tov[i] / tov_ma30[i] >= 3.0)
        events.append(dict(code=code, name=nm, i=i, date=ds[i], board=bd,
                           lvl=lvl, dev3=d3, dev10=d10, hits10=cnt10,
                           mcap=c[i]*osh[i]/1e8, turn=tov[i],
                           turn_spike=bool(turn_spike),
                           mom20=(c[i]/c[i-20]-1) if i >= 20 else np.nan))

ev = pd.DataFrame(events)
print(f"异动事件: 共{len(ev)}  L1={sum(ev.lvl=='L1')}  L2(严重)={sum(ev.lvl=='L2')}  "
      f"涉及{ev['code'].nunique()}只", flush=True)
print(f"  板块分布: {dict(ev['board'].value_counts())}", flush=True)


# ---------- 买入节点 ----------
def entry_point(code, i, mode):
    """返回 (买入日索引 k, 买入价). 全部为 T+1 及以后, 不含 T 日"""
    p = paths[code]; o, c, h, l, ma = p['o'], p['c'], p['h'], p['l'], None
    n = len(c)
    if mode == 'T1_OPEN':
        k = i+1;  return (k, o[k]) if k < n else None
    if mode == 'T1_CLOSE':
        k = i+1;  return (k, c[k]) if k < n else None
    if mode == 'T2_OPEN':
        k = i+2;  return (k, o[k]) if k < n else None
    if mode == 'T3_OPEN':
        k = i+3;  return (k, o[k]) if k < n else None
    if mode == 'T5_OPEN':
        k = i+5;  return (k, o[k]) if k < n else None
    if mode == 'MA5_BACK':
        ser = pd.Series(c).rolling(5).mean().values
        for k in range(i+1, min(i+11, n-1)):
            if not np.isnan(ser[k]) and l[k] <= ser[k]:
                return (k+1, o[k+1]) if k+1 < n else None
        return None
    return None


MODES = ['T1_OPEN', 'T1_CLOSE', 'T2_OPEN', 'T3_OPEN', 'T5_OPEN', 'MA5_BACK']

rows = []
for _, r in ev.iterrows():
    code = r['code']; p = paths[code]; c = p['c']; n = len(c)
    for mode in MODES:
        e = entry_point(code, int(r['i']), mode)
        if not e: continue
        k, px = e
        if px <= 0: continue
        for N in NS:
            if k+N >= n: continue
            seg = c[k:k+N+1]
            rows.append(dict(code=code, name=r['name'], date=r['date'], lvl=r['lvl'],
                             board=r['board'], mode=mode, N=N,
                             dev3=r['dev3'], mcap=r['mcap'], turn=r['turn'],
                             turn_spike=r['turn_spike'], mom20=r['mom20'],
                             ret=float(c[k+N]/px - 1.0),
                             mdd=float(np.min(seg)/px - 1.0)))

# ---------- 基准: 同池随机日, 同样的买入节点口径 ----------
base_rows = []
codes_in = ev['code'].unique().tolist()
for code in codes_in:
    p = paths[code]; o, c = p['o'], p['c']; n = len(c)
    for i in range(20, n-12, 5):
        for N in NS:
            k = i+1
            if k+N >= n or o[k] <= 0: continue
            seg = c[k:k+N+1]
            base_rows.append(dict(mode='BASE', N=N, ret=float(c[k+N]/o[k]-1.0),
                                  mdd=float(np.min(seg)/o[k]-1.0)))

res = pd.DataFrame(rows)
base = pd.DataFrame(base_rows)
res.to_csv("backtest_events_v5.csv", index=False, encoding='utf-8-sig')
print(f"回测样本: {len(res)}  基准样本: {len(base)}", flush=True)


def st(d):
    if len(d) < 25: return None
    r = d['ret'].values
    w, ls = r[r > 0], r[r <= 0]
    wr = len(w)/len(r)
    aw = float(w.mean()) if len(w) else 0.
    al = float(ls.mean()) if len(ls) else 0.
    return dict(n=len(r), wr=wr, aw=aw, al=al, pl=(aw/abs(al)) if al else np.inf,
                exp=wr*aw+(1-wr)*al, med=float(np.median(r)),
                mdd=float(d['mdd'].mean()), worst=float(r.min()))


def tval(r):
    r = np.asarray(r, float); sd = r.std(ddof=1)
    return (r.mean()/(sd/np.sqrt(len(r)))) if sd > 0 and len(r) > 1 else 0.


print("\n" + "#"*120)
print("# A. 买入节点扫描 —— 全部异动事件 (超额 = 策略 - 同持有期基准, 成本双边0.3%)")
print("#"*120)
for N in NS:
    b = st(base[base['N'] == N]); bexp = b['exp'] if b else 0
    print(f"\n─── 持有{N}日 (基准期望 {bexp*100:+.2f}%) " + "─"*56)
    print(f"{'买入节点':<14}{'样本':>7}{'胜率':>8}{'均盈':>8}{'均亏':>8}{'盈亏比':>7}"
          f"{'中位数':>8}{'期望':>8}{'超额':>8}{'成本后':>8}{'t值':>7}")
    for m in MODES:
        s = st(res[(res['mode'] == m) & (res['N'] == N)])
        if not s: continue
        e = s['exp'] - bexp
        t = tval(res[(res['mode'] == m) & (res['N'] == N)]['ret'].values)
        print(f"{m:<14}{s['n']:>7}{s['wr']*100:>7.1f}%{s['aw']*100:>7.2f}%{s['al']*100:>7.2f}%"
              f"{s['pl']:>7.2f}{s['med']*100:>7.2f}%{s['exp']*100:>7.2f}%{e*100:>+7.2f}%"
              f"{(e-0.003)*100:>+7.2f}%{t:>7.2f}")

print("\n" + "#"*120)
print("# B. 分监管等级 (L1普通异动 vs L2严重异常波动)")
print("#"*120)
for N in [3, 5]:
    b = st(base[base['N'] == N]); bexp = b['exp'] if b else 0
    print(f"\n─── 持有{N}日 " + "─"*66)
    print(f"{'等级':<6}{'买入节点':<12}{'样本':>7}{'胜率':>8}{'均盈':>8}{'均亏':>8}"
          f"{'盈亏比':>7}{'期望':>8}{'超额':>8}{'成本后':>8}")
    for lvl in ['L1', 'L2']:
        for m in ['T1_OPEN', 'T2_OPEN', 'T5_OPEN', 'MA5_BACK']:
            s = st(res[(res['mode'] == m) & (res['N'] == N) & (res['lvl'] == lvl)])
            if not s: continue
            e = s['exp'] - bexp
            print(f"{lvl:<6}{m:<12}{s['n']:>7}{s['wr']*100:>7.1f}%{s['aw']*100:>7.2f}%"
                  f"{s['al']*100:>7.2f}%{s['pl']:>7.2f}{s['exp']*100:>7.2f}%"
                  f"{e*100:>+7.2f}%{(e-0.003)*100:>+7.2f}%")

print("\n" + "#"*120)
print("# C. 异动强度分层 (3日累计偏离值越大 = 涨得越猛, 监管关注度越高)")
print("#"*120)
N = 3
b = st(base[base['N'] == N]); bexp = b['exp'] if b else 0
for m in ['T1_OPEN', 'T2_OPEN', 'MA5_BACK']:
    sub = res[(res['mode'] == m) & (res['N'] == N)]
    if len(sub) < 60: continue
    q = sub['dev3'].quantile([.33, .67]).values
    print(f"\n=== 买点 {m} · 持有{N}日 ===")
    print(f"{'偏离强度':<20}{'样本':>7}{'胜率':>8}{'盈亏比':>8}{'期望':>8}{'超额':>8}{'成本后':>8}")
    for lab, msk in [(f'弱 (<{q[0]*100:.0f}%)', sub['dev3'] < q[0]),
                     (f'中 ({q[0]*100:.0f}~{q[1]*100:.0f}%)', (sub['dev3'] >= q[0]) & (sub['dev3'] <= q[1])),
                     (f'强 (>{q[1]*100:.0f}%)', sub['dev3'] > q[1])]:
        s = st(sub[msk])
        if not s: continue
        e = s['exp'] - bexp
        print(f"{lab:<20}{s['n']:>7}{s['wr']*100:>7.1f}%{s['pl']:>8.2f}"
              f"{s['exp']*100:>7.2f}%{e*100:>+7.2f}%{(e-0.003)*100:>+7.2f}%")

print("\n" + "#"*120)
print("# D. 叠加 v3 因子 (低换手<5.2% + 非小盘>60亿) 与 放量异动")
print("#"*120)
for N in [3, 5]:
    b = st(base[base['N'] == N]); bexp = b['exp'] if b else 0
    print(f"\n─── 持有{N}日 " + "─"*62)
    print(f"{'买点':<11}{'过滤条件':<22}{'样本':>7}{'胜率':>8}{'盈亏比':>8}{'期望':>8}{'超额':>8}{'成本后':>8}")
    for m in ['T1_OPEN', 'T2_OPEN', 'MA5_BACK']:
        sub = res[(res['mode'] == m) & (res['N'] == N)]
        for lab, msk in [
            ('无过滤', sub['ret'].notna()),
            ('低换手+非小盘', (sub['turn'] < 0.052) & (sub['mcap'] >= 60)),
            ('放量异动(换手>3倍均)', sub['turn_spike'] == True),
            ('非放量', sub['turn_spike'] == False),
        ]:
            s = st(sub[msk])
            if not s: continue
            e = s['exp'] - bexp
            print(f"{m:<11}{lab:<22}{s['n']:>7}{s['wr']*100:>7.1f}%{s['pl']:>8.2f}"
                  f"{s['exp']*100:>7.2f}%{e*100:>+7.2f}%{(e-0.003)*100:>+7.2f}%")

print("\n" + "#"*120)
print("# E. 时间分段稳定性 (H1 2026-01~04 vs H2 05~08), 持有3日")
print("#"*120)
res['half'] = np.where(res['date'] < '2026-05-01', 'H1', 'H2')
print(f"{'买入节点':<14}{'H1样本':>8}{'H1期望':>9}{'H2样本':>8}{'H2期望':>9}{'同向?':>7}")
for m in MODES:
    a = st(res[(res['mode'] == m) & (res['N'] == 3) & (res['half'] == 'H1')])
    c2 = st(res[(res['mode'] == m) & (res['N'] == 3) & (res['half'] == 'H2')])
    if not a or not c2:
        print(f"{m:<14}{'样本不足':>8}"); continue
    same = '✓' if np.sign(a['exp']) == np.sign(c2['exp']) else '✗'
    print(f"{m:<14}{a['n']:>8}{a['exp']*100:>8.2f}%{c2['n']:>8}{c2['exp']*100:>8.2f}%{same:>7}")

print("\n明细: backtest_events_v5.csv", flush=True)
