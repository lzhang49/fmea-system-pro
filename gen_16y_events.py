# -*- coding: utf-8 -*-
"""
16年龙回头事件生成 + k值全样本计算
- 数据：cache16y/（2009-2026，不复权日线）
- 入池口径：10日累计涨幅 >= 100%
- 冷却期：60日
- 输出：events_lht_16y.csv + opt_full_16y.csv
"""
import os, gzip, pickle, time
import numpy as np
import pandas as pd

CACHE_DIR = 'cache16y'
BT_START = '2010-01-01'
WARMUP = 60
POOL_LOOK = 10     # 10日涨幅口径
COOLDOWN = 60      # 事件冷却期
PEAK_WIN = 40      # 入池后找峰值窗口
DD_CONFIRM = 0.10  # 回撤确认阈值
FEE_BUY = 0.0003
FEE_SELL = 0.0013
HOLD_DAYS = 3
K_RANGE = list(range(3, 16))  # k=3~15


def zt_thr(c):
    return 0.20 if c.startswith(('68', '30')) else 0.10


def board_of(c):
    return '科创' if c.startswith('68') else ('创业' if c.startswith('30') else '主板')


def load_all():
    paths = {}
    t0 = time.time()
    files = sorted(os.listdir(CACHE_DIR))
    print("载入缓存: %d 只" % len(files), flush=True)
    for fn in files:
        if not fn.endswith('.pkl'):
            continue
        code = fn[:-4]
        fp = os.path.join(CACHE_DIR, fn)
        try:
            with gzip.open(fp, 'rb') as f:
                df = pickle.load(f)
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
        cs = pd.Series(c)
        paths[code] = dict(
            o=df['open'].values.astype(np.float64),
            h=df['high'].values.astype(np.float64),
            l=df['low'].values.astype(np.float64), c=c,
            d=df['date'].values.astype(str),
            v=df['volume'].values.astype(np.float64),
            tov=df['turnover'].values.astype(np.float64),
            osh=df['outstanding_share'].values.astype(np.float64),
            run=s.groupby((s == 0).cumsum()).cumsum().values,
            isz=isz,
            ma10=cs.rolling(10, min_periods=10).mean().values,
            ma20=cs.rolling(20, min_periods=20).mean().values,
            ma60=cs.rolling(60, min_periods=60).mean().values,
        )
    print("载入 %d 只, %.0fs" % (len(paths), time.time() - t0), flush=True)
    return paths


def build_events(paths):
    """构建妖股池事件（10日涨幅>=100%）"""
    EV = []
    for code, p in paths.items():
        c, h, l, o, d = p['c'], p['h'], p['l'], p['o'], p['d']
        v, tov, osh, run = p['v'], p['tov'], p['osh'], p['run']
        ma10, ma20, isz = p['ma10'], p['ma20'], p['isz']
        n = len(c)
        r10 = np.full(n, np.nan)
        r10[POOL_LOOK:] = c[POOL_LOOK:] / np.maximum(c[:-POOL_LOOK], 1e-9) - 1.0
        cand = np.where((r10 >= 1.00) & (np.arange(n) >= WARMUP))[0]
        last = -10 ** 9
        for t_in in cand:
            if t_in - last < COOLDOWN:
                continue
            if d[t_in] < BT_START or t_in + PEAK_WIN + 30 >= n:
                continue
            last = t_in
            pp, pi, ci = -1.0, -1, -1
            for j in range(t_in, min(t_in + PEAK_WIN, n)):
                if h[j] > pp:
                    pp, pi = h[j], j
                if j > pi and pp > 0 and c[j] <= pp * (1 - DD_CONFIRM):
                    ci = j
                    break
            if ci < 0 or pi + 20 >= n:
                continue
            sig = pi + 5
            if sig + 8 >= n:
                continue
            v5 = v[sig - 4:sig + 1].mean()
            vp = v[max(pi - 2, 0):pi + 1].mean()
            EV.append(dict(
                code=code, sig=sig, date=d[sig], year=d[t_in][:4], t_in=t_in,
                peak_i=pi, dd5=c[sig] / pp - 1.0,
                run_max=int(run[max(t_in - POOL_LOOK, 0):t_in + 1].max()),
                mcap=c[sig] * osh[sig] / 1e8, turn=tov[sig] * 100.0,
                board=board_of(code), r10=r10[t_in],
                zt_in10=int(isz[max(t_in - 9, 0):t_in + 1].sum()),
                vshrink=v5 / vp if vp > 0 else np.nan,
                above10=bool(np.isfinite(ma10[sig]) and c[sig] > ma10[sig]),
                above20=bool(np.isfinite(ma20[sig]) and c[sig] > ma20[sig]),
                peak_gain=pp / c[max(t_in - POOL_LOOK, 0)] - 1.0))
    E = pd.DataFrame(EV)
    print("妖股池事件: %d 个" % len(E))
    return E


def build_k_samples(E, paths):
    """为每个事件计算k=3~15的买卖收益"""
    records = []
    for _, row in E.iterrows():
        code = row['code']
        if code not in paths:
            continue
        df_dates = paths[code]['d']
        df_o = paths[code]['o']
        df_c = paths[code]['c']
        peak_i = int(row['peak_i'])
        n = len(df_dates)

        base = {
            'code': code,
            'board': row['board'],
            'run_max': int(row['run_max']),
            'mcap': float(row['mcap']),
            'turn': float(row['turn']),
            'peak_gain': float(row['peak_gain']),
            'zt_in10': int(row['zt_in10']),
            'vshrink': float(row['vshrink']) if pd.notna(row['vshrink']) else np.nan,
            'peak_i': peak_i,
        }

        for k in K_RANGE:
            buy_i = peak_i + k + 1  # 峰值后第k日信号 → 第k+1日开盘买
            sell_i = buy_i + HOLD_DAYS  # 持HOLD_DAYS日收盘卖
            if buy_i >= n or sell_i >= n:
                continue
            buy_p = float(df_o[buy_i])
            sell_p = float(df_c[sell_i])
            if buy_p <= 0:
                continue
            raw_ret = sell_p / buy_p - 1
            net_ret = (1 + raw_ret) * (1 - FEE_SELL) / (1 + FEE_BUY) - 1
            peak_p = float(df_c[peak_i])
            dd_from_peak = buy_p / peak_p - 1
            buy_date = str(df_dates[buy_i])

            rec = base.copy()
            rec.update({
                'k': k,
                'buy_date': buy_date,
                'buy_price': buy_p,
                'sell_price': sell_p,
                'raw_ret': raw_ret,
                'net_ret': net_ret,
                'dd_from_peak': dd_from_peak,
            })
            records.append(rec)

    D = pd.DataFrame(records)
    print("k值样本: %d 条" % len(D))
    return D


def add_index_env(D):
    """加载创业板指数环境数据（pos_250, ma60, vol20, ma20_slope）"""
    import akshare as ak
    # 拉创业板指（399006）16年数据
    idx_df = ak.stock_zh_index_daily(symbol="sz399006")
    idx_df['date'] = pd.to_datetime(idx_df['date']).dt.strftime('%Y-%m-%d')
    idx_df = idx_df.sort_values('date').reset_index(drop=True)
    idx_df['close'] = idx_df['close'].astype(float)

    def pct_rank(series, window=250):
        return series.rolling(window).apply(lambda x: (x.rank(pct=True).iloc[-1]) * 100)

    idx_df['pos_250'] = pct_rank(idx_df['close'], 250)
    idx_df['ma20'] = idx_df['close'].rolling(20).mean()
    idx_df['ma60'] = idx_df['close'].rolling(60).mean()
    idx_df['vol20'] = idx_df['close'].pct_change().rolling(20).std() * np.sqrt(250) * 100
    idx_df['ma20_slope'] = idx_df['ma20'].pct_change(5) * 100

    date_to_idx = {r['date']: i for i, r in idx_df.iterrows()}

    def get_env(date_str):
        if date_str not in date_to_idx:
            return (np.nan, False, np.nan, np.nan)
        i = date_to_idx[date_str]
        r = idx_df.iloc[i]
        return (r['pos_250'], r['close'] > r['ma60'], r['vol20'], r['ma20_slope'])

    env_data = D['buy_date'].apply(get_env)
    D['pos_250'] = env_data.apply(lambda x: x[0])
    D['above_ma60'] = env_data.apply(lambda x: x[1])
    D['vol20'] = env_data.apply(lambda x: x[2])
    D['ma20_slope'] = env_data.apply(lambda x: x[3])
    D = D.dropna(subset=['pos_250'])
    print("有指数环境的样本: %d 条" % len(D))
    return D


def main():
    t0 = time.time()
    paths = load_all()

    print("\n=== 构建妖股池事件 ===", flush=True)
    E = build_events(paths)
    E.to_csv('events_lht_16y.csv', index=False, encoding='utf-8-sig')
    print("保存: events_lht_16y.csv (%d行)" % len(E))

    print("\n=== 计算k值全样本 ===", flush=True)
    D = build_k_samples(E, paths)

    print("\n=== 添加指数环境 ===", flush=True)
    D = add_index_env(D)

    D.to_csv('opt_full_16y.csv', index=False, encoding='utf-8-sig')
    print("保存: opt_full_16y.csv (%d行)" % len(D))

    # 年度分布
    print("\n=== 年度事件分布 ===")
    yr = E.groupby('year').size()
    for y, n in yr.items():
        print("  %s: %d 个事件" % (y, n))

    print("\n总用时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()