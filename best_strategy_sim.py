# -*- coding: utf-8 -*-
"""
最优策略 · 10万本金逐笔实盘模拟 v3
全部基于日期查找，不依赖索引对齐
"""
import pandas as pd
import numpy as np
import os, gzip, pickle

CACHE_DIR = 'cache5y'
FEE_BUY = 0.0003
FEE_SELL = 0.0013
HOLD_DAYS = 3
INIT_CAP = 100000.0

D = pd.read_csv('opt_full_data.csv', dtype={'code': str})

def filter_best(df):
    d = df.copy()
    d = d[d['k'] == 5]
    d = d[d['run_max'] <= 5]
    d = d[d['dd_from_peak'] <= -0.15]
    d = d[d['dd_from_peak'] >= -0.30]
    d = d[d['mcap'] >= 30]
    d = d[d['board'] == '创业']
    d = d[d['ma20_slope'] > 0.3]
    return d.sort_values('buy_date').reset_index(drop=True)

signals = filter_best(D)
print(f'最优策略信号数: {len(signals)}')

# 加载股票缓存
stock_cache = {}
stock_dates = {}  # code -> date list
def load_stock(code):
    fp = os.path.join(CACHE_DIR, f'{code}.pkl')
    if not os.path.exists(fp): return None
    try:
        with gzip.open(fp, 'rb') as f:
            df = pickle.load(f)
        return df.reset_index(drop=True) if isinstance(df, pd.DataFrame) else df
    except: return None

for code in signals['code'].unique():
    df = load_stock(code)
    if df is not None:
        stock_cache[code] = df
        stock_dates[code] = df['date'].astype(str).tolist()

# 交易日历（用所有股票的并集排序）
all_dates = set()
for code in stock_dates:
    all_dates.update(stock_dates[code])
cal_dates = sorted(all_dates)
date_set = set(cal_dates)
print(f'交易日总数: {len(cal_dates)}')
print(f'信号区间: {signals["buy_date"].min()} ~ {signals["buy_date"].max()}')

def get_nth_trading_day(code, from_date, n):
    """从from_date之后（含）第n个交易日的日期和价格"""
    dates = stock_dates[code]
    if from_date not in date_set: return None
    # from_date在股票K线中的位置
    if from_date not in dates: return None
    idx = dates.index(from_date)
    target_idx = idx + n
    if target_idx >= len(dates): return None
    target_date = dates[target_idx]
    df = stock_cache[code]
    return {'date': target_date, 'idx': target_idx, 'df_idx': target_idx}

# ============== 实盘级逐笔模拟 ==============
cash = INIT_CAP
# holdings: (code, sell_date, shares, cost_per_share)
holdings = []
trades = []
equity_ts = []

for date in cal_dates:
    # 到期卖出
    new_holdings = []
    for h in holdings:
        code, sell_date, shares, cost = h
        if sell_date <= date:
            # 卖出
            df = stock_cache[code]
            dates = stock_dates[code]
            if sell_date not in dates:
                # 找下一个交易日
                future = [d for d in dates if d >= sell_date]
                if not future:
                    new_holdings.append(h)
                    continue
                sell_date = future[0]
            si = dates.index(sell_date)
            sell_price = float(df.iloc[si]['close'])
            sell_amount = sell_price * shares
            fee = sell_amount * FEE_SELL
            net_proceeds = sell_amount - fee
            pnl = net_proceeds - cost * shares
            ret = pnl / (cost * shares) * 100
            cash += net_proceeds
            
            # 找买入日
            buy_date_idx = si - HOLD_DAYS
            if buy_date_idx >= 0:
                buy_date = dates[buy_date_idx]
                buy_price = float(df.iloc[buy_date_idx]['open'])
            else:
                buy_date = 'unknown'
                buy_price = 0
            
            trades.append({
                'code': code,
                'buy_date': buy_date,
                'buy_price': round(buy_price, 2),
                'shares': int(shares),
                'buy_amount': round(buy_price * shares, 2),
                'buy_fee': round(buy_price * shares * FEE_BUY, 2),
                'sell_date': sell_date,
                'sell_price': round(sell_price, 2),
                'sell_amount': round(sell_amount, 2),
                'sell_fee': round(fee, 2),
                'pnl': round(pnl, 2),
                'return_pct': round(ret, 2),
                'hold_days': HOLD_DAYS,
            })
        else:
            new_holdings.append(h)
    holdings = new_holdings
    
    # 当日信号
    today_sigs = signals[signals['buy_date'] == date]
    if len(today_sigs) > 0 and cash > 1000:
        n = len(today_sigs)
        per_trade = cash / n
        for _, r in today_sigs.iterrows():
            code = r['code']
            if code not in stock_cache: continue
            df = stock_cache[code]
            dates = stock_dates[code]
            
            if date not in dates: continue  # 股票当日停牌跳过
            buy_idx = dates.index(date)
            buy_price = float(df.iloc[buy_idx]['open'])
            if buy_price <= 0: continue
            
            shares = int(per_trade / buy_price / 100) * 100
            if shares < 100: continue
            
            buy_amount = buy_price * shares
            fee = buy_amount * FEE_BUY
            total_cost = buy_amount + fee
            
            if total_cost > cash:
                shares = int(cash / (buy_price * (1 + FEE_BUY)) / 100) * 100
                if shares < 100: continue
                buy_amount = buy_price * shares
                fee = buy_amount * FEE_BUY
                total_cost = buy_amount + fee
            
            cash -= total_cost
            
            # 卖出日：买入日+HOLD_DAYS个交易日
            sell_idx = buy_idx + HOLD_DAYS
            if sell_idx >= len(dates): continue
            sell_date = dates[sell_idx]
            
            cost_per_share = total_cost / shares
            holdings.append((code, sell_date, shares, cost_per_share))
    
    # 总资产（持仓按当日收盘价估值）
    hv = 0
    for h in holdings:
        code, sdate, sh, cost = h
        df = stock_cache[code]
        dates = stock_dates[code]
        # 用当日收盘价估市值
        if date in dates:
            cur_i = dates.index(date)
            hv += float(df.iloc[cur_i]['close']) * sh
        else:
            # 停牌用上一个价格
            past = [d for d in dates if d <= date]
            if past:
                cur_i = dates.index(past[-1])
                hv += float(df.iloc[cur_i]['close']) * sh
    equity_ts.append(cash + hv)

trades_df = pd.DataFrame(trades)
trades_df = trades_df.sort_values('buy_date').reset_index(drop=True)
trades_df.insert(0, 'trade_no', range(1, len(trades_df)+1))

print(f'\n成交笔数: {len(trades_df)}')
print(f'期末资金: ¥{cash:,.2f}')
print(f'总收益率: {(cash/INIT_CAP-1)*100:.1f}%')

# 资金曲线
eq_df = pd.DataFrame({'date': cal_dates, 'equity': equity_ts})
start_d = signals['buy_date'].min()
eq_df = eq_df[eq_df['date'] >= start_d].reset_index(drop=True)

peak = eq_df['equity'].cummax()
dd = (eq_df['equity'] / peak - 1) * 100
max_dd = dd.min()
print(f'最大回撤: {max_dd:.1f}%')

if len(trades_df) > 0:
    print(f'平均单笔收益: {trades_df["return_pct"].mean():.2f}%')
    print(f'胜率: {(trades_df["return_pct"]>0).mean()*100:.1f}%')
    
    # 验证一下和理论收益的偏差
    print(f'\n理论均值: {signals["net_ret"].mean()*100:.2f}%')
    print(f'实盘均收: {trades_df["return_pct"].mean():.2f}%')

    # 年度汇总
    trades_df['year'] = trades_df['buy_date'].str[:4].astype(int)
    yearly = trades_df.groupby('year').agg(
        n=('trade_no','count'),
        pnl_total=('pnl','sum'),
        win_rate=('pnl', lambda x:(x>0).mean()*100),
        avg_ret=('return_pct','mean'),
    ).reset_index()
    
    print(f'\n=== 年度汇总 ===')
    for _, r in yearly.iterrows():
        pnl = float(r['pnl_total'])
        sign = '+' if pnl >= 0 else ''
        print(f'  {int(r["year"])}年: {int(r["n"]):>2d}笔  {sign}{pnl:>10,.0f}元  胜率{r["win_rate"]:.1f}%  均收{r["avg_ret"]:+.2f}%')

# 抽查验证
print(f'\n=== 抽查5笔验证 ===')
for _, r in trades_df.head(5).iterrows():
    code = r['code']
    dates = stock_dates[code]
    df = stock_cache[code]
    if r['buy_date'] in dates and r['sell_date'] in dates:
        bi = dates.index(r['buy_date'])
        si = dates.index(r['sell_date'])
        buy_p = float(df.iloc[bi]['open'])
        sell_p = float(df.iloc[si]['close'])
        ok_buy = abs(buy_p - r['buy_price']) < 0.001
        ok_sell = abs(sell_p - r['sell_price']) < 0.001
        # 计算收益
        calc_ret = ((sell_p * (1 - FEE_SELL)) / (buy_p * (1 + FEE_BUY)) - 1) * 100
        ok_ret = abs(calc_ret - r['return_pct']) < 0.1
        print(f'  {code} {r["buy_date"]}→{r["sell_date"]}  '
              f'买入{buy_p:.2f}(记录{r["buy_price"]:.2f}{"✓" if ok_buy else "✗"})  '
              f'卖出{sell_p:.2f}(记录{r["sell_price"]:.2f}{"✓" if ok_sell else "✗"})  '
              f'收益{calc_ret:+.2f}%(记录{r["return_pct"]:+.2f}%{"✓" if ok_ret else "✗"})')

# 保存
trades_df.to_csv('best_strategy_trades.csv', index=False, encoding='utf-8-sig')
eq_df.to_csv('best_strategy_equity.csv', index=False, encoding='utf-8-sig')
print(f'\n已保存: best_strategy_trades.csv ({len(trades_df)}行)')
print(f'已保存: best_strategy_equity.csv ({len(eq_df)}行)')