# -*- coding: utf-8 -*-
"""
10万本金 · 个股买卖点明细
- 每笔交易的买入日、买入价、买入股数、买入金额
- 卖出日、卖出价、卖出金额
- 手续费、印花税、净盈亏、收益率
- 同日多只按可用资金等分
- 10万本金滚动操作
"""
import pandas as pd
import numpy as np
import os, gzip, pickle
from collections import defaultdict

T = pd.read_csv('trades_lht_main.csv', dtype={'code': str})
T['date'] = T['date'].astype(str)
T = T.sort_values('date').reset_index(drop=True)

CACHE_DIR = 'cache5y'

def load_stock(code):
    fp = os.path.join(CACHE_DIR, f'{code}.pkl')
    if not os.path.exists(fp): return None
    try:
        with gzip.open(fp, 'rb') as f:
            df = pickle.load(f)
        return df.reset_index(drop=True) if isinstance(df, pd.DataFrame) else df
    except: return None

# 缓存价格数据
stock_cache = {}
for code in T['code'].unique():
    df = load_stock(code)
    if df is not None: stock_cache[code] = df

print(f'加载股票数: {len(stock_cache)}')

# 交易日历
sample_df = load_stock('000001')
cal_dates = sample_df['date'].astype(str).tolist()
date_idx = {d: i for i, d in enumerate(cal_dates)}

# ===== 精确模拟 =====
INIT_CAP = 100000.0
FEE_BUY = 0.0003   # 买入手续费万3
FEE_SELL = 0.0013  # 卖出千1（印花税+佣金）

cash = INIT_CAP
holdings = []  # (code, sell_idx, shares, avg_cost)
trades_detail = []
equity_ts = []

for i, date in enumerate(cal_dates):
    # 到期卖出
    new_holdings = []
    for h in holdings:
        code, sell_idx, shares, cost = h
        if sell_idx <= i:
            df = stock_cache[code]
            sell_price = float(df.iloc[sell_idx]['close'])
            sell_amount = sell_price * shares
            fee = sell_amount * FEE_SELL
            net_proceeds = sell_amount - fee
            pnl = net_proceeds - cost * shares
            ret = pnl / (cost * shares) * 100
            cash += net_proceeds
            
            # 买入日 = sell_idx - 3 - 1（T+1买，持3日卖）
            buy_idx = sell_idx - 3
            buy_date = str(df.iloc[buy_idx]['date'])
            buy_price = float(df.iloc[buy_idx]['open'])
            sell_date = str(df.iloc[sell_idx]['date'])
            
            trades_detail.append({
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
                'hold_days': 3,
            })
        else:
            new_holdings.append(h)
    holdings = new_holdings
    
    # 当日信号
    sigs = T[T['date'] == date]
    if len(sigs) > 0 and cash > 100:  # 至少100元才操作
        n = len(sigs)
        per_trade = cash / n  # 等分
        for _, r in sigs.iterrows():
            code = r['code']
            if code not in stock_cache: continue
            df = stock_cache[code]
            
            # 信号日是date，T+1开盘买入 → date的下一日
            if date not in date_idx: continue
            sig_idx = date_idx[date]
            buy_idx = sig_idx + 1  # T+1
            if buy_idx >= len(df): continue
            
            buy_price = float(df.iloc[buy_idx]['open'])
            if buy_price <= 0: continue
            
            # 计算能买多少股（100股整手）
            amount = per_trade
            shares = int(amount / buy_price / 100) * 100
            if shares < 100: continue
            
            buy_amount = buy_price * shares
            fee = buy_amount * FEE_BUY
            total_cost = buy_amount + fee
            
            if total_cost > cash:
                # 重新计算能买多少
                shares = int(cash / (buy_price * (1 + FEE_BUY)) / 100) * 100
                if shares < 100: continue
                buy_amount = buy_price * shares
                fee = buy_amount * FEE_BUY
                total_cost = buy_amount + fee
            
            cash -= total_cost
            
            # 持3日收盘卖：买入日+3个交易日
            sell_idx = buy_idx + 3
            if sell_idx >= len(df): continue
            
            cost_per_share = total_cost / shares
            holdings.append((code, sell_idx, shares, cost_per_share))
    
    # 总资产
    hv = 0
    for h in holdings:
        code, si, sh, cost = h
        df = stock_cache[code]
        if si < len(df):
            hv += float(df.iloc[si]['close']) * sh  # 用卖出价估算
    equity_ts.append(cash + hv)

trades_df = pd.DataFrame(trades_detail)
trades_df = trades_df.sort_values('buy_date').reset_index(drop=True)
trades_df.insert(0, 'trade_no', range(1, len(trades_df)+1))

print(f'\n成交笔数: {len(trades_df)}')
print(f'期末资金: {cash:,.2f} 元')
print(f'总收益率: {(cash/INIT_CAP-1)*100:.1f}%')

# ===== 输出明细 =====
print(f'\n{"="*100}')
print(f'  【个股买卖点明细】 10万本金滚动操作（前20笔）')
print(f'{"="*100}')
print()

cols_show = ['trade_no', 'code', 'buy_date', 'buy_price', 'shares', 'buy_amount',
             'sell_date', 'sell_price', 'sell_amount', 'pnl', 'return_pct']

# 格式化打印
header = f'  {"序号":>4s} {"代码":<8s} {"买入日":<12s} {"买入价":>8s} {"股数":>8s} {"买入额":>10s} {"卖出日":<12s} {"卖出价":>8s} {"卖出额":>10s} {"盈亏":>10s} {"收益率":>7s}'
print(header)
print(f'  {"-"*95}')

for _, r in trades_df.head(20).iterrows():
    print(f'  {int(r["trade_no"]):>4d} {r["code"]:<8s} {r["buy_date"]:<12s} {r["buy_price"]:>8.2f} {r["shares"]:>8d} '
          f'{r["buy_amount"]:>10.2f} {r["sell_date"]:<12s} {r["sell_price"]:>8.2f} {r["sell_amount"]:>10.2f} '
          f'{r["pnl"]:>+10.2f} {r["return_pct"]:>+6.2f}%')

if len(trades_df) > 20:
    print(f'  ... 共 {len(trades_df)} 笔，完整数据见CSV')

# ===== 年度汇总 =====
trades_df['year'] = trades_df['buy_date'].str[:4].astype(int)
yearly = trades_df.groupby('year').agg(
    n=('trade_no', 'count'),
    total_buy=('buy_amount', 'sum'),
    total_sell=('sell_amount', 'sum'),
    total_pnl=('pnl', 'sum'),
    win_rate=('pnl', lambda x: (x>0).mean()*100),
    avg_ret=('return_pct', 'mean'),
).reset_index()
yearly['year'] = yearly['year'].astype(int)
yearly['n'] = yearly['n'].astype(int)

print(f'\n{"="*70}')
print(f'  【年度汇总】')
print(f'{"="*70}')
print(f'  {"年份":<6s} {"笔数":>5s} {"总买入额":>12s} {"总卖出额":>12s} {"总盈亏":>12s} {"胜率":>7s} {"均收益":>8s}')
print(f'  {"-"*70}')
for _, r in yearly.iterrows():
    print(f'  {int(r["year"]):<6d} {int(r["n"]):>5d} {r["total_buy"]:>12,.2f} {r["total_sell"]:>12,.2f} '
          f'{r["total_pnl"]:>+12,.2f} {r["win_rate"]:>6.1f}% {r["avg_ret"]:>+7.2f}%')

# ===== 保存 =====
trades_df.to_csv('trades_detail_10w.csv', index=False, encoding='utf-8-sig')
print(f'\n明细已保存: trades_detail_10w.csv ({len(trades_df)}行)')

# 也保存一个资金曲线
eq_df = pd.DataFrame({'date': cal_dates, 'equity': equity_ts})
start_d = T['date'].min()
end_d = T['date'].max()
eq_df = eq_df[(eq_df['date'] >= start_d) & (eq_df['date'] <= end_d)]
eq_df.to_csv('equity_detail_10w.csv', index=False, encoding='utf-8-sig')
print(f'资金曲线: equity_detail_10w.csv ({len(eq_df)}行)')

# ===== 最大盈利/亏损 =====
print(f'\n{"="*70}')
print(f'  【TOP 10 盈利交易】')
print(f'{"="*70}')
top_win = trades_df.nlargest(10, 'pnl')[['trade_no','code','buy_date','sell_date','pnl','return_pct']]
for _, r in top_win.iterrows():
    print(f'  #{int(r["trade_no"]):>3d} {r["code"]} {r["buy_date"]}→{r["sell_date"]}  {r["pnl"]:>+10,.2f}元 ({r["return_pct"]:>+6.2f}%)')

print(f'\n{"="*70}')
print(f'  【TOP 10 亏损交易】')
print(f'{"="*70}')
top_loss = trades_df.nsmallest(10, 'pnl')[['trade_no','code','buy_date','sell_date','pnl','return_pct']]
for _, r in top_loss.iterrows():
    print(f'  #{int(r["trade_no"]):>3d} {r["code"]} {r["buy_date"]}→{r["sell_date"]}  {r["pnl"]:>+10,.2f}元 ({r["return_pct"]:>+6.2f}%)')
