# -*- coding: utf-8 -*-
"""
拉取全 A 股 16 年日线（不复权），供 2010-2026 龙回头策略回测使用。
- 股票池：全 A（沪深主板/创业板/科创板），排除北交所
- 复权：adjust='' 不复权（涨停判定与流通市值口径需要真实盘面价）
- 时间：2009-01-01 ~ 2026-08-10（多拉1年暖机）
- 断点续传：已存在的 pkl 直接跳过
- 并发：12 线程
"""
import os, time, sys
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE = 'cache16y'
START = '2009-01-01'
END   = '2026-08-10'
WORKERS = 12
KEEP = ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'outstanding_share']

os.makedirs(CACHE, exist_ok=True)


def sina_sym(c):
    return ('sh' if c.startswith('6') else 'sz') + c


def fetch(code):
    p = os.path.join(CACHE, code + '.pkl')
    if os.path.exists(p):
        return 'skip'
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=sina_sym(code), adjust='')
            if df is None or len(df) == 0:
                return 'empty'
            df = df[[c for c in KEEP if c in df.columns]].copy()
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df[(df['date'] >= START) & (df['date'] <= END)]
            if len(df) < 120:
                return 'short'
            for c in df.columns:
                if c != 'date':
                    df[c] = df[c].astype('float32')
            df.reset_index(drop=True).to_pickle(p, compression='gzip')
            return 'ok'
        except Exception as e:
            if attempt == 2:
                return 'fail:%s' % type(e).__name__
            time.sleep(1.0 + attempt * 0.5)


def main():
    lst = ak.stock_info_a_code_name()
    codes = sorted(set(str(c).zfill(6) for c in lst['code']))
    # 只保留沪深：60/68/00/30 开头；排除北交所(4/8/920)与其它
    codes = [c for c in codes if c.startswith(('600', '601', '603', '605', '688',
                                               '000', '001', '002', '003', '300', '301'))]
    print('目标股票数: %d' % len(codes), flush=True)
    print('时间范围: %s ~ %s' % (START, END), flush=True)
    print('缓存目录: %s' % CACHE, flush=True)

    stat = {}
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, c): c for c in codes}
        for f in as_completed(futs):
            r = f.result()
            stat[r] = stat.get(r, 0) + 1
            done += 1
            if done % 200 == 0 or done == len(codes):
                el = time.time() - t0
                remain = el / done * (len(codes) - done) if done > 0 else 0
                print('  进度 %d/%d (%.1f%%)  用时%.0fs  预计剩余%.0fs  %s'
                      % (done, len(codes), done/len(codes)*100,
                         el, remain, stat), flush=True)

    print('\n完成: %s' % stat)
    print('缓存文件数: %d' % len(os.listdir(CACHE)))
    print('总用时 %.1f 分钟' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()