# -*- coding: utf-8 -*-
"""对比原版与新(实盘过滤)版交易明细, 找出被跳过的/新增的/顺延的交易"""
import csv, os
BASE = r"C:/Users/EDY/WorkBuddy/2026-08-10-10-57-30"

pairs = [
    ("S1_best","① 最优"), ("S2_base","② 基准"), ("S3_robust","③ 稳健"),
    ("S4_cyb_gaobiao","④ 创业板高标"), ("S5_low_ma60","⑤ 低标MA60"), ("S6_cyb_highvol","⑥ 高波"),
]

for sid,name in pairs:
    op=f"{BASE}/sim_{sid}_trades.csv"
    np_=f"{BASE}/sim_rt_{sid}_trades.csv"
    with open(op,encoding='utf-8-sig') as f: old=list(csv.DictReader(f))
    with open(np_,encoding='utf-8-sig') as f: new=list(csv.DictReader(f))
    old_map={(t['code'],t['buy_date']):t for t in old}
    new_map={(t['code'],t['buy_date']):t for t in new}
    skipped=[t for k,t in old_map.items() if k not in new_map]
    added=[t for k,t in new_map.items() if k not in old_map]
    # 顺延卖出: 同一(code,buy_date)但sell_date不同
    postponed=[]
    for k,t in new_map.items():
        if k in old_map and old_map[k]['sell_date']!=t['sell_date']:
            postponed.append((old_map[k],t))
    print(f"\n=== {name} ({sid}) 原{len(old)}笔 → 新{len(new)}笔 | 跳过{len(skipped)} 新增{len(added)} 顺延卖出{len(postponed)} ===")
    for t in skipped:
        print(f"  [跳过-买不进] {t['code']} {t['buy_date']} 原收益{t['return_pct']}%")
    for t in added:
        print(f"  [新增] {t['code']} {t['buy_date']} 卖{t['sell_date']} 收益{t['return_pct']}%")
    for ot,nt in postponed[:8]:
        print(f"  [顺延] {ot['code']} 买{ot['buy_date']} 原卖{ot['sell_date']}({ot['return_pct']}%) → 新卖{nt['sell_date']}({nt['return_pct']}%)")
    if len(postponed)>8: print(f"  ... 其余{len(postponed)-8}笔顺延省略")