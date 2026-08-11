# -*- coding: utf-8 -*-
"""
龙回头战法 · 每日收盘扫描 + 飞书推送 一体化脚本（6策略汇总版）
==============================================================
流程: 增量更新缓存 -> 逐个扫描S1~S6 -> 汇总 -> 推送到飞书私聊

用法:
  python lht_daily_push.py                # 完整流程(更新+6策略扫描+推送)
  python lht_daily_push.py --no-update    # 跳过缓存更新(只扫描+推送)
  python lht_daily_push.py --dry-run      # 只扫描不推送(终端显示结果)
  python lht_daily_push.py --stdout       # 输出消息文本到stdout(cron用)

依赖: akshare, pandas, requests (Hermes venv 已装)
飞书凭证: 从 ~/AppData/Local/hermes/.env 读取 FEISHU_APP_ID/SECRET
"""
import os, sys, subprocess, json, time, argparse

# 数据工作目录(缓存/扫描脚本所在处)
WORKDIR = r'C:/Users/EDY/WorkBuddy/2026-08-10-10-57-30'
# 本脚本可能被 cron 从 ~/AppData/Local/hermes/scripts/ 调用, 数据脚本路径固定用 WORKDIR
BASE = WORKDIR
HERMES_ENV = os.path.join(os.path.expanduser('~'), 'AppData/Local/hermes/.env')

FEISHU_CHAT_ID = 'oc_a832709f95c92b26aaff41e5bb8e14c9'  # 用户飞书私聊

# 6个策略顺序 + 中文名
STRATEGY_ORDER = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
STRATEGY_NAMES = {
    'S1': '① 最优(卡玛第一)',
    'S2': '② 原主规则(基准)',
    'S3': '③ 稳健(所有市场)',
    'S4': '④ 创业板高标',
    'S5': '⑤ 低标+MA60',
    'S6': '⑥ 创业板高波',
}


def load_env():
    env = {}
    try:
        with open(HERMES_ENV, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    except Exception as e:
        print(f'读取.env失败: {e}')
    return env


def feishu_token(app_id, app_secret):
    import requests
    r = requests.post('https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
                      json={'app_id': app_id, 'app_secret': app_secret}, timeout=15)
    j = r.json()
    if j.get('code') != 0:
        raise RuntimeError(f'飞书token失败: {j}')
    return j['tenant_access_token']


def feishu_send_text(token, chat_id, text):
    import requests
    r = requests.post(
        f'https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id',
        headers={'Authorization': f'Bearer {token}',
                 'Content-Type': 'application/json; charset=utf-8'},
        json={'receive_id': chat_id, 'msg_type': 'text',
              'content': json.dumps({'text': text}, ensure_ascii=False)},
        timeout=15)
    j = r.json()
    if j.get('code') != 0:
        raise RuntimeError(f'飞书发送失败: {j}')
    return j


def read_scan():
    """读取 lht_scan_result.csv (若存在)"""
    fp = os.path.join(BASE, 'lht_scan_result.csv')
    if not os.path.exists(fp):
        return []
    import csv
    with open(fp, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def run_scan(strategy):
    """运行单个策略扫描, 返回 (asof, strat_name, hits_list, stdout_tail)"""
    res_fp = os.path.join(BASE, 'lht_scan_result.csv')
    if os.path.exists(res_fp):
        os.remove(res_fp)
    r = subprocess.run(
        [sys.executable, os.path.join(BASE, 'lht_daily_scan.py'),
         '--strategy', strategy],
        capture_output=True, text=True, timeout=900)
    asof = ''
    strat_name = STRATEGY_NAMES.get(strategy, strategy)
    for line in (r.stdout or '').splitlines():
        if '扫描基准日' in line:
            parts = line.split('信号日):')
            if len(parts) > 1:
                rest = parts[1].strip()
                asof = rest.split('策略:')[0].strip() if '策略:' in rest else rest
            if '策略:' in line:
                strat_name = line.split('策略:')[1].strip()
            break
    hits = read_scan()
    tail = r.stdout[-300:] if r.stdout else ''
    ok = (r.returncode == 0)
    return asof, strat_name, hits, tail, ok


def format_message_all(asof, results):
    """results: {sid: (name, hits)} 汇总一条消息"""
    lines = []
    lines.append('📈 龙回头战法 · 每日信号（6策略汇总）')
    lines.append(f'📅 信号日: {asof}')
    lines.append('─' * 28)
    total = sum(len(h) for _, h in results.values())
    if total == 0:
        lines.append('✅ 今日6个策略均无满足条件的信号')
        lines.append('（T+1 无需操作，继续等待）')
    else:
        lines.append(f'⚡ 共命中 {total} 只信号（T+1 开盘买入）')
        lines.append('')
        for sid in STRATEGY_ORDER:
            if sid not in results:
                continue
            name, hits = results[sid]
            lines.append(f'【{name}】{len(hits)} 只')
            if not hits:
                lines.append('  · 无信号')
            else:
                for h in hits:
                    dd = float(h.get('dd_from_peak', 0)) * 100
                    mcap = h.get('mcap', '')
                    try:
                        mcap = f'{float(mcap):.1f}'
                    except Exception:
                        pass
                    lines.append(f"  · {h['code']} 板{h['run_max']} 回调{dd:.1f}% 市值{mcap}亿")
                    lines.append(f"    峰值{h.get('peak_date','')} 买入{h.get('buy_date','')}")
            lines.append('')
        lines.append('⚠️ 仅供参考，非投资建议')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-update', action='store_true', help='跳过缓存更新')
    ap.add_argument('--dry-run', action='store_true', help='只扫描不推送')
    ap.add_argument('--stdout', action='store_true', help='只输出消息文本到stdout(供cron no_agent投递)')
    args = ap.parse_args()

    t0 = time.time()
    # 1. 更新缓存
    if not args.no_update:
        print('==> 步骤1/3: 增量更新缓存...', flush=True)
        r = subprocess.run([sys.executable, os.path.join(BASE, 'lht_update_cache.py')],
                           capture_output=True, text=True, timeout=1800)
        print(r.stdout[-500:] if r.stdout else '')
        if r.returncode != 0:
            print('更新缓存异常:', r.stderr[-300:] if r.stderr else '')
    else:
        print('==> 步骤1/3: 跳过缓存更新', flush=True)

    # 2. 逐个扫描6个策略
    print('==> 步骤2/3: 扫描6策略信号...', flush=True)
    results = {}
    asof_main = ''
    for sid in STRATEGY_ORDER:
        name = STRATEGY_NAMES[sid]
        print(f'  [{sid}] {name} ...', end=' ', flush=True)
        asof, sname, hits, tail, ok = run_scan(sid)
        if not asof_main and asof:
            asof_main = asof
        results[sid] = (sname, hits)
        status = f'{len(hits)}只' if ok else '失败'
        print(status)

    # 3. 推送
    msg = format_message_all(asof_main, results)
    if args.stdout:
        print(msg)
        return
    print('─' * 30)
    print(msg)
    if args.dry_run:
        print('\n(dry-run 模式, 未推送飞书)')
        return
    env = load_env()
    app_id = env.get('FEISHU_APP_ID'); app_secret = env.get('FEISHU_APP_SECRET')
    if not app_id or not app_secret:
        print('!! 未找到飞书凭证, 跳过推送')
        return
    print('==> 步骤3/3: 推送到飞书...', flush=True)
    try:
        token = feishu_token(app_id, app_secret)
        feishu_send_text(token, FEISHU_CHAT_ID, msg)
        print(f'✅ 已推送飞书 (用时{time.time()-t0:.0f}s)')
    except Exception as e:
        print(f'❌ 推送失败: {e}')


if __name__ == '__main__':
    main()
