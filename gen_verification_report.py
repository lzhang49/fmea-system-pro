# -*- coding: utf-8 -*-
"""
生成详细版验证报告 - 供其他AI核验参数正确性
包含完整数据口径、筛选逻辑、计算公式、逐笔明细
"""
import pandas as pd
import numpy as np
import json

# ===== 加载数据 =====
D = pd.read_csv('opt_full_data.csv', dtype={'code': str})
with open('sim_all_summary.json', 'r', encoding='utf-8') as f:
    summary = json.load(f)

# 6组策略定义
STRATEGIES = [
    {
        'id': 'S1_best',
        'name': '① 最优（卡玛第一）',
        'params': {'k':5,'run_max_cap':5,'dd_min':15,'dd_max':30,'mcap_min':30,
                   'ma60_filter':False,'vol_mode':'all','regime':'up','board':'创业'}
    },
    {
        'id': 'S2_base',
        'name': '② 原主规则（基准）',
        'params': {'k':5,'run_max_cap':3,'dd_min':20,'dd_max':35,'mcap_min':0,
                   'ma60_filter':False,'vol_mode':'all','regime':'all','board':'all'}
    },
    {
        'id': 'S3_robust',
        'name': '③ 稳健推荐（全板块+MA60+50亿）',
        'params': {'k':5,'run_max_cap':7,'dd_min':15,'dd_max':40,'mcap_min':50,
                   'ma60_filter':True,'vol_mode':'all','regime':'all','board':'all'}
    },
    {
        'id': 'S4_cyb_gaobiao',
        'name': '④ 创业板高标+上升期',
        'params': {'k':5,'run_max_cap':0,'dd_min':15,'dd_max':35,'mcap_min':30,
                   'ma60_filter':False,'vol_mode':'all','regime':'up','board':'创业'}
    },
    {
        'id': 'S5_low_ma60',
        'name': '⑤ 低标+MA60过滤',
        'params': {'k':5,'run_max_cap':3,'dd_min':20,'dd_max':35,'mcap_min':0,
                   'ma60_filter':True,'vol_mode':'all','regime':'all','board':'all'}
    },
    {
        'id': 'S6_cyb_highvol',
        'name': '⑥ 创业板高波+上升期',
        'params': {'k':5,'run_max_cap':5,'dd_min':15,'dd_max':30,'mcap_min':30,
                   'ma60_filter':False,'vol_mode':'high','regime':'up','board':'创业'}
    },
]

def filter_signals(df, params):
    d = df.copy()
    if params.get('k', 0) > 0:
        d = d[d['k'] == params['k']]
    if params.get('run_max_cap', 0) > 0:
        d = d[d['run_max'] <= params['run_max_cap']]
    d = d[(d['dd_from_peak'] <= -params.get('dd_min', 0)/100) & 
          (d['dd_from_peak'] >= -params.get('dd_max', 100)/100)]
    if params.get('mcap_min', 0) > 0:
        d = d[d['mcap'] >= params['mcap_min']]
    if params.get('ma60_filter', False):
        d = d[d['above_ma60'] == True]
    if params.get('vol_mode', 'all') == 'high':
        med = df['vol20'].median()
        d = d[d['vol20'] >= med]
    if params.get('regime', 'all') == 'up':
        d = d[d['ma20_slope'] > 0.3]
    if params.get('board', 'all') != 'all':
        d = d[d['board'] == params['board']]
    return d.sort_values('buy_date').reset_index(drop=True)

# 加载每组交易明细
all_trades = {}
all_signals = {}
for s in STRATEGIES:
    sid = s['id']
    all_trades[sid] = pd.read_csv(f'sim_{sid}_trades.csv', dtype={'code': str})
    all_signals[sid] = filter_signals(D, s['params'])

# ===== 生成HTML =====
html_parts = []

html_parts.append('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>龙回头战法 · 策略参数验证报告（详细版）</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, "Microsoft YaHei", "PingFang SC", sans-serif;
    background: #0a0e17; color: #d0d7e2;
    padding: 20px; line-height: 1.7; font-size: 14px;
  }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { color: #fff; font-size: 24px; margin-bottom: 8px; text-align:center; }
  h2 { color: #f7c948; font-size: 18px; margin: 30px 0 12px; border-left: 4px solid #f7c948; padding-left: 10px; }
  h3 { color: #5cd6a4; font-size: 15px; margin: 18px 0 10px; }
  p { margin: 8px 0; color: #9ba6ba; }
  code { background: #1a2030; padding: 2px 6px; border-radius: 3px; color: #ff8c5a; font-family: Consolas, monospace; font-size: 13px; }
  .card {
    background: #111827; border: 1px solid #1e2533;
    border-radius: 8px; padding: 20px; margin: 16px 0;
  }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: center; border-bottom: 1px solid #1e2533; }
  th { background: #1a2030; color: #7a8599; font-weight: 600; }
  td { color: #d0d7e2; }
  tr:hover td { background: #151d2b; }
  .up { color: #ff6b35; }
  .down { color: #5cd6a4; }
  .neutral { color: #9ba6ba; }
  .tag {
    display: inline-block; padding: 2px 8px; margin: 2px;
    border-radius: 12px; font-size: 12px;
    background: #1a2030; color: #9ba6ba; border: 1px solid #2a3447;
  }
  .tag.active { background: #ff6b3520; color: #ff8c5a; border-color: #ff6b3550; }
  .formula {
    background: #0d1320; border: 1px solid #2a3447; border-radius: 6px;
    padding: 12px 16px; margin: 10px 0; font-family: Consolas, monospace;
    color: #5cd6a4; font-size: 13px; white-space: pre-wrap;
  }
  .note {
    background: #f7c94810; border-left: 3px solid #f7c948;
    padding: 10px 14px; margin: 12px 0; color: #d4c06a; font-size: 13px;
  }
  .strategy-section {
    border: 1px solid #2a3447; border-radius: 8px;
    padding: 16px; margin: 20px 0; background: #0e1522;
  }
  .strategy-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid #1e2533;
  }
  .strategy-name { font-size: 16px; font-weight: 600; color: #fff; }
  .strategy-stats { font-size: 13px; }
  .strategy-stats span { margin-left: 15px; }
  .param-grid {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 12px 0;
  }
  .param-item {
    background: #111827; padding: 8px 10px; border-radius: 4px;
    border: 1px solid #1e2533; font-size: 12px;
  }
  .param-label { color: #7a8599; }
  .param-value { color: #f7c948; font-weight: 600; }
  .toc { background: #0e1522; border: 1px solid #1e2533; border-radius: 8px; padding: 16px 20px; margin: 16px 0; }
  .toc ol { margin-left: 20px; color: #9ba6ba; }
  .toc li { padding: 3px 0; }
  .toc a { color: #5cd6a4; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="container">
<h1>🐉 龙回头战法 · 策略参数验证报告（详细版）</h1>
<p style="text-align:center; color:#7a8599;">供第三方AI核验参数配置、筛选逻辑、计算方法及交易明细之正确性</p>
''')

# 目录
html_parts.append('''
<div class="toc">
  <h3 style="margin-top:0;">📋 目录</h3>
  <ol>
    <li><a href="#ch1">数据基础与口径说明</a></li>
    <li><a href="#ch2">核心计算公式</a></li>
    <li><a href="#ch3">参数筛选规则详解</a></li>
    <li><a href="#ch4">6组策略参数配置总表</a></li>
    <li><a href="#ch5">6组策略逐笔交易明细</a></li>
    <li><a href="#ch6">实盘模拟规则与手续费假设</a></li>
    <li><a href="#ch7">理论回测 vs 实盘模拟 差异说明</a></li>
  </ol>
</div>
''')

# 第一章：数据基础
html_parts.append('''
<a id="ch1"></a>
<h2>第一章 · 数据基础与口径说明</h2>
<div class="card">

<h3>1.1 数据源</h3>
<p>全部数据来自通达信（.pkl 本地缓存），时间跨度约 5 年（2021-2026），覆盖沪深 A 股所有正常交易个股（本次样本板块：主板/创业/科创，无北证）。</p>

<h3>1.2 入池标准（异常监控池）</h3>
<p>从全部 A 股中筛选满足以下条件的股票，称为"入池事件"：</p>
<div class="formula">入池条件：10个交易日窗口内，个股累计涨幅 ≥ 100%
等效于：(当前收盘价 / 10日前收盘价) - 1 ≥ 1.0
（注：为窗口期累计涨幅，非要求每日连涨）</div>

<p>共筛选出 <b>493 起</b>确认入池事件（去重后，每只股票每次翻倍只算一次事件）。</p>

<h3>1.3 事件核心字段说明</h3>
<table>
  <thead><tr><th>字段名</th><th>含义</th><th>单位/类型</th><th>说明</th></tr></thead>
  <tbody>
    <tr><td><code>code</code></td><td>股票代码</td><td>字符串</td><td>6位数字</td></tr>
    <tr><td><code>board</code></td><td>所属板块</td><td>字符串</td><td>主板 / 创业 / 科创（本次样本无北证）</td></tr>
    <tr><td><code>run_max</code></td><td>入池时最高连板数</td><td>整数</td><td>翻倍过程中最多连续多少个涨停板</td></tr>
    <tr><td><code>mcap</code></td><td>流通市值</td><td>亿元</td><td>信号日（峰值后第5日）流通市值</td></tr>
    <tr><td><code>turn</code></td><td>换手率</td><td>%</td><td>信号日（峰值后第5日）换手率</td></tr>
    <tr><td><code>peak_gain</code></td><td>峰值涨幅</td><td>倍</td><td>从起点到峰值的总涨幅倍数（1.5 = +150%）</td></tr>
    <tr><td><code>peak_i</code></td><td>峰值日索引</td><td>整数</td><td>在个股K线中的位置（内部计算用）</td></tr>
    <tr><td><code>k</code></td><td>相位值</td><td>整数 3~15</td><td>峰值后第 k+1 个交易日开盘买入（信号在峰值后第k日收盘确认，次日T+1买入）</td></tr>
    <tr><td><code>buy_date</code></td><td>买入日期</td><td>日期字符串</td><td>YYYY-MM-DD 格式</td></tr>
    <tr><td><code>buy_price</code></td><td>买入价</td><td>元</td><td>开盘价买入</td></tr>
    <tr><td><code>sell_price</code></td><td>卖出价</td><td>元</td><td>持有3日后收盘价卖出</td></tr>
    <tr><td><code>raw_ret</code></td><td>原始收益率</td><td>小数</td><td>sell_price / buy_price - 1</td></tr>
    <tr><td><code>net_ret</code></td><td>净收益率（扣手续费）</td><td>小数</td><td>已扣除买入万3 + 卖出千1.3</td></tr>
    <tr><td><code>dd_from_peak</code></td><td>距峰值回撤幅度</td><td>小数</td><td>负值，如 -0.25 = 回撤25%</td></tr>
    <tr><td><code>pos_250</code></td><td>250日分位</td><td>万分位 0~10000</td><td>创业板指数收盘价250日百分位×100（注意：本字段为市场环境标签，取自创业板指数而非个股；存储为万分位，如 9640 = 96.4%分位）</td></tr>
    <tr><td><code>above_ma60</code></td><td>是否在MA60上方</td><td>布尔</td><td>True = 站上60日均线</td></tr>
    <tr><td><code>vol20</code></td><td>20日平均波动率</td><td>%</td><td>过去20日日涨跌幅标准差×√250（年化）</td></tr>
    <tr><td><code>ma20_slope</code></td><td>MA20斜率</td><td>%/日</td><td>20日均线的日变化率，>0.3 视为上升趋势</td></tr>
    <tr><td><code>zt_in10</code></td><td>10日内涨停天数</td><td>整数</td><td>入池前10天内有几个涨停</td></tr>
    <tr><td><code>vshrink</code></td><td>缩量比</td><td>倍</td><td>回调期成交量 vs 峰值成交量的比例</td></tr>
  </tbody>
</table>

<h3>1.4 全样本规模</h3>
<p><code>opt_full_data.csv</code> 共 <b>6,409 行</b> = 493 个入池事件 × 13 个相位（k=3 到 k=15）。每行代表一次"如果在峰值后第k+1日开盘买入并持有3日"的模拟交易结果。</p>

</div>
''')

# 第二章：计算公式
html_parts.append('''
<a id="ch2"></a>
<h2>第二章 · 核心计算公式</h2>
<div class="card">

<h3>2.1 单笔收益率</h3>
<div class="formula">买入金额 = buy_price × shares
买入手续费 = 买入金额 × 0.0003  (万3)
买入总成本 = 买入金额 + 买入手续费

卖出金额 = sell_price × shares
卖出手续费 = 卖出金额 × 0.0013  (千1.3 = 印花税千1 + 佣金万3)
卖出净收入 = 卖出金额 - 卖出手续费

净盈亏 = 卖出净收入 - 买入总成本
净收益率 = 净盈亏 / 买入总成本</div>

<h3>2.2 理论回测（等权日频复利）</h3>
<p>同一日有多只信号时，取当日所有信号的净收益率算术平均，作为当日收益率。然后按日复利计算资金曲线。</p>
<div class="formula">当日收益率 = mean(同日所有信号的 net_ret)
当日净值 = 前一日净值 × (1 + 当日收益率)
总收益 = 最终净值 - 1</div>

<h3>2.3 年化收益率</h3>
<div class="formula">年化收益 = (最终净值)^(1/年数) - 1
其中 年数 = (最后一个信号日 - 第一个信号日) / 365.25</div>

<h3>2.4 最大回撤</h3>
<div class="formula">每日回撤 = 当日净值 / 历史最高净值 - 1
最大回撤 = min(每日回撤)  (为负值)</div>

<h3>2.5 卡玛比率 (Calmar Ratio)</h3>
<div class="formula">卡玛比率 = 年化收益率 / |最大回撤|

注：若最大回撤绝对值 < 0.5%，卡玛比率取 99（截断处理，避免除零误差）</div>

<h3>2.6 胜率</h3>
<div class="formula">胜率 = 盈利交易笔数 / 总交易笔数 × 100%</div>

<h3>2.7 盈亏比</h3>
<div class="formula">盈亏比 = 平均盈利 / |平均亏损|
  其中 平均盈利 = 所有盈利交易的收益率均值
       平均亏损 = 所有亏损交易的收益率均值</div>

<h3>2.8 夏普比率</h3>
<div class="formula">夏普 = 日收益率均值 / 日收益率标准差 × √250
  （基于理论等权日频序列计算，无风险利率取0）</div>

</div>
''')

# 第三章：参数筛选规则
html_parts.append('''
<a id="ch3"></a>
<h2>第三章 · 参数筛选规则详解</h2>
<div class="card">

<p>所有筛选条件为"AND"关系（同时满足）。下表列出每个参数的含义、取值范围和筛选逻辑。</p>

<table>
  <thead>
    <tr><th>参数名</th><th>变量</th><th>含义</th><th>筛选公式</th><th>可选值</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>买入相位</td><td><code>k</code></td>
      <td>峰值后第k+1日开盘买入（信号在峰值后第k日收盘确认）</td>
      <td><code>k == K</code></td>
      <td>3~15</td>
    </tr>
    <tr>
      <td>连板数上限</td><td><code>run_max_cap</code></td>
      <td>入池时最高连板数不超过该值，0=不限制</td>
      <td><code>run_max &lt;= CAP</code>（当CAP>0时）</td>
      <td>0, 3, 5, 7</td>
    </tr>
    <tr>
      <td>回调幅度下限</td><td><code>dd_min</code></td>
      <td>买入时距峰值至少回调多少</td>
      <td><code>dd_from_peak &lt;= -dd_min/100</code></td>
      <td>10, 15, 20, 25（单位：%）</td>
    </tr>
    <tr>
      <td>回调幅度上限</td><td><code>dd_max</code></td>
      <td>买入时距峰值最多回调多少</td>
      <td><code>dd_from_peak &gt;= -dd_max/100</code></td>
      <td>30, 35, 40, 45, 50（单位：%）</td>
    </tr>
    <tr>
      <td>市值下限</td><td><code>mcap_min</code></td>
      <td>流通市值不低于该值，0=不限制</td>
      <td><code>mcap &gt;= mcap_min</code>（当mcap_min>0时）</td>
      <td>0, 30, 50（单位：亿）</td>
    </tr>
    <tr>
      <td>MA60过滤</td><td><code>ma60_filter</code></td>
      <td>只选站上60日均线的信号</td>
      <td><code>above_ma60 == True</code></td>
      <td>True / False</td>
    </tr>
    <tr>
      <td>波动率筛选</td><td><code>vol_mode</code></td>
      <td>按20日年化波动率分层</td>
      <td><code>vol20 &gt;= 全样本中位数</code>（high模式）</td>
      <td>all / high</td>
    </tr>
    <tr>
      <td>市场状态</td><td><code>regime</code></td>
      <td>MA20斜率判断市场趋势</td>
      <td><code>ma20_slope &gt; 0.3</code>（上升期）</td>
      <td>all / up</td>
    </tr>
    <tr>
      <td>板块限制</td><td><code>board</code></td>
      <td>仅交易指定板块</td>
      <td><code>board == 板块名</code></td>
      <td>all / 创业（本次优化网格仅覆盖这两项）</td>
    </tr>
    <tr>
      <td>峰值涨幅下限</td><td><code>peak_gain_min</code></td>
      <td>入池事件峰值涨幅不低于该值（倍），0=不限制</td>
      <td><code>peak_gain &gt;= peak_gain_min</code>（当peak_gain_min>0时）</td>
      <td>0, 1.5（倍）</td>
    </tr>
    <tr>
      <td>持有期</td><td><code>HOLD_DAYS</code></td>
      <td>买入后持有N个交易日，收盘价卖出</td>
      <td>—</td>
      <td>固定 = 3 日</td>
    </tr>
  </tbody>
</table>

<div class="note">
<b>重要提示</b>：回调幅度 dd_from_peak 是<b>负值</b>，例如 -0.25 表示距峰值回撤了 25%。
筛选时 dd_min 是"至少回调多少"（下界），dd_max 是"最多回调多少"（上界）。<br>
例：dd_min=15, dd_max=30 → 筛选条件为 -30% ≤ dd_from_peak ≤ -15%
</div>

</div>
''')

# 第四章：6组策略参数总表
html_parts.append('''
<a id="ch4"></a>
<h2>第四章 · 6组策略参数配置总表</h2>

<table>
  <thead>
    <tr>
      <th>参数</th>
      <th>①最优</th>
      <th>②基准</th>
      <th>③稳健</th>
      <th>④创业板高标</th>
      <th>⑤低标+MA60</th>
      <th>⑥创业板高波</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>k值（买入相位）</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td></tr>
    <tr><td>连板数上限</td><td>≤5</td><td>≤3</td><td>≤7</td><td>不限</td><td>≤3</td><td>≤5</td></tr>
    <tr><td>回调下限 dd_min</td><td>15%</td><td>20%</td><td>15%</td><td>15%</td><td>20%</td><td>15%</td></tr>
    <tr><td>回调上限 dd_max</td><td>30%</td><td>35%</td><td>40%</td><td>35%</td><td>35%</td><td>30%</td></tr>
    <tr><td>市值下限</td><td>≥30亿</td><td>不限</td><td>≥50亿</td><td>≥30亿</td><td>不限</td><td>≥30亿</td></tr>
    <tr><td>MA60过滤</td><td>关</td><td>关</td><td>开</td><td>关</td><td>开</td><td>关</td></tr>
    <tr><td>波动率</td><td>全部</td><td>全部</td><td>全部</td><td>全部</td><td>全部</td><td>高波</td></tr>
    <tr><td>市场状态</td><td>上升期</td><td>全部</td><td>全部</td><td>上升期</td><td>全部</td><td>上升期</td></tr>
    <tr><td>板块</td><td>创业板</td><td>全部</td><td>全部</td><td>创业板</td><td>全部</td><td>创业板</td></tr>
    <tr><td>持有期</td><td>3日</td><td>3日</td><td>3日</td><td>3日</td><td>3日</td><td>3日</td></tr>
  </tbody>
</table>

<h3>6组策略回测结果对比（理论等权复利）</h3>
<table>
  <thead>
    <tr>
      <th>策略</th>
      <th>信号数</th>
      <th>信号天数</th>
      <th>总收益</th>
      <th>年化</th>
      <th>最大回撤</th>
      <th>卡玛</th>
      <th>胜率</th>
      <th>均单笔</th>
      <th>夏普</th>
    </tr>
  </thead>
  <tbody>
''')

for s in STRATEGIES:
    sid = s['id']
    sigs = all_signals[sid]
    if len(sigs) < 3: continue
    
    # 重新计算理论指标
    daily = sigs.groupby('buy_date')['net_ret'].mean().reset_index()
    equity = 1.0
    peak_eq = 1.0
    max_dd = 0
    for _, r in daily.iterrows():
        equity *= (1 + r['net_ret'])
        peak_eq = max(peak_eq, equity)
        dd = (equity / peak_eq - 1) * 100
        max_dd = min(max_dd, dd)
    
    n_days = len(daily)
    start = pd.Timestamp(daily['buy_date'].iloc[0])
    end = pd.Timestamp(daily['buy_date'].iloc[-1])
    years = max((end - start).days / 365.25, 0.5)
    ann_ret = (equity**(1/years) - 1) * 100
    win_rate = (sigs['net_ret'] > 0).mean() * 100
    avg_ret = sigs['net_ret'].mean() * 100
    sharpe = daily['net_ret'].mean() / daily['net_ret'].std() * np.sqrt(250) if daily['net_ret'].std() > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd < -0.5 else 99
    
    html_parts.append(f'''    <tr>
      <td>{s["name"]}</td>
      <td>{len(sigs)}</td>
      <td>{n_days}</td>
      <td class="up">+{(equity-1)*100:.1f}%</td>
      <td class="up">+{ann_ret:.1f}%</td>
      <td class="down">{max_dd:.1f}%</td>
      <td class="up"><b>{calmar:.2f}</b></td>
      <td>{win_rate:.1f}%</td>
      <td>{avg_ret:.2f}%</td>
      <td>{sharpe:.2f}</td>
    </tr>
''')

html_parts.append('''  </tbody>
</table>
''')

# 第五章：逐笔交易明细
html_parts.append('''
<a id="ch5"></a>
<h2>第五章 · 6组策略逐笔交易明细（10万本金实盘级）</h2>
<p>以下为实盘级模拟逐笔明细：10万本金起步，同日多票资金等分，100股整手取整，扣除精确手续费。</p>
''')

for s in STRATEGIES:
    sid = s['id']
    sname = s['name']
    tdf = all_trades[sid]
    stats = summary[sid]['stats']
    
    html_parts.append(f'''
<div class="strategy-section">
  <div class="strategy-header">
    <div class="strategy-name">{sname}</div>
    <div class="strategy-stats">
      <span>成交 <b>{int(stats["n_trades"])}</b> 笔</span>
      <span>期末 <b>{stats["final_equity"]/10000:.1f}万</b></span>
      <span class="up">+{stats["total_return_pct"]:.1f}%</span>
      <span class="down">回撤 {stats["max_dd_pct"]:.1f}%</span>
      <span>胜率 {stats["win_rate_pct"]:.1f}%</span>
    </div>
  </div>
  
  <h3>参数配置</h3>
  <div class="param-grid">
    <div class="param-item"><div class="param-label">k值</div><div class="param-value">{s["params"]["k"]}</div></div>
    <div class="param-item"><div class="param-label">连板上限</div><div class="param-value">{"≤"+str(s["params"]["run_max_cap"]) if s["params"]["run_max_cap"]>0 else "不限"}</div></div>
    <div class="param-item"><div class="param-label">回调范围</div><div class="param-value">{s["params"]["dd_min"]}%~{s["params"]["dd_max"]}%</div></div>
    <div class="param-item"><div class="param-label">市值下限</div><div class="param-value">{str(s["params"]["mcap_min"])+"亿" if s["params"]["mcap_min"]>0 else "不限"}</div></div>
    <div class="param-item"><div class="param-label">MA60</div><div class="param-value">{"开" if s["params"]["ma60_filter"] else "关"}</div></div>
    <div class="param-item"><div class="param-label">波动率</div><div class="param-value">{"高波" if s["params"]["vol_mode"]=="high" else "全部"}</div></div>
    <div class="param-item"><div class="param-label">市场状态</div><div class="param-value">{"上升期" if s["params"]["regime"]=="up" else "全部"}</div></div>
    <div class="param-item"><div class="param-label">板块</div><div class="param-value">{s["params"]["board"] if s["params"]["board"]!="all" else "全部"}</div></div>
    <div class="param-item"><div class="param-label">持有期</div><div class="param-value">3日</div></div>
    <div class="param-item"><div class="param-label">买入方式</div><div class="param-value">开盘价</div></div>
  </div>
  
  <h3>逐笔交易明细</h3>
  <div style="overflow-x:auto;">
  <table>
    <thead>
      <tr>
        <th>序号</th>
        <th>代码</th>
        <th>买入日期</th>
        <th>买入价</th>
        <th>股数</th>
        <th>买入金额</th>
        <th>卖出日期</th>
        <th>卖出价</th>
        <th>卖出金额</th>
        <th>持仓天数</th>
        <th>盈亏(元)</th>
        <th>收益率</th>
      </tr>
    </thead>
    <tbody>
''')
    
    for _, r in tdf.iterrows():
        ret_cls = 'up' if r['pnl'] >= 0 else 'down'
        ret_sign = '+' if r['pnl'] >= 0 else ''
        html_parts.append(f'''      <tr>
        <td>{int(r["trade_no"])}</td>
        <td>{r["code"]}</td>
        <td>{r["buy_date"]}</td>
        <td>{r["buy_price"]:.2f}</td>
        <td>{int(r["shares"])}</td>
        <td>{r["buy_amount"]:,.2f}</td>
        <td>{r["sell_date"]}</td>
        <td>{r["sell_price"]:.2f}</td>
        <td>{r["sell_amount"]:,.2f}</td>
        <td>{int(r["hold_days"])}</td>
        <td class="{ret_cls}">{ret_sign}{r["pnl"]:,.2f}</td>
        <td class="{ret_cls}">{ret_sign}{r["return_pct"]:.2f}%</td>
      </tr>
''')
    
    html_parts.append('''    </tbody>
  </table>
  </div>
</div>
''')

# 第六章：实盘模拟规则
html_parts.append('''
<a id="ch6"></a>
<h2>第六章 · 实盘模拟规则与手续费假设</h2>
<div class="card">

<h3>6.1 初始条件</h3>
<ul style="color:#9ba6ba; margin-left:20px;">
  <li>初始资金：100,000 元</li>
  <li>时间范围：以该组第一个买入信号日为起点，最后一个卖出日为终点</li>
  <li>交易标的：仅 A 股（不含基金、债券）</li>
</ul>

<h3>6.2 买入规则</h3>
<ul style="color:#9ba6ba; margin-left:20px;">
  <li>买入时间：信号日开盘价成交（T+1 开盘买入，即峰值后第k+1日）</li>
  <li>仓位分配：当日有 N 只信号时，可用资金 N 等分，每只各买 1/N</li>
  <li>数量取整：100 股整数倍向下取整</li>
  <li>资金不足处理：若等分后单只资金不足买 100 股，则跳过该票</li>
  <li>停牌处理：信号日股票停牌则跳过</li>
</ul>

<h3>6.3 卖出规则</h3>
<ul style="color:#9ba6ba; margin-left:20px;">
  <li>持有期：固定 3 个交易日（含买入日吗？不含！买入日算第0天，第3个交易日收盘卖）</li>
  <li>卖出时间：到期日收盘价成交</li>
  <li>到期日停牌：顺延至下一交易日卖出</li>
</ul>

<h3>6.5 成交可行性说明（重要）</h3>
<div class="note">
本模拟为<b>理想成交模型</b>：买入按信号日开盘价、卖出按到期日收盘价成交，未建模涨停排队买不进、跌停封死卖不出的机制（仅处理停牌）。
已用真实K线（open/high/low/close）对全部 209 笔实盘交易逐笔核验：<br>
<b>买入日：0 笔开盘涨停/一字涨停</b>（全部可按开盘价成交）；<b>卖出日：0 笔一字跌停封死</b>（全部可按收盘价成交）。<br>
其中 5 笔（③稳健：603268/000795/000753/600126/603626）卖出日收盘跌停但盘中打开、早盘开盘价远高于收盘价，实际可按更优价格卖出——按收盘价成交属于<b>保守估计</b>。<br>
<b>结论：本报告全部买卖点在数据范围内均可成交，但模型本身未含涨跌停成交过滤，若换用其他数据集需重新核验。</b>
</div>

<h3>6.4 手续费假设</h3>
<table>
  <thead><tr><th>项目</th><th>费率</th><th>收取方向</th><th>说明</th></tr></thead>
  <tbody>
    <tr><td>佣金</td><td>万分之三 (0.03%)</td><td>双向</td><td>券商佣金，含经手费等杂费</td></tr>
    <tr><td>印花税</td><td>千分之一 (0.1%)</td><td>卖出</td><td>国家征收，仅卖出时收取</td></tr>
    <tr><td><b>合计买入</b></td><td><b>万3 (0.03%)</b></td><td>—</td><td>—</td></tr>
    <tr><td><b>合计卖出</b></td><td><b>千1.3 (0.13%)</b></td><td>—</td><td>佣金万3 + 印花税千1</td></tr>
    <tr><td><b>双边合计</b></td><td><b>约千1.6</b></td><td>—</td><td>买入万3 + 卖出千1.3</td></tr>
  </tbody>
</table>

<h3>6.5 资金曲线计算</h3>
<p>每日总资产 = 现金余额 + 所有持仓股票当日收盘价估值。按日记录，计算最大回撤。</p>

</div>
''')

# 第七章：理论vs实盘差异
html_parts.append('''
<a id="ch7"></a>
<h2>第七章 · 理论回测 vs 实盘模拟 差异说明</h2>
<div class="card">

<p>理论回测（第四章数据）与实盘模拟（第五章数据）存在差异，原因如下：</p>

<table>
  <thead>
    <tr><th>差异来源</th><th>理论回测</th><th>实盘模拟</th><th>影响方向</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>资金利用方式</td>
      <td>每笔交易都全仓投入（复利假设）</td>
      <td>同日多票等分资金 + 整手取整导致部分闲置</td>
      <td>实盘 < 理论</td>
    </tr>
    <tr>
      <td>空仓期</td>
      <td>只计算有信号的日子，空仓期不计入</td>
      <td>空仓期资金趴在账上，拉低年化</td>
      <td>实盘年化 < 理论年化</td>
    </tr>
    <tr>
      <td>手续费</td>
      <td>已扣除（固定费率法）</td>
      <td>已扣除（精确到每分钱）</td>
      <td>基本一致</td>
    </tr>
    <tr>
      <td>仓位管理</td>
      <td>等权平均（每票权重相同）</td>
      <td>资金等分（金额相同，股数不同）</td>
      <td>略有差异</td>
    </tr>
    <tr>
      <td>价格类型</td>
      <td>买入=开盘价，卖出=收盘价</td>
      <td>同左</td>
      <td>一致</td>
    </tr>
    <tr>
      <td>停牌/跳空</td>
      <td>按实际K线已有体现</td>
      <td>停牌顺延卖出</td>
      <td>基本一致</td>
    </tr>
    <tr>
      <td>涨停/跌停成交</td>
      <td>未建模（理想成交：开盘买、收盘卖）</td>
      <td>未建模（理想成交：开盘买、收盘卖）</td>
      <td>一致；本批209笔经核验均无买入一字涨停/卖出封死跌停，可成交</td>
    </tr>
  </tbody>
</table>

<div class="note">
<b>结论</b>：实盘收益通常低于理论收益，主要因为空仓期资金闲置和同日多票分流。
理论卡玛是"信号质量"的衡量，实盘卡玛还包含资金效率因素。
两者的胜率和单笔平均收益应该高度一致（因为每笔交易本身的盈亏计算方式相同）。
</div>

<h3>已知 Bug 与修复记录</h3>
<table>
  <thead>
    <tr><th>Bug描述</th><th>影响范围</th><th>原因</th><th>修复方案</th><th>修复状态</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>索引错位</td>
      <td>实盘模拟所有策略</td>
      <td>用上证指数日历索引定位个股K线，新股/停牌导致偏移</td>
      <td>改为每只股票独立维护日期列表，用日期字符串精确查找</td>
      <td style="color:#5cd6a4;">✅ 已修复</td>
    </tr>
    <tr>
      <td>pos_250 字段单位/来源描述不符</td>
      <td>环境字段说明（1.3节）</td>
      <td>脚本对百分位两次×100 存为万分位(0~10000)；且字段取自创业板指数而非个股，报告原描述未注明</td>
      <td>字段说明已改为：万分位 0~10000，创业板指数250日分位×100，标注市场环境标签</td>
      <td style="color:#5cd6a4;">✅ 已修正（6组策略均未用该字段筛选，对结果零影响）</td>
    </tr>
  </tbody>
</table>

<h3>前视偏差（未来函数）质检说明</h3>
<div class="note">
<b>结论：492/493 事件无前视，1 个事件（301318）有轻微前视，对结果零影响。</b><br><br>
本报告采用<b>回落确认制</b>构建信号：<br>
① 入池后 40 日内找到最高价作为峰值（peak_i，确认时点已知）；<br>
② 收盘价从峰值回撤 ≥10% 确认回调（确认日 ci，同样已知）；<br>
③ 信号日 = 峰值后第 k 日（sig = peak_i + 5），该日所有信息在当日收盘时均已知。<br>
因此常规事件<b>不存在前视</b>：信号日的买入决策只用到了当日及之前的信息。<br><br>
<b>例外事件 301318（唯一）</b>：该事件的确认日（回撤确认）晚于信号日 2 天，即信号日生成时，回撤 ≥10% 的确认尚未发生——存在轻微前视。<br>
<b>影响评估</b>：该事件全部 13 个相位均为亏损，且 6 组策略均未选中该事件（无一笔实盘交易涉及 301318）。<br>
<b>结论：对报告全部数字零影响。</b> 已逐事件核验，其余 492 个事件确认日 ≤ 信号日，无前视。
</div>

</div>

<p style="text-align:center; color:#5a6579; margin-top: 40px; padding: 20px 0; border-top: 1px solid #1e2533;">
  — 报告结束 · 请其他AI核验以上所有参数、公式和数据是否正确 —
</p>
</div>
</body>
</html>''')

# 写入文件
html_content = '\n'.join(html_parts)
with open('verification_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f'详细验证报告已生成: verification_report.html')
print(f'文件大小: {len(html_content):,} 字符')
