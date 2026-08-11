# -*- coding: utf-8 -*-
"""
生成 trades_detail_10w.html — 10万本金个股买卖点明细页
按年分组、可排序、含年度汇总、TOP盈亏
"""
import pandas as pd
import json

T = pd.read_csv('trades_detail_10w.csv', dtype={'code': str})
EQ = pd.read_csv('equity_detail_10w.csv')

INIT_CAP = 100000
final_equity = EQ['equity'].iloc[-1]
total_return = (final_equity / INIT_CAP - 1) * 100
peak = EQ['equity'].cummax()
dd = (EQ['equity'] / peak - 1) * 100
max_dd = dd.min()

# 年度汇总
T['year'] = T['buy_date'].str[:4].astype(int)
yearly = T.groupby('year').agg(
    n=('trade_no', 'count'),
    total_buy=('buy_amount', 'sum'),
    total_sell=('sell_amount', 'sum'),
    total_pnl=('pnl', 'sum'),
    win_rate=('pnl', lambda x: (x>0).mean()*100),
    avg_ret=('return_pct', 'mean'),
    max_win=('return_pct', 'max'),
    max_loss=('return_pct', 'min'),
).reset_index()

# 按年分组的交易明细
years_data = {}
for y in sorted(T['year'].unique(), reverse=True):
    yd = T[T['year'] == y].sort_values('buy_date')
    rows = []
    for _, r in yd.iterrows():
        rows.append({
            'no': int(r['trade_no']),
            'code': r['code'],
            'buy_date': r['buy_date'],
            'buy_price': round(r['buy_price'], 2),
            'shares': int(r['shares']),
            'buy_amount': round(r['buy_amount'], 2),
            'buy_fee': round(r['buy_fee'], 2),
            'sell_date': r['sell_date'],
            'sell_price': round(r['sell_price'], 2),
            'sell_amount': round(r['sell_amount'], 2),
            'sell_fee': round(r['sell_fee'], 2),
            'pnl': round(r['pnl'], 2),
            'ret': round(r['return_pct'], 2),
            'hold': int(r['hold_days']),
        })
    years_data[int(y)] = rows

# TOP盈亏
top_win = T.nlargest(10, 'pnl')[['trade_no','code','buy_date','sell_date','pnl','return_pct']].to_dict('records')
top_loss = T.nsmallest(10, 'pnl')[['trade_no','code','buy_date','sell_date','pnl','return_pct']].to_dict('records')

# 资金曲线数据（采样 200 点）
eq_sampled = EQ.iloc[::max(1, len(EQ)//200)]
equity_data = [{'d': r['date'], 'v': round(r['equity'], 2)} for _, r in eq_sampled.iterrows()]

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>10万本金 · 龙回头战法个股买卖点明细</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; background: #0f1419; color: #e6e6e6; line-height: 1.6; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 30px 20px; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; background: linear-gradient(90deg, #ff6b35, #f7c948); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .subtitle {{ color: #8a94a6; margin-bottom: 30px; font-size: 14px; }}
  
  /* 概览卡片 */
  .overview {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 30px; }}
  .card {{ background: #1a1f2e; border-radius: 12px; padding: 20px; border: 1px solid #2a3142; }}
  .card .label {{ font-size: 12px; color: #8a94a6; margin-bottom: 8px; }}
  .card .value {{ font-size: 24px; font-weight: 700; }}
  .card .value.up {{ color: #ef4444; }}
  .card .value.down {{ color: #22c55e; }}
  
  /* 图表区 */
  .chart-box {{ background: #1a1f2e; border-radius: 12px; padding: 20px; margin-bottom: 30px; border: 1px solid #2a3142; }}
  .chart-box h2 {{ font-size: 16px; margin-bottom: 16px; color: #e6e6e6; display: flex; align-items: center; gap: 8px; }}
  .chart-box h2::before {{ content: ""; width: 4px; height: 18px; background: linear-gradient(180deg, #ff6b35, #f7c948); border-radius: 2px; }}
  #equityChart {{ width: 100%; height: 320px; }}
  
  /* 年度汇总表 */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #252d3d; color: #8a94a6; font-weight: 500; padding: 10px 12px; text-align: right; border-bottom: 1px solid #2a3142; }}
  th:first-child {{ text-align: center; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #1e2533; text-align: right; }}
  td:first-child {{ text-align: center; }}
  tr:hover td {{ background: #1e2533; }}
  .up {{ color: #ef4444; }}
  .down {{ color: #22c55e; }}
  
  /* 分年明细 */
  .year-section {{ margin-bottom: 30px; }}
  .year-header {{ display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; background: #1a1f2e; border-radius: 10px; cursor: pointer; border: 1px solid #2a3142; margin-bottom: 2px; }}
  .year-header:hover {{ background: #252d3d; }}
  .year-title {{ font-size: 16px; font-weight: 600; display: flex; align-items: center; gap: 12px; }}
  .year-badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; }}
  .year-badge.win {{ background: rgba(239,68,68,0.15); color: #ef4444; }}
  .year-badge.loss {{ background: rgba(34,197,94,0.15); color: #22c55e; }}
  .year-stats {{ font-size: 13px; color: #8a94a6; display: flex; gap: 20px; }}
  .year-stats span b {{ color: #e6e6e6; font-weight: 600; }}
  .arrow {{ transition: transform 0.2s; color: #8a94a6; font-size: 12px; }}
  .year-header.active .arrow {{ transform: rotate(90deg); }}
  .year-content {{ display: none; overflow-x: auto; background: #1a1f2e; border-radius: 0 0 10px 10px; border: 1px solid #2a3142; border-top: none; }}
  .year-content.show {{ display: block; }}
  
  /* TOP排行 */
  .top-section {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
  
  /* 说明 */
  .note {{ background: #1a1f2e; border-left: 3px solid #ff6b35; padding: 16px 20px; border-radius: 0 8px 8px 0; font-size: 13px; color: #8a94a6; line-height: 1.8; }}
  .note b {{ color: #e6e6e6; }}
  
  @media (max-width: 1000px) {{
    .overview {{ grid-template-columns: repeat(2, 1fr); }}
    .top-section {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">
  <h1>🐉 龙回头战法 · 个股买卖点明细</h1>
  <div class="subtitle">10万本金起步 · 5年实盘级逐笔滚动 · 含手续费 · 2021.08 - 2026.03</div>
  
  <!-- 概览 -->
  <div class="overview">
    <div class="card">
      <div class="label">初始资金</div>
      <div class="value">¥100,000</div>
    </div>
    <div class="card">
      <div class="label">期末资金</div>
      <div class="value up">¥{final_equity:,.0f}</div>
    </div>
    <div class="card">
      <div class="label">总收益率</div>
      <div class="value up">+{total_return:.1f}%</div>
    </div>
    <div class="card">
      <div class="label">总交易笔数</div>
      <div class="value">{len(T)} 笔</div>
    </div>
    <div class="card">
      <div class="label">最大回撤</div>
      <div class="value down">{max_dd:.1f}%</div>
    </div>
  </div>
  
  <!-- 资金曲线 -->
  <div class="chart-box">
    <h2>📈 资金曲线</h2>
    <canvas id="equityChart"></canvas>
  </div>
  
  <!-- 年度汇总 -->
  <div class="chart-box">
    <h2>📊 年度汇总</h2>
    <table>
      <thead>
        <tr>
          <th>年份</th>
          <th>笔数</th>
          <th>总买入额</th>
          <th>总卖出额</th>
          <th>总盈亏</th>
          <th>胜率</th>
          <th>平均收益</th>
          <th>最大单笔盈利</th>
          <th>最大单笔亏损</th>
        </tr>
      </thead>
      <tbody>
'''

for _, r in yearly.iterrows():
    y = int(r['year'])
    pnl_cls = 'up' if r['total_pnl'] >= 0 else 'down'
    pnl_sign = '+' if r['total_pnl'] >= 0 else ''
    avg_cls = 'up' if r['avg_ret'] >= 0 else 'down'
    avg_sign = '+' if r['avg_ret'] >= 0 else ''
    html += f'''        <tr>
          <td><b>{y}</b></td>
          <td>{int(r['n'])}</td>
          <td>¥{r['total_buy']:,.0f}</td>
          <td>¥{r['total_sell']:,.0f}</td>
          <td class="{pnl_cls}">{pnl_sign}¥{r['total_pnl']:,.0f}</td>
          <td>{r['win_rate']:.1f}%</td>
          <td class="{avg_cls}">{avg_sign}{r['avg_ret']:.2f}%</td>
          <td class="up">+{r['max_win']:.2f}%</td>
          <td class="down">{r['max_loss']:.2f}%</td>
        </tr>
'''

html += '''      </tbody>
    </table>
  </div>
  
  <!-- TOP 盈亏 -->
  <div class="top-section">
    <div class="chart-box">
      <h2>🏆 TOP 10 盈利交易</h2>
      <table>
        <thead>
          <tr><th>#</th><th>代码</th><th>买入日</th><th>卖出日</th><th>盈亏</th><th>收益率</th></tr>
        </thead>
        <tbody>
'''

for i, r in enumerate(top_win, 1):
    html += f'''          <tr>
            <td>{i}</td>
            <td>{r['code']}</td>
            <td>{r['buy_date']}</td>
            <td>{r['sell_date']}</td>
            <td class="up">+¥{r['pnl']:,.0f}</td>
            <td class="up">+{r['return_pct']:.2f}%</td>
          </tr>
'''

html += '''        </tbody>
      </table>
    </div>
    <div class="chart-box">
      <h2>💔 TOP 10 亏损交易</h2>
      <table>
        <thead>
          <tr><th>#</th><th>代码</th><th>买入日</th><th>卖出日</th><th>盈亏</th><th>收益率</th></tr>
        </thead>
        <tbody>
'''

for i, r in enumerate(top_loss, 1):
    html += f'''          <tr>
            <td>{i}</td>
            <td>{r['code']}</td>
            <td>{r['buy_date']}</td>
            <td>{r['sell_date']}</td>
            <td class="down">¥{r['pnl']:,.0f}</td>
            <td class="down">{r['return_pct']:.2f}%</td>
          </tr>
'''

html += '''        </tbody>
      </table>
    </div>
  </div>
  
  <!-- 分年明细 -->
  <h2 style="font-size:18px; margin-bottom:16px; display:flex; align-items:center; gap:8px;">
    <span style="width:4px; height:20px; background:linear-gradient(180deg,#ff6b35,#f7c948); border-radius:2px; display:inline-block;"></span>
    📋 逐笔明细（按年分组，点击展开）
  </h2>
'''

for y in sorted(years_data.keys(), reverse=True):
    rows = years_data[y]
    yr = yearly[yearly['year'] == y].iloc[0]
    pnl_sign = '+' if yr['total_pnl'] >= 0 else ''
    pnl_cls = 'win' if yr['total_pnl'] >= 0 else 'loss'
    html += f'''
  <div class="year-section">
    <div class="year-header" onclick="toggleYear({y})">
      <div class="year-title">
        <span class="arrow">▶</span>
        {y}年
        <span class="year-badge {pnl_cls}">{pnl_sign}{yr['total_pnl']:,.0f}元</span>
      </div>
      <div class="year-stats">
        <span><b>{int(yr['n'])}</b> 笔</span>
        <span>胜率 <b>{yr['win_rate']:.1f}%</b></span>
        <span>均收益 <b class="{ 'up' if yr['avg_ret']>=0 else 'down' }">{ '+' if yr['avg_ret']>=0 else '' }{yr['avg_ret']:.2f}%</b></span>
      </div>
    </div>
    <div class="year-content" id="year-{y}">
      <table>
        <thead>
          <tr>
            <th>序号</th>
            <th>代码</th>
            <th>买入日</th>
            <th>买入价</th>
            <th>股数</th>
            <th>买入额</th>
            <th>卖出日</th>
            <th>卖出价</th>
            <th>卖出额</th>
            <th>手续费</th>
            <th>盈亏</th>
            <th>收益率</th>
            <th>持仓</th>
          </tr>
        </thead>
        <tbody>
'''
    for r in rows:
        pnl_cls = 'up' if r['pnl'] >= 0 else 'down'
        pnl_sign = '+' if r['pnl'] >= 0 else ''
        ret_sign = '+' if r['ret'] >= 0 else ''
        total_fee = r['buy_fee'] + r['sell_fee']
        html += f'''          <tr>
            <td>{r['no']}</td>
            <td><b>{r['code']}</b></td>
            <td>{r['buy_date']}</td>
            <td>{r['buy_price']:.2f}</td>
            <td>{r['shares']:,}</td>
            <td>¥{r['buy_amount']:,.0f}</td>
            <td>{r['sell_date']}</td>
            <td>{r['sell_price']:.2f}</td>
            <td>¥{r['sell_amount']:,.0f}</td>
            <td>{total_fee:,.0f}</td>
            <td class="{pnl_cls}">{pnl_sign}¥{r['pnl']:,.0f}</td>
            <td class="{pnl_cls}">{ret_sign}{r['ret']:.2f}%</td>
            <td>{r['hold']}日</td>
          </tr>
'''
    html += '''        </tbody>
      </table>
    </div>
  </div>
'''

# 说明部分
html += f'''
  <div class="note">
    <b>📝 说明：</b><br>
    1. <b>初始资金</b>：10万元，T+1开盘买入，持有3日收盘卖出<br>
    2. <b>手续费</b>：买入万3，卖出千1.3（印花税+佣金），双边合计约千1.6<br>
    3. <b>仓位规则</b>：同日多只信号时按可用资金等分，单只买入不足100股跳过<br>
    4. <b>为什么只有65笔？</b>：主规则共105个信号，其中部分信号因同期资金被占用（持有其他票）未能参与；另有整手取整、最小买入额限制导致部分信号被过滤<br>
    5. <b>为什么收益128.8%≠之前的675%？</b>：之前的模拟是收益率累乘的理想化假设（假设全仓复利、每笔都能100%参与），本次是真金白银逐笔滚动的实盘级模拟，受限于：信号频率低（年均13笔）、大量空仓期、同日多票分流仓位<br>
    6. <b>策略特征</b>：典型"低频率高胜率"模式，空仓期是收益被稀释的主因；若叠加指数环境过滤提高仓位集中度，卡玛比率可进一步优化
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
  // 年份展开/折叠
  function toggleYear(y) {{
    const content = document.getElementById('year-' + y);
    const header = content.previousElementSibling;
    content.classList.toggle('show');
    header.classList.toggle('active');
  }}
  // 默认展开最近一年
  toggleYear({max(years_data.keys())});
  
  // 资金曲线
  const eqData = {json.dumps(equity_data, ensure_ascii=False)};
  const labels = eqData.map(d => d.d);
  const values = eqData.map(d => d.v);
  
  // 计算回撤
  let peak = 0;
  const ddValues = values.map(v => {{
    peak = Math.max(peak, v);
    return (v / peak - 1) * 100;
  }});
  
  const ctx = document.getElementById('equityChart').getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 320);
  grad.addColorStop(0, 'rgba(255,107,53,0.3)');
  grad.addColorStop(1, 'rgba(255,107,53,0)');
  
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [
        {{
          label: '资金曲线',
          data: values,
          borderColor: '#ff6b35',
          backgroundColor: grad,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
          yAxisID: 'y',
        }},
        {{
          label: '回撤(%)',
          data: ddValues,
          borderColor: '#22c55e',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.3,
          yAxisID: 'y1',
          fill: false,
        }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ intersect: false, mode: 'index' }},
      plugins: {{
        legend: {{ labels: {{ color: '#8a94a6' }} }},
        tooltip: {{
          backgroundColor: '#1a1f2e',
          borderColor: '#2a3142',
          borderWidth: 1,
          titleColor: '#e6e6e6',
          bodyColor: '#8a94a6',
          callbacks: {{
            label: function(ctx) {{
              if (ctx.datasetIndex === 0) return ' 资金: ¥' + ctx.parsed.y.toLocaleString();
              return ' 回撤: ' + ctx.parsed.y.toFixed(2) + '%';
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{ color: '#8a94a6', maxTicksLimit: 10 }},
          grid: {{ color: '#1e2533' }}
        }},
        y: {{
          position: 'left',
          ticks: {{ 
            color: '#ff6b35',
            callback: v => '¥' + (v/10000).toFixed(0) + '万'
          }},
          grid: {{ color: '#1e2533' }}
        }},
        y1: {{
          position: 'right',
          ticks: {{ 
            color: '#22c55e',
            callback: v => v.toFixed(0) + '%'
          }},
          grid: {{ display: false }}
        }}
      }}
    }}
  }});
</script>
</body>
</html>
'''

with open('trades_detail_10w.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f'报告已生成: trades_detail_10w.html')
print(f'交易笔数: {len(T)}')
print(f'期末资金: ¥{final_equity:,.2f}')
print(f'总收益率: +{total_return:.1f}%')
print(f'最大回撤: {max_dd:.1f}%')
