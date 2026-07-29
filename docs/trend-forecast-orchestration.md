# A 股趋势 Agent 编排集成说明

当前仓库已加入 `src/agent_system` 趋势 Agent，用于把 A 股行情、均线基线、三个研究模型、五分类裁决和次日反馈组织成可追踪研究闭环。

当前落地范围：

- 定义 A 股请求、均线预测、统一模型信号和五分类裁决结果结构。
- 增加 ClickHouse Tick 聚合和分钟 K 线 Provider，配置来自 `CLICKHOUSE_*` 环境变量。
- 增加分钟行情质量门禁。
- 增加快照冻结与 SHA-256 标识。
- 已复制三个研究模型脚本到 `researcher_strategy/`。
- 保留三个研究模型适配器：`wavelet`、`analog`、`logistic_6f`，并增加均线基线。
- 增加均线主导的可配置权重、五分类确定性裁决和次日反馈。
- 增加 JSON 运行记录仓库。
- 增加 API：`POST /api/v1/trend-forecast/runs`、`POST /api/v1/trend-forecast/daily-cycle`。
- 增加运行记录查询：`GET /api/v1/trend-forecast/runs/{run_id}`。

当前不会改变现有 A 股分析主流程。研究模型只能通过适配器读取同一行情快照；新闻、财经博主、散户情绪和交易执行不进入当前趋势研究链路。兼容字段 `action` 固定输出 `abstain`。`bot/` 只保留核心模块导入所需的 `BotMessage` 兼容类型，不提供通知推送和 Bot 平台功能。

## 五分类和权重

默认权重为：均线基线 `0.70`，小波 `0.10`，历史相似日 `0.10`，六因子 `0.10`。四项权重必须合计为 `1.0`，且均线权重必须大于任一研究模型。Web 设置页的 Agent 分类可以修改这四项配置。

模型方向先统一为 `+1/0/-1`，综合分为：

```text
综合分 = Σ(实际生效权重 × 方向值 × 校准置信度)
```

| 综合分 | 状态 |
| --- | --- |
| `>= 0.60` | 强多头控制 |
| `0.20～<0.60` | 弱多头控制 |
| `-0.20～<0.20` | 多空拉锯 |
| `-0.60～<=-0.20` | 弱空头控制 |
| `<= -0.60` | 强空头控制 |

研究模型失败时，剩余有效模型按配置权重重新归一化，并在结果中保存配置权重与实际权重。

当 `MA_AGENT_ENABLED=true` 时，现有 `main.py --schedule` / runtime scheduler 会在每日分析任务完成后调用 A 股趋势 Agent。目标日期通过 `src/core/trading_calendar.py` 计算，不使用简单的自然日加一天；如果交易日历依赖不可用，只跳过周末并在日志中保留降级事实。当前自动化默认关闭。

可选依赖：

```bash
pip install -r requirements-trend-forecast.txt
```

真实接入还需要：

1. 配置只读 ClickHouse 账户和分钟行情表。
2. 安装 `requirements-trend-forecast.txt` 以启用六因子逻辑回归和真实 ClickHouse/LangGraph 依赖。
3. 如需真实 LangGraph runtime，在 `trend_forecast_graph.py` 中替换当前本地执行器。
4. 如需完全复用 `predict_tomorrow.py` 原文件，需要先把其导入即执行逻辑拆成函数，并补齐 `factor_lr_backtest.py` 与 `bull_bear_backtest.py`。
