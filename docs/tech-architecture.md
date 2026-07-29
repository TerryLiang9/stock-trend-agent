# xuanling 技术架构 & Agent 系统详解

---

## 一、技术栈总览

```
┌────────────────────────────────────────────────────────┐
│                     前端层                              │
│  React 18 · TypeScript · Vite 7 · Tailwind CSS         │
│  React Router · Recharts · next-themes · i18n          │
│  Electron (桌面端)                                       │
├────────────────────────────────────────────────────────┤
│                     API 层                              │
│  FastAPI · Uvicorn · Pydantic · SSE 流式               │
├────────────────────────────────────────────────────────┤
│                   AI 编排层                              │
│  AgentOrchestrator (多Agent流水线)                       │
│  ReAct Loop Runner (思考→行动→观察→思考)               │
│  SkillManager (YAML 策略热加载)                          │
│  StrategyEngine (多策略共识合成)                          │
├────────────────────────────────────────────────────────┤
│                   LLM 适配层                             │
│  LiteLLM Router (统一多模型接入)                          │
│  支持: DeepSeek · OpenAI · Claude · Gemini · MiniMax     │
│  功能: 多 Key 负载均衡 · 自动 fallback · Token 计费       │
├────────────────────────────────────────────────────────┤
│                   工具层 (18个工具)                       │
│  行情 · K线 · 技术指标 · 筹码 · 新闻 · 资金流 · ...       │
├────────────────────────────────────────────────────────┤
│                   数据源层 (7个+熔断)                     │
│  TickFlow (WS) · Efinance · AkShare · Tencent ·  ...    │
├────────────────────────────────────────────────────────┤
│                   存储层                                 │
│  SQLite + SQLAlchemy · ClickHouse (可选)                 │
│  本地日志 + JSONL 持久化                                  │
└────────────────────────────────────────────────────────┘
```

**语言/运行时**: Python 3.10+ / TypeScript 5 / Node.js 20+
**模型接入**: LiteLLM（统一兼容 20+ LLM 厂商）
**部署**: 本地 Windows/Linux · Docker容器 · GitHub Actions 定时任务

---

## 二、Agent 系统核心原理

整个 Agent 系统的设计回答了一个问题：

> 怎么让 LLM 不只是"聊天"，而是像一个真正的分析师一样，**自主调用工具获取数据 → 分维度分析 → 综合决策**？

### 2.1 核心：ReAct 循环（思考-行动-观察）

这是 Agent 最底层的运行机制。不是一次性问答，而是一个**循环**：

```
第 1 步: LLM 思考 → 决定调用 get_realtime_quote("688981")
          ↓
        工具执行 → 返回 {price: 54.53, change: -1.25%, ...}
          ↓
第 2 步: LLM 看到结果 → 继续思考 → 决定调用 analyze_trend("688981")
          ↓
        工具执行 → 返回 {ma_alignment: "bearish", bias: -3.2%, ...}
          ↓
第 3 步: LLM 看到结果 → 继续思考 → 决定调用 search_stock_news("688981")
          ↓
        工具执行 → 返回 6 条新闻
          ↓
第 N 步: LLM 综合所有结果 → 输出最终 JSON 报告
```

**关键参数**：每个 Agent 有最大步数限制（`max_steps`），防止无限循环、控制成本。每步 = 1 次 LLM API 调用 ≈ 2-10 秒。

### 2.2 单 Agent 模式 vs 多 Agent 编排

项目支持两种架构，通过 `AGENT_ARCH` 切换：

#### 单 Agent（`AGENT_ARCH=single`）

一个 Agent 完成所有工作：

```
用户问题 → [ Agent: 取数据→技术分析→情报搜索→写报告 ] → 回答
          └── 一个 ReAct 循环，最多 10 步
```

**优点**：简单、快（~4-8 次 LLM 调用）。适合简单问答。
**缺点**：单一 Agent 既要懂技术面又要懂基本面，prompt 太长容易分心。

#### 多 Agent 编排（`AGENT_ARCH=multi`）

多个专精 Agent 串行协作，通过 `AgentOrchestrator` 协调：

```
用户问题
    │
    ▼
┌──────────────────┐
│ TechnicalAgent   │  "我只看技术面"
│ 工具: 行情/K线/   │  输出: 趋势判断 + 关键价位 + 形态识别
│ 均线/量价/筹码    │
└────────┬─────────┘
         │ context 传递
         ▼
┌──────────────────┐
│ IntelAgent       │  "我只看情报面"
│ 工具: 新闻搜索/   │  输出: 利好/利空因素 + 情绪判断
│ 公告检索/资金流   │
└────────┬─────────┘
         │ context 传递
         ▼
┌──────────────────┐
│ RiskAgent        │  "我只找风险"
│ 工具: 新闻/行情   │  输出: 风险等级 + 是否否决买入
│ /基本面          │  (仅 full/specialist 模式)
└────────┬─────────┘
         │ context 传递
         ▼
┌──────────────────┐
│ SkillAgents      │  "每人一个策略"
│ (specialist模式)  │  每人用自己的交易策略独立评估
│ 最多 3 个         │  (仅 specialist 模式)
└────────┬─────────┘
         │ context + 多策略意见
         ▼
┌──────────────────┐
│ DecisionAgent    │  "我综合所有人意见"
│ 工具: 无          │  输出: 最终决策仪表盘 JSON
│ (纯合成)          │
└──────────────────┘
```

**每个 Agent 都是一个独立的 ReAct 循环**。Agent 之间通过共享的 `AgentContext` 传递数据——上游 Agent 的分析结果写入 context，下游 Agent 读取。

### 2.3 Pipeline 模式选择

| 模式 | Agent 链路 | 适用场景 |
|------|-----------|----------|
| `quick` | Technical → Decision | 快速看盘，最低延迟 |
| `standard` | Technical → Intel → Decision | 日常分析，兼顾深度和速度 |
| `full` | Technical → Intel → Risk → Decision | 深度研报，带风险否决 |
| `specialist` | + 策略专项 Agent（最多 3 个） | 多策略对比，最全面 |

### 2.4 优雅降级机制

Pipeline 跑一半失败了怎么办？不直接崩溃，而是**优雅降级**：

- **Intel/Risk 失败** → 非关键阶段，跳过，继续执行
- **SkillAgent 失败** → 跳过，用其他 Agent 的结果
- **整个 Pipeline 超时** → 用已完成阶段的数据拼一个降级报告，标注"部分数据缺失"
- **预算不足** → 剩余时间 < 15 秒时，跳过后续阶段，生成当前已有结果

### 2.5 工具系统

Agent 的"手"——让它能获取真实数据而不是编造。

```python
# 每个工具的定义
@tool(name="get_realtime_quote", category="data",
      description="获取股票实时行情，包括价格、涨跌幅、量比、换手率、PE、PB、市值")
def get_realtime_quote(stock_code: str) -> dict:
    ...
```

**18 个注册工具**，分 5 类：

| 类别 | 工具 | 谁用 |
|------|------|------|
| **行情数据** | `get_realtime_quote` 实时行情、`get_daily_history` 日线K线、`get_chip_distribution` 筹码分布、`get_analysis_context` 分析上下文、`get_stock_info` 基本信息 | Technical |
| **技术分析** | `analyze_trend` 趋势分析、`calculate_ma` 均线计算、`get_volume_analysis` 量能分析、`analyze_pattern` K线形态 | Technical |
| **情报搜索** | `search_stock_news` 新闻搜索、`search_comprehensive_intel` 综合情报 | Intel |
| **资金市场** | `get_capital_flow` 主力资金流向、`get_market_indices` 大盘指数、`get_sector_rankings` 板块排名 | Intel |
| **趋势预测** | `wavelet_trend` 小波、`historical_analog` 相似日、`logistic_6factor` 六因子 | 趋势预测页 |

### 2.6 策略系统：Skills

策略是整个系统最大的亮点——**交易策略不用写代码，写 YAML 文件就行**：

```yaml
# strategies/shrink_pullback.yaml
name: shrink_pullback
display_name: 缩量回踩
description: 检测缩量回踩均线支撑信号
category: trend
required_tools:
  - get_daily_history
  - analyze_trend
  - get_realtime_quote
instructions: |
  **缩量回踩策略**
  1. 前提：MA5 > MA10 > MA20
  2. 价格回踩至 MA5（误差 1%）或 MA10（误差 2%）
  3. 成交量 < 5日均量的 70%（缩量）
  4. 反弹信号确认后入场
```

内置了 **20 个策略**（多头趋势、缩量回踩、均线金叉、龙头战法、缠论、波浪理论...），用户也可以自己加 YAML 文件到 `strategies/` 目录，系统自动加载。

**在 specialist 模式下**，SkillRouter 会根据标的特征和市场状态自动选择最多 3 个最匹配的策略，每个策略由一个 SkillAgent 独立评估。最后 StrategyEngine 做多策略共识合成——哪几个策略看多、哪几个看空、冲突在哪、加权置信度多少。

### 2.7 多源数据 + 熔断保护

Agent 调用 `get_realtime_quote` 时，底层不是只调一个 API：

```
优先级链（从高到低）:
TickFlow (WS) → Tencent → AkShare_新浪 → Efinance → AkShare_东财 → Tushare
     │               │
     ▼               ▼
  失败? → 自动下一个   成功? → 返回数据

同时:
- 连续失败 3 次 → 熔断 5 分钟 → 标记为不可用
- 熔断过期 → 自动恢复 → 重新尝试
```

用户/Agent 完全不感知底层切换。这就是为什么那个「获取实时行情」的工具成功率很高——7 个源兜底。

---

## 三、一次 Agent Chat 的完整数据流

以用户问 "688981 中芯国际现在能买吗？" 为例（`standard` 模式）：

```
1. POST /api/v1/agent/chat/stream
   └── message: "688981 中芯国际现在能买吗？"

2. AgentOrchestrator.execute_turn()
   ├── 解析股票代码 → 688981
   ├── 加载历史对话
   └── 执行 Pipeline ─────────────────────────────────────┐
                                                          │
   3. TechnicalAgent.run()  (ReAct 循环, max 6步)          │
      Step 1: LLM决策 → 调 get_realtime_quote("688981")    │
              → 数据层: Tencent 返回 价格/涨跌/量比/...      │
      Step 2: LLM决策 → 调 get_daily_history("688981")     │
              → 数据层: 取120天K线                          │
      Step 3: LLM决策 → 调 analyze_trend("688981")         │
              → 计算: MA排列/乖离率/RSI/MACD/...            │
      Step 4: LLM决策 → 调 get_chip_distribution("688981") │
              → 数据层: 筹码集中度/获利比例                   │
      Step 5: LLM综合 → 输出 {signal: "hold", trend: "偏空" │
               support: 49.5, resistance: 55.0, ...}       │
                                                          │
   4. IntelAgent.run()  (ReAct 循环, max 4步)              │
      Step 1: LLM决策 → 调 search_stock_news("688981")     │
              → Tavily API 返回最新新闻                      │
      Step 2: LLM决策 → 调 get_capital_flow("688981")      │
              → 主力资金流向数据                              │
      Step 3: LLM综合 → 输出 {sentiment: "中性偏空",         │
               risk_alerts: [...], positive_catalysts: [...]}│
                                                          │
   5. DecisionAgent.run()  (ReAct 循环, max 3步, 无工具)    │
      Step 1: 读取 Technical + Intel 的结果                 │
      Step 2: 综合研判                                      │
      Step 3: 输出最终 JSON 仪表盘 ←───────────────────────┘
        {
          core_conclusion: { signal: "hold", position: "观望" },
          intelligence: { risk_alerts: [...], catalysts: [...] },
          battle_plan: { ideal_buy: 49.5, stop_loss: 47.0, ... }
        }

6. 前端收到 SSE 事件流 → 逐步渲染结果
   - stage_start / stage_done → 显示当前Agent进度
   - tool_start / tool_done → 显示"正在获取行情..."
   - done → 显示最终结果
```

**总计**：约 8-12 次 LLM 调用，~30-60 秒完成。

---

## 四、关键设计决策

### 为什么用多 Agent 而不是一个大 Agent？

一个大 Agent + 超长 prompt = **prompt 膨胀、注意力分散、容易跳过关键步骤**。拆成专精 Agent 后，每个 Agent 的 system prompt 短且聚焦，指令遵循率高得多。

### 为什么 Agent 是串行不是并行？

因为下游 Agent 依赖上游的结果——Intel 需要知道 Technical 判断的趋势方向才能针对性搜新闻（利好/利空过滤），Decision 需要所有上游结果才能综合。串行是业务逻辑决定的，不是技术限制。

### 为什么用 LiteLLM 而不是直接调 API？

LiteLLM 提供了三项关键能力：
1. **统一接口** — 换模型只改一行配置，不改代码
2. **多 Key 负载均衡** — 多个 API Key 轮询，绕过速率限制
3. **成本追踪** — 每次调用的 token 消耗和费用自动记录

### 为什么用 YAML 定义策略而不是代码？

方便非技术人员（研究员、交易员）自定义分析逻辑。加一个新策略 = 写一个 YAML 文件 + 系统自动加载，零代码部署。
