# A 股均线趋势 Agent 闭环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一个只服务 A 股的交易 Agent：以均线模型为首个可解释基线，复用并校准现有研究模型，融合截止时点前的新闻情报形成多模型裁决，盘后自动核验、反思和优化参数，并在严格风控、幂等和熔断保护下自动执行交易。

**Architecture:** 使用“市场/Tick 数据与新闻快照 → 多模型独立预测 → 规则化裁决 → 风控与策略选择 → 订单计划 → Broker 执行 → 成交回报 → 次日后验 → 反思与走步回测 → 受控晋级”的闭环。数值模型与新闻模型分别保存原始输出，裁决器只消费标准信号信封；LLM 可以解释新闻和提出反思，但不能绕过确定性风控、直接改代码、直接改 champion 参数或直接构造实盘订单。

**Tech Stack:** Python 3、FastAPI、Pydantic、ClickHouse (`clickhouse-connect`)、pandas、现有 `DatabaseManager`/SQLAlchemy、LangGraph、项目现有新闻检索/情报服务、可注入 Broker Gateway、pytest、项目现有调度器。

## Global Constraints

- 仅支持 A 股证券代码，统一规范为 `600519.SH`、`000001.SZ`、`920xxx.BJ`；拒绝港股、美股等市场输入。
- 均线模型作为首个基线；保留 `wavelet`、`analog`、`logistic_6f`，但必须统一为相同目标周期或在裁决时显式做目标映射，禁止直接平均语义不同的概率。
- 新闻只允许使用 `published_at <= data_cutoff` 的内容；发布日期未知、正文缺失或来源可信度不足的数据只能作为警告，不能形成强交易信号。
- 当前不需要任何 `bot/` 功能：不得新增或修改机器人命令、消息推送、群聊交互、Bot 定时通知或 Bot 交易确认入口；运行结果只通过 API、CLI、数据库和日志提供。
- 默认每天收盘后运行一次，建议 `15:10 Asia/Shanghai`；仅在 A 股交易日执行。
- 预测目标固定为“下一交易日收盘价相对本次 `data_cutoff` 参考价的方向”，方向枚举为 `bullish/neutral/bearish`。
- 默认中性区间为 `±0.5%`，必须记录在参数版本中，不允许散落为源码魔法数字。
- ClickHouse 数据选择顺序为 Tick 优先、分钟 K 线显式降级；若 Tick 表存在但字段映射或成交量语义未配置完整，则阻断而不是猜测。
- Tick 聚合必须在 ClickHouse 侧完成；同一运行只查询一次并冻结快照，模型、后验评估和回放不得各自重新取数。
- 严禁未来数据：所有查询必须满足 `event_time <= data_cutoff`，特征和调参必须采用时间序列切分，禁止随机打乱训练/验证集。
- “自动反思改正”不等于运行时自行修改源码或直接覆盖生产参数；只能生成有版本的 challenger，经最小样本量和 walk-forward 门禁后晋级。
- 自动交易默认 `paper` 模式；`live` 必须同时满足显式环境开关、Broker 适配器可用、账户白名单、单笔/单日限额、仓位上限、交易时段、价格偏离、重复单和熔断检查。
- 任何数据质量失败、模型结果不完整、裁决置信度不足、新闻重大风险、Broker 状态不明或成交回报超时，默认不下单。
- 新配置同步更新 `.env.example` 和专题文档；用户可见 API/调度行为同步更新 `docs/CHANGELOG.md`。
- 未经用户明确确认，不执行 `git commit`、`git tag` 或 `git push`；下列提交命令仅作为实施阶段的建议检查点。

---

## 目标判断标准

首版固定使用下列可解释规则，先建立稳定基线，再谈复杂模型：

| 判断项 | 多头条件 | 空头条件 | 其他 |
| --- | --- | --- | --- |
| 均线排列 | `MA_short > MA_mid > MA_long` | `MA_short < MA_mid < MA_long` | 视为混合 |
| 均线斜率 | 短、中期斜率均为正 | 短、中期斜率均为负 | 降低置信度 |
| 价格位置 | 参考价高于中期均线 | 参考价低于中期均线 | 视为中性证据 |
| 交叉事件 | 短均线上穿中均线 | 短均线下穿中均线 | 无交叉不加分 |
| 量能确认 | 当日量 / 近 5 日均量达到参数阈值 | 只作为确认，不单独产生方向 | Tick 无可靠量语义时禁用 |

每项输出 `-1/0/+1` 证据分，按参数版本中的权重计算总分。总分达到正阈值输出 `bullish`，达到负阈值输出 `bearish`，否则输出 `neutral`。默认参数：`MA5/MA10/MA20`、斜率窗口 `3`、量比阈值 `1.2`、方向阈值 `2.0`、中性收益带 `0.5%`。

## 多模型裁决与交易标准

所有模型统一输出 `SignalEnvelope(model_name, target_date, direction, confidence, calibrated_probability, evidence, warnings, snapshot_id)`。原始研究模型目标不一致时，先用各自历史后验做概率校准和目标映射；缺少足够校准样本的模型只能作为观察项，权重为零。

裁决采用确定性策略，不让 LLM直接投票：

- 先执行数据质量和模型完整性检查；可用模型少于 2 个时输出 `abstain`。
- 对可用模型采用经过版本化回测得到的非负权重，权重总和为 1；新闻信号权重上限默认 `0.20`。
- 多空两侧加权分差小于 `0.20`、最高置信度低于 `0.60`、或两个高置信模型方向相反时输出 `abstain`。
- 新闻出现停牌、监管调查、重大诉讼、财务造假、退市风险等阻断标签时，只允许 `reduce/exit/abstain`，禁止新开多头仓位。
- 裁决结果映射为 `buy/hold/reduce/exit/abstain`；A 股 MVP 不产生裸做空订单。

自动交易的默认硬限制：单笔资金不超过可用资金 `2%`、单股总仓位不超过净资产 `10%`、当日新增风险敞口不超过净资产 `20%`、单日亏损达到 `2%` 后熔断、委托价偏离最新价超过 `1%` 拒绝、同一 `decision_id` 只能产生一个有效订单。以上值均为版本化配置，实盘开启前必须由用户按实际账户风险偏好确认。

## 文件结构

### 新建

- `src/agent_system/config/ma_agent.py`：读取并校验均线周期、数据源模式、评估和晋级门槛。
- `src/agent_system/schemas/ma_prediction.py`：趋势请求、特征、预测、后验、反思和参数版本契约。
- `src/agent_system/data/tick_aggregation.py`：Tick/分钟查询模板与标准日 K 聚合契约。
- `src/agent_system/models/moving_average.py`：纯函数均线特征和确定性趋势分类器。
- `src/agent_system/news/provider.py`：复用现有新闻检索服务并冻结截止时点前的新闻快照。
- `src/agent_system/news/analyzer.py`：将新闻转换为带来源引用的结构化情报信号。
- `src/agent_system/adjudication/policy.py`：多模型目标映射、权重、冲突处理和 abstain 规则。
- `src/agent_system/strategies/selector.py`：把裁决结果、市场状态和持仓状态映射为策略与订单意图。
- `src/agent_system/execution/broker.py`：Broker Gateway 协议及统一订单/成交回报结构。
- `src/agent_system/execution/paper_broker.py`：默认模拟成交实现。
- `src/agent_system/execution/risk_gate.py`：下单前确定性风控、熔断和急停。
- `src/agent_system/execution/order_service.py`：幂等订单计划、提交、查询和对账。
- `src/agent_system/evaluation/outcome.py`：到期预测的真实方向、收益和命中判定。
- `src/agent_system/optimization/reflection.py`：按错误类型生成结构化反思与 challenger 参数。
- `src/agent_system/optimization/walk_forward.py`：仅使用历史切片的走步验证与晋级门禁。
- `src/agent_system/repositories/ma_agent_repository.py`：预测、后验、反思和参数版本持久化。
- `src/services/ma_trend_agent_service.py`：预测、日终反馈、参数晋级和回滚的应用服务。
- `scripts/run_ma_trend_agent.py`：可被 Windows 任务计划或项目调度器调用的幂等 CLI。
- `tests/agent_system/`：均线 Agent 的数据、模型、闭环、API 和调度测试。

### 修改

- `src/agent_system/data/provider.py`：将接口改为加载标准行情快照，支持 `tick/minute` 来源元数据。
- `src/agent_system/data/clickhouse_provider.py`：增加 Tick 优先查询、分钟降级、连接复用和列映射。
- `src/agent_system/data/quality_gate.py`：增加交易时段、完整交易日、最新数据和 Tick 聚合质量检查。
- `src/agent_system/data/snapshot_store.py`：快照中记录查询模板版本、源粒度和参数版本。
- `src/agent_system/graph/state.py`：从三模型 State 收敛为单模型预测/反馈闭环 State。
- `src/agent_system/graph/trend_forecast_graph.py`：扩展为行情、新闻、多模型、裁决、策略选择、风控和执行编排。
- `src/agent_system/schemas/request.py`：收敛请求字段，增加 `mode=predict/evaluate/daily_cycle`。
- `api/v1/endpoints/trend_forecast.py`：保留现有 URL，提供预测、日终闭环、历史和参数查询接口。
- `src/storage.py`：注册预测、新闻快照、裁决、参数、订单、成交和审计持久化表及唯一索引。
- `src/services/runtime_scheduler.py`、`main.py`：把每日闭环任务挂入现有运行时调度器。
- `.env.example`、`requirements-trend-forecast.txt`、`README.md`、`docs/trend-forecast-orchestration.md`、`docs/CHANGELOG.md`：同步配置、依赖、入口和边界。

### 明确不修改

- `bot/`：本阶段不接入任何机器人平台。
- `src/notification.py` 及各通知发送器：不推送预测、裁决、订单或成交消息。
- Bot 相关配置、依赖、API 路由和测试：不纳入本计划改动面。

### 保留并改造

- `src/agent_system/adapters/wavelet_adapter.py`：保留原始输出，增加统一预测周期和校准元数据。
- `src/agent_system/adapters/analog_adapter.py`：保留相似日输出，限制只读统一快照。
- `src/agent_system/adapters/logistic_adapter.py`：保留六因子结果，消除脚本导入副作用并返回强类型结果。
- `researcher_strategy/`：作为研究算法来源保留；运行时只能通过适配器调用，禁止调用脚本 `main()`。

---

### Task 1: 固定业务契约和配置边界

**Files:**
- Create: `src/agent_system/config/ma_agent.py`
- Create: `src/agent_system/schemas/ma_prediction.py`
- Modify: `src/agent_system/schemas/request.py`
- Modify: `.env.example`
- Test: `tests/agent_system/test_ma_agent_config.py`
- Test: `tests/agent_system/test_ma_prediction_schema.py`

**Interfaces:**
- Produces: `MaAgentConfig.from_env() -> MaAgentConfig`
- Produces: `MaParameters`, `MaTrendPrediction`, `MaPredictionOutcome`, `MaReflection`, `MaParameterVersion`
- Produces: `TrendForecastRequest(symbol, as_of, target_date, data_cutoff, mode, market_data)`

- [ ] **Step 1: 写配置失败测试**

```python
def test_tick_mode_requires_tick_table_and_volume_semantics(monkeypatch):
    monkeypatch.setenv("MA_AGENT_DATA_MODE", "tick")
    monkeypatch.delenv("CLICKHOUSE_TICK_TABLE", raising=False)
    with pytest.raises(ValueError, match="CLICKHOUSE_TICK_TABLE"):
        MaAgentConfig.from_env()

def test_default_parameters_are_versionable():
    config = MaAgentConfig.from_mapping({"MA_AGENT_DATA_MODE": "minute"})
    assert config.parameters.short_window == 5
    assert config.parameters.mid_window == 10
    assert config.parameters.long_window == 20
    assert config.parameters.neutral_band_pct == 0.5
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/agent_system/test_ma_agent_config.py -v`

Expected: FAIL，原因是 `MaAgentConfig` 尚不存在。

- [ ] **Step 3: 实现强类型配置和 Schema**

```python
class MaParameters(BaseModel):
    short_window: int = Field(default=5, ge=2)
    mid_window: int = Field(default=10, ge=3)
    long_window: int = Field(default=20, ge=5)
    slope_window: int = Field(default=3, ge=2)
    volume_ratio_threshold: float = Field(default=1.2, gt=0)
    direction_score_threshold: float = Field(default=2.0, gt=0)
    neutral_band_pct: float = Field(default=0.5, ge=0)

    @model_validator(mode="after")
    def validate_windows(self):
        if not self.short_window < self.mid_window < self.long_window:
            raise ValueError("均线周期必须满足 short < mid < long")
        return self
```

`.env.example` 增加：

```dotenv
MA_AGENT_ENABLED=false
MA_AGENT_SCHEDULE_TIME=15:10
MA_AGENT_DATA_MODE=tick
MA_AGENT_SYMBOLS=600519.SH
MA_AGENT_MIN_EVALUATED_SAMPLES=60
MA_AGENT_MIN_IMPROVEMENT_PCT=3.0
MA_AGENT_AUTO_PROMOTE=false
MA_AGENT_NEWS_ENABLED=true
MA_AGENT_NEWS_WEIGHT_CAP=0.20
MA_AGENT_TRADING_MODE=paper
MA_AGENT_LIVE_CONFIRMED=false
MA_AGENT_KILL_SWITCH=false
MA_AGENT_MAX_ORDER_EQUITY_PCT=2.0
MA_AGENT_MAX_SYMBOL_POSITION_PCT=10.0
MA_AGENT_MAX_DAILY_NEW_EXPOSURE_PCT=20.0
MA_AGENT_DAILY_LOSS_LIMIT_PCT=2.0
MA_AGENT_MAX_PRICE_DEVIATION_PCT=1.0
CLICKHOUSE_TICK_TABLE=
CLICKHOUSE_MINUTE_TABLE=
CLICKHOUSE_TICK_VOLUME_SEMANTICS=incremental
```

- [ ] **Step 4: 验证 Schema 拒绝非 A 股和非法时间窗口**

Run: `python -m pytest tests/agent_system/test_ma_agent_config.py tests/agent_system/test_ma_prediction_schema.py -v`

Expected: PASS；`AAPL`、`hk00700`、无时区 `data_cutoff` 均被拒绝。

- [ ] **Step 5: 建议提交检查点（需用户确认）**

```bash
git add .env.example src/agent_system/config/ma_agent.py src/agent_system/schemas/request.py src/agent_system/schemas/ma_prediction.py tests/agent_system
git commit -m "feat: define moving average agent contracts"
```

### Task 2: 实现 ClickHouse Tick 优先的数据层

**Files:**
- Create: `src/agent_system/data/tick_aggregation.py`
- Modify: `src/agent_system/data/provider.py`
- Modify: `src/agent_system/data/clickhouse_provider.py`
- Modify: `src/agent_system/data/quality_gate.py`
- Modify: `src/agent_system/data/snapshot_store.py`
- Test: `tests/agent_system/test_clickhouse_market_data_provider.py`
- Test: `tests/agent_system/test_market_data_quality_gate.py`

**Interfaces:**
- Consumes: `MaAgentConfig`
- Produces: `MarketDataProvider.load_market_data(symbol, history_start, data_cutoff) -> MarketDataResult`
- Produces metadata: `source_granularity`, `fallback_reason`, `query_id`, `query_template_version`, `row_count`, `minimum_event_time`, `maximum_event_time`
- Produces: `validate_market_rows(rows, data_cutoff, expected_symbol) -> DataQualityResult`

- [ ] **Step 1: 写 Tick 查询契约测试**

```python
def test_tick_query_is_bounded_and_parameterized(fake_client, config):
    provider = ClickHouseMarketDataProvider(config=config, client=fake_client)
    provider.load_market_data(
        symbol="600519.SH",
        history_start=aware("2025-01-01 00:00:00"),
        data_cutoff=aware("2026-07-21 15:00:00"),
    )
    sql, params = fake_client.last_query
    assert "event_time <= {data_cutoff:" in sql
    assert params["symbol"] == "600519.SH"
    assert "600519.SH" not in sql
```

- [ ] **Step 2: 写 Tick 聚合语义测试**

```python
def test_incremental_tick_volume_uses_sum():
    sql = build_tick_to_minute_sql(mapping(), volume_semantics="incremental")
    assert "sum(volume_delta)" in sql

def test_cumulative_tick_volume_uses_ordered_difference():
    sql = build_tick_to_minute_sql(mapping(), volume_semantics="cumulative")
    assert "max(volume_total) - min(volume_total)" in sql
```

- [ ] **Step 3: 实现数据选择规则**

```python
def load_market_data(self, *, symbol, history_start, data_cutoff):
    if self.config.data_mode == "tick":
        return self._load_tick_aggregated_minutes(symbol, history_start, data_cutoff)
    if self.config.data_mode == "minute":
        return self._load_minutes(symbol, history_start, data_cutoff)
    raise ValueError(f"不支持的数据模式: {self.config.data_mode}")
```

`tick` 模式查询失败时默认阻断；只有显式配置 `MA_AGENT_ALLOW_MINUTE_FALLBACK=true` 才允许降级，并必须写入 `fallback_reason`。

- [ ] **Step 4: 增加数据质量门禁**

质量测试必须覆盖：未来 Tick、重复事件、乱序、午休数据、集合竞价标记、非完整交易日、少于 `long_window + slope_window` 个有效日、当日最后数据早于 15:00、跨证券污染。

Run: `python -m pytest tests/agent_system/test_clickhouse_market_data_provider.py tests/agent_system/test_market_data_quality_gate.py -v`

Expected: PASS；任何未来数据和不明确成交量语义均阻断运行。

- [ ] **Step 5: 冻结可回放快照**

```python
snapshot = snapshot_store.freeze_rows(
    run_id=run_id,
    rows=result.rows,
    metadata={
        **result.metadata,
        "parameter_version": parameter_version,
        "data_cutoff": data_cutoff.isoformat(),
    },
)
```

快照哈希必须由标准化内容与元数据版本共同决定；相同输入产生相同 SHA-256。

### Task 3: 实现可解释的均线基线模型

**Files:**
- Create: `src/agent_system/models/__init__.py`
- Create: `src/agent_system/models/moving_average.py`
- Test: `tests/agent_system/test_moving_average_model.py`

**Interfaces:**
- Consumes: 标准日 K DataFrame、`MaParameters`
- Produces: `compute_ma_features(frame, params) -> MaFeatureSet`
- Produces: `predict_ma_trend(features, params) -> MaTrendPrediction`

- [ ] **Step 1: 写无前视特征测试**

```python
def test_future_rows_do_not_change_prediction(sample_daily):
    base = MovingAverageModel(DEFAULT_PARAMS).predict(sample_daily.iloc[:30])
    with_future = MovingAverageModel(DEFAULT_PARAMS).predict(sample_daily.iloc[:31], cutoff_index=29)
    assert with_future == base
```

- [ ] **Step 2: 写三种方向和理由测试**

```python
@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [("uptrend", "bullish"), ("sideways", "neutral"), ("downtrend", "bearish")],
)
def test_direction_is_deterministic(request, fixture_name, expected):
    result = MovingAverageModel(DEFAULT_PARAMS).predict(request.getfixturevalue(fixture_name))
    assert result.direction == expected
    assert result.evidence
    assert 0.0 <= result.confidence <= 1.0
```

- [ ] **Step 3: 实现特征和打分纯函数**

```python
score = (
    alignment_score
    + slope_score
    + price_position_score
    + crossover_score
    + volume_confirmation_score
)
direction = (
    "bullish" if score >= params.direction_score_threshold
    else "bearish" if score <= -params.direction_score_threshold
    else "neutral"
)
```

输出必须包含 `MA5/MA10/MA20`、斜率、价格乖离、交叉状态、量比、每项证据分、总分、参数版本、数据快照 ID 和参考价。

- [ ] **Step 4: 运行模型测试**

Run: `python -m pytest tests/agent_system/test_moving_average_model.py -v`

Expected: PASS；输入不足时返回明确失败，不使用填充数据伪造结果。

### Task 4: 建立预测、后验、反思和参数版本存储

**Files:**
- Modify: `src/storage.py`
- Create: `src/agent_system/repositories/ma_agent_repository.py`
- Test: `tests/agent_system/test_ma_agent_repository.py`

**Interfaces:**
- Produces: `save_prediction(prediction) -> int`
- Produces: `list_due_predictions(as_of_date) -> list[MaTrendPrediction]`
- Produces: `save_outcome(outcome) -> int`
- Produces: `save_reflection(reflection) -> int`
- Produces: `get_champion_parameters() -> MaParameterVersion`
- Produces: `create_challenger(parameters, parent_version, reason) -> MaParameterVersion`
- Produces: `promote(version_id, expected_champion_id) -> MaParameterVersion`

- [ ] **Step 1: 写幂等和版本并发测试**

```python
def test_prediction_is_idempotent(repo, prediction):
    first = repo.save_prediction(prediction)
    second = repo.save_prediction(prediction)
    assert first == second

def test_promote_uses_compare_and_swap(repo):
    champion = repo.get_champion_parameters()
    challenger = repo.create_challenger(DEFAULT_PARAMS, champion.id, "miss cluster")
    repo.promote(challenger.id, expected_champion_id=champion.id)
    with pytest.raises(ParameterVersionConflict):
        repo.promote(challenger.id, expected_champion_id=champion.id)
```

- [ ] **Step 2: 增加四张表**

表及唯一键：

```text
ma_predictions: UNIQUE(symbol, target_date, data_cutoff, parameter_version_id)
ma_prediction_outcomes: UNIQUE(prediction_id)
ma_reflections: UNIQUE(prediction_id, reflection_engine_version)
ma_parameter_versions: UNIQUE(version_name), status in champion/challenger/rejected/retired
```

预测记录必须保存原始特征 JSON、证据 JSON、快照哈希、参考价、目标日期、创建时间；后验记录保存目标收盘价、收益率、实际方向、是否命中和无法评估原因。

- [ ] **Step 3: 运行仓储测试**

Run: `python -m pytest tests/agent_system/test_ma_agent_repository.py -v`

Expected: PASS；重复调度不会产生重复预测或重复后验。

### Task 5: 实现每日预测与结果核验闭环

**Files:**
- Create: `src/agent_system/evaluation/outcome.py`
- Create: `src/services/ma_trend_agent_service.py`
- Modify: `src/agent_system/graph/state.py`
- Modify: `src/agent_system/graph/trend_forecast_graph.py`
- Test: `tests/agent_system/test_ma_trend_agent_service.py`
- Test: `tests/agent_system/test_ma_agent_workflow.py`

**Interfaces:**
- Produces: `MaTrendAgentService.predict(request) -> MaTrendPrediction`
- Produces: `MaTrendAgentService.evaluate_due(as_of) -> list[MaPredictionOutcome]`
- Produces: `MaTrendAgentService.run_daily_cycle(as_of) -> DailyCycleResult`

- [ ] **Step 1: 写下一交易日后验测试**

```python
def test_outcome_uses_target_close_and_saved_reference_price():
    outcome = evaluate_prediction(
        predicted="bullish",
        reference_price=100.0,
        target_close=101.0,
        neutral_band_pct=0.5,
    )
    assert outcome.actual_direction == "bullish"
    assert outcome.return_pct == 1.0
    assert outcome.is_correct is True
```

- [ ] **Step 2: 定义每日闭环顺序**

```python
def run_daily_cycle(self, as_of):
    evaluated = self.evaluate_due(as_of)
    reflections = self.reflect(evaluated)
    promotion = self.validate_and_promote(reflections, as_of)
    predictions = [self.predict_for_symbol(symbol, as_of) for symbol in self.config.symbols]
    return DailyCycleResult(evaluated=evaluated, reflections=reflections,
                            promotion=promotion, predictions=predictions)
```

顺序必须先评估昨日预测，再基于当前 champion 及已通过门禁的新版本生成明日预测；非交易日返回 `skipped`，不能错误推进 target date。

- [ ] **Step 3: 用现有交易日历计算目标日期**

禁止使用 `date + timedelta(days=1)`。调用项目现有 A 股交易日历，覆盖周五、法定节假日和补班日测试。

- [ ] **Step 4: 运行闭环测试**

Run: `python -m pytest tests/agent_system/test_ma_trend_agent_service.py tests/agent_system/test_ma_agent_workflow.py -v`

Expected: PASS；数据失败时不创建预测，昨日数据未到齐时 outcome 保持可重试状态。

### Task 6: 实现受控反思和参数迭代

**Files:**
- Create: `src/agent_system/optimization/reflection.py`
- Create: `src/agent_system/optimization/walk_forward.py`
- Test: `tests/agent_system/test_ma_reflection.py`
- Test: `tests/agent_system/test_ma_walk_forward.py`

**Interfaces:**
- Produces: `classify_miss(prediction, outcome) -> MissReason`
- Produces: `propose_challenger(champion, recent_history) -> MaParameterVersion | None`
- Produces: `walk_forward_validate(candidate, champion, bars) -> PromotionDecision`

- [ ] **Step 1: 写错误归因测试**

```python
def test_false_bullish_with_large_positive_bias_is_late_entry():
    reason = classify_miss(prediction=false_bullish(bias_pct=8.0), outcome=bearish_outcome())
    assert reason.code == "overextended_entry"

def test_single_miss_does_not_create_challenger():
    assert propose_challenger(champion, recent_history=[one_miss]) is None
```

- [ ] **Step 2: 固定允许调整的参数空间**

```python
SEARCH_SPACE = {
    "short_window": [3, 5, 7],
    "mid_window": [8, 10, 12],
    "long_window": [20, 30, 60],
    "slope_window": [3, 5],
    "volume_ratio_threshold": [1.0, 1.2, 1.5],
    "direction_score_threshold": [1.5, 2.0, 2.5],
    "neutral_band_pct": [0.3, 0.5, 1.0],
}
```

反思只能在该离散空间内生成候选；不得通过 LLM 返回任意代码、SQL 或未校验参数。

- [ ] **Step 3: 实现时间序列 walk-forward 门禁**

晋级必须同时满足：

```text
evaluated_samples >= MA_AGENT_MIN_EVALUATED_SAMPLES（默认 60）
candidate_hit_rate >= champion_hit_rate + 3 个百分点
candidate_max_consecutive_misses <= champion_max_consecutive_misses
至少覆盖 3 个滚动窗口，且每个窗口都不低于 champion 5 个百分点以上
```

所有比较使用相同证券集合、相同时间区间、相同交易成本假设和相同快照版本。首版 `MA_AGENT_AUTO_PROMOTE=false`，仅记录建议；开启后仍必须满足门禁。

- [ ] **Step 4: 运行反思与泄漏测试**

Run: `python -m pytest tests/agent_system/test_ma_reflection.py tests/agent_system/test_ma_walk_forward.py -v`

Expected: PASS；验证窗口之后的数据改变时，较早窗口的评分不变。

### Task 7: 接入新闻情报并冻结可追溯快照

**Files:**
- Create: `src/agent_system/news/__init__.py`
- Create: `src/agent_system/news/provider.py`
- Create: `src/agent_system/news/analyzer.py`
- Create: `src/agent_system/schemas/news_signal.py`
- Modify: `src/services/intelligence_service.py`
- Test: `tests/agent_system/test_news_signal_provider.py`
- Test: `tests/agent_system/test_news_signal_analyzer.py`

**Interfaces:**
- Produces: `NewsProvider.load(symbol, published_after, data_cutoff) -> list[NewsItem]`
- Produces: `NewsAnalyzer.analyze(items, data_cutoff) -> NewsSignalEnvelope`
- `NewsSignalEnvelope` 包含 `direction`、`confidence`、`risk_tags`、`citations`、`snapshot_id`、`prompt_version`。

- [ ] **Step 1: 写时间截止和来源去重测试**

```python
def test_news_after_cutoff_is_excluded(provider):
    items = provider.load("600519.SH", start, cutoff)
    assert all(item.published_at <= cutoff for item in items)

def test_same_canonical_url_is_deduplicated(provider):
    items = provider.normalize([same_story_a, same_story_b])
    assert len(items) == 1
```

- [ ] **Step 2: 复用现有情报服务并冻结新闻快照**

只从现有 `intelligence_service` 获取数据，标准化标题、正文摘要、发布时间、来源、URL 和可信度；提示词输入和输出均保存内容哈希，不保存 API Key。发布日期未知的新闻标记 `unusable_for_trading=true`。

- [ ] **Step 3: 实现强类型新闻分析**

```python
class NewsSignalEnvelope(BaseModel):
    direction: Literal["bullish", "neutral", "bearish", "abstain"]
    confidence: float = Field(ge=0, le=1)
    risk_tags: list[Literal[
        "suspension", "regulatory_investigation", "fraud",
        "delisting", "major_litigation", "earnings_warning"
    ]]
    citations: list[NewsCitation]
    snapshot_id: str
```

LLM 输出必须通过 Pydantic 校验；无引用、数字不可追溯、内容越过 `data_cutoff` 或出现交易指令时，结果降级为 `abstain`。

- [ ] **Step 4: 运行新闻测试**

Run: `python -m pytest tests/agent_system/test_news_signal_provider.py tests/agent_system/test_news_signal_analyzer.py -v`

Expected: PASS；网络失败只使新闻分支失败，不伪造中性新闻。

### Task 8: 实现多模型校准与确定性裁决

**Files:**
- Modify: `src/agent_system/schemas/model_result.py`
- Modify: `src/agent_system/adapters/wavelet_adapter.py`
- Modify: `src/agent_system/adapters/analog_adapter.py`
- Modify: `src/agent_system/adapters/logistic_adapter.py`
- Create: `src/agent_system/adjudication/__init__.py`
- Create: `src/agent_system/adjudication/calibration.py`
- Create: `src/agent_system/adjudication/policy.py`
- Create: `src/agent_system/schemas/adjudication.py`
- Test: `tests/agent_system/test_model_target_mapping.py`
- Test: `tests/agent_system/test_adjudication_policy.py`

**Interfaces:**
- Consumes: `moving_average`、`wavelet`、`analog`、`logistic_6f`、`news` 的独立信号信封。
- Produces: `calibrate_model(envelopes, historical_outcomes, version) -> list[CalibratedSignal]`
- Produces: `AdjudicationPolicy.decide(signals, holdings, risk_context) -> AdjudicationDecision`

- [ ] **Step 1: 写目标不一致和冲突测试**

```python
def test_uncalibrated_model_has_zero_weight(policy):
    result = policy.decide([uncalibrated_wavelet, calibrated_ma], empty_holdings, clean_risk)
    assert result.weights["wavelet"] == 0.0

def test_high_confidence_opposition_abstains(policy):
    result = policy.decide([bullish_ma(0.9), bearish_analog(0.9)], empty_holdings, clean_risk)
    assert result.action == "abstain"
    assert result.reason_code == "high_confidence_conflict"
```

- [ ] **Step 2: 统一信号信封但保留原始目标**

```python
class SignalEnvelope(BaseModel):
    model_name: Literal["moving_average", "wavelet", "analog", "logistic_6f", "news"]
    original_target_type: str
    mapped_target_type: Literal["next_session_close_vs_cutoff"]
    direction: Literal["bullish", "neutral", "bearish", "abstain"]
    confidence: float = Field(ge=0, le=1)
    calibrated_probability: float | None
    calibration_version: str | None
    evidence: dict[str, Any]
    warnings: list[str]
```

- [ ] **Step 3: 实现裁决与阻断规则**

裁决输出必须包含每个模型的原始方向、校准后概率、实际权重、冲突、新闻阻断标签、持仓状态、最终动作和 `abstain` 原因。模型权重只能来自已保存的回测版本，不能让 LLM 在单次运行中临时决定。

- [ ] **Step 4: 运行裁决测试**

Run: `python -m pytest tests/agent_system/test_model_target_mapping.py tests/agent_system/test_adjudication_policy.py -v`

Expected: PASS；少于两个可校准模型、重大新闻风险或强冲突时均不产生买入动作。

### Task 9: 实现策略选择、自动交易风控与订单执行

**Files:**
- Create: `src/agent_system/strategies/__init__.py`
- Create: `src/agent_system/strategies/selector.py`
- Create: `src/agent_system/execution/__init__.py`
- Create: `src/agent_system/execution/broker.py`
- Create: `src/agent_system/execution/paper_broker.py`
- Create: `src/agent_system/execution/risk_gate.py`
- Create: `src/agent_system/execution/order_service.py`
- Create: `src/agent_system/schemas/execution.py`
- Modify: `src/storage.py`
- Test: `tests/agent_system/test_strategy_selector.py`
- Test: `tests/agent_system/test_pretrade_risk_gate.py`
- Test: `tests/agent_system/test_order_service.py`

**Interfaces:**
- Produces: `StrategySelector.select(decision, market_state, holdings) -> StrategyIntent`
- Produces: `PreTradeRiskGate.check(intent, account, quote, controls) -> RiskDecision`
- Produces: `OrderService.execute(intent, decision_id) -> OrderExecutionResult`
- Produces protocol: `BrokerGateway.get_account()`, `get_positions()`, `get_quote(symbol)`, `submit_order(order)`, `get_order(client_order_id)`, `cancel_order(client_order_id)`。

- [ ] **Step 1: 写默认拒绝和幂等测试**

```python
def test_live_order_requires_all_live_gates(order_service):
    with pytest.raises(LiveTradingDisabled):
        order_service.execute(buy_intent, decision_id="d-1")

def test_same_decision_never_submits_twice(order_service, fake_broker):
    first = order_service.execute(buy_intent, decision_id="d-1")
    second = order_service.execute(buy_intent, decision_id="d-1")
    assert first.client_order_id == second.client_order_id
    assert fake_broker.submit_count == 1
```

- [ ] **Step 2: 实现策略映射和订单意图**

```python
ACTION_TO_STRATEGY = {
    "buy": "trend_follow_entry",
    "hold": "hold_and_trail",
    "reduce": "risk_reduction",
    "exit": "risk_exit",
    "abstain": "no_trade",
}
```

订单数量只能由账户净资产、可用资金、当前持仓、止损距离和风险预算计算；模型或 LLM 不得直接给出股数。A 股买入数量按 100 股整数倍处理，卖出遵循实际持仓和交易制度约束。

- [ ] **Step 3: 实现确定性风控、Paper Broker 与审计表**

```python
if controls.kill_switch:
    return RiskDecision.reject("kill_switch")
if controls.mode == "live" and not controls.live_confirmed:
    return RiskDecision.reject("live_not_confirmed")
if order.notional > min(account.equity * 0.02, account.cash):
    return RiskDecision.reject("single_order_limit")
if account.daily_pnl_pct <= -2.0:
    return RiskDecision.reject("daily_loss_circuit_breaker")
```

新增 `trade_decisions`、`order_intents`、`broker_orders`、`broker_fills`、`risk_events`、`position_snapshots`；所有外部提交使用稳定 `client_order_id=sha256(account_id+decision_id+action)`。

- [ ] **Step 4: 增加实盘 Broker 接入门禁**

首个可运行交付使用 `PaperBrokerGateway`。实盘适配器只有在用户提供具体券商/API、认证方式、委托/撤单/查询接口、交易时段和沙箱账户后才能实现；在此之前 `MA_AGENT_TRADING_MODE=live` 必须启动失败，不能悄悄回退到模拟成交或假装成交。

- [ ] **Step 5: 运行执行层测试**

Run: `python -m pytest tests/agent_system/test_strategy_selector.py tests/agent_system/test_pretrade_risk_gate.py tests/agent_system/test_order_service.py -v`

Expected: PASS；重复请求、超限、过期行情、熔断、急停、未知 Broker 状态和 `abstain` 均不会提交订单。

### Task 10: 暴露 API、CLI 和每日调度

**Files:**
- Modify: `api/v1/endpoints/trend_forecast.py`
- Create: `scripts/run_ma_trend_agent.py`
- Modify: `src/services/runtime_scheduler.py`
- Modify: `main.py`
- Test: `tests/agent_system/test_ma_agent_api.py`
- Test: `tests/agent_system/test_ma_agent_scheduler.py`

**Interfaces:**
- `POST /api/v1/trend-forecast/runs`：执行单股行情、新闻、多模型裁决和可选订单计划。
- `POST /api/v1/trend-forecast/daily-cycle`：执行到期评估、反思、可选晋级、新预测和交易编排。
- `GET /api/v1/trend-forecast/runs/{run_id}`：读取可追溯运行结果。
- `GET /api/v1/trend-forecast/parameters`：列出 champion/challenger 历史。
- `POST /api/v1/trend-forecast/parameters/{version_id}/rollback`：显式回滚到历史版本。
- `GET /api/v1/trend-forecast/orders/{client_order_id}`：读取订单、成交和风控轨迹。
- `POST /api/v1/trend-forecast/trading/kill-switch`：管理员启停急停开关；开启后只允许撤单和减仓。

- [ ] **Step 1: 写 API 契约测试**

```python
def test_run_response_contains_models_adjudication_and_execution(client, auth_headers):
    response = client.post("/api/v1/trend-forecast/runs", headers=auth_headers, json=request_json)
    assert response.status_code == 200
    body = response.json()
    assert set(body["models"]) == {"moving_average", "wavelet", "analog", "logistic_6f", "news"}
    assert body["adjudication"]["action"] in {"buy", "hold", "reduce", "exit", "abstain"}
    assert body["execution"]["mode"] in {"paper", "live", "none"}
```

- [ ] **Step 2: 实现幂等 CLI**

```powershell
python scripts/run_ma_trend_agent.py --mode daily-cycle --as-of 2026-07-21T15:10:00+08:00
```

同一交易日重复执行必须复用唯一键并返回 `already_completed`，除非显式使用只允许管理员调用的 `--force`。

- [ ] **Step 3: 接入现有调度器**

仅在 `MA_AGENT_ENABLED=true` 时注册任务；调度器复用全局运行锁，避免 API 手工触发与定时任务并发。单个证券失败记录错误后继续其他证券，最终返回批次级 `partial_success`。交易时段任务还需单独注册订单查询/撤单/成交对账，不得用一次提交结果推定成交。

- [ ] **Step 4: 运行 API 和调度测试**

Run: `python -m pytest tests/agent_system/test_ma_agent_api.py tests/agent_system/test_ma_agent_scheduler.py -v`

Expected: PASS；关闭开关时不注册任务，重复触发不产生重复记录。

### Task 11: 完成多模型交易闭环文档、依赖和全量验收

**Files:**
- Modify: `requirements-trend-forecast.txt`
- Modify: `README.md`
- Modify: `docs/trend-forecast-orchestration.md`
- Modify: `docs/CHANGELOG.md`
- Test: `tests/agent_system/test_complete_trading_surface.py`

**Interfaces:**
- Produces: 五个独立信号源、一个裁决结果、一个策略意图和零或一个幂等订单。
- Produces: Paper 模式可端到端运行；Live 模式在 Broker 未配置时 fail-closed。

- [ ] **Step 1: 写完整运行表面测试**

```python
def test_runtime_model_inventory_is_explicit():
    assert available_signal_sources() == (
        "moving_average", "wavelet", "analog", "logistic_6f", "news"
    )

def test_live_mode_without_broker_fails_closed(app_config):
    app_config.trading_mode = "live"
    with pytest.raises(ConfigurationError, match="Broker"):
        build_trading_runtime(app_config)

def test_agent_system_does_not_depend_on_bot_package():
    imports = collect_python_imports(Path("src/agent_system"))
    assert not any(name == "bot" or name.startswith("bot.") for name in imports)
```

`collect_python_imports` 只作为测试辅助函数放在 `tests/agent_system/conftest.py`，使用 `ast.walk` 收集 `Import` 和 `ImportFrom`，不在生产代码中新增文件扫描逻辑。

- [ ] **Step 2: 收敛依赖并检查模型调用面**

```powershell
rg -n "wavelet|analog|logistic_6f|researcher_strategy|submit_order|TRADING_MODE" src api scripts tests docs -S
```

保留各研究模型实际需要的依赖；将绘图等仅研究用途依赖与运行依赖分开。`requirements-trend-forecast.txt` 只保留运行闭环所需包，研究/回测依赖单列，避免生产安装无关工具。

- [ ] **Step 3: 更新首页和专题文档**

README 只保留首页级说明：A 股多模型趋势、ClickHouse/新闻数据、Paper 默认模式、启动命令和 API 入口。详细字段、Tick 映射、新闻截止规则、裁决、反馈、参数晋级/回滚、Broker 对接、风控、急停、订单对账和排障写入 `docs/trend-forecast-orchestration.md`。`docs/CHANGELOG.md` 的 `[Unreleased]` 使用扁平条目。

- [ ] **Step 4: 执行完整验证**

```powershell
python -m pytest tests/agent_system -v
python -m pytest -m "not network"
python -m compileall -q src\agent_system src\services\ma_trend_agent_service.py api\v1\endpoints\trend_forecast.py scripts\run_ma_trend_agent.py
```

Expected: 全部 PASS；五个信号源均通过适配器运行，任何交易提交都经过裁决、风控和幂等订单服务。

- [ ] **Step 5: 建议提交检查点（需用户确认）**

```bash
git add src/agent_system src/services/ma_trend_agent_service.py src/storage.py api/v1/endpoints/trend_forecast.py scripts/run_ma_trend_agent.py tests/agent_system .env.example requirements-trend-forecast.txt README.md docs
git commit -m "feat: add multi-model trading agent loop"
```

---

## 分阶段交付顺序

1. **M1 数据与均线基线（Task 1–3）**：能从固定快照稳定输出均线趋势，尚不自动调参或交易。
2. **M2 每日反馈闭环（Task 4–5）**：能保存预测、次日核验并形成结构化错误归因。
3. **M3 受控参数优化（Task 6）**：能生成 challenger 并通过 walk-forward 比较，默认不自动晋级。
4. **M4 新闻与多模型裁决（Task 7–8）**：接入可追溯新闻，统一并校准五个信号源，冲突时主动 abstain。
5. **M5 自动交易模拟盘（Task 9–10）**：策略选择、风控、Paper Broker、订单对账、API 和调度闭环可运行。
6. **M6 实盘准备与验收（Task 11）**：完成文档、依赖和全量验证；选定 Broker 后补充对应实盘适配器及沙箱验收。

每个里程碑均可独立验收；不要在 M1 未建立无前视基线前启动“自动学习”。

## 验收标准

- 同一证券、同一截止时间、同一快照和同一参数版本得到完全一致的趋势结果。
- Tick 模式使用 ClickHouse 数据库侧聚合，并能证明查询未读取 `data_cutoff` 之后的数据。
- 每条预测都有参考价、目标交易日、快照哈希、特征、证据、参数版本和运行 ID。
- 下一交易日数据到齐后自动生成后验，明确标记 `hit/miss/neutral/unable`。
- 判断错误会生成结构化反思；单次错误不会直接修改 champion。
- challenger 只在固定参数空间内产生，并经过无数据泄漏的 walk-forward 门禁。
- 参数晋级与回滚可追溯；默认关闭自动晋级。
- API 与调度重复调用幂等，单股失败不会丢失其他股票结果。
- 新闻快照不含截止时点之后的信息，每条有效新闻判断都有来源引用。
- 最终运行表面明确包含 `moving_average`、`wavelet`、`analog`、`logistic_6f` 和 `news`，并保存各自原始输出与校准版本。
- 多模型目标不同或高置信冲突时不会直接平均，而是目标映射、校准或 `abstain`。
- Paper 模式能完成订单提交、成交回报和持仓对账；Live 模式在任何配置或 Broker 状态不明时 fail-closed。
- 急停、日亏损熔断、仓位/金额/价格偏离限制、重复单保护均有回归测试。
- `src/agent_system`、自动交易服务和调度任务不依赖 `bot/`，也不会发送机器人通知。

## 已知前置确认项

实施 Task 2 前必须从实际 ClickHouse DDL 确认以下字段，不能凭计划猜测：Tick 表名、证券代码列、事件时间列、成交价列、成交量列、成交额列、事件排序列、买卖方向列（如有）、集合竞价/盘后标记，以及成交量是单笔增量还是日内累计。若只有分钟表，可先以 `MA_AGENT_DATA_MODE=minute` 完成 M1，再单独接 Tick。

实施真实自动交易前还必须确认：具体券商/Broker API、认证方式、账户类型、是否提供沙箱、委托类型、撤单/查询/成交推送协议、限频、交易时段、最小价格变动、A 股 T+1 与特殊证券交易规则。未确认前只交付 Paper 模式，Live 模式保持不可启动。

## 自审结果

- 需求覆盖：判断标准、市场数据判断方向、策略筛选预留、参数调整、交易决策输出、每日反馈、错误反思、ClickHouse 历史+当日、Tick 优先均有对应任务。
- 范围已扩展：新闻、多模型裁决和自动交易均有实施任务；Bot 功能、无门禁在线自学习、LLM 直接下单和未选定 Broker 的伪实盘实现仍明确排除。
- 类型一致：统一使用 `MaParameters`、`MaTrendPrediction`、`MaPredictionOutcome`、`MaReflection`、`MaParameterVersion`；方向统一为 `bullish/neutral/bearish`。
- 无前视保护：查询截止、特征 cutoff、下一交易日计算和 walk-forward 验证均有明确测试。
