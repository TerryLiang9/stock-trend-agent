# A 股五分类趋势与可配置模型权重 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将均线模型设为 70% 主权重、三个研究模型各设为 10%，允许在 Web 设置页修改并由后端严格校验，同时把预测结果升级为可解释的五分类研究结论。

**Architecture:** 新增独立的权重配置值对象与五分类纯函数，四个模型先通过适配器产生同目标的标准信号，再由裁决器计算配置权重、实际生效权重、贡献值和综合分。权重继续复用系统配置 API 持久化，Web 使用专用组合编辑器完成跨字段校验；旧裁决字段保留兼容，但交易动作固定为 `abstain`。

**Tech Stack:** Python 3.10+、FastAPI、Pydantic v2、pytest、React、TypeScript、Vitest、Testing Library、ClickHouse。

## Global Constraints

- 默认权重固定为 `moving_average=0.70`、`wavelet=0.10`、`analog=0.10`、`logistic_6f=0.10`。
- 四项权重分别在 `[0, 1]`，总和必须为 `1.0 ± 0.000001`，且均线权重大于任一研究模型权重。
- 五分类阈值固定为 `-0.60`、`-0.20`、`0.20`、`0.60`。
- 本阶段不接入新闻、财经博主、散户情绪、自动调权和自动交易。
- 结果只表达研究模式；兼容字段 `action` 固定为 `abstain`。
- 新配置必须同步 `.env.example`、配置注册表、中文帮助和专题文档。
- 未经用户明确确认，不执行计划中的 `git commit` 命令。

---

## File Structure

### 新增文件

- `src/agent_system/adjudication/trend_classifier.py`：综合分与五分类纯函数。
- `src/agent_system/schemas/model_signal.py`：四个技术模型统一信号契约。
- `apps/dsa-web/src/components/settings/TrendModelWeightEditor.tsx`：四权重组合编辑器。
- `apps/dsa-web/src/components/settings/__tests__/TrendModelWeightEditor.test.tsx`：权重编辑器测试。
- `apps/dsa-web/src/api/trendForecast.ts`：趋势预测 API 客户端。
- `apps/dsa-web/src/types/trendForecast.ts`：五分类响应类型。
- `apps/dsa-web/src/components/trend/TrendForecastResultCard.tsx`：五分类结果卡。
- `apps/dsa-web/src/components/trend/__tests__/TrendForecastResultCard.test.tsx`：结果卡测试。
- `apps/dsa-web/src/pages/TrendForecastPage.tsx`：输入 A 股代码、发起预测并展示结果。
- `apps/dsa-web/src/pages/__tests__/TrendForecastPage.test.tsx`：趋势页交互测试。
- `tests/agent_system/test_trend_classifier.py`：五分类和贡献计算测试。
- `tests/agent_system/test_model_signal_mapping.py`：三个研究适配器的强类型映射测试。
- `tests/agent_system/test_trend_weight_config.py`：权重配置测试。

### 修改文件

- `src/agent_system/config/ma_agent.py`：读取并校验四个权重，关闭新闻和交易默认值。
- `src/agent_system/schemas/adjudication.py`：追加五分类、权重和贡献字段。
- `src/agent_system/adapters/wavelet_adapter.py`：显式输出标准方向。
- `src/agent_system/adapters/analog_adapter.py`：显式输出标准方向。
- `src/agent_system/adapters/logistic_adapter.py`：统一次日收盘目标并输出标准方向。
- `src/agent_system/adjudication/policy.py`：使用五分类器，不再使用固定内嵌权重。
- `src/services/ma_trend_agent_service.py`：移除字符串猜测和新闻参与，传入配置权重。
- `src/agent_system/schemas/request.py`：允许省略目标日并由后端交易日历推导。
- `src/core/config_registry.py`：向系统设置 Schema 注册四个权重。
- `src/services/system_config_service.py`：增加四权重跨字段校验。
- `api/v1/endpoints/trend_forecast.py`：声明稳定响应并保持旧路由。
- `apps/dsa-web/src/pages/SettingsPage.tsx`：在 Agent 分类挂载组合编辑器。
- `apps/dsa-web/src/components/settings/index.ts`：导出权重编辑器。
- `apps/dsa-web/src/utils/systemConfigI18n.ts`：中文字段名称和说明。
- `.env.example`：加入默认权重并保持新闻、自动化、通知和交易关闭。
- `docs/trend-forecast-orchestration.md`：记录五分类公式和配置方式。
- `docs/CHANGELOG.md`：在 `[Unreleased]` 添加扁平条目。

---

### Task 1: 权重配置值对象与环境配置

**Files:**
- Modify: `src/agent_system/config/ma_agent.py`
- Create: `tests/agent_system/test_trend_weight_config.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `ModelWeights(moving_average, wavelet, analog, logistic_6f)`。
- Produces: `ModelWeights.as_dict() -> dict[str, float]`。
- Produces: `MaAgentConfig.model_weights: ModelWeights`。

- [ ] **Step 1: 编写失败测试**

```python
from pydantic import ValidationError
import pytest

from src.agent_system.config.ma_agent import MaAgentConfig, ModelWeights


def test_default_weights_make_moving_average_dominant():
    weights = ModelWeights()
    assert weights.as_dict() == {
        "moving_average": 0.70,
        "wavelet": 0.10,
        "analog": 0.10,
        "logistic_6f": 0.10,
    }


@pytest.mark.parametrize("values", [
    {"moving_average": 0.6, "wavelet": 0.2, "analog": 0.1, "logistic_6f": 0.05},
    {"moving_average": 0.25, "wavelet": 0.25, "analog": 0.25, "logistic_6f": 0.25},
])
def test_weights_reject_invalid_total_or_non_dominant_ma(values):
    with pytest.raises(ValidationError):
        ModelWeights(**values)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/agent_system/test_trend_weight_config.py -v`

Expected: FAIL，提示 `ModelWeights` 尚不存在。

- [ ] **Step 3: 实现最小权重对象和环境变量读取**

```python
class ModelWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    moving_average: float = Field(default=0.70, ge=0, le=1)
    wavelet: float = Field(default=0.10, ge=0, le=1)
    analog: float = Field(default=0.10, ge=0, le=1)
    logistic_6f: float = Field(default=0.10, ge=0, le=1)

    @model_validator(mode="after")
    def validate_weights(self) -> "ModelWeights":
        values = self.as_dict()
        if abs(sum(values.values()) - 1.0) > 1e-6:
            raise ValueError("四个趋势模型权重之和必须为 1.0")
        research_max = max(self.wavelet, self.analog, self.logistic_6f)
        if self.moving_average <= research_max:
            raise ValueError("均线基线权重必须大于任一研究模型权重")
        return self

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()
```

在 `MaAgentConfig.from_mapping()` 中读取：

```python
model_weights=ModelWeights(
    moving_average=float(values.get("MA_AGENT_WEIGHT_MOVING_AVERAGE") or 0.70),
    wavelet=float(values.get("MA_AGENT_WEIGHT_WAVELET") or 0.10),
    analog=float(values.get("MA_AGENT_WEIGHT_ANALOG") or 0.10),
    logistic_6f=float(values.get("MA_AGENT_WEIGHT_LOGISTIC_6F") or 0.10),
)
```

同时将默认值收敛为 `news_enabled=False`、`trading_mode="disabled"`。

- [ ] **Step 4: 更新 `.env.example`**

```dotenv
# 四个技术模型权重之和必须为 1；均线权重必须高于任一研究模型
MA_AGENT_WEIGHT_MOVING_AVERAGE=0.70
MA_AGENT_WEIGHT_WAVELET=0.10
MA_AGENT_WEIGHT_ANALOG=0.10
MA_AGENT_WEIGHT_LOGISTIC_6F=0.10

# 当前阶段不启用新闻和交易
MA_AGENT_NEWS_ENABLED=false
MA_AGENT_TRADING_MODE=disabled
```

- [ ] **Step 5: 运行测试**

Run: `python -m pytest tests/agent_system/test_trend_weight_config.py tests/agent_system/test_ma_agent_core.py -v`

Expected: PASS。

- [ ] **Step 6: 用户确认后提交**

```bash
git add src/agent_system/config/ma_agent.py tests/agent_system/test_trend_weight_config.py .env.example
git commit -m "feat: add configurable trend model weights"
```

---

### Task 2: 统一三个研究模型的目标与强类型输出

**Files:**
- Create: `src/agent_system/schemas/model_signal.py`
- Modify: `src/agent_system/adapters/wavelet_adapter.py`
- Modify: `src/agent_system/adapters/analog_adapter.py`
- Modify: `src/agent_system/adapters/logistic_adapter.py`
- Create: `tests/agent_system/test_model_signal_mapping.py`

**Interfaces:**
- Produces: `StandardModelSignal(model_name, target_type, direction, confidence, evidence, warnings)`。
- All adapters produce target type `next_session_close_vs_cutoff`。

- [ ] **Step 1: 编写方向映射与目标一致性测试**

```python
from src.agent_system.adapters.wavelet_adapter import map_wavelet_signal
from src.agent_system.adapters.analog_adapter import map_analog_signal


def test_wavelet_mapping_is_explicit():
    assert map_wavelet_signal("偏多") == "bullish"
    assert map_wavelet_signal("观望") == "neutral"
    assert map_wavelet_signal("偏空") == "bearish"


def test_analog_mapping_is_explicit():
    assert map_analog_signal("UP") == "bullish"
    assert map_analog_signal("FLAT") == "neutral"
    assert map_analog_signal("DOWN") == "bearish"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/agent_system/test_model_signal_mapping.py -v`

Expected: FAIL，显式映射函数尚不存在。

- [ ] **Step 3: 新增统一信号契约**

```python
class StandardModelSignal(BaseModel):
    model_name: Literal["moving_average", "wavelet", "analog", "logistic_6f"]
    target_type: Literal["next_session_close_vs_cutoff"]
    direction: Literal["bullish", "neutral", "bearish"]
    confidence: float = Field(ge=0, le=1)
    evidence: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: 为小波与相似日增加精确映射**

```python
def map_wavelet_signal(value: str) -> str:
    return {"偏多": "bullish", "观望": "neutral", "偏空": "bearish"}[value]


def map_analog_signal(value: str) -> str:
    return {"UP": "bullish", "FLAT": "neutral", "DOWN": "bearish"}[value.upper()]
```

适配器应把映射后的结构放入明确字段，未知值抛出 `ValueError`，不得使用子串匹配。

- [ ] **Step 5: 统一六因子标签**

把 `logistic_adapter.py` 的标签从下一日 `close > open` 改为下一日收盘相对当前日收盘：

```python
df["next_return"] = df["close"].shift(-1) / df["close"] - 1.0
df["label"] = (df["next_return"] > 0).astype(int)
```

概率差小于 `0.10` 输出 `neutral`，否则按概率较高的一侧输出 `bullish` 或 `bearish`；原始概率保留在证据中。

- [ ] **Step 6: 运行适配器测试**

Run: `python -m pytest tests/agent_system/test_model_signal_mapping.py -v`

Expected: PASS。

- [ ] **Step 7: 用户确认后提交**

```bash
git add src/agent_system/schemas/model_signal.py src/agent_system/adapters tests/agent_system/test_model_signal_mapping.py
git commit -m "refactor: unify trend model signal targets"
```

---

### Task 3: 五分类器与模型贡献计算

**Files:**
- Create: `src/agent_system/adjudication/trend_classifier.py`
- Modify: `src/agent_system/schemas/adjudication.py`
- Create: `tests/agent_system/test_trend_classifier.py`

**Interfaces:**
- Produces: `classify_trend_score(score: float) -> TrendClassification`。
- Produces: `calculate_weighted_trend(signals, configured_weights) -> WeightedTrendResult`。

- [ ] **Step 1: 编写全部边界测试**

```python
import pytest
from src.agent_system.adjudication.trend_classifier import classify_trend_score


@pytest.mark.parametrize(("score", "state"), [
    (1.0, "strong_bullish_control"),
    (0.60, "strong_bullish_control"),
    (0.599999, "weak_bullish_control"),
    (0.20, "weak_bullish_control"),
    (0.0, "bull_bear_tug_of_war"),
    (-0.20, "weak_bearish_control"),
    (-0.60, "strong_bearish_control"),
    (-1.0, "strong_bearish_control"),
])
def test_five_level_boundaries(score, state):
    assert classify_trend_score(score).trend_state == state
```

另加测试验证研究模型失败后实际权重归一化之和为 `1.0`，以及只有均线时 `limited_evidence=True`。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/agent_system/test_trend_classifier.py -v`

Expected: FAIL，分类器尚不存在。

- [ ] **Step 3: 实现五分类纯函数**

```python
def classify_trend_score(score: float) -> TrendClassification:
    if not isfinite(score) or score < -1 or score > 1:
        raise ValueError("综合趋势分必须是 -1 到 1 的有限数")
    if score >= 0.60:
        return TrendClassification("strong_bullish_control", "强多头控制", "只保留做多方向研究")
    if score >= 0.20:
        return TrendClassification("weak_bullish_control", "弱多头控制", "做多为主，降低强度")
    if score <= -0.60:
        return TrendClassification("strong_bearish_control", "强空头控制", "只保留做空方向研究")
    if score <= -0.20:
        return TrendClassification("weak_bearish_control", "弱空头控制", "做空为主，降低强度")
    return TrendClassification("bull_bear_tug_of_war", "多空拉锯", "双向/观望研究")
```

- [ ] **Step 4: 实现有效权重和贡献值**

对执行成功且目标一致的信号计算：

```python
effective_weight = configured_weight / sum_available_configured_weights
contribution = effective_weight * direction_value * confidence
weighted_score = sum(contributions)
```

响应必须包含每个模型的配置权重、实际权重、方向、置信度和贡献值。

- [ ] **Step 5: 运行测试**

Run: `python -m pytest tests/agent_system/test_trend_classifier.py -v`

Expected: PASS。

- [ ] **Step 6: 用户确认后提交**

```bash
git add src/agent_system/adjudication/trend_classifier.py src/agent_system/schemas/adjudication.py tests/agent_system/test_trend_classifier.py
git commit -m "feat: classify A-share trend into five levels"
```

---

### Task 4: 接入预测服务和 API 运行记录

**Files:**
- Modify: `src/agent_system/adjudication/policy.py`
- Modify: `src/services/ma_trend_agent_service.py`
- Modify: `src/agent_system/schemas/request.py`
- Modify: `api/v1/endpoints/trend_forecast.py`
- Modify: `tests/agent_system/test_ma_agent_core.py`
- Create: `tests/agent_system/test_ma_trend_agent_service.py`

**Interfaces:**
- Consumes: `MaAgentConfig.model_weights`、四个 `StandardModelSignal`。
- Produces: `adjudication.trend_state`、`trend_state_label`、`weighted_score`、`research_mode`、两组权重和贡献列表。

- [ ] **Step 1: 编写服务回归测试**

测试应断言：

```python
assert result["adjudication"]["trend_state"] in {
    "strong_bullish_control",
    "weak_bullish_control",
    "bull_bear_tug_of_war",
    "weak_bearish_control",
    "strong_bearish_control",
}
assert result["adjudication"]["action"] == "abstain"
assert "news" not in result["models"]
assert result["adjudication"]["configured_weights"]["moving_average"] == 0.70
```

另加测试模拟一个研究适配器失败，验证预测仍完成且有效权重总和为 `1.0`；请求省略 `target_date` 时，验证服务通过 `get_next_trading_date("cn", as_of.date())` 生成目标日。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/agent_system/test_ma_trend_agent_service.py -v`

Expected: FAIL，当前结果仍是三分类且包含新闻。

- [ ] **Step 3: 删除服务中的启发式映射和新闻执行**

移除 `_direction_from_research_output()`、`_probability_from_research_output()` 和 `analyze_news()` 调用。四个技术模型统一进入裁决；失败模型只保存错误信封，不进入贡献计算。

- [ ] **Step 4: 替换裁决实现**

`AdjudicationPolicy.decide()` 接收 `configured_weights`，调用 `calculate_weighted_trend()`，兼容字段设置为：

```python
action="abstain"
direction="bullish" if score >= 0.20 else "bearish" if score <= -0.20 else "neutral"
confidence=abs(score)
reason_code="five_level_weighted_consensus"
```

- [ ] **Step 5: 让后端推导可选目标日**

将 `TrendForecastRequest.target_date` 改为 `date | None = None`。服务开始预测时使用：

```python
target_date = request.target_date or get_next_trading_date("cn", request.as_of.date())
```

之后模型、响应和运行记录都使用该确定值。

- [ ] **Step 6: 验证 API 和运行记录回放**

Run: `python -m pytest tests/agent_system/test_ma_agent_core.py tests/agent_system/test_ma_trend_agent_service.py tests/test_api_schema_pydantic.py -v`

Expected: 新旧字段测试全部 PASS。

- [ ] **Step 7: 用户确认后提交**

```bash
git add src/agent_system/adjudication/policy.py src/services/ma_trend_agent_service.py src/agent_system/schemas/request.py api/v1/endpoints/trend_forecast.py tests/agent_system
git commit -m "feat: return five-level trend forecasts"
```

---

### Task 5: 把四个权重注册到系统配置 API

**Files:**
- Modify: `src/core/config_registry.py`
- Modify: `src/services/system_config_service.py`
- Modify: `tests/test_config_registry.py`
- Modify: `tests/test_system_config_service.py`
- Modify: `tests/test_system_config_api.py`

**Interfaces:**
- Produces four editable Agent-category settings named `MA_AGENT_WEIGHT_*`。
- System config validate/update rejects invalid totals atomically。

- [ ] **Step 1: 编写 Schema 与跨字段校验测试**

测试四项都位于 Agent 分类、默认值正确，并验证 `0.6+0.2+0.1+0.05` 返回总和错误，`0.25` 四等分返回均线不占主导错误，默认值保存成功。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config_registry.py tests/test_system_config_service.py tests/test_system_config_api.py -k "trend_weight" -v`

Expected: FAIL，配置项尚未注册。

- [ ] **Step 3: 注册配置字段**

四项使用：

```python
{
    "category": "agent",
    "data_type": "number",
    "ui_control": "number",
    "is_editable": True,
    "validation": {"min": 0.0, "max": 1.0},
}
```

显示顺序连续排列，默认值按 `0.70/0.10/0.10/0.10`。

- [ ] **Step 4: 增加原子跨字段校验**

验证必须基于“已保存配置 + 本次草稿”的有效配置图，而不是只检查本次提交字段；任一错误时不写入任何权重。

- [ ] **Step 5: 运行测试**

Run: `python -m pytest tests/test_config_registry.py tests/test_system_config_service.py tests/test_system_config_api.py -k "trend_weight or config_schema" -v`

Expected: PASS。

- [ ] **Step 6: 用户确认后提交**

```bash
git add src/core/config_registry.py src/services/system_config_service.py tests/test_config_registry.py tests/test_system_config_service.py tests/test_system_config_api.py
git commit -m "feat: expose trend weights in system config"
```

---

### Task 6: Web 权重组合编辑器

**Files:**
- Create: `apps/dsa-web/src/components/settings/TrendModelWeightEditor.tsx`
- Create: `apps/dsa-web/src/components/settings/__tests__/TrendModelWeightEditor.test.tsx`
- Modify: `apps/dsa-web/src/components/settings/index.ts`
- Modify: `apps/dsa-web/src/pages/SettingsPage.tsx`
- Modify: `apps/dsa-web/src/utils/systemConfigI18n.ts`

**Interfaces:**
- Consumes: existing `SystemConfigItem[]` and `onChange(key, value)` draft mechanism。
- Produces: four validated string drafts using existing system config save API。
- Produces: `onValidityChange(isValid: boolean)`，供设置页在组合非法时禁用统一保存按钮。

- [ ] **Step 1: 编写组件测试**

覆盖默认显示 `70%/10%/10%/10%`、合计 100%、合计不等于 100% 的错误、均线不占主导的错误，以及修正后错误消失。

- [ ] **Step 2: 运行测试确认失败**

Run: `cd apps/dsa-web && npm test -- TrendModelWeightEditor.test.tsx --run`

Expected: FAIL，组件尚不存在。

- [ ] **Step 3: 实现组合编辑器**

组件提供四个 `type="number"` 输入框，`min=0`、`max=100`、`step=1`，UI 使用百分比，写回环境配置时除以 100。每次输入后调用 `onValidityChange`；组件底部实时显示：

```text
当前合计：100%
均线基线必须高于任一研究模型；四项合计必须为100%。
仅用于趋势研究，不构成交易指令。
```

- [ ] **Step 4: 接入设置页**

在 `activeCategory === 'agent'` 时展示该编辑器；四个原始配置字段从通用字段列表中过滤，避免重复编辑。保存继续使用现有 `setDraftValue()` 和统一保存按钮；权重组合非法时统一保存按钮 `disabled`。

- [ ] **Step 5: 增加中文字段文本**

为四个 key 增加“均线基线权重、小波模型权重、历史相似日权重、六因子模型权重”及用途说明。

- [ ] **Step 6: 运行组件和设置页测试**

Run: `cd apps/dsa-web && npm test -- TrendModelWeightEditor.test.tsx SettingsPage.test.tsx --run`

Expected: PASS。

- [ ] **Step 7: 用户确认后提交**

```bash
git add apps/dsa-web/src/components/settings apps/dsa-web/src/pages/SettingsPage.tsx apps/dsa-web/src/utils/systemConfigI18n.ts
git commit -m "feat: edit trend model weights in web settings"
```

---

### Task 7: Web 五分类结果类型与展示卡

**Files:**
- Create: `apps/dsa-web/src/types/trendForecast.ts`
- Create: `apps/dsa-web/src/api/trendForecast.ts`
- Create: `apps/dsa-web/src/components/trend/TrendForecastResultCard.tsx`
- Create: `apps/dsa-web/src/components/trend/__tests__/TrendForecastResultCard.test.tsx`
- Create: `apps/dsa-web/src/pages/TrendForecastPage.tsx`
- Create: `apps/dsa-web/src/pages/__tests__/TrendForecastPage.test.tsx`
- Modify: `apps/dsa-web/src/App.tsx`
- Modify: `apps/dsa-web/src/components/layout/SidebarNav.tsx`
- Modify: `apps/dsa-web/src/components/layout/__tests__/SidebarNav.test.tsx`
- Modify: `apps/dsa-web/src/i18n/uiText.ts`

**Interfaces:**
- Consumes: `GET /api/v1/trend-forecast/runs/{runId}`。
- Consumes: `POST /api/v1/trend-forecast/runs`。
- Produces: accessible five-level result card and `/trend-forecast` page。

- [ ] **Step 1: 编写结果卡测试**

测试强多头显示“强多头控制”“只保留做多方向研究”、综合分和四模型贡献；测试 `limitedEvidence` 显示“证据有限”；测试失败模型显示失败原因但不显示为零贡献。页面测试输入 `688001.SH`、点击“开始预测”后调用一次 API 并显示返回结果。

- [ ] **Step 2: 运行测试确认失败**

Run: `cd apps/dsa-web && npm test -- TrendForecastResultCard.test.tsx TrendForecastPage.test.tsx --run`

Expected: FAIL，类型和组件尚不存在。

- [ ] **Step 3: 定义 TypeScript 契约和 API 客户端**

```typescript
export type TrendState =
  | 'strong_bullish_control'
  | 'weak_bullish_control'
  | 'bull_bear_tug_of_war'
  | 'weak_bearish_control'
  | 'strong_bearish_control';
```

响应类型必须包含 `weightedScore`、`researchMode`、`configuredWeights`、`effectiveWeights`、`contributions` 和 `limitedEvidence`。

- [ ] **Step 4: 实现结果卡**

卡片以中文状态为主标题，展示研究模式、综合分、各模型方向/置信度/贡献。颜色不得成为唯一信息载体；使用中文状态文字和正负号。

- [ ] **Step 5: 实现页面、路由和导航**

新增 `/trend-forecast` 路由和“A 股趋势”导航项。页面只提供 A 股代码输入、预测按钮、加载/失败状态和结果卡；请求体中的 `as_of`、`data_cutoff` 使用当前北京时间并省略 `target_date`，由后端交易日历推导，禁止前端用自然日猜测交易日。页面不得出现交易按钮。

- [ ] **Step 6: 运行测试**

Run: `cd apps/dsa-web && npm test -- TrendForecastResultCard.test.tsx TrendForecastPage.test.tsx SidebarNav.test.tsx --run`

Expected: PASS。

- [ ] **Step 7: 用户确认后提交**

```bash
git add apps/dsa-web/src/types/trendForecast.ts apps/dsa-web/src/api/trendForecast.ts apps/dsa-web/src/components/trend apps/dsa-web/src/pages/TrendForecastPage.tsx apps/dsa-web/src/pages/__tests__/TrendForecastPage.test.tsx apps/dsa-web/src/App.tsx apps/dsa-web/src/components/layout/SidebarNav.tsx apps/dsa-web/src/components/layout/__tests__/SidebarNav.test.tsx apps/dsa-web/src/i18n/uiText.ts
git commit -m "feat: display five-level trend result"
```

---

### Task 8: 文档、完整回归与交付

**Files:**
- Modify: `docs/trend-forecast-orchestration.md`
- Modify: `docs/CHANGELOG.md`
- Review only: `README.md`

**Interfaces:**
- Documents: weight formula, five boundaries, failure normalization, Web edit path and research-only scope。

- [ ] **Step 1: 更新专题文档**

增加四权重表、综合分公式、五分类区间、配置保存规则和示例 API 响应。明确新闻、舆情、通知、自动化和交易不参与当前流程。

- [ ] **Step 2: 更新 Changelog**

在 `[Unreleased]` 直接增加扁平条目：

```markdown
- [新功能] A 股趋势预测支持均线主导的可配置四模型权重和五分类研究结果。
- [改进] Web 设置页可组合编辑趋势模型权重，并校验总和与均线主导约束。
```

- [ ] **Step 3: 核对 README**

只修正与实际能力矛盾的首页级描述，不把阈值、字段契约和排障细节复制到 README。

- [ ] **Step 4: 运行后端验证**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\agent_system tests\test_config_registry.py tests\test_system_config_service.py tests\test_system_config_api.py -v
.\.venv\Scripts\python.exe -m compileall -q src\agent_system src\services\ma_trend_agent_service.py api\v1\endpoints\trend_forecast.py
```

Expected: 所有目标测试 PASS，compileall exit code 0。

- [ ] **Step 5: 运行 Web 验证**

```powershell
Set-Location apps\dsa-web
npm run lint
npm run build
npm test -- TrendModelWeightEditor.test.tsx TrendForecastResultCard.test.tsx TrendForecastPage.test.tsx SettingsPage.test.tsx SidebarNav.test.tsx --run
```

Expected: lint、build 和测试均 PASS。

- [ ] **Step 6: 执行手工 API 冒烟**

使用内联分钟行情调用 `POST /api/v1/trend-forecast/runs`，确认：

- 返回且只返回五个状态之一。
- `configured_weights` 为 `0.70/0.10/0.10/0.10`。
- `effective_weights` 总和为 `1.0`。
- `models` 不含 `news`。
- `action` 为 `abstain`。
- 读取 `GET /runs/{run_id}` 得到相同分类和权重版本。

- [ ] **Step 7: 记录未验证项、风险和回滚方式**

未连接真实 ClickHouse 时明确记录在线行情未验证。主要风险是六因子目标变更导致历史结果不可直接横向比较；回滚时恢复旧裁决器和配置读取即可，新增 API 字段为追加字段，不要求数据迁移。

- [ ] **Step 8: 用户确认后提交**

```bash
git add docs/trend-forecast-orchestration.md docs/CHANGELOG.md README.md
git commit -m "docs: document five-level trend weighting"
```

---

## Acceptance Criteria

- 默认情况下均线贡献占 70%，三个研究模型各占 10%。
- 用户能够在 Web 设置页修改四权重，非法组合无法保存，合法组合在下一次预测生效。
- 任一成功预测只产生五个规定状态之一，并返回中文名称、综合分、研究模式和模型贡献。
- 单研究模型失败不阻断预测，剩余有效权重正确归一化；均线失败则预测失败。
- 新闻、财经博主、散户情绪和交易执行不参与该链路。
- 旧 API 路由和核心字段保持兼容，运行记录可以复盘配置权重与实际权重。
- 后端目标测试、Python 编译、Web lint/build/组件测试全部通过。
