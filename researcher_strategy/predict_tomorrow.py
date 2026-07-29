"""用最优 6 因子逻辑回归预测 2026-07-16，基于 07-15 日内数据"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from factor_lr_backtest import load_all_data, build_daily_ohlcv
from bull_bear_backtest import compute_bull_bear_factors
import warnings
warnings.filterwarnings("ignore")

TOP6 = ["wvad", "obv_ratio", "cmf_intraday", "buy_sell_change", "buy_sell_vol_ratio", "close_30m_momentum"]

print("=" * 60)
print("  逻辑回归预测 2026-07-16（基于 07-15 日内数据）")
print("=" * 60)

# 加载全部数据
df_1m = load_all_data()
print(f"  1m 数据范围: {df_1m.index[0]} ~ {df_1m.index[-1]}")

df_daily = build_daily_ohlcv(df_1m)
df_bb = compute_bull_bear_factors(df_1m)

df = df_daily.merge(df_bb.drop(columns=["open", "close"], errors="ignore"), on="date")
df["buy_sell_change"] = df["buy_sell_vol_ratio"].pct_change()
df["obv_change"] = df["obv_ratio"].diff()
df["cmf_change"] = df["cmf_intraday"].diff()
df["power_change"] = df["power_ratio"].pct_change()
df["daily_return"] = df["close"].pct_change()

# 标签（仅用于历史训练）
df["next_open"] = df["open"].shift(-1)
df["next_close"] = df["close"].shift(-1)
df["label"] = (df["next_close"] > df["next_open"]).astype(int)
df["next_direction"] = (df["next_close"] > df["close"]).astype(int)

# 分离：有标签的历史数据 → 训练；最新一天 → 预测
df_train = df.dropna(subset=["label"] + TOP6).reset_index(drop=True)
df_today = df[df["date"] == df["date"].max()].iloc[0]

print(f"  历史训练样本: {len(df_train)} 天")
print(f"  最新一天: {df_today['date']}")
print(f"  今日({df_today['date']}): {'阳线' if df_today['close'] > df_today['open'] else '阴线'}")
print(f"    开 {df_today['open']:.2f}  高 {df_today['high']:.2f}")
print(f"    低 {df_today['low']:.2f}  收 {df_today['close']:.2f}")
print(f"    收益率: {(df_today['close']/df_today['open']-1)*100:+.2f}%")

# 训练
X_train = df_train[TOP6].values
y_train = df_train["label"].values
scaler = StandardScaler()
X_s = scaler.fit_transform(X_train)
model = LogisticRegression(max_iter=3000, C=1.0, random_state=42)
model.fit(X_s, y_train)

print(f"\n  因子权重:")
for name, w in zip(TOP6, model.coef_[0]):
    print(f"    {'+' if w > 0 else '-'} {name:<25s} {w:+.4f}")

print(f"\n  今日(07-15)因子值:")
for name in TOP6:
    print(f"    {name:<25s} = {df_today[name]:+.4f}")

# 预测
X_pred = df_today[TOP6].values.reshape(1, -1)
X_pred_s = scaler.transform(X_pred)
prob = model.predict_proba(X_pred_s)[0]
pred = model.predict(X_pred_s)[0]

# 置信度阈值：概率差 < 阈值 → 多空都做
THRESHOLD = 0.10
diff = abs(prob[1] - prob[0])

print(f"\n{'='*60}")
print(f"  预测 2026-07-16")
print(f"{'='*60}")
if diff < THRESHOLD:
    print(f"  信号: 多空都做（概率差距 {diff*100:.1f}% < 阈值 {THRESHOLD*100:.0f}%）")
else:
    direction = "阳线（做多）" if pred == 1 else "阴线（做空）"
    print(f"  信号: {direction}")
print(f"  阳线概率: {prob[1]*100:.1f}%  阴线概率: {prob[0]*100:.1f}%")
print(f"  置信度: {'低' if diff < THRESHOLD else '中等' if diff < 0.20 else '较强'}")

# 模型训练准确率（in-sample）
train_pred = model.predict(X_s)
train_acc = (train_pred == y_train).mean()
train_base = max(y_train.mean(), 1-y_train.mean())
print(f"\n  训练集准确率: {train_acc:.2%} (基准 {train_base:.2%})")
