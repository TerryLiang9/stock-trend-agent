# -*- coding: utf-8 -*-
"""ClickHouse Tick 聚合为分钟 K 线的受信任 SQL 模板。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TickColumnMapping:
    symbol: str = "symbol"
    event_time: str = "event_time"
    price: str = "price"
    volume: str = "volume"
    amount: str = "amount"
    sequence: str = "sequence"


def build_tick_to_minute_sql(
    *,
    database: str,
    table: str,
    mapping: TickColumnMapping = TickColumnMapping(),
    volume_semantics: str = "incremental",
) -> str:
    """生成只允许服务端配置展开的 Tick 聚合 SQL。"""
    if volume_semantics not in {"incremental", "cumulative"}:
        raise ValueError("成交量语义必须为 incremental 或 cumulative")
    volume_expr = (
        f"sum({mapping.volume}) AS volume"
        if volume_semantics == "incremental"
        else f"max({mapping.volume}) - min({mapping.volume}) AS volume"
    )
    return f"""
SELECT
    {mapping.symbol} AS symbol,
    toStartOfMinute({mapping.event_time}) AS datetime,
    argMin({mapping.price}, ({mapping.event_time}, {mapping.sequence})) AS open,
    max({mapping.price}) AS high,
    min({mapping.price}) AS low,
    argMax({mapping.price}, ({mapping.event_time}, {mapping.sequence})) AS close,
    {volume_expr},
    sum({mapping.amount}) AS amount
FROM {database}.{table}
PREWHERE {mapping.symbol} = {{symbol:String}}
WHERE {mapping.event_time} >= {{history_start:DateTime64(3)}}
  AND {mapping.event_time} <= {{data_cutoff:DateTime64(3)}}
GROUP BY {mapping.symbol}, toStartOfMinute({mapping.event_time})
ORDER BY datetime ASC
""".strip()
