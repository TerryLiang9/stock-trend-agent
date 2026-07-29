"""行情采集器 — 订阅 L2 WS，落地「十档订单簿快照 + 逐笔成交 + 逐笔委托」历史数据。

数据来源已确认为全量快照（按 symbol 做 key），每只票对象含：
  - 顶层快照字段：last_price/open/high/low/pre_close/up_limit/down_limit/volume/amount/time
  - orderbook：十档 bidPrice/askPrice/bidVol/askVol + transactionNum/stockStatus
  - l2_trade：逐笔成交（滚动窗口，需按 seq/tradeIndex 去重）
  - l2_order：逐笔委托（滚动窗口，需按 entrustNo 去重）

落地结构（CSV，pandas 直接可读）：
  data/<YYYYMMDD>/<symbol>/snapshot.csv   十档订单簿快照，一行一个快照时刻
  data/<YYYYMMDD>/<symbol>/trades.csv     逐笔成交（已去重、已判主动方向）
  data/<YYYYMMDD>/<symbol>/orders.csv     逐笔委托（已去重）

设计要点：
  1. 去重：逐笔是滚动窗口，相邻消息重叠。成交按 seq(为0时回退 tradeIndex+time)
     去重，委托按 entrustNo 去重，避免成交量/委托被重复记录。
  2. 快照去重：按 (symbol, 快照time) 记一行，同一 time 重复推送不重复落盘。
  3. 主动方向：A股逐笔成交，buyNo>sellNo → 主动买(B)，sellNo>buyNo → 主动卖(S)。
     这是做空因子里「主动性卖盘占比」的基础。
  4. 缓冲 + 定时落盘（追加写），降低磁盘 IO。
  5. 断线自动重连（指数退避）；Ctrl+C 优雅退出并做最后一次落盘。

用法：
  python collector.py                      # 采集默认 watchlist，落到 ./data
  python collector.py --env DEV            # 用 DEV 环境
  python collector.py --out /path/to/data  # 自定义落地目录

需要：pip install websockets
"""

import argparse
import asyncio
import csv
import json
import os
import signal
import time
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

try:
    import websockets
except ImportError:
    print("请安装: pip install websockets")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
WATCHLIST = ["688981.SH", "688449.SH", "688798.SH", "688469.SH", "688536.SH",
             "688396.SH", "688126.SH", "688311.SH"]

ENVS = {
    "PROD": "ws://192.168.251.1:26080/ws/quote",
    "DEV":  "ws://192.168.251.1:26080/ws/quote",
}

FLUSH_INTERVAL = 3.0     # 落盘间隔（秒）
STATS_INTERVAL = 30.0    # 打印统计间隔（秒）
DEPTH = 10               # 记录几档盘口
RECONNECT_MAX = 30.0     # 重连最大退避（秒）

# 收盘自动退出配置 ----------------------------------------------------------
BEIJING = ZoneInfo("Asia/Shanghai")
CLOSE_H, CLOSE_M = 15, 0     # A股下午收盘 15:00（含 14:57-15:00 尾盘集合竞价）
EXIT_GRACE_SEC = 30          # 收盘后再等这么久，确保收尾快照落盘后再退出


def is_trading_day(dt: datetime) -> bool:
    """是否交易日（仅按周一~周五判断，不含法定节假日）。"""
    return dt.weekday() < 5


def market_session_over(now: datetime) -> tuple[bool, str]:
    """判断当前是否应视为"已收盘/无盘"。返回 (是否收盘, 原因)。"""
    if not is_trading_day(now):
        return True, "今日非交易日（周末）"
    close_dt = now.replace(hour=CLOSE_H, minute=CLOSE_M, second=0, microsecond=0)
    if now >= close_dt + timedelta(seconds=EXIT_GRACE_SEC):
        return True, f"已过收盘时间 {CLOSE_H:02d}:{CLOSE_M:02d}"
    return False, ""

# CSV 列定义 -----------------------------------------------------------------
SNAPSHOT_COLS = (
    ["symbol", "quote_time", "recv_ts", "last_price", "open", "high", "low",
     "pre_close", "up_limit", "down_limit", "volume", "amount",
     "transaction_num", "stock_status"]
    + [f"{side}{i}_{f}" for i in range(1, DEPTH + 1)
       for side in ("bid", "ask") for f in ("price", "vol")]
)
TRADE_COLS = ["symbol", "quote_time", "recv_ts", "price", "volume", "amount",
              "buy_no", "sell_no", "trade_type", "trade_flag", "trade_index",
              "seq", "active_side"]
ORDER_COLS = ["symbol", "quote_time", "recv_ts", "price", "volume", "entrust_no",
              "entrust_type", "entrust_direction", "seq"]


# ---------------------------------------------------------------------------
# CSV 落地器：管理文件句柄 + 表头 + 追加写
# ---------------------------------------------------------------------------
class CsvWriter:
    """按 (日期, symbol, 类型) 追加写 CSV，首次写入自动建目录和表头。"""

    def __init__(self, root: str):
        self.root = root
        self._writers = {}   # key -> (file_obj, csv_writer)
        self._cols = {"snapshot": SNAPSHOT_COLS, "trades": TRADE_COLS, "orders": ORDER_COLS}

    def _get(self, symbol: str, kind: str):
        date = datetime.now().strftime("%Y%m%d")
        key = (date, symbol, kind)
        if key not in self._writers:
            d = os.path.join(self.root, date, symbol)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{kind}.csv")
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            f = open(path, "a", newline="", encoding="utf-8")
            w = csv.DictWriter(f, fieldnames=self._cols[kind], extrasaction="ignore")
            if new:
                w.writeheader()
            self._writers[key] = (f, w)
        return self._writers[key]

    def write_rows(self, symbol: str, kind: str, rows: list):
        if not rows:
            return
        f, w = self._get(symbol, kind)
        w.writerows(rows)

    def flush(self):
        for f, _ in self._writers.values():
            f.flush()

    def close(self):
        for f, _ in self._writers.values():
            try:
                f.close()
            except Exception:
                pass
        self._writers.clear()


# ---------------------------------------------------------------------------
# 解析：把一只票的对象拆成 snapshot / trades / orders 行
# ---------------------------------------------------------------------------
def parse_snapshot(sym: str, obj: dict, recv_ts: int) -> dict:
    ob = obj.get("orderbook", {}) or {}
    row = {
        "symbol": sym,
        "quote_time": obj.get("time"),
        "recv_ts": recv_ts,
        "last_price": obj.get("last_price"),
        "open": obj.get("open"),
        "high": obj.get("high"),
        "low": obj.get("low"),
        "pre_close": obj.get("pre_close"),
        "up_limit": obj.get("up_limit"),
        "down_limit": obj.get("down_limit"),
        "volume": obj.get("volume"),
        "amount": obj.get("amount"),
        "transaction_num": ob.get("transactionNum"),
        "stock_status": ob.get("stockStatus"),
    }
    # 十档：优先用 orderbook 的数组（10档），否则回退顶层 bid1..5 字段
    bid_p, ask_p = ob.get("bidPrice") or [], ob.get("askPrice") or []
    bid_v, ask_v = ob.get("bidVol") or [], ob.get("askVol") or []
    for i in range(1, DEPTH + 1):
        row[f"bid{i}_price"] = bid_p[i - 1] if i <= len(bid_p) else obj.get(f"bid{i}_price")
        row[f"bid{i}_vol"]   = bid_v[i - 1] if i <= len(bid_v) else obj.get(f"bid{i}_vol")
        row[f"ask{i}_price"] = ask_p[i - 1] if i <= len(ask_p) else obj.get(f"ask{i}_price")
        row[f"ask{i}_vol"]   = ask_v[i - 1] if i <= len(ask_v) else obj.get(f"ask{i}_vol")
    return row


def active_side(buy_no, sell_no) -> str:
    """A股逐笔成交主动方向：委托号更大者为主动方（后到的吃单方）。"""
    try:
        if buy_no > sell_no:
            return "B"   # 主动买
        if sell_no > buy_no:
            return "S"   # 主动卖
    except TypeError:
        pass
    return "?"


def parse_trade(sym: str, t: dict, recv_ts: int) -> dict:
    return {
        "symbol": sym,
        "quote_time": t.get("time"),
        "recv_ts": recv_ts,
        "price": t.get("price"),
        "volume": t.get("volume"),
        "amount": t.get("amount"),
        "buy_no": t.get("buyNo"),
        "sell_no": t.get("sellNo"),
        "trade_type": t.get("tradeType"),
        "trade_flag": t.get("tradeFlag"),
        "trade_index": t.get("tradeIndex"),
        "seq": t.get("seq"),
        "active_side": active_side(t.get("buyNo"), t.get("sellNo")),
    }


def parse_order(sym: str, o: dict, recv_ts: int) -> dict:
    return {
        "symbol": sym,
        "quote_time": o.get("time"),
        "recv_ts": recv_ts,
        "price": o.get("price"),
        "volume": o.get("volume"),
        "entrust_no": o.get("entrustNo"),
        "entrust_type": o.get("entrustType"),
        "entrust_direction": o.get("entrustDirection"),  # 1=买 2=卖
        "seq": o.get("seq"),
    }


def trade_key(t: dict):
    """成交去重键：seq>0 用 seq；seq=0（盘后/竞价）回退 tradeIndex+time。"""
    seq = t.get("seq")
    if seq:
        return ("seq", seq)
    return ("idx", t.get("tradeIndex"), t.get("time"))


# ---------------------------------------------------------------------------
# 采集器主体
# ---------------------------------------------------------------------------
class Collector:
    def __init__(self, uri: str, symbols: list, out_dir: str, auto_exit: bool = True):
        self.uri = uri
        self.symbols = symbols
        self.writer = CsvWriter(out_dir)
        self.stop = False
        self.auto_exit = auto_exit    # 收盘后是否自动退出
        self.closed_reason = None     # 触发退出的原因（用于报告）

        # 去重状态（当日累积）
        self._seen_trades = defaultdict(set)   # symbol -> set(trade_key)
        self._seen_orders = defaultdict(set)   # symbol -> set(entrust_no)
        self._last_snap_time = {}              # symbol -> 上次落盘的 quote_time

        # 缓冲区
        self._buf = defaultdict(lambda: defaultdict(list))  # symbol -> kind -> [rows]

        # 统计
        self.n_msgs = 0
        self.n_snap = 0
        self.n_trades = 0
        self.n_orders = 0
        self._day = datetime.now().strftime("%Y%m%d")

        # 推送间隔统计（相邻两条消息的到达时间差，单位 ms）
        self._last_recv = None      # 上一条消息的本地到达时间
        self._itv_sum = 0.0         # 当前统计窗口内的间隔累加
        self._itv_cnt = 0           # 当前统计窗口内的间隔样本数
        self._itv_min = None
        self._itv_max = None

    # --- 处理一条消息 ---
    def handle_message(self, data: dict, recv_ts: int):
        # 跨天：重置去重状态（新的一天新文件）
        today = datetime.now().strftime("%Y%m%d")
        if today != self._day:
            self._seen_trades.clear()
            self._seen_orders.clear()
            self._last_snap_time.clear()
            self._day = today

        self.n_msgs += 1

        # 推送间隔：相邻两条消息的本地到达时间差（超时空档不计入）
        if self._last_recv is not None:
            dt = recv_ts - self._last_recv
            if dt >= 0:
                self._itv_sum += dt
                self._itv_cnt += 1
                self._itv_min = dt if self._itv_min is None else min(self._itv_min, dt)
                self._itv_max = dt if self._itv_max is None else max(self._itv_max, dt)
        self._last_recv = recv_ts

        for sym in self.symbols:
            obj = data.get(sym)
            if not isinstance(obj, dict):
                continue

            # 1) 快照：按 quote_time 去重
            qt = obj.get("time")
            if qt is not None and self._last_snap_time.get(sym) != qt:
                self._buf[sym]["snapshot"].append(parse_snapshot(sym, obj, recv_ts))
                self._last_snap_time[sym] = qt
                self.n_snap += 1

            # 2) 逐笔成交：按 key 去重
            seen_t = self._seen_trades[sym]
            for t in obj.get("l2_trade") or []:
                k = trade_key(t)
                if k in seen_t:
                    continue
                seen_t.add(k)
                self._buf[sym]["trades"].append(parse_trade(sym, t, recv_ts))
                self.n_trades += 1

            # 3) 逐笔委托：按 entrustNo 去重
            seen_o = self._seen_orders[sym]
            for o in obj.get("l2_order") or []:
                eno = o.get("entrustNo")
                if eno in seen_o:
                    continue
                seen_o.add(eno)
                self._buf[sym]["orders"].append(parse_order(sym, o, recv_ts))
                self.n_orders += 1

    # --- 落盘 ---
    def flush(self):
        for sym, kinds in self._buf.items():
            for kind, rows in kinds.items():
                if rows:
                    self.writer.write_rows(sym, kind, rows)
                    rows.clear()
        self.writer.flush()

    # --- 收盘报告 ---
    def _report_close(self, reason: str):
        print(f"\n{'='*60}")
        print(f"  检测到收盘：{reason}")
        print(f"  收盘前最后一条快照时间（各票 quote_time）：")
        for sym in self.symbols:
            qt = self._last_snap_time.get(sym)
            if qt:
                ts = datetime.fromtimestamp(qt / 1000, BEIJING).strftime("%H:%M:%S")
                print(f"    {sym:<12} {ts}  (quote_time={qt})")
            else:
                print(f"    {sym:<12} 今日无数据")
        print(f"{'='*60}")

    # --- 后台定时落盘 + 统计 ---
    async def _periodic(self):
        last_stats = time.time()
        while not self.stop:
            await asyncio.sleep(FLUSH_INTERVAL)
            self.flush()

            # 收盘自动退出判断
            if self.auto_exit:
                closed, reason = market_session_over(datetime.now(BEIJING))
                if closed:
                    self.closed_reason = reason
                    self._report_close(reason)
                    self.stop = True
                    break

            now = time.time()
            if now - last_stats >= STATS_INTERVAL:
                # 推送间隔：本窗口均值 / min / max（反映服务端实际推送节奏）
                if self._itv_cnt:
                    itv = (f"itv≈{self._itv_sum / self._itv_cnt:.0f}ms "
                           f"[{self._itv_min:.0f}~{self._itv_max:.0f}]")
                else:
                    itv = "itv=n/a"
                print(f"[{datetime.now():%H:%M:%S}] msgs={self.n_msgs} "
                      f"snap={self.n_snap} trades={self.n_trades} orders={self.n_orders} {itv}")
                # 重置窗口统计，让下次打印反映最近这段的节奏
                self._itv_sum = 0.0
                self._itv_cnt = 0
                self._itv_min = self._itv_max = None
                last_stats = now

    # --- 接收循环（含重连）---
    async def run(self):
        periodic_task = asyncio.create_task(self._periodic())
        backoff = 1.0
        url = f"{self.uri}?symbols={','.join(self.symbols)}"
        print(f"采集启动: {url}")
        print(f"落地目录: {os.path.abspath(self.writer.root)}\n")

        while not self.stop:
            try:
                # proxy=None: 局域网直连，绕过 SOCKS/HTTP 代理
                async with websockets.connect(url, ping_interval=20, proxy=None) as ws:
                    backoff = 1.0  # 连上后重置退避
                    while not self.stop:
                        try:
                            # timeout=5：无数据时也能每 5s 检查一次 stop（收盘可及时退出）
                            msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue
                        recv_ts = int(time.time() * 1000)
                        try:
                            data = json.loads(msg)
                            if isinstance(data, dict):
                                self.handle_message(data, recv_ts)
                        except Exception as e:
                            print(f"  解析失败(跳过): {e}")
            except Exception as e:
                if self.stop:
                    break
                print(f"  连接断开: {e}  {backoff:.0f}s 后重连…")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX)

        periodic_task.cancel()
        self.flush()
        self.writer.close()
        tag = f"（{self.closed_reason}）" if self.closed_reason else ""
        print(f"\n已停止{tag}。累计 snap={self.n_snap} trades={self.n_trades} orders={self.n_orders}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="L2 行情采集器")
    ap.add_argument("--env", default="DEV", choices=list(ENVS), help="环境")
    ap.add_argument("--out", default="../data/collector", help="落地根目录")
    ap.add_argument("--symbols", default=",".join(WATCHLIST), help="逗号分隔的股票代码")
    ap.add_argument("--no-auto-exit", action="store_true",
                    help="关闭收盘自动退出（默认收盘后自动报告并退出）")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    collector = Collector(ENVS[args.env], symbols, args.out,
                          auto_exit=not args.no_auto_exit)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _sigint(*_):
        print("\n收到停止信号，正在落盘…")
        collector.stop = True
    signal.signal(signal.SIGINT, _sigint)

    try:
        loop.run_until_complete(collector.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
