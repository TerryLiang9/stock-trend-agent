"""原始行情录制器 — 把实时 WS 数据「完整、无损」存下来，供回放/审计/重算因子。

与 collector.py 的区别：
  - collector.py 把消息**拆成字段**写 CSV（snapshot/trades/orders），是有损的、schema 固定。
  - 本文件把**整条消息原样**存成 NDJSON（每行一个 JSON），无损、无 schema、可回放。

为什么选 NDJSON + gzip（最适合原始流式行情的数据结构）：
  1. 结构匹配：消息是嵌套+变长（十档 + 若干条逐笔），天然是"每条一个 JSON 文档"，
     不是二维表。JSONL 原样保真，新增字段也不用改代码。
  2. 流式友好：一条一行、追加即写，天然适合实时流；逐行读取即可回放。
  3. 高压缩比：逐笔是滚动窗口、相邻消息大量重复，gzip 常压到 1/10 以下。
  每行是一个「信封」：{"recv_ts": 本地接收毫秒, "data": 原始消息对象}
  （recv_ts 用于和消息内 time 做延迟分析、以及精确回放。）

落地结构：
  raw/<YYYYMMDD>/raw_<YYYYMMDD>.jsonl.gz      （--rotate hour 时按小时切：raw_<YYYYMMDD>_<HH>.jsonl.gz）

用法：
  python raw_recorder.py                       # 录制默认 watchlist 到 ./raw
  python raw_recorder.py --env DEV --rotate hour
  python raw_recorder.py --no-gzip             # 存明文 .jsonl（便于直接查看）

回放/读取：
  from raw_recorder import iter_records
  for rec in iter_records("raw/20260702/raw_20260702.jsonl.gz"):
      recv_ts, data = rec["recv_ts"], rec["data"]

需要：pip install websockets
"""

import argparse
import asyncio
import gzip
import json
import os
import signal
import time
from datetime import datetime

try:
    import websockets
except ImportError:
    print("请安装: pip install websockets")
    raise SystemExit(1)

import collector_auto_exit as col   # 复用 ENVS / WATCHLIST / BEIJING / 收盘判断
#import collector as col   # 复用 ENVS / WATCHLIST / BEIJING / 收盘判断

FLUSH_INTERVAL = 2.0
STATS_INTERVAL = 30.0

# ---------------------------------------------------------------------------
# 读取器（回放用）
# ---------------------------------------------------------------------------
def iter_records(path: str):
    """逐行读取 NDJSON（自动识别 .gz），yield 每条 {"recv_ts":..., "data":...}。"""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# 录制器
# ---------------------------------------------------------------------------
class RawRecorder:
    def __init__(self, uri, symbols, out_dir, use_gzip=True, rotate="day", auto_exit=True):
        self.uri = uri
        self.symbols = symbols
        self.out_dir = out_dir
        self.use_gzip = use_gzip
        self.rotate = rotate           # "day" | "hour"
        self.auto_exit = auto_exit
        self.stop = False
        self.closed_reason = None

        self._fh = None                # 当前文件句柄
        self._key = None               # 当前文件的 (date[,hour]) 键
        self._buf = []                 # 待写缓冲（原始行文本）
        self.n = 0                     # 累计记录数
        self.n_bytes = 0               # 累计原始字节（未压缩）

    # --- 文件切换（按日/小时）---
    def _target(self):
        now = datetime.now(col.BEIJING)
        date = now.strftime("%Y%m%d")
        if self.rotate == "hour":
            key = (date, now.strftime("%H"))
            name = f"raw_{date}_{key[1]}.jsonl"
        else:
            key = (date,)
            name = f"raw_{date}.jsonl"
        if self.use_gzip:
            name += ".gz"
        d = os.path.join(self.out_dir, date)
        return key, os.path.join(d, name)

    def _ensure_file(self):
        key, path = self._target()
        if key != self._key:
            self._close_file()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # 追加模式：gzip 追加会生成多成员 gzip，读取时可正常拼接
            self._fh = (gzip.open(path, "at", encoding="utf-8") if self.use_gzip
                        else open(path, "a", encoding="utf-8"))
            self._key = key
            print(f"[{datetime.now():%H:%M:%S}] 写入文件: {os.path.abspath(path)}")
        return self._fh

    def _close_file(self):
        if self._fh:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    # --- 处理一条消息：原样入库 ---
    def handle(self, msg_text: str, recv_ts: int):
        try:
            data = json.loads(msg_text)          # 解析以校验+统一序列化（float 精度无损）
            line = json.dumps({"recv_ts": recv_ts, "data": data}, ensure_ascii=False)
        except Exception:
            # 非 JSON：仍无损保留原文
            line = json.dumps({"recv_ts": recv_ts, "raw": msg_text}, ensure_ascii=False)
        self._buf.append(line)
        self.n += 1
        self.n_bytes += len(line) + 1

    def flush(self):
        if not self._buf:
            return
        fh = self._ensure_file()
        fh.write("\n".join(self._buf) + "\n")
        fh.flush()
        self._buf.clear()

    async def _periodic(self):
        last_stats = time.time()
        while not self.stop:
            await asyncio.sleep(FLUSH_INTERVAL)
            self.flush()
            if self.auto_exit:
                closed, reason = col.market_session_over(datetime.now(col.BEIJING))
                if closed:
                    self.closed_reason = reason
                    print(f"\n检测到收盘：{reason}，录制退出。")
                    self.stop = True
                    break
            now = time.time()
            if now - last_stats >= STATS_INTERVAL:
                mb = self.n_bytes / 1e6
                print(f"[{datetime.now():%H:%M:%S}] 记录={self.n} 原始≈{mb:.1f}MB"
                      f"{'（gzip压缩后更小）' if self.use_gzip else ''}")
                last_stats = now

    async def run(self):
        periodic = asyncio.create_task(self._periodic())
        url = f"{self.uri}?symbols={','.join(self.symbols)}"
        print(f"原始录制启动: {url}")
        print(f"落地目录: {os.path.abspath(self.out_dir)}  格式: "
              f"{'NDJSON.gz' if self.use_gzip else 'NDJSON'}  切分: 按{self.rotate}\n")
        backoff = 1.0
        while not self.stop:
            try:
                async with websockets.connect(url, ping_interval=20, proxy=None) as ws:
                    backoff = 1.0
                    while not self.stop:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue
                        self.handle(msg, int(time.time() * 1000))
            except Exception as e:
                if self.stop:
                    break
                print(f"  连接断开: {e}  {backoff:.0f}s 后重连…")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, col.RECONNECT_MAX)
        periodic.cancel()
        self.flush()
        #print("jiluzhong")
        self._close_file()
        tag = f"（{self.closed_reason}）" if self.closed_reason else ""
        print(f"\n已停止{tag}。累计记录 {self.n} 条。")


def main():
    ap = argparse.ArgumentParser(description="原始行情录制器（完整无损 NDJSON）")
    ap.add_argument("--env", default="DEV", choices=list(col.ENVS))
    ap.add_argument("--out", default="../data/raw", help="落地根目录")
    ap.add_argument("--symbols", default=",".join(col.WATCHLIST))
    ap.add_argument("--rotate", default="day", choices=["day", "hour"], help="文件切分粒度")
    ap.add_argument("--no-gzip", action="store_true", help="存明文 NDJSON，不压缩")
    ap.add_argument("--no-auto-exit", action="store_true", help="收盘不自动退出")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    rec = RawRecorder(col.ENVS[args.env], symbols, args.out,
                      use_gzip=not args.no_gzip, rotate=args.rotate,
                      auto_exit=not args.no_auto_exit)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _sigint(*_):
        print("\n收到停止信号，正在落盘…")
        rec.stop = True
    signal.signal(signal.SIGINT, _sigint)

    try:
        loop.run_until_complete(rec.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
