# -*- coding: utf-8 -*-
"""Runtime scheduler service for long-lived API/Web/Desktop processes."""

from __future__ import annotations

import logging
import os
import threading
import _thread
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Set

from src.config import Config, get_config
from src.scheduler import Scheduler, normalize_schedule_times
from src.services.runtime_feature_flags import automation_enabled

logger = logging.getLogger(__name__)
CLI_SCHEDULER_OWNER_ENV = "DSA_CLI_SCHEDULER_OWNS_SCHEDULE"
RUNTIME_SCHEDULER_FORCE_ENABLED_ENV = "DSA_RUNTIME_SCHEDULER_FORCE_ENABLED"
RUNTIME_SCHEDULER_RUN_IMMEDIATELY_ENV = "DSA_RUNTIME_SCHEDULER_RUN_IMMEDIATELY"
RUNTIME_SCHEDULER_SUPPRESS_START_ENV = "DSA_RUNTIME_SCHEDULER_SUPPRESS_START"
RUNTIME_SCHEDULER_ARGS_ENV = "DSA_RUNTIME_SCHEDULER_ARGS"
_RUNTIME_ANALYSIS_LOCK = threading.Lock()
SCHEDULE_ARGS_OVERRIDE_KEYS = {
    "no_notify",
    "no_market_review",
    "dry_run",
    "force_run",
    "single_notify",
    "no_context_snapshot",
    "workers",
}


def run_with_global_analysis_lock(
    task_runner: Callable[[Config, Any, Optional[List[str]]], Any],
    config: Config,
    args: Any,
    stock_codes: Optional[List[str]] = None,
    *,
    blocking: bool = True,
) -> bool:
    """Execute a task while holding the shared runtime analysis lock."""
    if not _RUNTIME_ANALYSIS_LOCK.acquire(blocking=blocking):
        return False
    try:
        task_runner(config, args, stock_codes)
    finally:
        _RUNTIME_ANALYSIS_LOCK.release()
    return True


def _agent_event_monitor_interval_seconds(config: Config) -> int:
    """Return the validated Event Monitor polling interval in seconds."""
    interval_minutes = getattr(config, "agent_event_monitor_interval_minutes", 5)
    try:
        interval_minutes = max(1, int(interval_minutes))
    except (TypeError, ValueError):  # pragma: no cover - defensive branch
        logger.warning(
            "Invalid AGENT_EVENT_MONITOR_INTERVAL_MINUTES=%r; use fallback 5",
            interval_minutes,
        )
        interval_minutes = 5
    return interval_minutes * 60


def build_agent_event_monitor_background_tasks(
    config: Config,
    *,
    config_provider: Callable[[], Config],
) -> List[Dict[str, Any]]:
    """Build scheduler background tasks used by the runtime scheduler."""
    if not getattr(config, "agent_event_monitor_enabled", False):
        return []

    from src.services.alert_worker import AlertWorker

    interval_seconds = _agent_event_monitor_interval_seconds(config)
    try:
        alert_worker = AlertWorker(config_provider=config_provider)
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.warning("Failed to initialize AlertWorker for event monitor: %s", exc)
        return []

    def event_monitor_task() -> None:
        stats = alert_worker.run_once()
        triggered_count = stats.get("triggered", 0)
        if triggered_count:
            logger.info("[EventMonitor] triggered %d alert(s)", triggered_count)

    return [{
        "task": event_monitor_task,
        "interval_seconds": interval_seconds,
        "run_immediately": True,
        "name": "agent_event_monitor",
    }]


_MA_TREND_TASK_NAME = "ma_trend_daily_cycle"


def _schedule_passed(hour: int, minute: int, now: datetime) -> bool:
    """Check if the scheduled time has passed today (catches up after restart)."""
    return now.hour > hour or (now.hour == hour and now.minute >= minute)


def _should_fire(task_name: str, hour: int, minute: int, now: datetime) -> bool:
    """Return True if today is a weekday, the schedule has passed, and not yet run."""
    if now.date().weekday() >= 5:
        return False
    if not _schedule_passed(hour, minute, now):
        return False
    from src.storage import get_db
    if get_db().has_run_today(task_name):
        return False
    return True


def _mark_fired(task_name: str) -> None:
    from src.storage import get_db
    get_db().mark_run_today(task_name)


def build_ma_trend_background_tasks(
    _config: Config,
) -> List[Dict[str, Any]]:
    """Build all MA trend background tasks.

    All tasks use DB-persisted run logs so a process restart does NOT
    prevent the task from firing later (catch-up via _schedule_passed).
    """
    import os
    from zoneinfo import ZoneInfo

    enabled = os.getenv("MA_AGENT_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return []

    schedule_time_str = os.getenv("MA_AGENT_SCHEDULE_TIME", "16:30").strip()
    try:
        schedule_hour, schedule_minute = map(int, schedule_time_str.split(":"))
    except (ValueError, TypeError):
        schedule_hour, schedule_minute = 16, 30

    def ma_trend_task() -> None:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        if not _should_fire(_MA_TREND_TASK_NAME, schedule_hour, schedule_minute, now):
            return
        _mark_fired(_MA_TREND_TASK_NAME)
        logger.info("[MaTrend] 触发每日趋势预测周期: %s", now.isoformat())
        try:
            from src.services.ma_trend_scheduler import run_daily_ma_cycle
            result = run_daily_ma_cycle()
            logger.info("[MaTrend] 每日趋势周期完成: %s", result.get("predict", {}).get("status"))
        except Exception as exc:
            logger.exception("[MaTrend] 每日趋势周期失败: %s", exc)

    # 早盘修正任务（09:25：先拉全球指数，再 LLM 修正昨日预测）
    morning_time_str = os.getenv("MA_AGENT_MORNING_REVISION_TIME", "09:25").strip()
    morning_enabled = os.getenv("MA_AGENT_MORNING_REVISION_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    try:
        morning_hour, morning_minute = map(int, morning_time_str.split(":"))
    except (ValueError, TypeError):
        morning_hour, morning_minute = 9, 25
    _MORNING_TASK_NAME = "ma_trend_morning_revision"

    def morning_revision_task() -> None:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        if not _should_fire(_MORNING_TASK_NAME, morning_hour, morning_minute, now):
            return
        _mark_fired(_MORNING_TASK_NAME)
        logger.info("[MaTrend] 触发早盘修正: %s", now.isoformat())
        try:
            from src.services.global_index_service import fetch_and_save_batch
            fetch_and_save_batch(["SOXX", "NIKKEI", "KOSPI", "HSI", "HSTECH"])
            from src.services.ma_trend_scheduler import run_morning_revision
            result = run_morning_revision()
            logger.info("[MaTrend] 早盘修正完成: revised=%s", result.get("revised", 0))
        except Exception as exc:
            logger.exception("[MaTrend] 早盘修正失败: %s", exc)

    # 午盘预测任务（11:35：用上午行情重跑模型+LLM，预测下午走势）
    midday_enabled = os.getenv("MA_AGENT_MIDDAY_PREDICTION_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    midday_time_str = os.getenv("MA_AGENT_MIDDAY_PREDICTION_TIME", "11:35").strip()
    try:
        midday_hour, midday_minute = map(int, midday_time_str.split(":"))
    except (ValueError, TypeError):
        midday_hour, midday_minute = 11, 35
    _MIDDAY_TASK_NAME = "ma_trend_midday_prediction"

    def midday_prediction_task() -> None:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        if not _should_fire(_MIDDAY_TASK_NAME, midday_hour, midday_minute, now):
            return
        _mark_fired(_MIDDAY_TASK_NAME)
        logger.info("[MaTrend] 触发午盘预测: %s", now.isoformat())
        try:
            from src.services.ma_trend_scheduler import run_daily_ma_agent
            result = run_daily_ma_agent(search_news=True, mode="midday")
            logger.info("[MaTrend] 午盘预测完成: status=%s llm=%s",
                        result.get("status"), result.get("llm_analyses", 0))
        except Exception as exc:
            logger.exception("[MaTrend] 午盘预测失败: %s", exc)

    tasks = [{
        "task": ma_trend_task,
        "interval_seconds": 60,
        "run_immediately": False,
        "name": _MA_TREND_TASK_NAME,
    }]

    if morning_enabled:
        tasks.append({
            "task": morning_revision_task,
            "interval_seconds": 60,
            "run_immediately": False,
            "name": _MORNING_TASK_NAME,
        })

    if midday_enabled:
        tasks.append({
            "task": midday_prediction_task,
            "interval_seconds": 60,
            "run_immediately": False,
            "name": _MIDDAY_TASK_NAME,
        })

    return tasks


_GLOBAL_INDEX_TASK_NAME = "global_index_fetch"


def build_global_index_background_tasks(
    _config: Config,
) -> List[Dict[str, Any]]:
    """Build background tasks that fetch global indices at scheduled times.

    Single time slot (Asia/Shanghai):
    - 09:25  SOXX + NIKKEI + KOSPI + HSI + HSTECH
      （日韩开盘 1.5h、港股开盘 25min、A 股 5min 后开盘，时效最佳）

    Uses DB-persisted run logs so restart doesn't miss the window.
    Controlled by GLOBAL_INDEX_ENABLED env var.
    """
    import os
    from zoneinfo import ZoneInfo

    enabled = os.getenv("GLOBAL_INDEX_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return []

    from src.services.global_index_service import _FETCH_SCHEDULES, fetch_and_save_batch, fetch_and_save_global_index

    def _build_slot_task(slot_time: str, codes: List[str], task_name: str):
        try:
            h, m = map(int, slot_time.split(":"))
        except (ValueError, TypeError):
            h, m = 9, 25

        def _task() -> None:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            if not _should_fire(task_name, h, m, now):
                return
            _mark_fired(task_name)
            logger.info("[全球指数] 触发 %s 拉取: %s", slot_time, codes)
            try:
                if len(codes) == 1:
                    fetch_and_save_global_index(codes[0])
                else:
                    fetch_and_save_batch(codes)
                logger.info("[全球指数] %s 拉取完成", slot_time)
            except Exception as exc:
                logger.exception("[全球指数] %s 拉取失败: %s", slot_time, exc)

        return _task

    tasks = []
    for slot_time, codes in _FETCH_SCHEDULES.items():
        task_name = f"{_GLOBAL_INDEX_TASK_NAME}_{slot_time.replace(':', '')}"
        tasks.append({
            "task": _build_slot_task(slot_time, codes, task_name),
            "interval_seconds": 60,
            "run_immediately": False,
            "name": task_name,
        })

    logger.info("[全球指数] 已注册 %d 个定时拉取任务: %s", len(tasks),
                 [t["name"] for t in tasks])
    return tasks


class RuntimeSchedulerService:
    """Manage scheduled analysis inside the current API/Web/Desktop process."""

    def __init__(
        self,
        *,
        config_provider: Callable[[], Config] = get_config,
        task_runner: Optional[Callable[[Config, Any, Optional[List[str]]], Any]] = None,
        owns_schedule: Optional[bool] = None,
        force_enabled: bool = False,
        run_immediately_in_background: bool = False,
        background_tasks_provider: Optional[Callable[[Config], List[Dict[str, Any]]]] = None,
        schedule_args_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._config_provider = config_provider
        self._task_runner = task_runner
        if owns_schedule is None:
            owns_schedule = os.getenv(CLI_SCHEDULER_OWNER_ENV, "").strip().lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }
        self._owns_schedule = owns_schedule
        self._force_enabled = force_enabled
        self._run_immediately_in_background = run_immediately_in_background
        self._background_tasks_provider = background_tasks_provider
        self._schedule_args_overrides = {
            key: value
            for key, value in (schedule_args_overrides or {}).items()
            if key in SCHEDULE_ARGS_OVERRIDE_KEYS
        }
        self._background_task_cache: Dict[str, Dict[str, Any]] = {}
        self._background_task_registered_names: Set[str] = set()
        self._lock = threading.RLock()
        self._run_lock = _RUNTIME_ANALYSIS_LOCK
        self._scheduler: Optional[Scheduler] = None
        self._thread: Optional[threading.Thread] = None
        self._enabled = False
        self._last_run_at: Optional[str] = None
        self._last_success_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._last_skipped_at: Optional[str] = None
        self._last_skip_reason: Optional[str] = None

    def _make_schedule_args(self) -> SimpleNamespace:
        defaults = {
            "schedule": True,
            "no_run_immediately": True,
            "no_notify": False,
            "no_market_review": False,
            "dry_run": False,
            "force_run": False,
            "single_notify": False,
            "no_context_snapshot": False,
            "market_review": False,
            "serve": False,
            "serve_only": True,
            "stocks": None,
            "workers": None,
        }
        defaults.update(self._schedule_args_overrides)
        return SimpleNamespace(**defaults)

    def _reload_config(self) -> Config:
        from main import _reload_runtime_config

        return _reload_runtime_config()

    def _record_analysis_busy_skip(self) -> None:
        self._last_skipped_at = datetime.now().isoformat()
        self._last_skip_reason = "analysis_already_running"
        logger.warning("Runtime scheduler skipped run: analysis already running")

    def _run_analysis_locked(self, stock_codes: Optional[List[str]]) -> None:
        try:
            config = self._reload_config()
            runner = self._task_runner
            if runner is None:
                from main import run_scheduled_analysis

                runner = run_scheduled_analysis
            self._last_run_at = datetime.now().isoformat()
            result = runner(config, self._make_schedule_args(), stock_codes)
            if result is False:
                raise RuntimeError("runtime scheduled analysis reported failure")
            self._last_success_at = datetime.now().isoformat()
            self._last_error = None
        except Exception as exc:  # noqa: BLE001 - scheduled runs must not kill API process.
            self._last_error = str(exc)
            logger.exception("Runtime scheduled analysis failed: %s", exc)

    def _run_analysis_once(self, stock_codes: Optional[List[str]] = None) -> bool:
        if not self._run_lock.acquire(blocking=False):
            self._record_analysis_busy_skip()
            return False
        try:
            self._run_analysis_locked(stock_codes)
        finally:
            self._run_lock.release()
        return True

    def _current_times(self) -> List[str]:
        config = self._config_provider()
        return normalize_schedule_times(
            getattr(config, "schedule_times", None),
            fallback_time=getattr(config, "schedule_time", "18:00"),
        )

    def _is_schedule_enabled(self, config: Config) -> bool:
        return automation_enabled() and (self._force_enabled or bool(getattr(config, "schedule_enabled", False)))

    def _current_background_tasks(self, config: Config) -> List[Dict[str, Any]]:
        if self._background_tasks_provider is not None:
            tasks = list(self._background_tasks_provider(config))
        else:
            tasks = self._current_agent_event_monitor_background_tasks(config)
        # Always merge MA trend background task (its own enabled check is internal)
        ma_trend_tasks = build_ma_trend_background_tasks(config)
        for entry in ma_trend_tasks:
            if not any(t.get("name") == entry.get("name") for t in tasks):
                tasks.append(entry)
        # Always merge global index fetch tasks (their own enabled check is internal)
        global_idx_tasks = build_global_index_background_tasks(config)
        for entry in global_idx_tasks:
            if not any(t.get("name") == entry.get("name") for t in tasks):
                tasks.append(entry)
        return tasks

    def _current_agent_event_monitor_background_tasks(self, config: Config) -> List[Dict[str, Any]]:
        name = "agent_event_monitor"
        if not getattr(config, "agent_event_monitor_enabled", False):
            self._background_task_cache.pop(name, None)
            self._background_task_registered_names.discard(name)
            return []

        cached = self._background_task_cache.get(name)
        if cached is None:
            entries = build_agent_event_monitor_background_tasks(
                config,
                config_provider=self._reload_config,
            )
            if not entries:
                self._background_task_cache.pop(name, None)
                self._background_task_registered_names.discard(name)
                return []
            cached = dict(entries[0])
            cached["name"] = name
            self._background_task_cache[name] = cached
            interval_seconds = int(cached["interval_seconds"])
        else:
            interval_seconds = _agent_event_monitor_interval_seconds(config)

        run_immediately = (
            bool(cached.get("run_immediately", False))
            and name not in self._background_task_registered_names
        )
        self._background_task_registered_names.add(name)
        return [{
            "task": cached["task"],
            "interval_seconds": interval_seconds,
            "run_immediately": run_immediately,
            "name": name,
        }]

    @staticmethod
    def _run_in_background_thread(target: Callable[[], None]) -> None:
        """Run a callback in a background thread without blocking startup."""
        try:
            _thread.start_new_thread(target, ())
            return
        except Exception:
            # Best-effort fallback for environments where the low-level thread API
            # is unavailable or restricted.
            thread = threading.Thread(target=target, daemon=True)
            thread.start()

    def start(self, *, run_immediately: bool = False) -> None:
        with self._lock:
            if not self._owns_schedule:
                self.stop()
                return
            config = self._config_provider()
            if not self._is_schedule_enabled(config):
                self.stop()
                return
            background_tasks = self._current_background_tasks(config)
            self.stop()
            times = normalize_schedule_times(
                getattr(config, "schedule_times", None),
                fallback_time=getattr(config, "schedule_time", "18:00"),
            )
            scheduler = Scheduler(
                schedule_time=getattr(config, "schedule_time", "18:00"),
                schedule_times=times,
                schedule_times_provider=self._current_times,
                register_signals=False,
            )
            if run_immediately and self._run_immediately_in_background:
                scheduler.set_daily_task(self._run_analysis_once, run_immediately=False)
            else:
                scheduler.set_daily_task(self._run_analysis_once, run_immediately=run_immediately)
            for entry in background_tasks:
                scheduler.add_background_task(
                    entry["task"],
                    interval_seconds=entry["interval_seconds"],
                    run_immediately=entry.get("run_immediately", False),
                    name=entry.get("name"),
                )
            if run_immediately and self._run_immediately_in_background:
                self._run_in_background_thread(self._run_analysis_once)
            thread = threading.Thread(
                target=scheduler.run,
                daemon=True,
                name="runtime-scheduler",
            )
            self._scheduler = scheduler
            self._thread = thread
            self._enabled = True
            thread.start()

    def stop(self) -> None:
        scheduler = self._scheduler
        if scheduler is not None:
            scheduler.stop()
        self._scheduler = None
        self._thread = None
        self._enabled = False

    def reconcile_from_config(
        self,
        *,
        run_immediately: bool = False,
        clear_enabled_override: bool = False,
    ) -> None:
        if clear_enabled_override:
            self._force_enabled = False
        if not self._owns_schedule:
            self.stop()
            return
        config = self._config_provider()
        if self._is_schedule_enabled(config):
            self.start(run_immediately=run_immediately)
        else:
            self.stop()

    def run_now(self) -> Dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            self._record_analysis_busy_skip()
            return {
                "accepted": False,
                "running": True,
                "reason": "analysis_already_running",
            }

        def run_and_release() -> None:
            try:
                self._run_analysis_locked(None)
            finally:
                self._run_lock.release()

        worker = threading.Thread(
            target=run_and_release,
            daemon=True,
            name="runtime-scheduler-run-now",
        )
        try:
            worker.start()
        except Exception:
            self._run_lock.release()
            raise
        return {"accepted": True, "running": True}

    def status(self) -> Dict[str, Any]:
        scheduler = self._scheduler
        jobs = scheduler.schedule.get_jobs() if scheduler is not None else []
        next_run = None
        if jobs:
            next_run = min(job.next_run for job in jobs).isoformat()
        if scheduler is not None:
            schedule_times = list(getattr(scheduler, "schedule_times", []))
        else:
            try:
                schedule_times = self._current_times()
            except Exception:  # pragma: no cover - defensive status fallback
                schedule_times = []
        running = self._run_lock.locked()
        return {
            "enabled": self._enabled,
            "running": running,
            "schedule_times": schedule_times,
            "next_run_at": next_run,
            "last_run_at": self._last_run_at,
            "last_success_at": self._last_success_at,
            "last_error": self._last_error,
            "last_skipped_at": self._last_skipped_at,
            "last_skip_reason": self._last_skip_reason,
        }
