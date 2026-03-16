"""Main entry point for the On-Chain Autonomous Trader."""

import argparse
import asyncio
import signal
import sys

import structlog

from bot.config import Settings, TradingMode, load_settings
from bot.dashboard import app as dashboard_module
from bot.dashboard.websocket import ws_manager
from bot.data.store import DataStore
from bot.engines.manager import EngineManager
from bot.engines.onchain_trader import OnChainTraderEngine
from bot.engines.portfolio_manager import PortfolioManager
from bot.exchanges.factory import ExchangeFactory
from bot.exchanges.rate_limiter import DEFAULT_EXCHANGE_LIMITS, RateLimiter
from bot.execution.preflight import PreFlightChecker
from bot.execution.resilient import ResilientExchange
from bot.monitoring.audit import AuditLogger
from bot.monitoring.logger import setup_logging
from bot.monitoring.telegram import TelegramNotifier
from bot.risk.manager import RiskManager

logger = structlog.get_logger()


class TradingBot:
    """Main orchestrator for the on-chain autonomous trader."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or load_settings()
        self._running = False
        self._exchanges = []
        self._store: DataStore | None = None
        self._risk_manager: RiskManager | None = None
        self._telegram: TelegramNotifier | None = None
        self._dashboard_task: asyncio.Task | None = None
        self._preflight_checker: PreFlightChecker | None = None
        self._audit_logger: AuditLogger = AuditLogger()
        self._portfolio_mgr: PortfolioManager | None = None
        self._engine_manager: EngineManager | None = None
        # Emergency stop state
        self._emergency_stopped: bool = False
        self._emergency_stopped_at: str | None = None
        self._emergency_reason: str | None = None

    async def initialize(self) -> None:
        """Initialize all bot components."""
        setup_logging(self._settings.log_level)
        logger.info("bot_initializing", mode=self._settings.trading_mode.value)

        # Initialize data store
        self._store = DataStore(database_url=self._settings.database_url)
        await self._store.initialize()

        # Connect audit logger to data store
        self._audit_logger.store = self._store

        # Create Binance spot exchange adapter
        self._init_exchanges()

        # Initialize risk manager
        self._risk_manager = RiskManager(
            max_position_size_pct=self._settings.max_position_size_pct,
            stop_loss_pct=self._settings.stop_loss_pct,
            daily_loss_limit_pct=self._settings.daily_loss_limit_pct,
            max_drawdown_pct=self._settings.max_drawdown_pct,
            max_concurrent_positions=self._settings.max_concurrent_positions,
        )

        # Run pre-flight checks for live trading mode
        if self._settings.trading_mode == TradingMode.LIVE:
            await self._run_preflight_checks()

        # Initialize Telegram notifier
        self._init_telegram()

        # Start dashboard server
        self._start_dashboard()

        # Provide settings, bot ref, and audit logger to dashboard
        dashboard_module.set_settings(self._settings)
        dashboard_module.set_trading_bot(self)
        dashboard_module.set_audit_logger(self._audit_logger)
        if self._store:
            dashboard_module.set_store_ref(self._store)

        # Initialize engine mode (always on)
        self._init_engine_mode()

        # Connect engine manager to dashboard
        if self._engine_manager:
            dashboard_module.set_engine_manager(self._engine_manager)

        logger.info(
            "bot_initialized",
            exchanges=[e.name for e in self._exchanges],
            symbols=self._settings.onchain_symbols,
        )

        # Notify startup via Telegram
        if self._telegram:
            await self._telegram.send_message(
                f"Bot started in {self._settings.trading_mode.value} mode. "
                f"Symbols: {', '.join(self._settings.onchain_symbols)}"
            )
            self._register_telegram_commands()
            await self._telegram.start_command_polling()

        # Audit log startup
        await self._audit_logger.log_bot_started(
            mode=self._settings.trading_mode.value,
            symbols=self._settings.onchain_symbols,
        )

    def _init_exchanges(self) -> None:
        """Create Binance spot adapter wrapped in ResilientExchange."""
        import bot.exchanges.binance  # noqa: F401

        if self._settings.binance_api_key:
            try:
                adapter = ExchangeFactory.create(
                    "binance",
                    api_key=self._settings.binance_api_key,
                    secret_key=self._settings.binance_secret_key,
                    testnet=self._settings.binance_testnet,
                )
                limiter = self._create_rate_limiter("binance")
                self._exchanges.append(ResilientExchange(
                    adapter,
                    rate_limiter=limiter,
                    rate_limit_enabled=self._settings.rate_limit_enabled,
                ))
            except ValueError:
                logger.warning("binance_adapter_not_available")

    def _create_rate_limiter(self, exchange_name: str) -> RateLimiter | None:
        if not self._settings.rate_limit_enabled:
            return None
        overrides = self._settings.exchange_rate_limits.get(exchange_name, {})
        defaults = DEFAULT_EXCHANGE_LIMITS.get(
            exchange_name, {"requests_per_second": 10.0, "burst_size": 20}
        )
        rps = overrides.get("requests_per_second", defaults["requests_per_second"])
        burst = int(overrides.get("burst_size", defaults["burst_size"]))
        return RateLimiter(
            max_requests_per_second=rps,
            burst_size=burst,
            name=exchange_name,
        )

    def _init_telegram(self) -> None:
        if self._settings.telegram_bot_token and self._settings.telegram_chat_id:
            self._telegram = TelegramNotifier(
                bot_token=self._settings.telegram_bot_token,
                chat_id=self._settings.telegram_chat_id,
            )
            logger.info("telegram_notifier_initialized")

    def _register_telegram_commands(self) -> None:
        if not self._telegram:
            return

        async def cmd_stop() -> str:
            result = await self.emergency_stop(reason="telegram_command")
            return (
                f"EMERGENCY STOP activated.\n"
                f"Cancelled orders: {result['cancelled_orders']}"
            )

        async def cmd_resume() -> str:
            result = await self.emergency_resume()
            if result["success"]:
                return "Trading RESUMED."
            return f"Cannot resume: {result.get('error', 'unknown')}"

        async def cmd_status() -> str:
            status = dashboard_module.get_state().get("status", "unknown")
            emergency = self.emergency_state
            portfolio = dashboard_module.get_state().get("portfolio", {})
            total_value = portfolio.get("total_value", 0)

            lines = [
                f"Bot Status: {status}",
                f"Emergency: {'ACTIVE' if emergency['active'] else 'inactive'}",
                f"Portfolio: ${total_value:,.2f}",
            ]
            if self._engine_manager:
                for name, engine in self._engine_manager.engines.items():
                    lines.append(
                        f"  {name}: {engine.status.value} | "
                        f"positions={engine.position_count} | "
                        f"PnL={engine.total_pnl:+.2f}"
                    )
            return "\n".join(lines)

        self._telegram.register_command("stop", cmd_stop)
        self._telegram.register_command("resume", cmd_resume)
        self._telegram.register_command("status", cmd_status)

    def _start_dashboard(self) -> None:
        try:
            import uvicorn

            config = uvicorn.Config(
                app=dashboard_module.app,
                host="0.0.0.0",
                port=self._settings.dashboard_port,
                log_level="warning",
            )
            server = uvicorn.Server(config)

            async def _serve_safe() -> None:
                try:
                    await server.serve()
                except SystemExit:
                    logger.warning("dashboard_server_exited")
                except Exception as e:
                    logger.warning("dashboard_server_error", error=str(e))

            self._dashboard_task = asyncio.ensure_future(_serve_safe())
            dashboard_module.update_state(status="running")
            logger.info("dashboard_started", port=self._settings.dashboard_port)
        except ImportError:
            logger.warning("uvicorn_not_installed_dashboard_disabled")
        except Exception as e:
            logger.warning("dashboard_start_failed", error=str(e))

    def _init_engine_mode(self) -> None:
        """Initialize the OnChainTrader engine."""
        capital = self._settings.engine_total_capital
        is_paper = self._settings.trading_mode == TradingMode.PAPER
        if is_paper:
            capital = self._settings.paper_initial_balance

        self._portfolio_mgr = PortfolioManager(
            total_capital=capital,
            engine_allocations=self._settings.engine_allocations,
            max_drawdown_pct=self._settings.engine_max_drawdown_pct,
        )
        self._engine_manager = EngineManager(
            self._portfolio_mgr, settings=self._settings
        )

        # Register OnChainTraderEngine
        engine = OnChainTraderEngine(
            portfolio_manager=self._portfolio_mgr,
            exchanges=self._exchanges,
            paper_mode=is_paper,
            settings=self._settings,
        )
        self._engine_manager.register(engine)

        # Wire DataStore and Telegram to engine
        if self._store is not None:
            engine.set_store(self._store)
        if self._telegram is not None:
            engine.set_telegram(self._telegram)

        # Register cycle-complete callback for WebSocket broadcast
        async def _broadcast_engine_cycle(result):
            try:
                await ws_manager.broadcast(
                    {"type": "engine_cycle", "payload": result.to_dict()}
                )
            except Exception:
                logger.debug("engine_cycle_broadcast_error", exc_info=True)

        engine.set_on_cycle_complete(_broadcast_engine_cycle)

        # Wire MetricsPersistence
        if (
            getattr(self._settings, "metrics_persistence_enabled", True)
            and self._store is not None
        ):
            from bot.engines.metrics_persistence import MetricsPersistence

            persistence = MetricsPersistence(
                data_store=self._store,
                tracker=self._engine_manager.tracker,
            )
            self._engine_manager.set_metrics_persistence(persistence)

        # Wire MarketRegimeDetector
        if getattr(self._settings, "regime_detection_enabled", True):
            from bot.risk.regime_detector import MarketRegimeDetector

            crisis_thresh = getattr(
                self._settings, "regime_crisis_threshold", 2.5,
            )
            detector = MarketRegimeDetector(
                volatility_service=None,
                crisis_threshold=crisis_thresh,
            )
            self._engine_manager.set_regime_detector(detector)
            engine.set_regime_detector(detector)

        logger.info(
            "engine_mode_initialized",
            engines=list(self._engine_manager.engines.keys()),
            total_capital=capital,
        )

    async def _run_engine_mode(self) -> None:
        """Run the engine system — each engine runs its own async loop."""
        if not self._engine_manager:
            logger.error("engine_manager_not_initialized")
            return

        self._running = True
        logger.info("engine_mode_started")

        await self._engine_manager.start_all()
        await self._engine_manager.start_background_loops()

        try:
            while self._running:
                if self._emergency_stopped:
                    await asyncio.sleep(1)
                    continue

                # Global drawdown check
                if self._portfolio_mgr and self._portfolio_mgr.is_drawdown_breached():
                    await self.emergency_stop(reason="global_drawdown_breached")
                    continue

                # Update dashboard with engine status + portfolio
                if self._engine_manager:
                    engine_status = self._engine_manager.get_status()
                    pm_summary = (
                        self._portfolio_mgr.get_summary()
                        if self._portfolio_mgr
                        else {}
                    )

                    # Build portfolio from exchange balances + engine positions
                    portfolio_data = dashboard_module.get_state()["portfolio"]
                    try:
                        balances = {}

                        def _get_ccxt(ex):
                            obj = ex
                            for _ in range(5):
                                inner = getattr(obj, '_exchange', None)
                                if inner is None:
                                    return None
                                if hasattr(inner, 'fetch_balance'):
                                    return inner
                                obj = inner
                            return None

                        seen_ccxt: set[int] = set()
                        all_exchanges = list(self._exchanges)
                        for engine in self._engine_manager.engines.values():
                            for ex in getattr(engine, '_exchanges', []):
                                all_exchanges.append(ex)
                        for ex in all_exchanges:
                            ccxt_ex = _get_ccxt(ex)
                            if ccxt_ex is None:
                                continue
                            if id(ccxt_ex) in seen_ccxt:
                                continue
                            seen_ccxt.add(id(ccxt_ex))
                            try:
                                raw = await ccxt_ex.fetch_balance()
                                for k, v in raw.get("total", {}).items():
                                    fv = float(v)
                                    if fv > 0:
                                        balances[k] = balances.get(k, 0) + fv
                            except Exception:
                                pass
                        cash_value = sum(
                            v for k, v in balances.items()
                            if k in ("USDT", "USD", "BUSD", "USDC")
                        )
                        total_upnl = 0.0
                        for engine in self._engine_manager.engines.values():
                            for pos in engine.positions.values():
                                total_upnl += pos.get("unrealized_pnl", 0.0)
                        total_value = cash_value + total_upnl
                        portfolio_data = {
                            "balances": balances,
                            "total_value": round(total_value, 2),
                            "cash_value": round(cash_value, 2),
                            "unrealized_pnl": round(total_upnl, 4),
                            "positions": [],
                        }
                    except Exception:
                        logger.debug("portfolio_update_error", exc_info=True)

                    # Get onchain signals for dashboard
                    onchain_engine = self._engine_manager.get_engine("onchain_trader")
                    onchain_signals = {}
                    if onchain_engine and hasattr(onchain_engine, "latest_signals"):
                        onchain_signals = onchain_engine.latest_signals

                    dashboard_module.update_state(
                        status="running",
                        engine_status=engine_status,
                        portfolio_manager=pm_summary,
                        portfolio=portfolio_data,
                        onchain_signals=onchain_signals,
                    )

                await asyncio.sleep(5)
        finally:
            await self._engine_manager.stop_all()

    async def run(self) -> None:
        """Run the bot (engine mode)."""
        await self._run_engine_mode()

    async def shutdown(self) -> None:
        """Graceful shutdown with timeout."""
        self._running = False
        timeout = getattr(self._settings, "shutdown_timeout_seconds", 30.0)

        if self._engine_manager:
            try:
                await asyncio.wait_for(
                    self._engine_manager.stop_all(), timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("engine_stop_timeout", timeout=timeout)
            except Exception:
                pass

        for exchange in self._exchanges:
            try:
                await exchange.close()
            except Exception:
                pass
        if self._telegram:
            try:
                await self._telegram.send_message("Bot shutting down.")
            except Exception:
                pass
        logger.info("bot_shutdown_complete")

    # ------------------------------------------------------------------
    # Emergency controls
    # ------------------------------------------------------------------

    @property
    def emergency_state(self) -> dict:
        return {
            "active": self._emergency_stopped,
            "activated_at": self._emergency_stopped_at,
            "reason": self._emergency_reason,
        }

    async def emergency_stop(self, reason: str = "manual") -> dict:
        from datetime import datetime, timezone
        self._emergency_stopped = True
        self._emergency_stopped_at = datetime.now(timezone.utc).isoformat()
        self._emergency_reason = reason

        cancelled = 0
        if self._engine_manager:
            for engine in self._engine_manager.engines.values():
                await engine.pause()

        dashboard_module.update_state(
            emergency=self.emergency_state,
            status="emergency_stopped",
        )

        if self._telegram:
            await self._telegram.send_message(
                f"EMERGENCY STOP: {reason}"
            )

        await self._audit_logger.log_emergency_stop(
            reason=reason,
            cancelled_orders=cancelled,
        )

        logger.critical("emergency_stop", reason=reason)
        return {"cancelled_orders": cancelled}

    async def emergency_resume(self) -> dict:
        if not self._emergency_stopped:
            return {"success": False, "error": "not_stopped"}

        self._emergency_stopped = False
        self._emergency_stopped_at = None
        self._emergency_reason = None

        if self._engine_manager:
            for engine in self._engine_manager.engines.values():
                await engine.resume()

        dashboard_module.update_state(
            emergency=self.emergency_state,
            status="running",
        )

        await self._audit_logger.log_emergency_resume(
            previous_reason="manual_resume",
        )

        logger.info("emergency_resumed")
        return {"success": True}

    async def emergency_close_all(self, reason: str = "manual") -> dict:
        """Close all open positions across all engines."""
        closed = []
        if self._engine_manager:
            for engine in self._engine_manager.engines.values():
                for symbol in list(engine.positions.keys()):
                    pos = engine.positions.get(symbol)
                    if pos:
                        closed.append({
                            "symbol": symbol,
                            "quantity": pos.get("quantity", 0),
                            "pnl": 0.0,
                        })
                        engine._remove_position(symbol)

        await self.emergency_stop(reason=f"close_all:{reason}")
        return {"closed_positions": closed}

    async def _run_preflight_checks(self) -> None:
        """Run pre-flight checks for live trading."""
        if not self._exchanges:
            logger.warning("preflight_no_exchanges")
            return

        self._preflight_checker = PreFlightChecker(
            exchanges=self._exchanges,
            symbols=self._settings.onchain_symbols,
            settings=self._settings,
        )
        results = await self._preflight_checker.run_all()
        dashboard_module.update_state(preflight=results)

        failures = [r for r in results if not r.get("passed", True)]
        if failures:
            logger.warning("preflight_failures", failures=failures)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="On-Chain Autonomous Trader")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=None,
        help="Trading mode (overrides env var)",
    )
    args = parser.parse_args()

    overrides = {}
    if args.mode:
        overrides["trading_mode"] = args.mode

    settings = load_settings(**overrides)
    # Always enable engine mode
    settings = load_settings(engine_mode=True, **overrides)

    bot = TradingBot(settings)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        await bot.initialize()
        await bot.run()

    def _signal_handler(sig, frame):
        logger.info("shutdown_signal_received", signal=sig)
        bot._running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt")
    finally:
        loop.run_until_complete(bot.shutdown())
        loop.close()


if __name__ == "__main__":
    main()
