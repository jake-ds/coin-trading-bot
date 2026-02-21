"""Main entry point for the Coin Trading Bot."""

import asyncio
import signal
import sys
import time

import structlog

from bot.config import Settings, TradingMode, load_settings
from bot.dashboard import app as dashboard_module
from bot.data.collector import DataCollector
from bot.data.store import DataStore
from bot.exchanges.factory import ExchangeFactory
from bot.execution.engine import ExecutionEngine
from bot.execution.paper_portfolio import PaperPortfolio
from bot.execution.position_manager import ExitType, PositionManager
from bot.execution.resilient import ResilientExchange
from bot.models import OrderSide, SignalAction, TradingSignal
from bot.monitoring.logger import setup_logging
from bot.monitoring.telegram import TelegramNotifier
from bot.risk.manager import RiskManager
from bot.strategies.base import strategy_registry
from bot.strategies.ensemble import SignalEnsemble
from bot.strategies.indicators import calculate_atr
from bot.strategies.regime import MarketRegimeDetector
from bot.strategies.trend_filter import TrendFilter

logger = structlog.get_logger()


class TradingBot:
    """Main orchestrator that ties all components together."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or load_settings()
        self._running = False
        self._exchanges = []
        self._store: DataStore | None = None
        self._collector: DataCollector | None = None
        self._risk_manager: RiskManager | None = None
        self._execution_engines: dict[str, ExecutionEngine] = {}
        self._telegram: TelegramNotifier | None = None
        self._dashboard_task: asyncio.Task | None = None
        self._paper_portfolio: PaperPortfolio | None = None
        self._position_manager: PositionManager | None = None
        self._signal_ensemble: SignalEnsemble | None = None
        self._trend_filter: TrendFilter | None = None
        self._regime_detector: MarketRegimeDetector | None = None
        self._cycle_lock: asyncio.Lock = asyncio.Lock()
        self._cycle_count: int = 0
        self._total_cycle_duration: float = 0.0
        self._last_cycle_time: float | None = None

    async def initialize(self) -> None:
        """Initialize all bot components."""
        setup_logging(self._settings.log_level)
        logger.info("bot_initializing", mode=self._settings.trading_mode.value)

        # Initialize data store
        self._store = DataStore(database_url=self._settings.database_url)
        await self._store.initialize()

        # Create exchange adapters (wrapped in ResilientExchange)
        self._init_exchanges()

        # Initialize data collector (collect all configured timeframes)
        self._collector = DataCollector(
            exchanges=self._exchanges,
            store=self._store,
            symbols=self._settings.symbols,
            timeframes=self._settings.timeframes,
            collection_interval=self._settings.loop_interval_seconds,
        )

        # Initialize risk manager
        self._risk_manager = RiskManager(
            max_position_size_pct=self._settings.max_position_size_pct,
            stop_loss_pct=self._settings.stop_loss_pct,
            daily_loss_limit_pct=self._settings.daily_loss_limit_pct,
            max_drawdown_pct=self._settings.max_drawdown_pct,
            max_concurrent_positions=self._settings.max_concurrent_positions,
        )

        # Initialize signal ensemble voting system
        self._signal_ensemble = SignalEnsemble(
            min_agreement=self._settings.signal_min_agreement,
            strategy_weights=self._settings.strategy_weights,
        )

        # Initialize trend filter for higher-timeframe confirmation
        self._trend_filter = TrendFilter()

        # Initialize market regime detector
        self._regime_detector = MarketRegimeDetector()

        # Initialize position manager (stop-loss, take-profit, trailing stop)
        self._position_manager = PositionManager(
            stop_loss_pct=self._settings.stop_loss_pct,
            take_profit_pct=self._settings.take_profit_pct,
            trailing_stop_enabled=self._settings.trailing_stop_enabled,
            trailing_stop_pct=self._settings.trailing_stop_pct,
        )

        # Initialize execution engines (one per exchange)
        is_paper = self._settings.trading_mode == TradingMode.PAPER
        if is_paper:
            self._paper_portfolio = PaperPortfolio(
                initial_balance=self._settings.paper_initial_balance,
                fee_pct=self._settings.paper_fee_pct,
            )
        for exchange in self._exchanges:
            self._execution_engines[exchange.name] = ExecutionEngine(
                exchange=exchange,
                store=self._store,
                paper_trading=is_paper,
                paper_portfolio=self._paper_portfolio,
            )

        # Initialize Telegram notifier (gracefully skip if not configured)
        self._init_telegram()

        # Start dashboard server as background task
        self._start_dashboard()

        # Import strategies to trigger registration
        self._load_strategies()

        logger.info(
            "bot_initialized",
            exchanges=[e.name for e in self._exchanges],
            strategies=[s.name for s in strategy_registry.get_active()],
            symbols=self._settings.symbols,
        )

        # Notify startup via Telegram
        if self._telegram:
            await self._telegram.send_message(
                f"Bot started in {self._settings.trading_mode.value} mode. "
                f"Symbols: {', '.join(self._settings.symbols)}"
            )

    def _init_exchanges(self) -> None:
        """Create exchange adapters from configuration, wrapped in ResilientExchange."""
        # Import to trigger registration
        import bot.exchanges.binance  # noqa: F401
        import bot.exchanges.upbit  # noqa: F401

        if self._settings.binance_api_key:
            try:
                adapter = ExchangeFactory.create(
                    "binance",
                    api_key=self._settings.binance_api_key,
                    secret_key=self._settings.binance_secret_key,
                    testnet=self._settings.binance_testnet,
                )
                self._exchanges.append(ResilientExchange(adapter))
            except ValueError:
                logger.warning("binance_adapter_not_available")

        if self._settings.upbit_api_key:
            try:
                adapter = ExchangeFactory.create(
                    "upbit",
                    api_key=self._settings.upbit_api_key,
                    secret_key=self._settings.upbit_secret_key,
                )
                self._exchanges.append(ResilientExchange(adapter))
            except ValueError:
                logger.warning("upbit_adapter_not_available")

    def _init_telegram(self) -> None:
        """Initialize Telegram notifier if bot_token and chat_id are configured."""
        if self._settings.telegram_bot_token and self._settings.telegram_chat_id:
            self._telegram = TelegramNotifier(
                bot_token=self._settings.telegram_bot_token,
                chat_id=self._settings.telegram_chat_id,
            )
            logger.info("telegram_notifier_initialized")
        else:
            logger.debug("telegram_not_configured_skipping")

    def _start_dashboard(self) -> None:
        """Start the dashboard uvicorn server as a background asyncio task."""
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

    def _load_strategies(self) -> None:
        """Import strategy modules to trigger auto-registration."""
        try:
            import bot.strategies.arbitrage.arbitrage_strategy  # noqa: F401
            import bot.strategies.dca.dca_strategy  # noqa: F401
            import bot.strategies.technical.bollinger  # noqa: F401
            import bot.strategies.technical.ma_crossover  # noqa: F401
            import bot.strategies.technical.macd  # noqa: F401
            import bot.strategies.technical.rsi  # noqa: F401
        except ImportError as e:
            logger.warning("strategy_import_error", error=str(e))

    @property
    def cycle_metrics(self) -> dict:
        """Return current cycle metrics."""
        avg_duration = (
            self._total_cycle_duration / self._cycle_count
            if self._cycle_count > 0
            else 0.0
        )
        return {
            "cycle_count": self._cycle_count,
            "average_cycle_duration": round(avg_duration, 4),
            "last_cycle_time": self._last_cycle_time,
        }

    async def run_trading_loop(self) -> None:
        """Run the main trading loop."""
        self._running = True
        logger.info("trading_loop_started")

        while self._running:
            if self._cycle_lock.locked():
                logger.warning(
                    "trading_cycle_skipped_overlap",
                    reason="previous cycle still running",
                )
            else:
                try:
                    async with self._cycle_lock:
                        start_time = time.monotonic()
                        await self._trading_cycle()
                        duration = time.monotonic() - start_time
                        self._cycle_count += 1
                        self._total_cycle_duration += duration
                        self._last_cycle_time = time.time()
                        # Update dashboard with latest cycle metrics
                        dashboard_module.update_state(
                            cycle_metrics=self.cycle_metrics,
                        )
                        logger.debug(
                            "trading_cycle_completed",
                            cycle_count=self._cycle_count,
                            duration=round(duration, 4),
                        )
                except Exception:
                    logger.error("trading_cycle_error", exc_info=True)
                    if self._telegram:
                        import traceback

                        tb = traceback.format_exc()
                        await self._telegram.notify_error(tb)

            await asyncio.sleep(self._settings.loop_interval_seconds)

    async def _trading_cycle(self) -> None:
        """Execute one trading cycle: check exits -> collect -> analyze -> risk check -> execute."""
        # Auto-reset daily PnL at start of each new day
        if self._risk_manager:
            self._risk_manager.check_and_reset_daily()

        # Update portfolio value from paper portfolio or default
        if self._risk_manager:
            if self._paper_portfolio:
                self._risk_manager.update_portfolio_value(
                    self._paper_portfolio.total_value
                )
            elif self._risk_manager._current_portfolio_value == 0:
                self._risk_manager.update_portfolio_value(10000.0)

        # Collect data
        if self._collector:
            await self._collector.collect_once()

        recent_trades = []

        # BEFORE running strategies, check exit conditions on all managed positions
        if self._position_manager and self._store:
            for symbol in list(self._position_manager.managed_symbols):
                candles = await self._store.get_candles(symbol, limit=1)
                if not candles:
                    continue
                current_price = candles[-1].close
                exit_signal = self._position_manager.check_exits(
                    symbol, current_price
                )
                if exit_signal:
                    await self._execute_exit(
                        exit_signal, recent_trades
                    )

        # Run strategies on each symbol using ensemble voting
        active_strategies = strategy_registry.get_active()

        for symbol in self._settings.symbols:
            if not self._store:
                continue

            candles = await self._store.get_candles(symbol, limit=200)
            if not candles:
                continue

            # Detect market regime and adapt strategies
            if self._regime_detector and candles:
                try:
                    if (
                        len(candles)
                        >= self._regime_detector.required_history_length
                    ):
                        regime = self._regime_detector.detect(candles)
                        for strategy in active_strategies:
                            strategy.adapt_to_regime(regime)
                        logger.debug(
                            "regime_applied",
                            symbol=symbol,
                            regime=regime.value,
                        )
                except Exception:
                    logger.warning(
                        "regime_detection_error",
                        symbol=symbol,
                        exc_info=True,
                    )

            # Determine trend from higher-timeframe candles
            trend_direction = None
            if self._trend_filter and self._store:
                try:
                    trend_candles = await self._store.get_candles(
                        symbol,
                        timeframe=self._settings.trend_timeframe,
                        limit=50,
                    )
                    if (
                        len(trend_candles)
                        >= self._trend_filter.required_history_length
                    ):
                        trend_direction = self._trend_filter.get_trend(
                            symbol, trend_candles
                        )
                except Exception:
                    logger.warning(
                        "trend_filter_error",
                        symbol=symbol,
                        exc_info=True,
                    )

            # Collect signals from all strategies and vote
            if self._signal_ensemble:
                signals = await self._signal_ensemble.collect_signals(
                    symbol, active_strategies, candles
                )
                signal = self._signal_ensemble.vote(
                    signals, symbol, trend_direction=trend_direction
                )
            else:
                continue

            # Risk check
            if self._risk_manager:
                signal = self._risk_manager.validate_signal(signal)

            if signal.action != signal.action.HOLD:
                # Execute on first available exchange
                for name, engine in self._execution_engines.items():
                    if self._risk_manager:
                        # Try ATR-based dynamic sizing first
                        try:
                            atr = calculate_atr(
                                candles,
                                period=self._settings.atr_period,
                            )
                        except Exception:
                            atr = None
                        if atr is not None and atr > 0:
                            qty = (
                                self._risk_manager
                                .calculate_dynamic_position_size(
                                    self._risk_manager
                                    ._current_portfolio_value
                                    or 10000,
                                    candles[-1].close,
                                    atr,
                                    risk_per_trade_pct=self._settings
                                    .risk_per_trade_pct,
                                    atr_multiplier=self._settings
                                    .atr_multiplier,
                                )
                            )
                        else:
                            # Fallback to fixed % sizing
                            qty = (
                                self._risk_manager
                                .calculate_position_size(
                                    self._risk_manager
                                    ._current_portfolio_value
                                    or 10000,
                                    candles[-1].close,
                                )
                            )
                    else:
                        qty = 0.01

                    if qty > 0:
                        order = await engine.execute_signal(signal, quantity=qty)
                        if order:
                            fill_price = order.filled_price or 0

                            # Track position in RiskManager
                            if self._risk_manager and fill_price > 0:
                                if order.side == OrderSide.BUY:
                                    self._risk_manager.add_position(
                                        order.symbol,
                                        order.quantity,
                                        fill_price,
                                    )
                                    # Register with PositionManager
                                    if self._position_manager:
                                        self._position_manager.add_position(
                                            order.symbol,
                                            fill_price,
                                            order.quantity,
                                        )
                                elif order.side == OrderSide.SELL:
                                    position = (
                                        self._risk_manager.get_position(
                                            order.symbol
                                        )
                                    )
                                    if position:
                                        pnl = (
                                            fill_price
                                            - position["entry_price"]
                                        ) * order.quantity
                                        self._risk_manager.record_trade_pnl(
                                            pnl
                                        )
                                    self._risk_manager.remove_position(
                                        order.symbol
                                    )
                                    # Remove from PositionManager
                                    if self._position_manager:
                                        self._position_manager.remove_position(
                                            order.symbol
                                        )

                            trade_info = {
                                "timestamp": (
                                    order.created_at.isoformat()
                                    if order.created_at
                                    else ""
                                ),
                                "symbol": order.symbol,
                                "side": order.side.value,
                                "quantity": order.quantity,
                                "price": fill_price,
                            }
                            recent_trades.append(trade_info)
                            # Notify via Telegram
                            if self._telegram:
                                await self._telegram.notify_trade(
                                    symbol=order.symbol,
                                    side=order.side.value,
                                    quantity=order.quantity,
                                    price=fill_price,
                                )
                    break

        # Update dashboard state after each cycle
        dashboard_module.update_state(
            status="running",
            trades=dashboard_module.get_state()["trades"] + recent_trades,
            metrics=dashboard_module.get_state()["metrics"],
            portfolio=dashboard_module.get_state()["portfolio"],
        )

    async def _execute_exit(
        self,
        exit_signal,
        recent_trades: list[dict],
    ) -> None:
        """Execute a position exit triggered by PositionManager."""
        sell_signal = TradingSignal(
            strategy_name="position_manager",
            symbol=exit_signal.symbol,
            action=SignalAction.SELL,
            confidence=1.0,
            metadata={
                "exit_type": exit_signal.exit_type.value,
                "exit_price": exit_signal.exit_price,
            },
        )
        for name, engine in self._execution_engines.items():
            order = await engine.execute_signal(
                sell_signal, quantity=exit_signal.quantity
            )
            if order:
                fill_price = order.filled_price or 0

                if self._risk_manager and fill_price > 0:
                    position = self._risk_manager.get_position(
                        exit_signal.symbol
                    )
                    if position:
                        pnl = (
                            fill_price - position["entry_price"]
                        ) * order.quantity
                        self._risk_manager.record_trade_pnl(pnl)

                    # For partial exits (TP1), update RiskManager position
                    # For full exits (SL, TP2, trailing), remove entirely
                    if exit_signal.exit_type == ExitType.TAKE_PROFIT_1:
                        if position:
                            remaining = (
                                position["quantity"] - order.quantity
                            )
                            if remaining > 1e-10:
                                self._risk_manager.add_position(
                                    exit_signal.symbol,
                                    remaining,
                                    position["entry_price"],
                                )
                            else:
                                self._risk_manager.remove_position(
                                    exit_signal.symbol
                                )
                    else:
                        self._risk_manager.remove_position(
                            exit_signal.symbol
                        )
                        # Remove from PositionManager on full exit
                        if self._position_manager:
                            self._position_manager.remove_position(
                                exit_signal.symbol
                            )

                trade_info = {
                    "timestamp": (
                        order.created_at.isoformat()
                        if order.created_at
                        else ""
                    ),
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "price": fill_price,
                }
                recent_trades.append(trade_info)
                if self._telegram:
                    await self._telegram.notify_trade(
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.quantity,
                        price=fill_price,
                    )
            break

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("bot_shutting_down")
        self._running = False

        # Notify shutdown via Telegram
        if self._telegram:
            await self._telegram.send_message("Bot shutting down.")

        # Cancel dashboard background task
        if self._dashboard_task and not self._dashboard_task.done():
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except asyncio.CancelledError:
                pass

        for exchange in self._exchanges:
            await exchange.close()

        if self._store:
            await self._store.close()

        dashboard_module.update_state(status="stopped")
        logger.info("bot_shutdown_complete")

    def stop(self) -> None:
        """Signal the bot to stop."""
        self._running = False


async def main() -> None:
    """Run the trading bot."""
    bot = TradingBot()

    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, bot.stop)

    try:
        await bot.initialize()
        await bot.run_trading_loop()
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested. Exiting.")
        sys.exit(0)
