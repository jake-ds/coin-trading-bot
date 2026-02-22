"""Main entry point for the Coin Trading Bot."""

import asyncio
import signal
import sys
import time

import structlog

from bot.config import Settings, TradingMode, load_settings
from bot.dashboard import app as dashboard_module
from bot.data.collector import DataCollector
from bot.data.order_book import OrderBookAnalyzer
from bot.data.store import DataStore
from bot.data.websocket_feed import WebSocketFeed
from bot.exchanges.factory import ExchangeFactory
from bot.execution.engine import ExecutionEngine
from bot.execution.paper_portfolio import PaperPortfolio
from bot.execution.position_manager import ExitType, PositionManager
from bot.execution.resilient import ResilientExchange
from bot.execution.smart_executor import SmartExecutor
from bot.models import OrderSide, SignalAction, TradingSignal
from bot.monitoring.logger import setup_logging
from bot.monitoring.strategy_tracker import StrategyTracker
from bot.monitoring.telegram import TelegramNotifier
from bot.risk.manager import RiskManager
from bot.risk.portfolio_risk import PortfolioRiskManager
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
        self._ws_feed: WebSocketFeed | None = None
        self._strategy_tracker: StrategyTracker | None = None
        self._portfolio_risk: PortfolioRiskManager | None = None
        self._order_book_analyzer: OrderBookAnalyzer | None = None
        self._cycle_lock: asyncio.Lock = asyncio.Lock()
        self._cycle_count: int = 0
        self._total_cycle_duration: float = 0.0
        self._last_cycle_time: float | None = None
        self._current_regime = None

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

        # Initialize portfolio-level risk manager
        self._portfolio_risk = PortfolioRiskManager(
            max_total_exposure_pct=self._settings.max_total_exposure_pct,
            max_correlation=self._settings.max_correlation,
            correlation_window=self._settings.correlation_window,
            max_positions_per_sector=self._settings.max_positions_per_sector,
            max_portfolio_heat=self._settings.max_portfolio_heat,
            sector_map=self._settings.sector_map,
        )

        # Initialize order book analyzer
        self._order_book_analyzer = OrderBookAnalyzer()

        # Initialize signal ensemble voting system
        self._signal_ensemble = SignalEnsemble(
            min_agreement=self._settings.signal_min_agreement,
            strategy_weights=self._settings.strategy_weights,
            order_book_analyzer=self._order_book_analyzer,
        )

        # Initialize trend filter for higher-timeframe confirmation
        self._trend_filter = TrendFilter()

        # Initialize market regime detector
        self._regime_detector = MarketRegimeDetector()

        # Initialize strategy performance tracker
        self._strategy_tracker = StrategyTracker(
            max_consecutive_losses=self._settings.strategy_max_consecutive_losses,
            min_win_rate_pct=self._settings.strategy_min_win_rate_pct,
            min_trades_for_evaluation=self._settings.strategy_min_trades_for_eval,
            re_enable_check_hours=self._settings.strategy_re_enable_check_hours,
            registry=strategy_registry,
        )

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
            # Create SmartExecutor for live trading with limit order optimization
            smart_exec = None
            if not is_paper:
                smart_exec = SmartExecutor(
                    exchange=exchange,
                    prefer_limit_orders=self._settings.prefer_limit_orders,
                    limit_order_timeout_seconds=self._settings.limit_order_timeout_seconds,
                    twap_chunk_count=self._settings.twap_chunk_count,
                )
            self._execution_engines[exchange.name] = ExecutionEngine(
                exchange=exchange,
                store=self._store,
                paper_trading=is_paper,
                paper_portfolio=self._paper_portfolio,
                smart_executor=smart_exec,
            )

        # Initialize WebSocket feed (if enabled and exchanges available)
        self._init_ws_feed()

        # Initialize Telegram notifier (gracefully skip if not configured)
        self._init_telegram()

        # Start dashboard server as background task
        self._start_dashboard()

        # Provide strategy registry to dashboard for toggle endpoint
        dashboard_module.set_strategy_registry(strategy_registry)

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

    def _init_ws_feed(self) -> None:
        """Initialize WebSocket feed if enabled and exchanges are available."""
        if not self._settings.websocket_enabled:
            logger.debug("websocket_feed_disabled")
            return
        if not self._exchanges:
            logger.debug("websocket_feed_no_exchanges")
            return

        # Use first exchange for the WebSocket feed
        self._ws_feed = WebSocketFeed(
            exchange=self._exchanges[0],
            symbols=self._settings.symbols,
            timeframes=self._settings.timeframes,
            poll_interval=self._settings.websocket_poll_interval,
            max_reconnect_delay=self._settings.websocket_max_reconnect_delay,
        )
        # Start feed as background task
        asyncio.ensure_future(self._ws_feed.start())
        logger.info(
            "websocket_feed_initialized",
            ws_supported=self._ws_feed.ws_supported,
        )

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

        # Check if disabled strategies should be re-enabled
        if self._strategy_tracker:
            self._strategy_tracker.check_re_enable()

        # Update portfolio value from paper portfolio or default
        if self._risk_manager:
            if self._paper_portfolio:
                portfolio_val = self._paper_portfolio.total_value
                self._risk_manager.update_portfolio_value(portfolio_val)
            elif self._risk_manager._current_portfolio_value == 0:
                portfolio_val = 10000.0
                self._risk_manager.update_portfolio_value(portfolio_val)
            else:
                portfolio_val = self._risk_manager._current_portfolio_value
            # Sync portfolio value to portfolio risk manager
            if self._portfolio_risk:
                self._portfolio_risk.update_portfolio_value(portfolio_val)

        # Collect data
        if self._collector:
            await self._collector.collect_once()

        recent_trades = []

        # BEFORE running strategies, check exit conditions on all managed positions
        if self._position_manager and self._store:
            for symbol in list(self._position_manager.managed_symbols):
                # Prefer WebSocket feed price for faster stop-loss checks
                current_price = None
                if self._ws_feed:
                    current_price = self._ws_feed.get_latest_price(symbol)
                if current_price is None:
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

            # Update price history for portfolio correlation calculation
            if self._portfolio_risk:
                try:
                    self._portfolio_risk.update_price_history(
                        symbol, candles
                    )
                except Exception:
                    pass

            # Detect market regime and adapt strategies
            if self._regime_detector and candles:
                try:
                    if (
                        len(candles)
                        >= self._regime_detector.required_history_length
                    ):
                        regime = self._regime_detector.detect(candles)
                        self._current_regime = regime
                        for strategy in active_strategies:
                            strategy.adapt_to_regime(regime)
                        # Update strategy tracker with regime
                        if self._strategy_tracker:
                            self._strategy_tracker.update_regime(regime)
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

            # Analyze order book for confidence modification
            ob_analysis = None
            if self._order_book_analyzer and self._exchanges:
                try:
                    ob_analysis = (
                        await self._order_book_analyzer.fetch_and_analyze(
                            self._exchanges[0], symbol
                        )
                    )
                except Exception:
                    logger.warning(
                        "order_book_analysis_error",
                        symbol=symbol,
                        exc_info=True,
                    )

            # Collect signals from all strategies and vote
            if self._signal_ensemble:
                signals = await self._signal_ensemble.collect_signals(
                    symbol, active_strategies, candles
                )
                signal = self._signal_ensemble.vote(
                    signals,
                    symbol,
                    trend_direction=trend_direction,
                    order_book_analysis=ob_analysis,
                )
            else:
                continue

            # Risk check (individual trade)
            if self._risk_manager:
                signal = self._risk_manager.validate_signal(signal)

            # Portfolio-level risk check (BUY only)
            if (
                signal.action == SignalAction.BUY
                and self._portfolio_risk
            ):
                try:
                    atr_for_check = calculate_atr(
                        candles, period=self._settings.atr_period
                    )
                except Exception:
                    atr_for_check = None
                # Estimate position value for check
                price = candles[-1].close
                est_qty = 0.01  # Will be refined later
                if self._risk_manager:
                    portfolio_val = (
                        self._risk_manager._current_portfolio_value
                        or 10000
                    )
                    if atr_for_check and atr_for_check > 0:
                        est_qty = (
                            self._risk_manager
                            .calculate_dynamic_position_size(
                                portfolio_val,
                                price,
                                atr_for_check,
                                risk_per_trade_pct=self._settings
                                .risk_per_trade_pct,
                                atr_multiplier=self._settings
                                .atr_multiplier,
                            )
                        )
                    else:
                        est_qty = (
                            self._risk_manager
                            .calculate_position_size(
                                portfolio_val, price
                            )
                        )
                est_value = est_qty * price
                # Normalize ATR as fraction of price for heat calc
                atr_ratio = (
                    atr_for_check / price
                    if atr_for_check and price > 0
                    else None
                )
                allowed, reason = (
                    self._portfolio_risk.validate_new_position(
                        symbol, est_value, atr=atr_ratio
                    )
                )
                if not allowed:
                    logger.info(
                        "portfolio_risk_rejected",
                        symbol=symbol,
                        reason=reason,
                    )
                    signal = TradingSignal(
                        strategy_name=signal.strategy_name,
                        symbol=signal.symbol,
                        action=SignalAction.HOLD,
                        confidence=0.0,
                        metadata={
                            **signal.metadata,
                            "rejected": True,
                            "reject_reason": f"portfolio_risk: {reason}",
                        },
                    )

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
                                    # Track in portfolio risk manager
                                    if self._portfolio_risk:
                                        pos_value = (
                                            order.quantity * fill_price
                                        )
                                        try:
                                            pos_atr = calculate_atr(
                                                candles,
                                                period=self._settings
                                                .atr_period,
                                            )
                                        except Exception:
                                            pos_atr = None
                                        atr_r = (
                                            pos_atr / fill_price
                                            if pos_atr
                                            and fill_price > 0
                                            else None
                                        )
                                        self._portfolio_risk.add_position(
                                            order.symbol,
                                            pos_value,
                                            atr=atr_r,
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
                                        # Record to strategy tracker
                                        if self._strategy_tracker:
                                            strat_names = signal.metadata.get(
                                                "agreeing_strategies", []
                                            )
                                            if strat_names:
                                                for sn in strat_names:
                                                    self._strategy_tracker.record_trade(
                                                        sn, pnl
                                                    )
                                            else:
                                                self._strategy_tracker.record_trade(
                                                    signal.strategy_name,
                                                    pnl,
                                                )
                                    self._risk_manager.remove_position(
                                        order.symbol
                                    )
                                    # Remove from PositionManager
                                    if self._position_manager:
                                        self._position_manager.remove_position(
                                            order.symbol
                                        )
                                    # Remove from portfolio risk
                                    if self._portfolio_risk:
                                        self._portfolio_risk.remove_position(
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

        # Build open positions data for dashboard
        open_positions = self._build_open_positions()

        # Save portfolio snapshot every 10 cycles for equity curve
        await self._maybe_save_portfolio_snapshot()

        # Update dashboard state after each cycle
        strategy_stats = (
            self._strategy_tracker.get_all_stats()
            if self._strategy_tracker
            else {}
        )

        # Get current regime string for dashboard
        current_regime = (
            self._current_regime.value
            if self._current_regime
            else None
        )

        dashboard_module.update_state(
            status="running",
            trades=dashboard_module.get_state()["trades"] + recent_trades,
            metrics=dashboard_module.get_state()["metrics"],
            portfolio=dashboard_module.get_state()["portfolio"],
            strategy_stats=strategy_stats,
            open_positions=open_positions,
            regime=current_regime,
            equity_curve=dashboard_module.get_state().get("equity_curve", []),
        )

    def _build_open_positions(self) -> list[dict]:
        """Build open positions list for dashboard display."""
        positions = []
        if not self._position_manager:
            return positions
        for symbol, pos in self._position_manager.positions.items():
            current_price = pos.highest_price_since_entry
            # Try to get latest price from WS feed
            if self._ws_feed:
                ws_price = self._ws_feed.get_latest_price(symbol)
                if ws_price:
                    current_price = ws_price
            unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            positions.append({
                "symbol": symbol,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "unrealized_pnl": round(unrealized_pnl, 2),
                "stop_loss": pos.stop_loss_price,
                "take_profit": pos.tp2_price,
            })
        return positions

    async def _maybe_save_portfolio_snapshot(self) -> None:
        """Save portfolio snapshot every 10 cycles for equity curve tracking."""
        if self._cycle_count == 0 or self._cycle_count % 10 != 0:
            return
        if not self._store:
            return

        total_value = 10000.0
        unrealized_pnl = 0.0

        if self._paper_portfolio:
            total_value = self._paper_portfolio.total_value
            unrealized_pnl = self._paper_portfolio.unrealized_pnl
        elif self._risk_manager:
            total_value = (
                self._risk_manager._current_portfolio_value or 10000.0
            )

        try:
            await self._store.save_portfolio_snapshot(
                total_value=total_value,
                unrealized_pnl=unrealized_pnl,
            )
            # Update equity curve in dashboard state
            eq_curve = dashboard_module.get_state().get("equity_curve", [])
            from datetime import datetime, timezone

            eq_curve.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_value": total_value,
            })
            # Keep only last 500 points
            if len(eq_curve) > 500:
                eq_curve = eq_curve[-500:]
            dashboard_module.update_state(equity_curve=eq_curve)
        except Exception:
            logger.warning("portfolio_snapshot_save_error", exc_info=True)

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
                        # Record exit to strategy tracker
                        if self._strategy_tracker:
                            self._strategy_tracker.record_trade(
                                "position_manager", pnl
                            )

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
                        # Remove from portfolio risk on full exit
                        if self._portfolio_risk:
                            self._portfolio_risk.remove_position(
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

        # Stop WebSocket feed
        if self._ws_feed:
            await self._ws_feed.stop()

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
