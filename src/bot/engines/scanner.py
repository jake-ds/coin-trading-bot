"""Token scanner engine — discovers trading opportunities via batch API calls.

Runs every 5 minutes (configurable), fetching tickers and funding rates in bulk
from connected exchanges.  Discovered opportunities are published to the shared
OpportunityRegistry so that other engines can dynamically expand their symbol lists.

API budget: 2-3 calls per cycle (fetchTickers + fetchFundingRates).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import structlog

from bot.engines.base import BaseEngine, DecisionStep, EngineCycleResult
from bot.engines.opportunity_registry import (
    Opportunity,
    OpportunityRegistry,
    OpportunityType,
)

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.portfolio_manager import PortfolioManager
    from bot.exchanges.base import ExchangeAdapter

logger = structlog.get_logger(__name__)

# Default TTLs
_DEFAULT_TTL_FUNDING = timedelta(hours=8)
_DEFAULT_TTL_VOLATILITY = timedelta(hours=1)
_DEFAULT_TTL_SPREAD = timedelta(minutes=5)
_DEFAULT_TTL_CORRELATION = timedelta(hours=24)


class TokenScannerEngine(BaseEngine):
    """Batch-scans connected exchanges for trading opportunities.

    This engine does **not** trade — it only discovers and publishes
    opportunities to the ``OpportunityRegistry``.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        exchanges: list[ExchangeAdapter] | None = None,
        paper_mode: bool = True,
        settings: Settings | None = None,
        registry: OpportunityRegistry | None = None,
    ):
        s = settings
        loop_interval = s.scanner_interval_seconds if s else 300.0
        super().__init__(
            portfolio_manager=portfolio_manager,
            exchanges=exchanges,
            loop_interval=loop_interval,
            max_positions=0,  # scanner never opens positions
            paper_mode=paper_mode,
        )
        self._registry = registry or OpportunityRegistry()
        self._min_volume_usdt = s.scanner_min_volume_usdt if s else 100_000.0
        self._top_n = s.scanner_top_n if s else 20
        self._enabled_scans: list[str] = (
            list(s.scanner_enabled_scans) if s else ["funding_rate", "volatility"]
        )
        # TTL config
        self._ttl_funding = timedelta(
            hours=s.scanner_ttl_funding_hours if s else 8
        )
        self._ttl_volatility = timedelta(
            hours=s.scanner_ttl_volatility_hours if s else 1
        )
        self._ttl_spread = timedelta(
            minutes=s.scanner_ttl_spread_minutes if s else 5
        )
        self._ttl_correlation = timedelta(
            hours=s.scanner_ttl_correlation_hours if s else 24
        )

        # Cache ticker data across scans within same cycle
        self._ticker_cache: dict[str, dict[str, Any]] = {}
        # Price history for correlation scan (symbol -> list of last prices)
        self._price_history: dict[str, list[float]] = {}
        # Unauthenticated ccxt client for public API calls (lazy init)
        self._public_exchange: Any | None = None

    @property
    def name(self) -> str:
        return "token_scanner"

    @property
    def description(self) -> str:
        return "Batch market scanner for dynamic opportunity discovery"

    @property
    def registry(self) -> OpportunityRegistry:
        return self._registry

    async def _run_cycle(self) -> EngineCycleResult:
        """Execute one scan cycle: fetch batch data and publish opportunities."""
        cycle_start = datetime.now(timezone.utc)
        actions: list[dict] = []
        signals: list[dict] = []
        decisions: list[DecisionStep] = []

        # Fetch batch ticker data from primary exchange
        self._ticker_cache = await self._fetch_all_tickers()
        ticker_count = len(self._ticker_cache)

        decisions.append(DecisionStep(
            label="배치 티커 조회",
            observation=f"{ticker_count}개 심볼 티커 수신",
            threshold=f"최소 거래량 ${self._min_volume_usdt:,.0f}",
            result=f"OK - {ticker_count}개 심볼" if ticker_count > 0 else "FAIL - 티커 없음",
            category="evaluate",
        ))

        if ticker_count == 0:
            return EngineCycleResult(
                engine_name=self.name,
                cycle_num=self._cycle_count + 1,
                timestamp=cycle_start.isoformat(),
                duration_ms=0.0,
                decisions=decisions,
            )

        # Filter by minimum volume
        filtered = {
            sym: data for sym, data in self._ticker_cache.items()
            if self._get_volume_usdt(data) >= self._min_volume_usdt
        }
        decisions.append(DecisionStep(
            label="거래량 필터링",
            observation=f"전체 {ticker_count}개 중 {len(filtered)}개 통과",
            threshold=f"24h 거래량 >= ${self._min_volume_usdt:,.0f}",
            result=f"FILTER - {len(filtered)}개 심볼 선별",
            category="evaluate",
        ))

        total_published = 0

        # 1. Funding rate scan
        if "funding_rate" in self._enabled_scans:
            count = await self._scan_funding_rates(filtered, decisions)
            total_published += count
            actions.append({
                "action": "scan_funding_rates",
                "opportunities_found": count,
            })

        # 2. Volatility scan
        if "volatility" in self._enabled_scans:
            count = self._scan_volatility(filtered, decisions)
            total_published += count
            actions.append({
                "action": "scan_volatility",
                "opportunities_found": count,
            })

        # 3. Cross-exchange spread scan
        if "cross_exchange_spread" in self._enabled_scans:
            count = await self._scan_spreads(filtered, decisions)
            total_published += count
            actions.append({
                "action": "scan_spreads",
                "opportunities_found": count,
            })

        # 4. Correlation scan
        if "correlation" in self._enabled_scans:
            count = self._scan_correlations(filtered, decisions)
            total_published += count
            actions.append({
                "action": "scan_correlations",
                "opportunities_found": count,
            })

        # Clean up expired entries
        expired_count = self._registry.clear_expired()

        signals.append({
            "total_symbols_scanned": ticker_count,
            "volume_filtered": len(filtered),
            "total_opportunities": total_published,
            "expired_cleaned": expired_count,
        })

        decisions.append(DecisionStep(
            label="스캔 완료 요약",
            observation=f"총 {total_published}개 기회 발견, {expired_count}개 만료 제거",
            threshold="N/A",
            result=f"PUBLISHED - {total_published}개 기회",
            category="execute" if total_published > 0 else "evaluate",
        ))

        return EngineCycleResult(
            engine_name=self.name,
            cycle_num=self._cycle_count + 1,
            timestamp=cycle_start.isoformat(),
            duration_ms=0.0,
            actions_taken=actions,
            positions=[],
            signals=signals,
            pnl_update=0.0,
            metadata={
                "symbols_scanned": ticker_count,
                "opportunities_published": total_published,
                "registry_summary": self._registry.get_summary(),
            },
            decisions=decisions,
        )

    # ------------------------------------------------------------------
    # Batch data fetching
    # ------------------------------------------------------------------

    async def _get_public_exchange(self) -> Any:
        """Lazy-init an unauthenticated ccxt client for public API calls."""
        if self._public_exchange is None:
            try:
                import ccxt.async_support as ccxt

                self._public_exchange = ccxt.binance({
                    "options": {"defaultType": "future"},
                })
            except Exception:
                return None
        return self._public_exchange

    async def _fetch_all_tickers(self) -> dict[str, dict[str, Any]]:
        """Fetch all tickers from the primary exchange (1 API call)."""
        # Try each connected exchange adapter first
        for exchange in self._exchanges:
            inner = getattr(exchange, "_exchange", None)
            if inner and hasattr(inner, "fetch_tickers"):
                try:
                    raw = await inner.fetch_tickers()
                    if isinstance(raw, dict) and raw:
                        return raw
                except Exception as e:
                    logger.debug(
                        "scanner_fetch_tickers_failed",
                        exchange=getattr(exchange, "name", "unknown"),
                        error=str(e),
                    )

            if hasattr(exchange, "get_all_tickers"):
                try:
                    result = await exchange.get_all_tickers()
                    if result:
                        return result
                except Exception as e:
                    logger.debug(
                        "scanner_get_all_tickers_failed",
                        exchange=getattr(exchange, "name", "unknown"),
                        error=str(e),
                    )

        # Fallback: unauthenticated public client
        pub = await self._get_public_exchange()
        if pub and hasattr(pub, "fetch_tickers"):
            try:
                raw = await pub.fetch_tickers()
                if isinstance(raw, dict) and raw:
                    logger.debug(
                        "scanner_tickers_via_public_client",
                        count=len(raw),
                    )
                    return raw
            except Exception as e:
                logger.debug(
                    "scanner_public_fetch_tickers_failed",
                    error=str(e),
                )
        return {}

    async def _fetch_all_funding_rates(self) -> dict[str, dict[str, Any]]:
        """Fetch all funding rates from a futures exchange (1 API call)."""
        # Try connected exchanges first
        for exchange in self._exchanges:
            if hasattr(exchange, "get_all_funding_rates"):
                try:
                    result = await exchange.get_all_funding_rates()
                    if result:
                        return result
                except Exception:
                    pass

            inner = getattr(exchange, "_exchange", None)
            if inner and hasattr(inner, "fetch_funding_rates"):
                try:
                    raw = await inner.fetch_funding_rates()
                    if isinstance(raw, dict) and raw:
                        return raw
                except Exception as e:
                    logger.debug(
                        "scanner_fetch_funding_rates_failed",
                        error=str(e),
                    )

        # Fallback: unauthenticated public client
        pub = await self._get_public_exchange()
        if pub and hasattr(pub, "fetch_funding_rates"):
            try:
                raw = await pub.fetch_funding_rates()
                if isinstance(raw, dict) and raw:
                    logger.debug(
                        "scanner_funding_rates_via_public_client",
                        count=len(raw),
                    )
                    return raw
            except Exception as e:
                logger.debug(
                    "scanner_public_funding_rates_failed",
                    error=str(e),
                )
        return {}

    # ------------------------------------------------------------------
    # Scan implementations
    # ------------------------------------------------------------------

    async def _scan_funding_rates(
        self,
        tickers: dict[str, dict[str, Any]],
        decisions: list[DecisionStep],
    ) -> int:
        """Scan funding rates — publish high-rate symbols."""
        rates = await self._fetch_all_funding_rates()
        if not rates:
            decisions.append(DecisionStep(
                label="펀딩비 스캔",
                observation="펀딩비 데이터 없음",
                threshold="N/A",
                result="SKIP - 펀딩비 API 없음",
                category="skip",
            ))
            return 0

        now = datetime.now(timezone.utc)
        expires = (now + self._ttl_funding).isoformat()
        opportunities: list[Opportunity] = []

        for symbol, rate_data in rates.items():
            if symbol not in tickers:
                continue
            funding_rate = float(
                rate_data.get("fundingRate", 0)
                or rate_data.get("funding_rate", 0)
                or 0
            )
            if funding_rate <= 0:
                continue

            # Score: annualized rate percentage (capped at 100)
            ann_pct = funding_rate * 3 * 365 * 100
            score = min(ann_pct, 100.0)

            opportunities.append(Opportunity(
                symbol=symbol,
                type=OpportunityType.FUNDING_RATE,
                score=score,
                metrics={
                    "funding_rate": funding_rate,
                    "annualized_pct": round(ann_pct, 2),
                },
                discovered_at=now.isoformat(),
                expires_at=expires,
                source_exchange=self._get_exchange_name(0),
            ))

        # Keep top N
        opportunities.sort(key=lambda o: o.score, reverse=True)
        opportunities = opportunities[: self._top_n]
        self._registry.publish(OpportunityType.FUNDING_RATE, opportunities)

        decisions.append(DecisionStep(
            label="펀딩비 스캔",
            observation=f"{len(rates)}개 심볼 펀딩비 확인, {len(opportunities)}개 양수",
            threshold=f"상위 {self._top_n}개, TTL {self._ttl_funding}",
            result=f"PUBLISH - {len(opportunities)}개 기회",
            category="execute" if opportunities else "evaluate",
        ))
        return len(opportunities)

    def _scan_volatility(
        self,
        tickers: dict[str, dict[str, Any]],
        decisions: list[DecisionStep],
    ) -> int:
        """Scan for high-volatility symbols using ticker data (0 API calls)."""
        now = datetime.now(timezone.utc)
        expires = (now + self._ttl_volatility).isoformat()
        opportunities: list[Opportunity] = []

        for symbol, data in tickers.items():
            pct_change = abs(float(data.get("percentage", 0) or 0))
            volume_usdt = self._get_volume_usdt(data)
            if pct_change <= 0 or volume_usdt <= 0:
                continue

            # Score: weighted by change% and volume
            # Normalize: 10% change = 50 score base, volume multiplier
            vol_factor = min(volume_usdt / 1_000_000, 2.0)  # cap at 2x
            score = min(pct_change * 5 * vol_factor, 100.0)

            opportunities.append(Opportunity(
                symbol=symbol,
                type=OpportunityType.VOLATILITY,
                score=score,
                metrics={
                    "change_pct": round(pct_change, 2),
                    "volume_usdt": round(volume_usdt, 0),
                    "last_price": float(data.get("last", 0) or 0),
                },
                discovered_at=now.isoformat(),
                expires_at=expires,
                source_exchange=self._get_exchange_name(0),
            ))

        opportunities.sort(key=lambda o: o.score, reverse=True)
        opportunities = opportunities[: self._top_n]
        self._registry.publish(OpportunityType.VOLATILITY, opportunities)

        decisions.append(DecisionStep(
            label="변동성 스캔",
            observation=f"{len(tickers)}개 심볼 분석, {len(opportunities)}개 고변동",
            threshold=f"상위 {self._top_n}개, TTL {self._ttl_volatility}",
            result=f"PUBLISH - {len(opportunities)}개 기회",
            category="execute" if opportunities else "evaluate",
        ))
        return len(opportunities)

    async def _scan_spreads(
        self,
        tickers: dict[str, dict[str, Any]],
        decisions: list[DecisionStep],
    ) -> int:
        """Scan for cross-exchange price spreads (1 additional API call)."""
        if len(self._exchanges) < 2:
            decisions.append(DecisionStep(
                label="스프레드 스캔",
                observation=f"거래소 {len(self._exchanges)}개",
                threshold="최소 2개 거래소 필요",
                result="SKIP - 거래소 부족",
                category="skip",
            ))
            return 0

        # Fetch tickers from second exchange
        second_tickers: dict[str, dict[str, Any]] = {}
        exchange_b = self._exchanges[1]
        inner = getattr(exchange_b, "_exchange", None)
        if inner and hasattr(inner, "fetch_tickers"):
            try:
                second_tickers = await inner.fetch_tickers()
            except Exception as e:
                logger.debug("scanner_second_exchange_tickers_failed", error=str(e))
        elif hasattr(exchange_b, "get_all_tickers"):
            try:
                second_tickers = await exchange_b.get_all_tickers()
            except Exception:
                pass

        if not second_tickers:
            decisions.append(DecisionStep(
                label="스프레드 스캔",
                observation="두 번째 거래소 티커 조회 실패",
                threshold="N/A",
                result="SKIP - 두 번째 거래소 데이터 없음",
                category="skip",
            ))
            return 0

        now = datetime.now(timezone.utc)
        expires = (now + self._ttl_spread).isoformat()
        opportunities: list[Opportunity] = []

        common_symbols = set(tickers.keys()) & set(second_tickers.keys())
        for symbol in common_symbols:
            price_a = float(tickers[symbol].get("last", 0) or 0)
            price_b = float(second_tickers[symbol].get("last", 0) or 0)
            if price_a <= 0 or price_b <= 0:
                continue

            mid = (price_a + price_b) / 2
            spread_pct = abs(price_a - price_b) / mid * 100

            if spread_pct < 0.1:  # minimum threshold
                continue

            score = min(spread_pct * 20, 100.0)
            opportunities.append(Opportunity(
                symbol=symbol,
                type=OpportunityType.CROSS_EXCHANGE_SPREAD,
                score=score,
                metrics={
                    "spread_pct": round(spread_pct, 4),
                    "price_a": price_a,
                    "price_b": price_b,
                    "exchange_a": self._get_exchange_name(0),
                    "exchange_b": self._get_exchange_name(1),
                },
                discovered_at=now.isoformat(),
                expires_at=expires,
                source_exchange=self._get_exchange_name(0),
            ))

        opportunities.sort(key=lambda o: o.score, reverse=True)
        opportunities = opportunities[: self._top_n]
        self._registry.publish(OpportunityType.CROSS_EXCHANGE_SPREAD, opportunities)

        decisions.append(DecisionStep(
            label="스프레드 스캔",
            observation=(
                f"공통 심볼 {len(common_symbols)}개, "
                f"{len(opportunities)}개 유의미한 스프레드"
            ),
            threshold=f"최소 0.1% 스프레드, 상위 {self._top_n}개",
            result=f"PUBLISH - {len(opportunities)}개 기회",
            category="execute" if opportunities else "evaluate",
        ))
        return len(opportunities)

    def _scan_correlations(
        self,
        tickers: dict[str, dict[str, Any]],
        decisions: list[DecisionStep],
    ) -> int:
        """Scan for correlated pairs using cached price history (0 API calls)."""
        # Update price history
        for symbol, data in tickers.items():
            price = float(data.get("last", 0) or 0)
            if price <= 0:
                continue
            if symbol not in self._price_history:
                self._price_history[symbol] = []
            self._price_history[symbol].append(price)
            # Keep bounded
            if len(self._price_history[symbol]) > 200:
                self._price_history[symbol] = self._price_history[symbol][-200:]

        # Need at least 30 data points for meaningful correlation
        min_history = 30
        eligible = [
            sym for sym, prices in self._price_history.items()
            if len(prices) >= min_history and sym in tickers
        ]

        # Sort by volume and take top 30 for O(435) pair computation
        eligible.sort(
            key=lambda s: self._get_volume_usdt(tickers.get(s, {})),
            reverse=True,
        )
        eligible = eligible[:30]

        if len(eligible) < 2:
            decisions.append(DecisionStep(
                label="상관관계 스캔",
                observation=f"적격 심볼 {len(eligible)}개 (최소 {min_history}개 가격 기록 필요)",
                threshold="최소 2개 심볼 필요",
                result="SKIP - 데이터 부족",
                category="skip",
            ))
            return 0

        now = datetime.now(timezone.utc)
        expires = (now + self._ttl_correlation).isoformat()
        opportunities: list[Opportunity] = []

        try:
            import numpy as np
        except ImportError:
            decisions.append(DecisionStep(
                label="상관관계 스캔",
                observation="numpy 불가",
                threshold="N/A",
                result="SKIP - numpy 미설치",
                category="skip",
            ))
            return 0

        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                sym_a = eligible[i]
                sym_b = eligible[j]
                prices_a = self._price_history[sym_a]
                prices_b = self._price_history[sym_b]
                n = min(len(prices_a), len(prices_b))
                if n < min_history:
                    continue

                a = np.array(prices_a[-n:], dtype=float)
                b = np.array(prices_b[-n:], dtype=float)
                corr = float(np.corrcoef(a, b)[0, 1])

                if abs(corr) < 0.7:
                    continue

                score = abs(corr) * 100
                pair_symbol = f"{sym_a}|{sym_b}"
                opportunities.append(Opportunity(
                    symbol=pair_symbol,
                    type=OpportunityType.CORRELATION,
                    score=score,
                    metrics={
                        "correlation": round(corr, 4),
                        "pair": [sym_a, sym_b],
                        "data_points": n,
                    },
                    discovered_at=now.isoformat(),
                    expires_at=expires,
                    source_exchange=self._get_exchange_name(0),
                ))

        opportunities.sort(key=lambda o: o.score, reverse=True)
        opportunities = opportunities[: self._top_n]
        self._registry.publish(OpportunityType.CORRELATION, opportunities)

        decisions.append(DecisionStep(
            label="상관관계 스캔",
            observation=f"상위 {len(eligible)}개 심볼, {len(opportunities)}개 상관 페어",
            threshold=f"|상관계수| >= 0.7, 상위 {self._top_n}개",
            result=f"PUBLISH - {len(opportunities)}개 페어",
            category="execute" if opportunities else "evaluate",
        ))
        return len(opportunities)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_volume_usdt(ticker_data: dict[str, Any]) -> float:
        """Extract 24h USDT volume from ticker data."""
        # ccxt uses quoteVolume for USDT volume
        vol = ticker_data.get("quoteVolume", 0)
        if vol:
            return float(vol)
        # Fallback: baseVolume * last price
        base_vol = float(ticker_data.get("baseVolume", 0) or 0)
        last = float(ticker_data.get("last", 0) or 0)
        return base_vol * last

    def _get_exchange_name(self, index: int) -> str:
        """Get exchange name by index, safely."""
        if index < len(self._exchanges):
            return getattr(self._exchanges[index], "name", f"exchange_{index}")
        return f"exchange_{index}"
