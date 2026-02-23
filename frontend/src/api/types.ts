export interface StatusResponse {
  status: string
  started_at: string | null
  cycle_metrics: {
    cycle_count: number
    average_cycle_duration: number
    last_cycle_time: number | null
  }
}

export interface PortfolioResponse {
  portfolio: {
    balances: Record<string, number>
    positions: unknown[]
    total_value: number
  }
}

export interface MetricsResponse {
  metrics: Record<string, number>
}

export interface TradesResponse {
  trades: Trade[]
  total: number
  page: number
  limit: number
  total_pages: number
}

export interface Trade {
  timestamp: string
  symbol: string
  side: 'BUY' | 'SELL'
  quantity: number
  price: number
  pnl?: number
  strategy?: string
}

export interface Position {
  symbol: string
  quantity: number
  entry_price: number
  current_price: number
  unrealized_pnl: number
  stop_loss: number
  take_profit: number
  opened_at?: string
  strategy?: string
}

export interface PositionsResponse {
  positions: Position[]
}

export interface StrategiesResponse {
  strategies: Record<string, StrategyStats>
}

export interface StrategyStats {
  total_pnl: number
  win_rate: number
  total_trades: number
  wins: number
  losses: number
  avg_pnl: number
  consecutive_losses: number
  active: boolean
  sharpe_ratio: number
  profit_factor: number
  disabled: boolean
  disabled_reason: string | null
  pnl_history: number[]
}

export interface ToggleResponse {
  name: string
  active: boolean
  success: boolean
  has_open_positions?: boolean
  open_position_count?: number
  warning?: string
}

export interface RegimeResponse {
  regime: string | null
}

export interface AnalyticsResponse {
  equity_curve: EquityPoint[]
  drawdown: DrawdownPoint[]
  trade_markers: TradeMarker[]
  monthly_returns: MonthlyReturn[]
  stats: AnalyticsStats
  range: string
}

export interface EquityPoint {
  timestamp: string
  total_value: number
}

export interface DrawdownPoint {
  timestamp: string
  drawdown_pct: number
}

export interface TradeMarker {
  timestamp: string
  value: number
  side: 'BUY' | 'SELL'
  symbol: string
  price: number
}

export interface MonthlyReturn {
  month: string
  return_pct: number
}

export interface AnalyticsStats {
  sharpe_ratio: number
  sortino_ratio: number
  max_drawdown_pct: number
  profit_factor: number
  avg_trade_pnl: number
  best_trade: number
  worst_trade: number
  total_return_pct: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number
}

export interface AuthStatusResponse {
  auth_enabled: boolean
  dev_mode: boolean
}

export interface LoginResponse {
  access_token: string
  refresh_token: string
  token_type: string
}

export interface RefreshResponse {
  access_token: string
  token_type: string
}

export interface WsStatusPayload {
  status: string
  started_at: string | null
  cycle_metrics: {
    cycle_count: number
    average_cycle_duration: number
    last_cycle_time: number | null
  }
  portfolio: {
    balances: Record<string, number>
    positions: unknown[]
    total_value: number
  }
  metrics: Record<string, number>
  regime: string | null
  trades: Trade[]
  strategy_stats: Record<string, StrategyStats>
  open_positions: Position[]
  cycle_log_latest: CycleLogEntry | null
}

// Cycle Log types
export interface StrategySignalEntry {
  name: string
  action: string
  confidence: number
}

export interface EnsembleResult {
  action: string
  confidence: number
  agreement: number
  reason: string | null
  agreeing_strategies: string[]
}

export interface RiskCheckResult {
  passed: boolean
  reason: string | null
  stage: string
}

export interface CycleOrderInfo {
  symbol: string
  side: string
  quantity: number
  price: number
}

export interface CycleSymbolDetail {
  price: number
  regime: string | null
  trend: string | null
  strategies: StrategySignalEntry[]
  ensemble: EnsembleResult
  risk_check: RiskCheckResult | null
  final_action: string
  order: CycleOrderInfo | null
}

export interface CycleLogEntry {
  cycle_num: number
  timestamp: string
  duration_ms: number | null
  symbols: Record<string, CycleSymbolDetail>
}

export interface CycleLogResponse {
  cycle_log: CycleLogEntry[]
}

// Settings types
export interface SettingItem {
  key: string
  value: unknown
  default: unknown
  section: string
  description: string
  type: string
  requires_restart: boolean
  options?: string[]
}

export interface SettingsResponse {
  settings: SettingItem[]
}

export interface SettingsUpdateResponse {
  success: boolean
  changed: string[]
  previous: Record<string, unknown>
}

// Engine types (V5 multi-engine system)
export type EngineStatusValue = 'stopped' | 'running' | 'paused' | 'error'

export interface EngineInfo {
  name: string
  description: string
  status: EngineStatusValue
  cycle_count: number
  total_pnl: number
  allocated_capital: number
  position_count: number
  max_positions: number
  loop_interval: number
  error: string | null
  // V5-003: description metadata and tracked symbols
  role_ko?: string
  role_en?: string
  description_ko?: string
  key_params?: string
  symbols?: string[] | string[][]
}

export type EnginesResponse = Record<string, EngineInfo>

export interface EngineActionResponse {
  success: boolean
  engine: string
  action: string
}

export interface EngineDecisionStep {
  label: string
  observation: string
  threshold: string
  result: string
  category: 'evaluate' | 'decide' | 'execute' | 'skip'
}

export interface EngineCycleLogEntry {
  engine_name: string
  cycle_num: number
  timestamp: string
  duration_ms: number
  actions_taken: Record<string, unknown>[]
  positions: Record<string, unknown>[]
  signals: Record<string, unknown>[]
  pnl_update: number
  metadata: Record<string, unknown>
  decisions: EngineDecisionStep[]
}

export interface EngineCycleLogResponse {
  engine: string
  cycle_log: EngineCycleLogEntry[]
}

export interface EnginePositionsResponse {
  engine: string
  positions: Record<string, unknown>[]
}

// V5-006: Performance tracking types
export interface EngineMetrics {
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number
  total_pnl: number
  avg_profit_per_trade: number
  sharpe_ratio: number
  max_drawdown: number
  profit_factor: number
  avg_hold_time_min: number
  cost_ratio: number
  best_trade: number
  worst_trade: number
  total_cost: number
}

export interface PerformanceSummary {
  engines: Record<string, EngineMetrics>
  totals: {
    total_pnl: number
    total_trades: number
    overall_sharpe: number
    overall_win_rate: number
    total_cost: number
  }
  window_hours: number
}

// Scanner / Opportunity types
export type OpportunityTypeName = 'funding_rate' | 'volatility' | 'cross_exchange_spread' | 'correlation'

export interface ScannerOpportunity {
  symbol: string
  type: OpportunityTypeName
  score: number
  metrics: Record<string, unknown>
  discovered_at: string
  expires_at: string
  source_exchange: string
}

export interface ScannerSummaryItem {
  count: number
  top_score: number
  symbols: string[]
}

export interface ScannerResponse {
  summary: Record<OpportunityTypeName, ScannerSummaryItem>
  opportunities: Record<OpportunityTypeName, ScannerOpportunity[]>
}
