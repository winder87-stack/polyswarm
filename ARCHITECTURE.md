# System Architecture

## Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                        POLYMARKET BOT                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ News Monitor │    │ External Odds│    │  Historical  │      │
│  │   Service    │    │   Service    │    │   Analysis   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │                    AI SWARM                          │      │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────┐ │      │
│  │  │ Claude │ │ Gemini │ │  GPT   │ │DeepSeek│ │Pplx│ │      │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────┘ │      │
│  │                    │                                 │      │
│  │         Weighted Consensus + Calibration             │      │
│  └──────────────────────────┬───────────────────────────┘      │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │                 TRADING SWARM                        │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │      │
│  │  │  Signal    │  │  Category  │  │   Entry    │     │      │
│  │  │ Generator  │  │ Specialist │  │  Timing    │     │      │
│  │  └────────────┘  └────────────┘  └────────────┘     │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │      │
│  │  │Contrarian  │  │  Position  │  │    Risk    │     │      │
│  │  │ Detector   │  │  Manager   │  │  Manager   │     │      │
│  │  └────────────┘  └────────────┘  └────────────┘     │      │
│  └──────────────────────────┬───────────────────────────┘      │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │                POLYMARKET CLIENT                     │      │
│  │  ┌────────────┐              ┌────────────┐         │      │
│  │  │ Gamma API  │              │  CLOB API  │         │      │
│  │  │(Market Data)│              │ (Trading)  │         │      │
│  │  └────────────┘              └────────────┘         │      │
│  └──────────────────────────────────────────────────────┘      │
│                             │                                   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │    POLYMARKET    │
                       │     (Polygon)    │
                       └──────────────────┘
```

┌─────────────────────────────────────────────────────────────────┐
│                     EXTERNAL DATA (ALL FREE)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Google News  │    │  PredictIt   │    │  Metaculus   │      │
│  │  (RSS/Free)  │    │ (API/Free)   │    │ (API/Free)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
│  No API keys required for any external data source              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

## Data Flow

### 1. Market Analysis Flow
```
Market Scanner
│
▼
┌────────────────┐
│ Get Markets    │ ──► Gamma API (public data)
│ Filter/Rank    │
└───────┬────────┘
│
▼
┌────────────────┐
│ News Context   │ ──► Google News (RSS)
│ External Odds  │ ──► PredictIt, Metaculus
└───────┬────────┘
│
▼
┌────────────────┐
│ AI Swarm Query │ ──► 5 models in parallel
│ (with context) │
└───────┬────────┘
│
▼
┌────────────────┐
│ Consensus Calc │ ──► Weighted average
│ + Calibration  │ ──► Historical adjustment
└───────┬────────┘
│
▼
┌────────────────┐
│ Signal Gen     │ ──► Edge, EV, Kelly sizing
└───────┬────────┘
│
▼
┌────────────────┐
│ Risk Check     │ ──► Position limits, drawdown
│ Timing Check   │ ──► Spread, momentum
│ Category Check │ ──► Category-specific rules
└───────┬────────┘
│
▼
┌────────────────┐
│ Execute Trade  │ ──► CLOB API (authenticated)
└────────────────┘
```

### 2. Position Management Flow
```
Every Hour:
│
▼
┌────────────────┐
│ Update Prices  │ ──► Gamma API
└───────┬────────┘
│
▼
┌────────────────┐
│ Check Each     │
│ Position       │
│ - Scale in?    │ ──► If down 15%+ and edge still good
│ - Scale out?   │ ──► If up 20%+
│ - Close?       │ ──► If edge gone or AI flipped
└───────┬────────┘
│
▼
┌────────────────┐
│ Reanalyze      │ ──► AI Swarm (if needed)
│ Markets        │
└───────┬────────┘
│
▼
┌────────────────┐
│ Execute        │ ──► Scale in/out/close
│ Actions        │
└────────────────┘
```

## Component Details

### AI Swarm (`src/agents/swarm_agent.py`)
- **Parallel Queries**: 5 AI models queried simultaneously
- **Response Caching**: 5-minute TTL to avoid redundant queries
- **Weighted Consensus**: Model-specific importance weights
- **Cost Tracking**: API usage monitoring across all providers
- **Chain-of-Thought**: Multi-step reasoning prompts
- **Debate Mode**: YES vs NO position arguments
- **Red Team Analysis**: Critical thinking and counter-arguments
- **Calibration**: Historical bias correction

### Trading Swarm (`src/agents/trading_swarm.py`)
- **Signal Generation**: Edge, EV, and Kelly calculations
- **Position Sizing**: Kelly criterion with uncertainty adjustment
- **Expected Value**: Risk-adjusted profit calculations
- **Trade Execution**: Paper/live mode with order splitting
- **Position Tracking**: Real-time P&L and exposure monitoring
- **News Integration**: Breaking news fast-path trading

### Risk Manager (`src/strategies/risk_manager.py`)
- **Multi-Layer Protection**: 15+ concurrent risk checks
- **Kelly with Uncertainty**: Confidence-adjusted position sizing
- **Drawdown Protection**: Progressive size reduction and trading pauses
- **Correlation Adjustment**: Reduces size for related positions
- **Sharpe Ratio Tracking**: Risk-adjusted performance metrics
- **Bankroll Management**: Dynamic limits based on current capital
- **Emergency Stops**: Automatic pause at critical thresholds

### Entry Timing (`src/strategies/entry_timing.py`)
- **Spread Scoring**: Bid-ask spread analysis for optimal entry
- **Momentum Detection**: Price trend and velocity analysis
- **Time-of-Day Optimization**: Market hours and liquidity patterns
- **Order Book Analysis**: Depth and slippage assessment
- **Order Splitting**: Large position execution across multiple fills
- **VIX Integration**: Volatility-based timing adjustments

### Position Manager (`src/strategies/position_manager.py`)
- **Position Lifecycle**: Entry → monitoring → exit
- **Scale In/Out Logic**: Profit-taking and averaging strategies
- **Stop Loss Monitoring**: Risk-based exit triggers
- **Reanalysis Triggers**: Market condition change detection
- **Partial Closures**: Profit-locking and risk reduction
- **Performance Attribution**: Individual position P&L tracking

### Category Specialist (`src/strategies/category_specialist.py`)
- **Per-Category Weights**: Optimized AI model importance
- **Custom Thresholds**: Category-specific edge requirements
- **Exposure Limits**: Category-based position constraints
- **News Source Selection**: Relevant information feeds
- **Time Preferences**: Optimal expiry ranges per category
- **Market Efficiency**: Different assumptions per category type

### Contrarian Detector (`src/strategies/contrarian_detector.py`)
- **Sentiment Divergence**: Public vs sharp money positioning
- **Overreaction Detection**: Extreme move reversal patterns
- **Consensus Traps**: Crowd behavior analysis
- **Sharp Money Tracking**: Professional positioning signals
- **Sentiment Analysis**: News and social media sentiment
- **Contrarian Scoring**: Statistical edge calculation

### News Monitor (`src/services/news_monitor.py`)
- **FREE Sources Only**: Google News RSS + generic RSS feeds
- **Deduplication**: Intelligent duplicate detection
- **Relevance Scoring**: Market-specific news ranking
- **Market Matching**: News-to-market association algorithms
- **Breaking News Detection**: High-impact event identification
- **Zero API Keys**: All sources are public RSS feeds

### External Odds (`src/services/external_odds.py`)
- **FREE Sources Only**: PredictIt + Metaculus public APIs
- **Consensus Calculation**: Cross-platform agreement analysis
- **Arbitrage Detection**: Risk-free profit opportunities
- **Sharp Money Signals**: Professional trader positioning
- **Market Efficiency**: Comparison with Polymarket pricing
- **Zero API Keys**: All sources are public APIs

### Historical Collector (`src/analysis/historical_collector.py`)
- **Market Data Storage**: Complete resolved market history
- **Price Snapshots**: Time-series price data collection
- **Trade Records**: Transaction-level data storage
- **Data Backfilling**: Historical data population
- **Scheduled Updates**: Daily resolved market collection
- **Performance Tracking**: Historical accuracy analysis

### Pattern Analyzer (`src/analysis/pattern_analyzer.py`)
- **Category Performance**: Per-category accuracy analysis
- **Price Pattern Recognition**: Movement and flip detection
- **Volume Signal Analysis**: Trading volume correlations
- **Calibration Curves**: Market pricing accuracy assessment
- **Profitable Pattern Discovery**: Statistical edge identification
- **Market Efficiency Scoring**: Pricing quality metrics

### AI Accuracy Tracker (`src/analysis/ai_accuracy_tracker.py`)
- **Prediction Storage**: All AI predictions with context
- **Resolution Tracking**: Outcome verification and updates
- **Brier Score Calculation**: Probabilistic accuracy metrics
- **Model Ranking**: Individual and ensemble performance
- **Calibration Analysis**: Bias detection and correction
- **Weight Optimization**: Optimal model weighting suggestions

## Database Schema

### `historical.db` - Historical Market Data
```sql
-- Resolved markets with outcomes
CREATE TABLE markets (
    condition_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    category TEXT,
    created_at TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution TEXT,  -- "YES", "NO", "INVALID"
    final_yes_price REAL,
    final_no_price REAL,
    total_volume REAL,
    total_liquidity REAL,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hourly price snapshots
CREATE TABLE price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    yes_price REAL NOT NULL,
    no_price REAL NOT NULL,
    volume_24h REAL DEFAULT 0,
    FOREIGN KEY (condition_id) REFERENCES markets(condition_id)
);

-- Individual trades (if available)
CREATE TABLE trade_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    side TEXT NOT NULL,  -- "BUY", "SELL"
    outcome TEXT NOT NULL,  -- "YES", "NO"
    price REAL NOT NULL,
    size REAL NOT NULL,
    is_taker BOOLEAN DEFAULT 0,
    FOREIGN KEY (condition_id) REFERENCES markets(condition_id)
);

-- Collection metadata
CREATE TABLE collection_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `ai_predictions.db` - AI Performance Tracking
```sql
-- All AI predictions with outcomes
CREATE TABLE predictions (
    prediction_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    condition_id TEXT NOT NULL,
    question TEXT,
    ai_probability REAL NOT NULL,  -- Ensemble prediction
    model_probabilities TEXT,  -- JSON: {"claude": 0.7, "gpt": 0.65, ...}
    confidence REAL,  -- Ensemble confidence (0-1)
    market_probability REAL,  -- Market price at prediction time
    volume REAL,
    hours_until_close REAL,
    category TEXT,

    -- Resolution data (filled later)
    resolution TEXT,  -- "YES", "NO", "INVALID"
    resolved_at TIMESTAMP,
    was_correct BOOLEAN,  -- Directional accuracy
    brier_score REAL,  -- Calibration metric
    profit_if_traded REAL,  -- Hypothetical P&L

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance cache
CREATE TABLE model_performance (
    model_name TEXT NOT NULL,
    date DATE NOT NULL,
    predictions_count INTEGER DEFAULT 0,
    accuracy REAL DEFAULT 0,
    brier_score REAL DEFAULT 0,
    avg_confidence REAL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_name, date)
);
```

### `calibration.db` - Model Calibration Data
```sql
-- Calibration curves by model and market bucket
CREATE TABLE calibration_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    price_bucket_start REAL NOT NULL,  -- 0.0, 0.1, 0.2, etc.
    price_bucket_end REAL NOT NULL,    -- 0.1, 0.2, 0.3, etc.
    predictions_count INTEGER DEFAULT 0,
    actual_yes_rate REAL,  -- What % actually resolved YES
    expected_yes_rate REAL,  -- What market priced (midpoint)
    calibration_error REAL,  -- actual - expected
    absolute_error REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(model_name, price_bucket_start)
);

-- Category-specific calibration
CREATE TABLE category_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    model_name TEXT NOT NULL,
    price_bucket_start REAL NOT NULL,
    predictions_count INTEGER DEFAULT 0,
    actual_yes_rate REAL,
    calibration_error REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(category, model_name, price_bucket_start)
);
```

## Configuration Files

### `.env` - Runtime Configuration
```bash
# AI API Keys
ANTHROPIC_API_KEY=[your-anthropic-key]
GOOGLE_API_KEY=[your-google-key]
OPENAI_API_KEY=[your-openai-key]
OPENROUTER_API_KEY=[your-openrouter-key]
PERPLEXITY_API_KEY=[your-perplexity-key]

# Polymarket Credentials
POLYGON_WALLET_PRIVATE_KEY=...
POLYGON_FUNDER_ADDRESS=...
POLYMARKET_SIGNATURE_TYPE=0
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...

# Trading Parameters
PAPER_TRADING=true
BANKROLL=1000
MIN_EDGE_THRESHOLD=0.08
KELLY_MULTIPLIER=0.25

# System Settings
LOG_LEVEL=INFO
SCAN_INTERVAL_MINUTES=30
```

### `config/categories.yaml` - Category-Specific Settings
```yaml
politics:
  enabled: true
  model_weights:
    perplexity: 1.5  # News critical
  min_edge: 0.08
  news_sources: [google_news, twitter, fivethirtyeight]

sports:
  enabled: true
  min_edge: 0.10  # More efficient
  preferred_time_to_expiry:
    min_days: 1
    max_days: 14
```

## API Integrations

### Polymarket APIs
- **Gamma API**: Public market data, statistics, historical prices
- **CLOB API**: Authenticated trading, order management, positions

### AI Provider APIs
- **Anthropic Claude**: Text generation, reasoning, analysis
- **Google Gemini**: Multimodal analysis, market context
- **OpenAI GPT**: General intelligence, pattern recognition
- **DeepSeek**: Fast inference, cost-effective analysis
- **Perplexity**: Real-time web search, news integration

### External Data Sources (All FREE)
- **Google News**: RSS feeds, no API keys required
- **PredictIt**: Public API, no authentication needed
- **Metaculus**: Public API, no authentication needed

### External Data Philosophy
All external data sources are FREE and require no API keys:

- **Simplifies setup and maintenance** - No API key management
- **No rate limit concerns** for external sources - Public APIs only
- **Perplexity (AI swarm) provides web search** including Reddit automatically
- **PredictIt and Metaculus** provide sufficient external odds for consensus

## Performance Metrics

### AI Model Metrics
- **Accuracy**: Directional prediction correctness
- **Brier Score**: Probabilistic calibration quality
- **Confidence Calibration**: Over/under-confidence detection
- **Response Time**: Query latency and throughput

### Trading Performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside deviation focus
- **Max Drawdown**: Peak-to-trough loss
- **Win Rate**: Percentage profitable trades
- **Profit Factor**: Gross profit / gross loss

### System Metrics
- **API Costs**: Usage tracking across providers
- **Trade Latency**: Signal to execution time
- **Market Coverage**: Percentage of markets analyzed
- **Data Freshness**: Age of market data and news

## Security Considerations

### API Key Management
- Environment variable storage only
- No keys in source code or logs
- Rotating key strategy for production

### Trading Safety
- Paper trading default mode
- Position size limits
- Daily loss limits with auto-stop
- Manual confirmation for large trades

### Data Privacy
- No user data collection
- Local database storage only
- No telemetry or tracking

## Deployment Architecture

### Development Environment
```
Local Machine
├── Python 3.11+
├── SQLite databases
├── API key management
└── Paper trading only
```

### Production Environment
```
Cloud Server/VM
├── Docker containerization
├── PostgreSQL (optional upgrade)
├── Automated key rotation
├── Live trading capability
└── Monitoring/alerts
```

### Scaling Considerations
- **Horizontal Scaling**: Multiple bot instances per category
- **Database Sharding**: Partition by market category or time
- **API Rate Limiting**: Distributed across multiple API keys
- **Caching Layer**: Redis for frequently accessed data

## Monitoring & Alerting

### Health Checks
- API connectivity status
- Database responsiveness
- Memory and CPU usage
- Trade execution success rate

### Performance Alerts
- Sharpe ratio degradation
- Increasing drawdown
- API rate limit approaches
- Unusual trade rejection rates

### Business Metrics
- Daily P&L tracking
- Position exposure monitoring
- Market coverage completeness
- AI model performance trends

This architecture provides a robust, scalable foundation for automated prediction market trading with sophisticated AI analysis, comprehensive risk management, and multi-source data integration.
