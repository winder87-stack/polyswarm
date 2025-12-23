# Polymarket AI Trading Bot - Quick Reference

## 🚀 Quick Start
```bash
cd ~/polymarket-trader
source venv/bin/activate

# Test everything works
python main.py test

# Run 24-hour paper trading test
python scripts/paper_trading_24h.py --hours 24 --interval 30

# Scan for opportunities
python main.py scan --min-edge 0.08 --limit 10
```

## 📁 File Map

### Core Trading
| File | Purpose |
|------|---------|
| `main.py` | CLI entry point |
| `src/agents/swarm_agent.py` | 5-model AI swarm queries |
| `src/agents/trading_swarm.py` | Signal generation + execution |
| `src/connectors/polymarket_client.py` | Polymarket API |
| `src/models/model_factory.py` | AI provider classes |

### Strategies
| File | Purpose |
|------|---------|
| `src/strategies/risk_manager.py` | Advanced risk limits, Kelly sizing, drawdown protection |
| `src/strategies/entry_timing.py` | Spread, momentum, time-of-day optimization |
| `src/strategies/position_manager.py` | Scale in/out, stop loss (TODO) |
| `src/strategies/category_specialist.py` | Per-category settings (TODO) |
| `src/strategies/contrarian_detector.py` | Contrarian signals (TODO) |

### Services
| File | Purpose |
|------|---------|
| `src/services/news_monitor.py` | Google News + RSS feeds (FREE) |
| `src/services/external_odds.py` | PredictIt + Metaculus (FREE) |

### Analysis
| File | Purpose |
|------|---------|
| `src/analysis/historical_collector.py` | Historical data SQLite |
| `src/analysis/pattern_analyzer.py` | Pattern recognition |
| `src/analysis/ai_accuracy_tracker.py` | Track AI predictions |
| `src/analysis/model_calibration.py` | Calibration adjustments (TODO) |

### Scripts
| File | Purpose |
|------|---------|
| `scripts/paper_trading_24h.py` | 24-hour paper test |
| `scripts/test_swarm.py` | Test AI models |
| `scripts/test_markets.py` | Test Polymarket |
| `scripts/generate_api_keys.py` | Generate PM credentials |
| `scripts/generate_wallet.py` | Create new wallet |
| `scripts/monitor_paper_trading.py` | Monitor running tests |
| `scripts/test_all_models.py` | Test all 5 AI models |

## 🔧 Configuration

### .env File
```bash
# AI APIs (need all 5 for full swarm)
ANTHROPIC_API_KEY=[your-anthropic-key]
GOOGLE_API_KEY=[your-google-key]
OPENAI_API_KEY=[your-openai-key]
OPENROUTER_API_KEY=[your-openrouter-key]
PERPLEXITY_API_KEY=[your-perplexity-key]

# Polymarket
POLYGON_WALLET_PRIVATE_KEY=...     # No 0x prefix
POLYGON_FUNDER_ADDRESS=0x...       # With 0x prefix
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...
POLYMARKET_SIGNATURE_TYPE=0

# Trading
PAPER_TRADING=true
BANKROLL=1000
MAX_POSITION_SIZE=100
MIN_EDGE_THRESHOLD=0.08
```

### AI Swarm (src/agents/swarm_agent.py)
```python
SWARM_MODELS = {
    "claude": (True, "claude", "claude-3-5-haiku-20241022"),
    "gemini": (True, "google", "gemini-pro"),
    "gpt": (True, "openai", "gpt-3.5-turbo"),
    "deepseek": (True, "openrouter", "deepseek/deepseek-v3.2"),
    "perplexity": (True, "perplexity", "sonar-pro"),
}

MODEL_WEIGHTS = {
    "claude": 1.3,      # Best reasoning
    "gemini": 1.3,      # Strong reasoning
    "gpt": 1.2,         # General intelligence
    "perplexity": 1.2,  # Real-time web search
    "deepseek": 1.0,    # Fast, cheap
}
```

### Category Settings (config/categories.yaml)
```yaml
politics:
  enabled: true
  model_weights:
    claude: 1.3
    perplexity: 1.5  # News important
  min_edge: 0.08
  max_exposure_pct: 30

sports:
  enabled: true
  min_edge: 0.10  # More efficient
  max_exposure_pct: 25

crypto:
  enabled: true
  min_edge: 0.12  # Volatile
  max_exposure_pct: 20
```

### Risk Limits (src/strategies/risk_manager.py)
```python
@dataclass
class RiskLimits:
    max_daily_loss: float = 200
    max_position_size: float = 100
    max_positions: int = 10
    max_exposure: float = 500
    min_edge: float = 0.08
    min_confidence: float = 0.5
    min_expected_value: float = 0.02
    kelly_fraction: float = 0.25
    max_drawdown_percent: float = 20
```

## 📊 Data Structures

### Market
```python
@dataclass
class Market:
    condition_id: str
    question: str
    yes_price: float      # 0.0 - 1.0
    no_price: float       # 0.0 - 1.0
    volume: float         # Total USD volume
    liquidity: float      # Available liquidity
    category: str
    end_date: str
    slug: str
    yes_token_id: str
    no_token_id: str
```

### TradingSignal
```python
@dataclass
class TradingSignal:
    market: Market
    direction: str                    # "YES" or "NO"
    probability: float                # AI estimate (0-1)
    market_probability: float         # Current price
    edge: float                       # probability - market_probability
    confidence: float                 # Model agreement (0-1)
    expected_value: float             # edge * confidence
    kelly_fraction: float             # Optimal bet size
    recommended_size: float           # USD amount
    reasoning: str
    model_votes: Dict[str, float]
    news_context: Optional[List[NewsItem]]
    timing_score: float
    contrarian_signals: List[ContrarianSignal]

    @property
    def is_actionable(self) -> bool:
        return self.edge >= 0.08 and self.confidence >= 0.5
```

## 🔌 API Endpoints
```python
# Polymarket
GAMMA_API = "https://gamma-api.polymarket.com"   # Market data (public)
CLOB_API = "https://clob.polymarket.com"         # Trading (auth required)

# AI Providers
ANTHROPIC = "https://api.anthropic.com"
GOOGLE = "https://generativelanguage.googleapis.com"
OPENAI = "https://api.openai.com/v1"
OPENROUTER = "https://openrouter.ai/api/v1"
PERPLEXITY = "https://api.perplexity.ai"

# External Odds
PREDICTIT = "https://www.predictit.org/api/marketdata/all/"
METACULUS = "https://www.metaculus.com/api2/"
```

## 📡 External Data Sources (All FREE - No API Keys)

### News Monitoring
- **Google News RSS** (free) - Real-time news feeds
- **NPR, NYT, ESPN RSS feeds** (free) - Major news outlets
- **No API keys required** - All RSS feeds are public

### External Odds
- **PredictIt**: `https://www.predictit.org/api/marketdata/all/` (free)
- **Metaculus**: `https://www.metaculus.com/api2/` (free)
- **No API keys required** - Public APIs only

### Web Search
- **Perplexity Sonar Pro** (included in AI swarm)
- Searches Reddit, news sites, etc. automatically
- No additional API keys needed

## 🛠️ CLI Commands
```bash
# Markets
python main.py markets                    # List top markets
python main.py markets --category politics --limit 20
python main.py analyze trump-2024         # Analyze specific market
python main.py analyze trump-2024 --deep  # Full analysis with debate

# Scanning
python main.py scan                       # Find opportunities
python main.py scan --min-edge 0.10 --min-confidence 0.6
python main.py scan-contrarian            # Contrarian opportunities
python main.py scan-news                  # News-driven opportunities

# Trading
python main.py trade --paper              # Paper trading
python main.py trade --paper --interval 30 --max-trades 10
python main.py trade --live               # REAL MONEY (careful!)

# Data & Analysis
python main.py collect-history --days 365
python main.py analyze-history
python main.py calibration-report
python main.py performance-report

# Status
python main.py status                     # Current positions + P&L
python main.py positions                  # Detailed positions
python main.py test                       # System test
```

## 🧪 Testing
```bash
# Test individual components
python scripts/test_markets.py            # Polymarket connection
python scripts/test_swarm.py              # AI models

# Test trading system
python main.py test                       # Full system test

# Paper trading
python scripts/paper_trading_24h.py       # 24-hour test
python scripts/paper_trading_24h.py --hours 1 --interval 10  # Quick test
```

## 🐛 Debugging

### Check AI Models
```python
from src.models.model_factory import model_factory, test_models
test_models()  # Tests all 5 models
```

### Check Polymarket
```python
from src.connectors.polymarket_client import polymarket
await polymarket.test_connection()
```

### Check Environment
```python
import os
keys = ["ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
        "OPENROUTER_API_KEY", "PERPLEXITY_API_KEY",
        "POLYGON_WALLET_PRIVATE_KEY", "POLYGON_FUNDER_ADDRESS"]
for k in keys:
    v = os.getenv(k)
    print(f"{'✅' if v else '❌'} {k}: {'Set' if v else 'Missing'}")
```

### View Logs
```bash
tail -f logs/trading_bot_*.log
tail -f logs/paper_trading_*.log
```

## ⚡ Common Patterns

### Analyze and Trade
```python
from src.agents.trading_swarm import TradingSwarm

swarm = TradingSwarm(paper_trading=True)
signal = await swarm.analyze_market(market)

if signal and signal.is_actionable:
    result = await swarm.execute_signal(signal)
    print(f"Trade: {result}")
```

### Scan Markets
```python
signals = await swarm.find_opportunities(
    min_edge=0.08,
    min_confidence=0.5,
    limit=10
)
for signal in signals:
    swarm.print_signal(signal)
```

### Check Timing
```python
from src.strategies.entry_timing import EntryTimingOptimizer

timing = EntryTimingOptimizer()
signal = await timing.analyze_timing(market, direction="YES")

if signal.should_trade_now:
    # Execute with size_multiplier adjustment
    size = base_size * signal.size_multiplier
```

### Manage Positions
```python
from src.strategies.position_manager import PositionManager

pm = PositionManager(swarm)
await pm.update_all_positions()

for pos in pm.positions.values():
    action = await pm.check_position_actions(pos)
    if action:
        print(f"Action needed: {action}")
```

## ⚠️ Important Notes

1. **Always paper trade first** - Run 24h+ paper test before live
2. **Start small** - Even with edge, variance is high
3. **Monitor drawdowns** - Stop at 25% drawdown
4. **Check API costs** - ~$0.03-0.08 per analysis
5. **News is critical** - Perplexity web search is your edge
6. **Categories differ** - Politics ≠ Sports ≠ Crypto
7. **Timing matters** - Don't chase, buy weakness

### Note on Reddit & Betfair
Reddit and Betfair APIs were removed to simplify setup.

- **Perplexity already searches Reddit** via web search
- **PredictIt and Metaculus** provide sufficient external odds
- **All external data sources are now FREE** with no API keys
