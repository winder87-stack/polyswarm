# 🎯 Polymarket AI Trading Bot

Automated prediction market trading bot using a 5-model AI swarm for [Polymarket](https://polymarket.com).

## Features

### 🤖 AI Swarm (5 Models)
- **Claude 3.5** - Best reasoning and analysis
- **Gemini Pro** - Strong reasoning capabilities
- **GPT-3.5/4** - General intelligence
- **DeepSeek V3.2** - Fast, cost-effective via OpenRouter
- **Perplexity Sonar Pro** - Real-time web search for news

### 📊 Trading Features
- Weighted consensus from multiple AI models
- Kelly Criterion position sizing with uncertainty adjustment
- Expected value calculations
- Smart entry timing (spread, momentum, time-of-day)
- Category-specific strategies (politics, sports, crypto)
- Active position management (scale in/out, stop loss)

### 📰 Market Intelligence
- Real-time news monitoring (Google News + RSS feeds)
- Multi-source odds aggregation (PredictIt + Metaculus)
- Contrarian signal detection
- Historical pattern analysis
- AI calibration tracking

### 🛡️ Risk Management
- Maximum position and exposure limits
- Drawdown protection with auto-pause
- Correlation-adjusted sizing
- Daily/weekly loss limits
- Paper trading mode

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo>
cd polymarket-trader
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required keys:
- `ANTHROPIC_API_KEY` - [Get here](https://console.anthropic.com/)
- `GOOGLE_API_KEY` - [Get here](https://aistudio.google.com/app/apikey)
- `OPENAI_API_KEY` - [Get here](https://platform.openai.com/api-keys)
- `OPENROUTER_API_KEY` - [Get here](https://openrouter.ai/keys)
- `PERPLEXITY_API_KEY` - [Get here](https://www.perplexity.ai/settings/api)

### 3. Setup Polymarket Wallet
```bash
# Generate new wallet (or use existing)
python scripts/generate_wallet.py

# Generate API credentials
python scripts/generate_api_keys.py

# Set token allowances
python scripts/set_allowances.py
```

### 4. Run Pre-Flight Check
```bash
# Comprehensive system check (recommended)
python main.py preflight

# Or run directly
python scripts/pre_flight_check.py
```

### 5. Run Paper Trading
```bash
# Quick 1-hour test
python scripts/paper_trading_24h.py --hours 1 --interval 10

# Full 24-hour test
python scripts/paper_trading_24h.py
```

## Usage

### List Markets
```bash
python main.py markets --limit 20
python main.py markets --category politics
```

### Analyze Specific Market
```bash
python main.py analyze will-trump-win-2024
python main.py analyze will-trump-win-2024 --deep  # Full analysis
```

### Scan for Opportunities
```bash
python main.py scan --min-edge 0.08
python main.py scan-contrarian
python main.py scan-news
```

### Paper Trading
```bash
python main.py trade --paper --interval 30
```

### Live Trading ⚠️
```bash
# CAUTION: Real money at risk!
python main.py trade --live --max-trades 5
```

## Testing

### Pre-Flight Check (Recommended)
Run comprehensive tests before paper trading:
```bash
python main.py preflight                  # All system checks
python scripts/pre_flight_check.py        # Direct script run
```

Tests include:
- ✅ Environment variables & API keys
- ✅ Python imports & dependencies
- ✅ Project structure & file integrity
- ✅ AI model connectivity (5 models)
- ✅ Polymarket API connection
- ✅ Core component initialization
- ✅ External data sources (FREE)
- ✅ End-to-end market analysis
- ✅ System resources & disk space

### Individual Tests
```bash
# Test specific components
python scripts/test_markets.py            # Polymarket connection
python scripts/test_swarm.py              # AI models
python main.py test                       # System components

# Paper trading tests
python scripts/paper_trading_24h.py --hours 1 --interval 10
```

## Project Structure
polymarket-trader/
├── main.py                      # CLI entry point
├── config/
│   └── categories.yaml          # Category settings
├── src/
│   ├── agents/                  # AI swarm + trading
│   ├── models/                  # AI provider classes
│   ├── connectors/              # Polymarket API
│   ├── strategies/              # Risk, timing, positions
│   ├── services/                # News, external odds
│   └── analysis/                # Historical, calibration
├── scripts/                     # Utility scripts
├── data/                        # SQLite databases
├── logs/                        # Trading logs
└── reports/                     # Generated reports

## Configuration

### AI Model Weights
```python
MODEL_WEIGHTS = {
    "claude": 1.3,      # Best reasoning
    "gemini": 1.3,      # Strong reasoning
    "gpt": 1.2,         # General intelligence
    "perplexity": 1.2,  # Web search advantage
    "deepseek": 1.0,    # Fast baseline
}
```

### Risk Limits
```python
RiskLimits(
    max_daily_loss=200,      # Stop after $200 loss
    max_position_size=100,   # $100 max per trade
    max_positions=10,        # Max 10 open positions
    max_exposure=500,        # $500 total exposure
    min_edge=0.08,           # 8% minimum edge
    min_confidence=0.5,      # 50% model agreement
    kelly_multiplier=0.25,   # Quarter Kelly
    max_drawdown_percent=20, # Pause at 20% drawdown
)
```

## External Data Sources

All external data sources are FREE and require no API keys:

| Source | Type | URL |
|--------|------|-----|
| Google News | News | RSS feeds |
| PredictIt | Odds | https://www.predictit.org/api/marketdata/all/ |
| Metaculus | Odds | https://www.metaculus.com/api2/ |
| NPR, NYT, ESPN | News | RSS feeds |

**Note**: Perplexity (in your AI swarm) also searches Reddit and other sites automatically via web search.

## Cost Estimates

| Model | Cost per 1M tokens | Typical query cost |
|-------|-------------------|-------------------|
| Claude 3.5 Haiku | $0.25 / $1.25 | ~$0.005 |
| Gemini Pro | $0.50 / $1.50 | ~$0.003 |
| GPT-3.5 Turbo | $0.50 / $1.50 | ~$0.003 |
| DeepSeek V3.2 | $0.24 / $0.38 | ~$0.002 |
| Perplexity Sonar | $3.00 / $15.00 | ~$0.01 |

**Total per market analysis: ~$0.02-0.05**

## Disclaimer

⚠️ **This is experimental software. Use at your own risk.**

- Prediction markets are risky - you can lose money
- Past performance does not guarantee future results
- AI models can be wrong
- Start with paper trading
- Never invest more than you can afford to lose

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

### Debug Configurations

Press `F5` to run debug, select configuration:

- **Python: Paper Trading** - Debug paper trading mode
- **Python: Test Markets** - Debug market fetching
- **Python: Current File** - Debug any open file

## 🔧 Quick Start

### 1. Generate Wallet

```bash
python scripts/generate_wallet.py
```
Save the private key securely!

### 2. Set Up Environment

Edit `.env` with your private key and AI API keys.

### 3. Generate Polymarket API Keys

```bash
python scripts/generate_api_keys.py
```
Copy the output back into `.env`.

### 4. Set Token Allowances

```bash
# Requires MATIC in wallet for gas
python scripts/set_allowances.py
```

### 5. Run Pre-Flight Check

```bash
# Comprehensive system check (recommended)
python main.py preflight

# Or run individual tests
python scripts/test_markets.py    # Polymarket connection
python scripts/test_swarm.py      # AI models
```

### 6. Start Paper Trading

```bash
python main.py trade --paper
```

## 💡 Cursor AI Prompts

Use these prompts in Cursor (Ctrl+L) to build out the project:

### Generate Missing Files

```
Look at the project structure in README.md. Create the files that are
missing in src/agents/swarm_agent.py. Follow the patterns in .cursorrules.
```

### Debug Issues

```
I'm getting this error: [paste error]

Check the CURSOR_CONTEXT.md for common solutions and fix the issue.
```

### Add New Feature

```
Add a new AI model provider for Groq. Follow the pattern in
src/models/model_factory.py and add it to the SWARM_MODELS config.
```

### Explain Code

```
Explain how the trading signal generation works in trading_swarm.py.
What determines if we should trade?
```

## ⚠️ Important Notes

1. **Paper Trading First**: Always test with `PAPER_TRADING=true`
2. **Never Commit .env**: Your private keys should never be in git
3. **Gas Fees**: You need MATIC on Polygon for transactions
4. **Geographic Restrictions**: Polymarket has geographic restrictions

## 📚 Resources

- [Polymarket Docs](https://docs.polymarket.com/)
- [py-clob-client](https://github.com/Polymarket/py-clob-client)
- [Polymarket Agents](https://github.com/Polymarket/agents)
- [Cursor IDE](https://cursor.sh/)

## 📄 License

MIT License - See LICENSE file
