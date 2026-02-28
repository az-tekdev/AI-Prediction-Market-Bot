# AI Prediction Market Bot

A production-ready Python trading bot that uses machine learning and AI to automatically trade on decentralized prediction markets (e.g., Polymarket on Polygon). The bot analyzes market data, news sentiment, and historical patterns to make informed buy/sell decisions on outcome shares.

## 🎯 Features

- **AI-Powered Predictions**: Multiple ML models (Logistic Regression, Random Forest) and optional LLM integration for outcome probability estimation
- **Real-Time Trading**: Automated execution of trades on prediction markets with risk management
- **Backtesting Engine**: Comprehensive backtesting framework with performance metrics and visualization
- **Risk Management**: Kelly criterion-based position sizing, stop-loss mechanisms, and portfolio risk limits
- **Data Integration**: Fetches market data, news, and sentiment analysis for informed decisions
- **Database Persistence**: SQLite database for trade history and performance tracking
- **Dry-Run Mode**: Safe testing without real capital
- **Modular Architecture**: Clean, maintainable codebase with type hints and comprehensive error handling

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Testing](#testing)
- [Docker](#docker)
- [Risk Disclaimer](#risk-disclaimer)
- [Support](#support)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker for containerized deployment

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/AI-Prediction-Market-Bot.git
cd AI-Prediction-Market-Bot
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration (see Configuration section)
```

## ⚙️ Configuration

Create a `.env` file in the project root with the following variables:

### Blockchain Configuration
```env
RPC_URL=https://polygon-rpc.com
WALLET_PRIVATE_KEY=your_private_key_here
CHAIN_ID=137
```

### Prediction Market API
```env
MARKET_API_URL=https://clob.polymarket.com
MARKET_API_KEY=your_api_key_here
```

### AI/ML Configuration
```env
OPENAI_API_KEY=your_openai_key_here  # Optional, for LLM predictions
USE_LLM_PREDICTIONS=false
```

### Trading Configuration
```env
MIN_LIQUIDITY=1000
MAX_POSITION_SIZE=0.1
STOP_LOSS_THRESHOLD=0.3
KELLY_FRACTION=0.25
```

### Risk Management
```env
MAX_DAILY_TRADES=10
MIN_PROBABILITY_THRESHOLD=0.55
MAX_PORTFOLIO_RISK=0.2
```

### Other Settings
```env
DRY_RUN=true  # Set to false for live trading
LOG_LEVEL=INFO
DB_PATH=data/trades.db
```

## 📖 Usage

### Training a Model

Train an AI model on historical data:

```bash
python main.py train --model-type logistic --model-path models/my_model.pkl
```

Options:
- `--model-type`: Choose `logistic` or `random_forest`
- `--model-path`: Path to save the trained model

### Running Backtests

Test your strategy on historical data:

```bash
python main.py backtest --market-id market_123 --model-type logistic --initial-capital 10000 --plot
```

Options:
- `--market-id`: Market identifier to backtest
- `--model-type`: Model type to use
- `--initial-capital`: Starting capital for backtest
- `--plot`: Generate visualization plots

### Live Trading

Start the bot in live trading mode:

```bash
python main.py trade --category politics --model-type logistic --model-path models/my_model.pkl
```

Options:
- `--category`: Filter markets by category (e.g., politics, sports)
- `--model-type`: Model type to use
- `--model-path`: Path to trained model file

**⚠️ Important**: Always test in `DRY_RUN=true` mode first!

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐
│   main.py       │  Entry point with CLI interface
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│Config │ │Database │  Configuration & Persistence
└───────┘ └─────────┘
    │
┌───▼──────────────┐
│  Data Fetcher    │  Market data, news, sentiment
└────────┬─────────┘
    │
┌───▼──────────────┐
│  AI Predictor    │  ML models for probability estimation
└────────┬─────────┘
    │
┌───▼──────────────┐
│     Trader       │  Risk management & trade execution
└────────┬─────────┘
    │
┌───▼──────────────┐
│   Backtest       │  Strategy evaluation
└──────────────────┘
```

### Module Structure

```
AI-Prediction-Market-Bot/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data_fetcher.py        # Market data & news fetching
│   ├── ai_predictor.py        # ML prediction models
│   ├── trader.py              # Trading execution & risk management
│   ├── backtest.py            # Backtesting engine
│   └── database.py            # SQLite persistence
├── tests/
│   ├── test_data_fetcher.py
│   ├── test_ai_predictor.py
│   └── test_trader.py
├── main.py                    # CLI entry point
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

### Key Components

#### 1. Data Fetcher (`data_fetcher.py`)
- Fetches real-time market data (odds, liquidity, volume)
- Retrieves news articles and sentiment analysis
- Provides historical data for backtesting

#### 2. AI Predictor (`ai_predictor.py`)
- **LogisticRegressionPredictor**: Fast, interpretable model
- **RandomForestPredictor**: Ensemble method for complex patterns
- **LLMPredictor**: Optional LLM-based predictions via OpenAI
- Feature extraction from market and external data

#### 3. Trader (`trader.py`)
- **RiskManager**: Kelly criterion position sizing, stop-loss logic
- **PredictionMarketTrader**: Trade execution (dry-run and live)
- Portfolio management and position tracking

#### 4. Backtest Engine (`backtest.py`)
- Historical strategy simulation
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Visualization of results

#### 5. Database (`database.py`)
- SQLite storage for trades, predictions, and performance metrics
- Query interface for analysis

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ai_predictor.py
```

## 🐳 Docker

### Build the Docker image:

```bash
docker build -t ai-prediction-bot .
```

### Run in container:

```bash
docker run -v $(pwd)/.env:/app/.env ai-prediction-bot python main.py --help
```

### Docker Compose (optional):

Create a `docker-compose.yml` for easier management:

```yaml
version: '3.8'
services:
  bot:
    build: .
    volumes:
      - ./.env:/app/.env
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - DRY_RUN=true
    command: python main.py trade
```

## 📊 Configuration Options

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `RPC_URL` | Blockchain RPC endpoint | `https://polygon-rpc.com` | Yes |
| `WALLET_PRIVATE_KEY` | Wallet private key | - | For live trading |
| `MARKET_API_URL` | Prediction market API URL | `https://clob.polymarket.com` | Yes |
| `MIN_LIQUIDITY` | Minimum market liquidity | `1000` | No |
| `MAX_POSITION_SIZE` | Max position as % of portfolio | `0.1` | No |
| `KELLY_FRACTION` | Kelly criterion fraction | `0.25` | No |
| `STOP_LOSS_THRESHOLD` | Stop loss probability shift | `0.3` | No |
| `MIN_PROBABILITY_THRESHOLD` | Min confidence to trade | `0.55` | No |
| `MAX_DAILY_TRADES` | Daily trade limit | `10` | No |
| `DRY_RUN` | Enable dry-run mode | `true` | No |

## ⚠️ Risk Disclaimer

**IMPORTANT**: Trading on prediction markets involves significant financial risk. This bot is provided for educational and research purposes only.

- **No Guarantees**: Past performance does not guarantee future results
- **Capital Risk**: You may lose all or part of your capital
- **Market Risk**: Prediction markets are volatile and unpredictable
- **Technical Risk**: Software bugs, network issues, or API failures may cause losses
- **Regulatory Risk**: Prediction markets may be restricted in your jurisdiction

**Use at your own risk. The authors and contributors are not responsible for any financial losses.**

## 📚 References & Inspiration

- [web3.py Documentation](https://web3py.readthedocs.io/) - Ethereum/Polygon blockchain interactions
- [Polymarket API](https://docs.polymarket.com/) - Prediction market platform
- [scikit-learn](https://scikit-learn.org/) - Machine learning models
- [PyTorch](https://pytorch.org/) - Deep learning framework
- Kelly Criterion - Position sizing methodology

## 📧 Support

- telegram: https://t.me/az_tekDev
- twitter:  https://x.com/az_tekDev
