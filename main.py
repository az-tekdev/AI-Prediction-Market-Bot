#!/usr/bin/env python3
"""
Main entry point for the AI Prediction Market Bot.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import Config
from src.data_fetcher import MarketDataFetcher, NewsDataFetcher
from src.ai_predictor import AIPredictor
from src.trader import PredictionMarketTrader
from src.backtest import BacktestEngine
from src.database import TradeDatabase

logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories."""
    directories = ['logs', 'data', 'models', 'backtest_results']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def run_backtest(args):
    """Run backtesting mode."""
    logger.info("Starting backtest mode...")
    
    # Initialize components
    data_fetcher = MarketDataFetcher(Config.MARKET_API_URL, Config.MARKET_API_KEY)
    predictor = AIPredictor(
        model_type=args.model_type,
        use_llm=Config.USE_LLM_PREDICTIONS,
        llm_api_key=Config.OPENAI_API_KEY
    )
    
    from src.trader import RiskManager
    risk_manager = RiskManager(
        max_position_size=Config.MAX_POSITION_SIZE,
        kelly_fraction=Config.KELLY_FRACTION,
        stop_loss_threshold=Config.STOP_LOSS_THRESHOLD
    )
    
    # Fetch historical data
    if args.market_id:
        historical_data = data_fetcher.get_historical_odds(args.market_id)
    else:
        logger.error("Market ID required for backtesting")
        return
    
    if historical_data.empty:
        logger.error("No historical data available")
        return
    
    # Run backtest
    engine = BacktestEngine(initial_capital=args.initial_capital)
    results = engine.run_backtest(
        historical_data,
        predictor,
        risk_manager,
        min_prob_threshold=Config.MIN_PROBABILITY_THRESHOLD,
        max_trades_per_day=Config.MAX_DAILY_TRADES
    )
    
    # Print results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print("="*50)
    
    # Save plot
    if args.plot:
        output_path = f"backtest_results/backtest_{args.market_id}.png"
        engine.plot_results(results, output_path)
        print(f"Plot saved to {output_path}")


def run_live_trading(args):
    """Run live trading mode."""
    logger.info("Starting live trading mode...")
    
    if Config.DRY_RUN:
        logger.warning("Running in DRY RUN mode - no real trades will be executed")
    
    # Initialize components
    data_fetcher = MarketDataFetcher(
        Config.MARKET_API_URL,
        Config.MARKET_API_KEY,
        Config.RPC_URL
    )
    news_fetcher = NewsDataFetcher(Config.NEWS_API_KEY)
    
    predictor = AIPredictor(
        model_type=args.model_type,
        model_path=args.model_path,
        use_llm=Config.USE_LLM_PREDICTIONS,
        llm_api_key=Config.OPENAI_API_KEY
    )
    
    trader = PredictionMarketTrader(
        Config.RPC_URL,
        Config.WALLET_PRIVATE_KEY,
        Config.CHAIN_ID,
        dry_run=Config.DRY_RUN
    )
    
    database = TradeDatabase(Config.DB_PATH)
    
    # Main trading loop
    logger.info("Starting trading loop...")
    try:
        while True:
            # Fetch markets
            markets = data_fetcher.get_markets(
                category=args.category,
                min_liquidity=Config.MIN_LIQUIDITY,
                limit=10
            )
            
            for market in markets:
                market_id = market['id']
                
                # Get market details
                market_details = data_fetcher.get_market_details(market_id)
                if not market_details:
                    continue
                
                # Get current odds
                odds = data_fetcher.get_market_odds(market_id)
                if not odds:
                    continue
                
                # Fetch news and sentiment
                news = news_fetcher.fetch_news(market['question'], limit=5)
                sentiment = news_fetcher.analyze_sentiment(
                    ' '.join([n.get('title', '') for n in news])
                )
                
                # Get prediction
                predicted_prob, metadata = predictor.predict(
                    market_details,
                    news,
                    sentiment
                )
                
                market_prob = odds.get('YES', 0.5)
                
                # Save prediction
                database.save_prediction({
                    'market_id': market_id,
                    'predicted_prob': predicted_prob,
                    'market_prob': market_prob,
                    'features': metadata.get('features', {}),
                    'model_type': args.model_type
                })
                
                # Check stop loss
                trader.check_stop_loss(market_id, market_prob)
                
                # Determine if we should trade
                edge = abs(predicted_prob - market_prob)
                if edge < 0.05:
                    logger.debug(f"Insufficient edge for {market_id}")
                    continue
                
                if predicted_prob < Config.MIN_PROBABILITY_THRESHOLD and \
                   predicted_prob > (1 - Config.MIN_PROBABILITY_THRESHOLD):
                    logger.debug(f"Probability too close to 0.5 for {market_id}")
                    continue
                
                # Determine direction
                if predicted_prob > market_prob:
                    outcome = 'YES'
                else:
                    outcome = 'NO'
                
                # Execute trade
                result = trader.execute_trade(
                    market_id,
                    outcome,
                    predicted_prob,
                    market_prob
                )
                
                if result['success']:
                    database.save_trade(result['trade'])
                    logger.info(f"Trade executed: {result['trade']}")
            
            # Sleep before next iteration
            import time
            time.sleep(60)  # Wait 1 minute between iterations
            
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Trading error: {e}", exc_info=True)


def train_model(args):
    """Train AI model on historical data."""
    logger.info("Training model...")
    
    # This would load training data from a file or database
    # For now, generate synthetic training data
    import pandas as pd
    import numpy as np
    
    # Generate synthetic training data
    n_samples = 1000
    X = pd.DataFrame({
        'current_yes_prob': np.random.uniform(0.3, 0.7, n_samples),
        'volume_24h': np.random.uniform(1000, 10000, n_samples),
        'liquidity': np.random.uniform(5000, 50000, n_samples),
        'time_to_resolution': np.random.uniform(1, 30, n_samples),
        'sentiment_score': np.random.uniform(-0.5, 0.5, n_samples),
        'news_count': np.random.randint(0, 20, n_samples)
    })
    
    # Generate target (outcome) - higher yes_prob and positive sentiment -> more likely YES
    y = ((X['current_yes_prob'] + X['sentiment_score'] * 0.2) > 0.5).astype(int)
    
    # Train model
    predictor = AIPredictor(model_type=args.model_type)
    metrics = predictor.train(pd.concat([X, pd.Series(y, name='outcome')], axis=1))
    
    # Save model
    model_path = args.model_path or f"models/{args.model_type}_model.pkl"
    predictor.save(model_path)
    
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*50)
    print(f"Model saved to {model_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Prediction Market Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Bot mode')
    
    # Backtest mode
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--market-id', type=str, help='Market ID to backtest')
    backtest_parser.add_argument('--model-type', type=str, default='logistic',
                                choices=['logistic', 'random_forest'],
                                help='Model type')
    backtest_parser.add_argument('--initial-capital', type=float, default=10000.0,
                                help='Initial capital')
    backtest_parser.add_argument('--plot', action='store_true',
                                help='Generate backtest plots')
    
    # Live trading mode
    live_parser = subparsers.add_parser('trade', help='Run live trading')
    live_parser.add_argument('--category', type=str, help='Market category filter')
    live_parser.add_argument('--model-type', type=str, default='logistic',
                            choices=['logistic', 'random_forest'],
                            help='Model type')
    live_parser.add_argument('--model-path', type=str, help='Path to saved model')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--model-type', type=str, default='logistic',
                             choices=['logistic', 'random_forest'],
                             help='Model type')
    train_parser.add_argument('--model-path', type=str, help='Path to save model')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    # Setup
    setup_directories()
    Config.setup_logging()
    Config.validate()
    
    # Run mode
    if args.mode == 'backtest':
        run_backtest(args)
    elif args.mode == 'trade':
        run_live_trading(args)
    elif args.mode == 'train':
        train_model(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
