"""
Example usage of the AI Prediction Market Bot.
This script demonstrates basic usage patterns.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_fetcher import MarketDataFetcher, NewsDataFetcher
from src.ai_predictor import AIPredictor
from src.trader import PredictionMarketTrader
from src.utils import format_currency, format_percentage, log_trade_summary


def example_data_fetching():
    """Example: Fetching market data."""
    print("\n=== Example: Data Fetching ===")
    
    fetcher = MarketDataFetcher(Config.MARKET_API_URL)
    markets = fetcher.get_markets(limit=5)
    
    print(f"Found {len(markets)} markets:")
    for market in markets:
        print(f"  - {market['question']} (ID: {market['id']})")
    
    if markets:
        market_id = markets[0]['id']
        details = fetcher.get_market_details(market_id)
        odds = fetcher.get_market_odds(market_id)
        
        print(f"\nMarket Details for {market_id}:")
        print(f"  Question: {details['question']}")
        print(f"  Current Odds: YES={odds['YES']:.2%}, NO={odds['NO']:.2%}")


def example_prediction():
    """Example: Making predictions."""
    print("\n=== Example: AI Prediction ===")
    
    predictor = AIPredictor(model_type="logistic")
    
    market_data = {
        'current_odds': {'YES': 0.65, 'NO': 0.35},
        'volume_24h': 5000,
        'liquidity': 10000,
        'end_date': '2024-12-31T00:00:00'
    }
    
    predicted_prob, metadata = predictor.predict(market_data)
    
    print(f"Market Probability (YES): {format_percentage(market_data['current_odds']['YES'])}")
    print(f"Predicted Probability (YES): {format_percentage(predicted_prob)}")
    print(f"Edge: {format_percentage(abs(predicted_prob - market_data['current_odds']['YES']))}")


def example_trading():
    """Example: Executing trades (dry-run)."""
    print("\n=== Example: Trading (Dry-Run) ===")
    
    trader = PredictionMarketTrader(
        Config.RPC_URL,
        dry_run=True
    )
    
    result = trader.execute_trade(
        market_id="example_market",
        outcome="YES",
        predicted_prob=0.70,
        market_prob=0.60
    )
    
    if result['success']:
        print("Trade executed successfully!")
        log_trade_summary(result['trade'])
    else:
        print(f"Trade failed: {result.get('reason', 'Unknown error')}")


def example_risk_management():
    """Example: Risk management calculations."""
    print("\n=== Example: Risk Management ===")
    
    from src.trader import RiskManager
    
    risk_manager = RiskManager(
        max_position_size=0.1,
        kelly_fraction=0.25
    )
    
    portfolio_value = 10000.0
    predicted_prob = 0.70
    market_prob = 0.60
    
    position_size = risk_manager.calculate_position_size(
        predicted_prob,
        market_prob,
        portfolio_value
    )
    
    print(f"Portfolio Value: {format_currency(portfolio_value)}")
    print(f"Predicted Probability: {format_percentage(predicted_prob)}")
    print(f"Market Probability: {format_percentage(market_prob)}")
    print(f"Recommended Position Size: {format_currency(position_size)}")
    print(f"Position as % of Portfolio: {format_percentage(position_size / portfolio_value)}")


def main():
    """Run all examples."""
    print("AI Prediction Market Bot - Example Usage")
    print("=" * 50)
    
    # Setup
    Config.setup_logging()
    
    try:
        example_data_fetching()
        example_prediction()
        example_trading()
        example_risk_management()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
