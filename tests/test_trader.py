"""
Tests for trading module.
"""
import pytest
from src.trader import PredictionMarketTrader, RiskManager


class TestRiskManager:
    """Test RiskManager class."""
    
    def test_init(self):
        """Test initialization."""
        rm = RiskManager()
        assert rm.max_position_size > 0
        assert rm.kelly_fraction > 0
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        rm = RiskManager(max_position_size=0.1, kelly_fraction=0.25)
        portfolio_value = 10000.0
        
        # Test with edge
        size = rm.calculate_position_size(
            predicted_prob=0.7,
            market_prob=0.6,
            portfolio_value=portfolio_value,
            confidence=1.0
        )
        assert size >= 0
        assert size <= portfolio_value * rm.max_position_size
    
    def test_should_stop_loss(self):
        """Test stop loss logic."""
        rm = RiskManager(stop_loss_threshold=0.3)
        
        # Should trigger stop loss
        assert rm.should_stop_loss(0.7, 0.4, 'YES')
        
        # Should not trigger
        assert not rm.should_stop_loss(0.7, 0.65, 'YES')


class TestPredictionMarketTrader:
    """Test PredictionMarketTrader class."""
    
    def test_init_dry_run(self):
        """Test initialization in dry run mode."""
        trader = PredictionMarketTrader(
            rpc_url="https://polygon-rpc.com",
            dry_run=True
        )
        assert trader.dry_run is True
        assert trader.account is None
    
    def test_execute_trade_dry_run(self):
        """Test trade execution in dry run mode."""
        trader = PredictionMarketTrader(
            rpc_url="https://polygon-rpc.com",
            dry_run=True
        )
        
        result = trader.execute_trade(
            market_id="test_market",
            outcome="YES",
            predicted_prob=0.7,
            market_prob=0.6
        )
        
        assert isinstance(result, dict)
        # In dry run, should succeed or fail gracefully
        assert 'success' in result
    
    def test_validate_trade(self):
        """Test trade validation."""
        trader = PredictionMarketTrader(
            rpc_url="https://polygon-rpc.com",
            dry_run=True
        )
        
        # Valid trade
        assert trader._validate_trade("market_1", "YES", 100.0)
        
        # Invalid outcome
        assert not trader._validate_trade("market_1", "MAYBE", 100.0)
        
        # Invalid amount
        assert not trader._validate_trade("market_1", "YES", -100.0)
