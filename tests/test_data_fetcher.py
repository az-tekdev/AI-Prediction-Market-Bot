"""
Tests for data fetching module.
"""
import pytest
from src.data_fetcher import MarketDataFetcher, NewsDataFetcher


class TestMarketDataFetcher:
    """Test MarketDataFetcher class."""
    
    def test_init(self):
        """Test initialization."""
        fetcher = MarketDataFetcher("https://api.example.com")
        assert fetcher.api_url == "https://api.example.com"
    
    def test_get_markets(self):
        """Test market fetching."""
        fetcher = MarketDataFetcher("https://api.example.com")
        markets = fetcher.get_markets(limit=5)
        assert isinstance(markets, list)
        assert len(markets) <= 5
    
    def test_get_market_details(self):
        """Test market details fetching."""
        fetcher = MarketDataFetcher("https://api.example.com")
        details = fetcher.get_market_details("market_1")
        assert details is not None
        assert "id" in details
    
    def test_get_market_odds(self):
        """Test odds fetching."""
        fetcher = MarketDataFetcher("https://api.example.com")
        odds = fetcher.get_market_odds("market_1")
        assert odds is not None
        assert "YES" in odds
        assert "NO" in odds
    
    def test_get_historical_odds(self):
        """Test historical odds fetching."""
        from datetime import datetime, timedelta
        fetcher = MarketDataFetcher("https://api.example.com")
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        historical = fetcher.get_historical_odds("market_1", start, end)
        assert not historical.empty
        assert "timestamp" in historical.columns
        assert "yes_probability" in historical.columns


class TestNewsDataFetcher:
    """Test NewsDataFetcher class."""
    
    def test_init(self):
        """Test initialization."""
        fetcher = NewsDataFetcher()
        assert fetcher is not None
    
    def test_fetch_news(self):
        """Test news fetching."""
        fetcher = NewsDataFetcher()
        news = fetcher.fetch_news("test query", limit=5)
        assert isinstance(news, list)
        assert len(news) <= 5
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        fetcher = NewsDataFetcher()
        sentiment = fetcher.analyze_sentiment("This is a positive text")
        assert "positive" in sentiment
        assert "negative" in sentiment
        assert "score" in sentiment
