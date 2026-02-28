"""
Data fetching module for prediction market data and external feeds.
"""
import requests
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from web3 import Web3
from web3.middleware import geth_poa_middleware

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetches real-time market data from prediction markets."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None, rpc_url: Optional[str] = None):
        """
        Initialize market data fetcher.
        
        Args:
            api_url: Base URL for the prediction market API
            api_key: Optional API key for authenticated requests
            rpc_url: Optional RPC URL for on-chain queries
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        self.w3 = None
        if rpc_url:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            # Add PoA middleware for Polygon
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    
    def get_markets(
        self,
        category: Optional[str] = None,
        min_liquidity: float = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch available markets.
        
        Args:
            category: Filter by category (e.g., 'politics', 'sports')
            min_liquidity: Minimum liquidity threshold
            limit: Maximum number of markets to return
            
        Returns:
            List of market dictionaries
        """
        try:
            # Simulated API call - replace with actual Polymarket API endpoint
            # In production, this would be: f"{self.api_url}/markets"
            params = {
                "limit": limit,
                "category": category,
                "min_liquidity": min_liquidity
            }
            
            # Mock response for demo purposes
            # Replace with actual API call:
            # response = self.session.get(f"{self.api_url}/markets", params=params)
            # response.raise_for_status()
            # return response.json()
            
            logger.info(f"Fetching markets with params: {params}")
            return self._mock_markets(limit)
            
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []
    
    def get_market_details(self, market_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific market.
        
        Args:
            market_id: Unique market identifier
            
        Returns:
            Market details dictionary or None
        """
        try:
            # Mock implementation - replace with actual API call
            logger.info(f"Fetching details for market: {market_id}")
            return self._mock_market_details(market_id)
            
        except Exception as e:
            logger.error(f"Error fetching market details: {e}")
            return None
    
    def get_market_odds(self, market_id: str) -> Optional[Dict[str, float]]:
        """
        Get current odds/probabilities for market outcomes.
        
        Args:
            market_id: Unique market identifier
            
        Returns:
            Dictionary mapping outcome to probability
        """
        try:
            # Mock implementation - replace with actual API call
            logger.info(f"Fetching odds for market: {market_id}")
            return {"YES": 0.65, "NO": 0.35}
            
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return None
    
    def get_historical_odds(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical odds data for backtesting.
        
        Args:
            market_id: Unique market identifier
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            DataFrame with timestamp and odds columns
        """
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(days=30)
            if not end_time:
                end_time = datetime.now()
            
            # Mock historical data generation
            dates = pd.date_range(start_time, end_time, freq='1H')
            data = {
                'timestamp': dates,
                'yes_probability': [0.5 + 0.15 * (i % 20) / 20 for i in range(len(dates))],
                'no_probability': [0.5 - 0.15 * (i % 20) / 20 for i in range(len(dates))],
                'volume': [1000 + (i % 100) * 10 for i in range(len(dates))]
            }
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching historical odds: {e}")
            return pd.DataFrame()
    
    def _mock_markets(self, limit: int) -> List[Dict[str, Any]]:
        """Generate mock market data for testing."""
        categories = ["politics", "sports", "crypto", "entertainment"]
        markets = []
        
        for i in range(min(limit, 10)):
            markets.append({
                "id": f"market_{i}",
                "question": f"Will event {i} happen?",
                "category": categories[i % len(categories)],
                "liquidity": 5000 + i * 1000,
                "volume_24h": 2000 + i * 500,
                "end_date": (datetime.now() + timedelta(days=7 + i)).isoformat(),
                "outcomes": ["YES", "NO"]
            })
        
        return markets
    
    def _mock_market_details(self, market_id: str) -> Dict[str, Any]:
        """Generate mock market details."""
        return {
            "id": market_id,
            "question": "Will this event happen?",
            "category": "politics",
            "liquidity": 10000,
            "volume_24h": 5000,
            "end_date": (datetime.now() + timedelta(days=7)).isoformat(),
            "outcomes": ["YES", "NO"],
            "current_odds": {"YES": 0.65, "NO": 0.35},
            "created_at": (datetime.now() - timedelta(days=1)).isoformat()
        }


class NewsDataFetcher:
    """Fetches news and sentiment data for market analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news data fetcher.
        
        Args:
            api_key: API key for news service
        """
        self.api_key = api_key
        self.session = requests.Session()
    
    def fetch_news(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles related to a query.
        
        Args:
            query: Search query
            limit: Maximum number of articles
            
        Returns:
            List of news article dictionaries
        """
        try:
            if not self.api_key:
                logger.warning("No news API key provided, returning mock data")
                return self._mock_news(query, limit)
            
            # In production, integrate with NewsAPI or similar
            # response = self.session.get(
            #     "https://newsapi.org/v2/everything",
            #     params={"q": query, "apiKey": self.api_key, "pageSize": limit}
            # )
            # response.raise_for_status()
            # return response.json().get("articles", [])
            
            return self._mock_news(query, limit)
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            # Mock sentiment analysis - in production, use OpenAI API or similar
            # For now, return mock positive sentiment
            return {
                "positive": 0.6,
                "negative": 0.2,
                "neutral": 0.2,
                "score": 0.4  # -1 to 1 scale
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "score": 0.0}
    
    def _mock_news(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock news data."""
        return [
            {
                "title": f"News article about {query}",
                "description": f"Recent developments regarding {query}",
                "url": f"https://example.com/news/{i}",
                "publishedAt": (datetime.now() - timedelta(hours=i)).isoformat(),
                "source": {"name": "Example News"}
            }
            for i in range(limit)
        ]
