"""
Configuration management for the AI Prediction Market Bot.
"""
import os
from typing import Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()


class Config:
    """Central configuration class for the bot."""
    
    # Blockchain Configuration
    RPC_URL: str = os.getenv("RPC_URL", "https://polygon-rpc.com")
    WALLET_PRIVATE_KEY: Optional[str] = os.getenv("WALLET_PRIVATE_KEY")
    CHAIN_ID: int = int(os.getenv("CHAIN_ID", "137"))
    
    # Prediction Market API
    MARKET_API_URL: str = os.getenv("MARKET_API_URL", "https://clob.polymarket.com")
    MARKET_API_KEY: Optional[str] = os.getenv("MARKET_API_KEY")
    
    # AI/ML Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    USE_LLM_PREDICTIONS: bool = os.getenv("USE_LLM_PREDICTIONS", "false").lower() == "true"
    
    # Trading Configuration
    MIN_LIQUIDITY: float = float(os.getenv("MIN_LIQUIDITY", "1000"))
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    STOP_LOSS_THRESHOLD: float = float(os.getenv("STOP_LOSS_THRESHOLD", "0.3"))
    KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))
    
    # Risk Management
    MAX_DAILY_TRADES: int = int(os.getenv("MAX_DAILY_TRADES", "10"))
    MIN_PROBABILITY_THRESHOLD: float = float(os.getenv("MIN_PROBABILITY_THRESHOLD", "0.55"))
    MAX_PORTFOLIO_RISK: float = float(os.getenv("MAX_PORTFOLIO_RISK", "0.2"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/bot.log")
    
    # Database
    DB_PATH: str = os.getenv("DB_PATH", "data/trades.db")
    
    # News/Sentiment APIs
    NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY")
    SENTIMENT_API_KEY: Optional[str] = os.getenv("SENTIMENT_API_KEY")
    
    # Dry Run Mode
    DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.DRY_RUN and not cls.WALLET_PRIVATE_KEY:
            raise ValueError("WALLET_PRIVATE_KEY is required for live trading")
        return True
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup logging configuration."""
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )
