"""
Utility functions for the AI Prediction Market Bot.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def format_currency(amount: float, decimals: int = 2) -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    return f"${amount:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.
    
    Args:
        value: Value to format (0-1 range)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_roi(initial: float, final: float) -> float:
    """
    Calculate return on investment.
    
    Args:
        initial: Initial capital
        final: Final capital
        
    Returns:
        ROI as decimal (e.g., 0.1 for 10%)
    """
    if initial == 0:
        return 0.0
    return (final - initial) / initial


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def validate_market_data(market_data: Dict[str, Any]) -> bool:
    """
    Validate market data structure.
    
    Args:
        market_data: Market data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['id', 'question', 'outcomes']
    return all(field in market_data for field in required_fields)


def log_trade_summary(trade: Dict[str, Any]) -> None:
    """
    Log a formatted trade summary.
    
    Args:
        trade: Trade dictionary
    """
    logger.info(
        f"Trade: {trade.get('outcome')} on {trade.get('market_id')} | "
        f"Amount: {format_currency(trade.get('amount', 0))} | "
        f"Predicted: {format_percentage(trade.get('predicted_prob', 0))} | "
        f"Market: {format_percentage(trade.get('market_prob', 0))}"
    )


def save_json(data: Any, filepath: str) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        return False


def load_json(filepath: str) -> Optional[Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return None
