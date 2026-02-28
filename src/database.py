"""
Database module for storing trade history and model data.
"""
import sqlite3
import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeDatabase:
    """SQLite database for storing trade history and metrics."""
    
    def __init__(self, db_path: str = "data/trades.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                amount REAL NOT NULL,
                predicted_prob REAL NOT NULL,
                market_prob REAL NOT NULL,
                tx_hash TEXT,
                status TEXT,
                pnl REAL,
                closed INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                predicted_prob REAL NOT NULL,
                market_prob REAL NOT NULL,
                features TEXT,
                model_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                portfolio_value REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_trade(self, trade: Dict[str, Any]) -> int:
        """
        Save a trade to the database.
        
        Args:
            trade: Trade dictionary
            
        Returns:
            Trade ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (
                timestamp, market_id, outcome, amount,
                predicted_prob, market_prob, tx_hash, status, pnl, closed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.get('timestamp', datetime.now().isoformat()),
            trade.get('market_id'),
            trade.get('outcome'),
            trade.get('amount'),
            trade.get('predicted_prob'),
            trade.get('market_prob'),
            trade.get('tx_hash'),
            trade.get('status', 'pending'),
            trade.get('pnl', 0.0),
            trade.get('closed', 0)
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.debug(f"Trade saved with ID: {trade_id}")
        return trade_id
    
    def save_prediction(self, prediction: Dict[str, Any]) -> int:
        """
        Save a prediction to the database.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            Prediction ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        import json
        features_json = json.dumps(prediction.get('features', {}))
        
        cursor.execute('''
            INSERT INTO predictions (
                timestamp, market_id, predicted_prob, market_prob,
                features, model_type
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            prediction.get('timestamp', datetime.now().isoformat()),
            prediction.get('market_id'),
            prediction.get('predicted_prob'),
            prediction.get('market_prob'),
            features_json,
            prediction.get('model_type')
        ))
        
        pred_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return pred_id
    
    def get_trades(
        self,
        market_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Retrieve trades from database.
        
        Args:
            market_id: Filter by market ID
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records
            
        Returns:
            DataFrame with trades
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if market_id:
            query += " AND market_id = ?"
            params.append(market_id)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_performance_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Performance metrics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0.0,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        }
