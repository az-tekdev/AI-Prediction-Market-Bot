"""
Backtesting module for strategy evaluation.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

from src.ai_predictor import AIPredictor
from src.trader import PredictionMarketTrader, RiskManager
from src.data_fetcher import MarketDataFetcher

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Engine for backtesting trading strategies."""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.02,  # 2% commission
        slippage: float = 0.01  # 1% slippage
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission: Trading commission rate
            slippage: Slippage rate
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = []
    
    def run_backtest(
        self,
        historical_data: pd.DataFrame,
        predictor: AIPredictor,
        risk_manager: RiskManager,
        min_prob_threshold: float = 0.55,
        max_trades_per_day: int = 10
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            historical_data: DataFrame with historical odds and outcomes
            predictor: AI predictor instance
            risk_manager: Risk manager instance
            min_prob_threshold: Minimum probability threshold for trades
            max_trades_per_day: Maximum trades per day
            
        Returns:
            Backtest results dictionary
        """
        logger.info("Starting backtest...")
        
        capital = self.initial_capital
        positions = []
        trades = []
        daily_trades = {}
        
        # Sort by timestamp
        data = historical_data.sort_values('timestamp').copy()
        
        for idx, row in data.iterrows():
            timestamp = pd.to_datetime(row['timestamp'])
            date_key = timestamp.date()
            
            # Check daily trade limit
            if date_key in daily_trades:
                if daily_trades[date_key] >= max_trades_per_day:
                    continue
            else:
                daily_trades[date_key] = 0
            
            # Get market data
            market_data = {
                'current_odds': {
                    'YES': row.get('yes_probability', 0.5),
                    'NO': row.get('no_probability', 0.5)
                },
                'volume_24h': row.get('volume', 0),
                'liquidity': row.get('volume', 0) * 2,
                'end_date': (timestamp + timedelta(days=7)).isoformat()
            }
            
            # Get prediction
            predicted_prob, metadata = predictor.predict(market_data)
            market_prob = row.get('yes_probability', 0.5)
            
            # Check if we should trade
            if abs(predicted_prob - market_prob) < 0.05:
                continue  # Not enough edge
            
            if predicted_prob < min_prob_threshold and predicted_prob > (1 - min_prob_threshold):
                continue  # Not confident enough
            
            # Determine direction
            if predicted_prob > market_prob:
                outcome = 'YES'
                entry_price = market_prob
            else:
                outcome = 'NO'
                entry_price = 1 - market_prob
            
            # Calculate position size
            position_size = risk_manager.calculate_position_size(
                predicted_prob, market_prob, capital, confidence=1.0
            )
            
            if position_size <= 0:
                continue
            
            # Apply slippage and commission
            cost = position_size * (1 + self.slippage + self.commission)
            
            if cost > capital:
                continue  # Not enough capital
            
            # Execute trade
            capital -= cost
            position = {
                'timestamp': timestamp,
                'outcome': outcome,
                'entry_price': entry_price,
                'predicted_prob': predicted_prob,
                'market_prob': market_prob,
                'size': position_size,
                'cost': cost
            }
            positions.append(position)
            daily_trades[date_key] += 1
            
            # Check for stop loss on existing positions
            self._check_stop_losses(positions, market_prob, risk_manager)
            
            # Close positions if we have outcome data
            if 'actual_outcome' in row and pd.notna(row['actual_outcome']):
                self._close_positions(positions, row['actual_outcome'], trades, capital)
        
        # Calculate final metrics
        results = self._calculate_metrics(capital, trades, positions)
        
        logger.info(f"Backtest completed. Final capital: ${capital:.2f}")
        return results
    
    def _check_stop_losses(
        self,
        positions: List[Dict],
        current_prob: float,
        risk_manager: RiskManager
    ) -> None:
        """Check and close positions that hit stop loss."""
        for position in positions:
            if position.get('closed'):
                continue
            
            entry_prob = position['market_prob']
            outcome = position['outcome']
            
            if risk_manager.should_stop_loss(entry_prob, current_prob, outcome):
                # Close position at loss
                position['closed'] = True
                position['exit_price'] = current_prob if outcome == 'YES' else (1 - current_prob)
                position['pnl'] = -position['cost'] * 0.5  # Assume 50% loss
                logger.debug(f"Stop loss triggered for position")
    
    def _close_positions(
        self,
        positions: List[Dict],
        actual_outcome: str,
        trades: List[Dict],
        capital: float
    ) -> None:
        """Close all open positions based on actual outcome."""
        for position in positions:
            if position.get('closed'):
                continue
            
            position['closed'] = True
            
            # Calculate P&L
            if position['outcome'] == actual_outcome:
                # Win: payout is based on entry price
                payout = position['size'] / position['entry_price']
                pnl = payout - position['cost']
            else:
                # Loss: lose the cost
                pnl = -position['cost']
            
            position['pnl'] = pnl
            capital += position['size'] / position['entry_price'] if position['outcome'] == actual_outcome else 0
            
            trades.append({
                'entry_time': position['timestamp'],
                'outcome': position['outcome'],
                'actual_outcome': actual_outcome,
                'pnl': pnl,
                'roi': pnl / position['cost']
            })
    
    def _calculate_metrics(
        self,
        final_capital: float,
        trades: List[Dict],
        positions: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate backtest performance metrics."""
        if not trades:
            return {
                'total_return': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        trades_df = pd.DataFrame(trades)
        
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns series for Sharpe ratio
        returns = trades_df['roi'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': np.mean(returns),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades_df
        }
    
    def plot_results(self, results: Dict[str, Any], output_path: str = "backtest_results.png") -> None:
        """
        Plot backtest results.
        
        Args:
            results: Backtest results dictionary
            output_path: Path to save plot
        """
        if 'trades' not in results or results['trades'].empty:
            logger.warning("No trades to plot")
            return
        
        trades_df = results['trades']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative returns
        cumulative_returns = (1 + trades_df['roi']).cumprod()
        axes[0, 0].plot(cumulative_returns)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True)
        
        # P&L distribution
        axes[0, 1].hist(trades_df['pnl'], bins=30, edgecolor='black')
        axes[0, 1].set_title('P&L Distribution')
        axes[0, 1].set_xlabel('Profit/Loss')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(0, color='r', linestyle='--')
        axes[0, 1].grid(True)
        
        # Win/Loss breakdown
        win_loss = [results['winning_trades'], results['losing_trades']]
        axes[1, 0].pie(win_loss, labels=['Wins', 'Losses'], autopct='%1.1f%%')
        axes[1, 0].set_title('Win/Loss Breakdown')
        
        # ROI over time
        axes[1, 1].scatter(range(len(trades_df)), trades_df['roi'], alpha=0.5)
        axes[1, 1].set_title('ROI per Trade')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('ROI')
        axes[1, 1].axhline(0, color='r', linestyle='--')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
        plt.close()
