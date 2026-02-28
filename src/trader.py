"""
Trading execution module for prediction market transactions.
"""
import logging
import time
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from decimal import Decimal
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages position sizing and risk limits."""
    
    def __init__(
        self,
        max_position_size: float = 0.1,
        kelly_fraction: float = 0.25,
        stop_loss_threshold: float = 0.3,
        max_portfolio_risk: float = 0.2
    ):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            kelly_fraction: Fraction of Kelly criterion to use
            stop_loss_threshold: Stop loss threshold (probability shift)
            max_portfolio_risk: Maximum total portfolio risk
        """
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction
        self.stop_loss_threshold = stop_loss_threshold
        self.max_portfolio_risk = max_portfolio_risk
    
    def calculate_position_size(
        self,
        predicted_prob: float,
        market_prob: float,
        portfolio_value: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate optimal position size using Kelly criterion.
        
        Args:
            predicted_prob: Model's predicted probability
            market_prob: Current market probability
            confidence: Confidence in prediction (0-1)
            
        Returns:
            Position size in base currency
        """
        if predicted_prob <= 0 or predicted_prob >= 1:
            return 0.0
        
        # Expected value calculation
        # If we think YES has 70% chance but market prices at 60%, we have edge
        edge = abs(predicted_prob - market_prob)
        
        if edge < 0.05:  # Minimum edge threshold
            return 0.0
        
        # Kelly criterion: f = (bp - q) / b
        # where b = odds, p = win prob, q = loss prob
        if predicted_prob > market_prob:
            # Betting on YES
            odds = market_prob / (1 - market_prob) if market_prob < 1 else 10
            win_prob = predicted_prob
            loss_prob = 1 - predicted_prob
        else:
            # Betting on NO
            odds = (1 - market_prob) / market_prob if market_prob > 0 else 10
            win_prob = 1 - predicted_prob
            loss_prob = predicted_prob
        
        kelly = (odds * win_prob - loss_prob) / odds if odds > 0 else 0
        kelly = max(0, kelly)  # No negative Kelly
        
        # Apply fraction and confidence
        position_fraction = min(
            kelly * self.kelly_fraction * confidence,
            self.max_position_size
        )
        
        position_size = portfolio_value * position_fraction
        
        return position_size
    
    def should_stop_loss(
        self,
        entry_prob: float,
        current_prob: float,
        position_direction: str
    ) -> bool:
        """
        Check if stop loss should be triggered.
        
        Args:
            entry_prob: Probability at entry
            current_prob: Current market probability
            position_direction: 'YES' or 'NO'
            
        Returns:
            True if stop loss should trigger
        """
        if position_direction == 'YES':
            prob_change = entry_prob - current_prob
        else:
            prob_change = current_prob - entry_prob
        
        return prob_change >= self.stop_loss_threshold


class PredictionMarketTrader:
    """Handles trading execution on prediction markets."""
    
    def __init__(
        self,
        rpc_url: str,
        private_key: Optional[str] = None,
        chain_id: int = 137,
        dry_run: bool = True
    ):
        """
        Initialize trader.
        
        Args:
            rpc_url: RPC URL for blockchain connection
            private_key: Wallet private key (None for dry run)
            chain_id: Chain ID (137 for Polygon)
            dry_run: If True, simulate trades without executing
        """
        self.dry_run = dry_run
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        self.account = None
        if private_key and not dry_run:
            self.account = Account.from_key(private_key)
            logger.info(f"Initialized trader with address: {self.account.address}")
        else:
            logger.info("Running in dry-run mode")
        
        self.risk_manager = RiskManager()
        self.trade_history = []
        self.portfolio_value = 1000.0  # Starting portfolio value
    
    def execute_trade(
        self,
        market_id: str,
        outcome: str,
        predicted_prob: float,
        market_prob: float,
        amount: Optional[float] = None,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Execute a trade on a prediction market.
        
        Args:
            market_id: Market identifier
            outcome: 'YES' or 'NO'
            predicted_prob: Model's predicted probability
            market_prob: Current market probability
            amount: Trade amount (None for auto-sizing)
            confidence: Confidence in prediction
            
        Returns:
            Trade result dictionary
        """
        try:
            # Calculate position size
            if amount is None:
                amount = self.risk_manager.calculate_position_size(
                    predicted_prob, market_prob, self.portfolio_value, confidence
                )
            
            if amount <= 0:
                logger.warning("Position size is zero, skipping trade")
                return {
                    'success': False,
                    'reason': 'zero_position_size'
                }
            
            # Validate trade
            if not self._validate_trade(market_id, outcome, amount):
                return {
                    'success': False,
                    'reason': 'validation_failed'
                }
            
            # Execute trade
            if self.dry_run:
                result = self._simulate_trade(market_id, outcome, amount)
            else:
                result = self._execute_on_chain(market_id, outcome, amount)
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'market_id': market_id,
                'outcome': outcome,
                'amount': amount,
                'predicted_prob': predicted_prob,
                'market_prob': market_prob,
                'result': result
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Trade executed: {trade_record}")
            return {
                'success': True,
                'trade': trade_record
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_trade(
        self,
        market_id: str,
        outcome: str,
        amount: float
    ) -> bool:
        """Validate trade parameters."""
        if outcome not in ['YES', 'NO']:
            logger.error(f"Invalid outcome: {outcome}")
            return False
        
        if amount <= 0:
            logger.error("Invalid amount")
            return False
        
        if amount > self.portfolio_value * self.risk_manager.max_position_size:
            logger.warning("Amount exceeds max position size")
            return False
        
        return True
    
    def _simulate_trade(
        self,
        market_id: str,
        outcome: str,
        amount: float
    ) -> Dict[str, Any]:
        """Simulate a trade without on-chain execution."""
        logger.info(f"[DRY RUN] Simulating trade: {outcome} {amount} on {market_id}")
        
        # Simulate transaction hash
        tx_hash = f"0x{''.join([hex(ord(c))[2:] for c in f'{market_id}{outcome}{time.time()}'])[:64]}"
        
        return {
            'tx_hash': tx_hash,
            'status': 'simulated',
            'gas_used': 150000,
            'gas_price': 30 * 10**9,  # 30 gwei
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_on_chain(
        self,
        market_id: str,
        outcome: str,
        amount: float
    ) -> Dict[str, Any]:
        """Execute trade on-chain."""
        if not self.account:
            raise ValueError("No account configured for on-chain execution")
        
        try:
            # In production, this would interact with the actual prediction market contract
            # For now, this is a placeholder structure
            
            # Example: Build transaction
            # contract_address = self._get_contract_address(market_id)
            # contract = self.w3.eth.contract(address=contract_address, abi=CONTRACT_ABI)
            # 
            # tx = contract.functions.buyShares(
            #     market_id, outcome, amount
            # ).build_transaction({
            #     'from': self.account.address,
            #     'nonce': self.w3.eth.get_transaction_count(self.account.address),
            #     'gas': 200000,
            #     'gasPrice': self.w3.eth.gas_price
            # })
            # 
            # signed_tx = self.account.sign_transaction(tx)
            # tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            # receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Executing on-chain trade: {outcome} {amount} on {market_id}")
            
            # Mock implementation
            tx_hash = self.w3.keccak(text=f"{market_id}{outcome}{time.time()}").hex()
            
            return {
                'tx_hash': tx_hash,
                'status': 'pending',
                'gas_used': 0,
                'gas_price': 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"On-chain execution error: {e}")
            raise
    
    def check_stop_loss(self, market_id: str, current_prob: float) -> bool:
        """
        Check if any open positions should trigger stop loss.
        
        Args:
            market_id: Market identifier
            current_prob: Current market probability
            
        Returns:
            True if stop loss was triggered
        """
        # Find open positions for this market
        open_positions = [
            t for t in self.trade_history
            if t['market_id'] == market_id and t.get('closed', False) is False
        ]
        
        for position in open_positions:
            entry_prob = position['market_prob']
            outcome = position['outcome']
            
            if self.risk_manager.should_stop_loss(entry_prob, current_prob, outcome):
                logger.warning(f"Stop loss triggered for {market_id}")
                # Close position
                self.close_position(market_id, position)
                return True
        
        return False
    
    def close_position(
        self,
        market_id: str,
        position: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Close an open position.
        
        Args:
            market_id: Market identifier
            position: Position dictionary from trade history
            
        Returns:
            Close trade result
        """
        try:
            if self.dry_run:
                result = self._simulate_trade(market_id, 'CLOSE', position['amount'])
            else:
                result = self._execute_on_chain(market_id, 'CLOSE', position['amount'])
            
            position['closed'] = True
            position['close_tx'] = result
            
            return {
                'success': True,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.portfolio_value
    
    def get_trade_history(self) -> list:
        """Get trade history."""
        return self.trade_history
