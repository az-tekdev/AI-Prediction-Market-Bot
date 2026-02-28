"""
AI prediction engine for outcome probability estimation.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pickle
import os

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available, using simple models")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch not available, LSTM models disabled")

logger = logging.getLogger(__name__)


class BasePredictor:
    """Base class for prediction models."""
    
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict outcome probability.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Probability of YES outcome (0-1)
        """
        raise NotImplementedError
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels (1 for YES, 0 for NO)
            
        Returns:
            Dictionary with training metrics
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        raise NotImplementedError


class LogisticRegressionPredictor(BasePredictor):
    """Logistic regression based predictor."""
    
    def __init__(self):
        """Initialize the predictor."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for LogisticRegressionPredictor")
        
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def _extract_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Extract and normalize features."""
        if self.feature_names is None:
            # Initialize with default features
            self.feature_names = [
                'current_yes_prob', 'volume_24h', 'liquidity',
                'time_to_resolution', 'sentiment_score', 'news_count'
            ]
        
        feature_vector = np.array([
            features.get(name, 0.0) for name in self.feature_names
        ]).reshape(1, -1)
        
        if hasattr(self.scaler, 'mean_'):
            feature_vector = self.scaler.transform(feature_vector)
        
        return feature_vector
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predict YES probability."""
        try:
            X = self._extract_features(features)
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5  # Default to neutral
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model."""
        try:
            self.feature_names = list(X.columns)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            train_proba = self.model.predict_proba(X_train)[:, 1]
            test_proba = self.model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'train_log_loss': log_loss(y_train, train_proba),
                'test_log_loss': log_loss(y_test, test_proba)
            }
            
            logger.info(f"Training completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {}
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']


class RandomForestPredictor(BasePredictor):
    """Random forest based predictor."""
    
    def __init__(self, n_estimators: int = 100):
        """Initialize the predictor."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for RandomForestPredictor")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            max_depth=10
        )
        self.feature_names = None
    
    def _extract_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Extract features."""
        if self.feature_names is None:
            self.feature_names = [
                'current_yes_prob', 'volume_24h', 'liquidity',
                'time_to_resolution', 'sentiment_score', 'news_count'
            ]
        
        return np.array([
            features.get(name, 0.0) for name in self.feature_names
        ]).reshape(1, -1)
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predict YES probability."""
        try:
            X = self._extract_features(features)
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model."""
        try:
            self.feature_names = list(X.columns)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            train_proba = self.model.predict_proba(X_train)[:, 1]
            test_proba = self.model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'train_log_loss': log_loss(y_train, train_proba),
                'test_log_loss': log_loss(y_test, test_proba)
            }
            
            logger.info(f"Training completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {}
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']


class LLMPredictor(BasePredictor):
    """LLM-based predictor using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM predictor."""
        self.api_key = api_key
        if not api_key:
            logger.warning("No OpenAI API key provided, LLM predictions disabled")
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predict using LLM."""
        if not self.api_key:
            return 0.5
        
        try:
            # In production, make actual API call to OpenAI
            # For now, return mock prediction based on features
            yes_prob = features.get('current_yes_prob', 0.5)
            sentiment = features.get('sentiment_score', 0.0)
            
            # Simple heuristic: adjust based on sentiment
            adjusted_prob = yes_prob + sentiment * 0.1
            return max(0.0, min(1.0, adjusted_prob))
            
        except Exception as e:
            logger.error(f"LLM prediction error: {e}")
            return 0.5
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """LLM models don't require traditional training."""
        logger.info("LLM predictor doesn't require training")
        return {}
    
    def save(self, path: str) -> None:
        """Save API key configuration."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.api_key or "")
    
    def load(self, path: str) -> None:
        """Load API key configuration."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.api_key = f.read().strip()


class AIPredictor:
    """Main AI prediction engine that manages multiple models."""
    
    def __init__(
        self,
        model_type: str = "logistic",
        model_path: Optional[str] = None,
        use_llm: bool = False,
        llm_api_key: Optional[str] = None
    ):
        """
        Initialize AI predictor.
        
        Args:
            model_type: Type of model ('logistic', 'random_forest')
            model_path: Path to saved model file
            use_llm: Whether to use LLM for predictions
            llm_api_key: OpenAI API key if using LLM
        """
        self.model_type = model_type
        self.model_path = model_path or f"models/{model_type}_model.pkl"
        self.use_llm = use_llm
        
        # Initialize base model
        if model_type == "logistic":
            self.model = LogisticRegressionPredictor()
        elif model_type == "random_forest":
            self.model = RandomForestPredictor()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize LLM if enabled
        self.llm_model = None
        if use_llm:
            self.llm_model = LLMPredictor(llm_api_key)
        
        # Load model if path exists
        if model_path and os.path.exists(model_path):
            try:
                self.model.load(model_path)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def predict(
        self,
        market_data: Dict[str, Any],
        news_data: Optional[List[Dict[str, Any]]] = None,
        sentiment: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Generate prediction for a market.
        
        Args:
            market_data: Market information dictionary
            news_data: Optional list of news articles
            sentiment: Optional sentiment analysis results
            
        Returns:
            Tuple of (predicted_probability, metadata)
        """
        try:
            # Extract features
            features = self._extract_features(market_data, news_data, sentiment)
            
            # Get base model prediction
            base_pred = self.model.predict(features)
            
            # Optionally use LLM for refinement
            if self.use_llm and self.llm_model:
                llm_pred = self.llm_model.predict(features)
                # Combine predictions (weighted average)
                final_pred = 0.7 * base_pred + 0.3 * llm_pred
            else:
                final_pred = base_pred
            
            metadata = {
                'base_prediction': base_pred,
                'features': features,
                'model_type': self.model_type,
                'llm_used': self.use_llm
            }
            
            return float(final_pred), metadata
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5, {'error': str(e)}
    
    def _extract_features(
        self,
        market_data: Dict[str, Any],
        news_data: Optional[List[Dict[str, Any]]],
        sentiment: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Extract features from market and external data."""
        # Current market odds
        current_yes_prob = market_data.get('current_odds', {}).get('YES', 0.5)
        
        # Market metrics
        volume_24h = market_data.get('volume_24h', 0)
        liquidity = market_data.get('liquidity', 0)
        
        # Time to resolution
        end_date_str = market_data.get('end_date')
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                time_to_resolution = (end_date - datetime.now()).total_seconds() / 86400  # days
            except:
                time_to_resolution = 7.0
        else:
            time_to_resolution = 7.0
        
        # Sentiment features
        sentiment_score = 0.0
        if sentiment:
            sentiment_score = sentiment.get('score', 0.0)
        
        # News count
        news_count = len(news_data) if news_data else 0
        
        return {
            'current_yes_prob': current_yes_prob,
            'volume_24h': volume_24h,
            'liquidity': liquidity,
            'time_to_resolution': time_to_resolution,
            'sentiment_score': sentiment_score,
            'news_count': news_count
        }
    
    def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the model on historical data.
        
        Args:
            training_data: DataFrame with features and target column 'outcome'
            
        Returns:
            Training metrics
        """
        try:
            if 'outcome' not in training_data.columns:
                raise ValueError("Training data must include 'outcome' column")
            
            X = training_data.drop('outcome', axis=1)
            y = training_data['outcome']
            
            metrics = self.model.train(X, y)
            
            # Save trained model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {}
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the model."""
        save_path = path or self.model_path
        self.model.save(save_path)
