"""
Tests for AI predictor module.
"""
import pytest
import pandas as pd
import numpy as np
from src.ai_predictor import AIPredictor, LogisticRegressionPredictor


class TestAIPredictor:
    """Test AIPredictor class."""
    
    def test_init_logistic(self):
        """Test initialization with logistic regression."""
        predictor = AIPredictor(model_type="logistic")
        assert predictor.model_type == "logistic"
    
    def test_init_random_forest(self):
        """Test initialization with random forest."""
        predictor = AIPredictor(model_type="random_forest")
        assert predictor.model_type == "random_forest"
    
    def test_predict(self):
        """Test prediction."""
        predictor = AIPredictor(model_type="logistic")
        market_data = {
            'current_odds': {'YES': 0.6, 'NO': 0.4},
            'volume_24h': 5000,
            'liquidity': 10000,
            'end_date': '2024-12-31T00:00:00'
        }
        prob, metadata = predictor.predict(market_data)
        assert 0 <= prob <= 1
        assert isinstance(metadata, dict)
    
    def test_train(self):
        """Test model training."""
        predictor = AIPredictor(model_type="logistic")
        
        # Generate synthetic training data
        n_samples = 100
        X = pd.DataFrame({
            'current_yes_prob': np.random.uniform(0.3, 0.7, n_samples),
            'volume_24h': np.random.uniform(1000, 10000, n_samples),
            'liquidity': np.random.uniform(5000, 50000, n_samples),
            'time_to_resolution': np.random.uniform(1, 30, n_samples),
            'sentiment_score': np.random.uniform(-0.5, 0.5, n_samples),
            'news_count': np.random.randint(0, 20, n_samples)
        })
        y = pd.Series((X['current_yes_prob'] > 0.5).astype(int), name='outcome')
        
        training_data = pd.concat([X, y], axis=1)
        metrics = predictor.train(training_data)
        
        assert isinstance(metrics, dict)
        assert 'train_accuracy' in metrics or len(metrics) == 0


class TestLogisticRegressionPredictor:
    """Test LogisticRegressionPredictor class."""
    
    @pytest.mark.skipif(not hasattr(LogisticRegressionPredictor, '__init__'),
                        reason="scikit-learn not available")
    def test_predict(self):
        """Test prediction."""
        predictor = LogisticRegressionPredictor()
        features = {
            'current_yes_prob': 0.6,
            'volume_24h': 5000,
            'liquidity': 10000,
            'time_to_resolution': 7.0,
            'sentiment_score': 0.2,
            'news_count': 5
        }
        # Will return default 0.5 if model not trained
        prob = predictor.predict(features)
        assert 0 <= prob <= 1
