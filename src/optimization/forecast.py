"""
Demand Forecasting Module for Inventory Optimization.

Provides multiple forecasting methods including moving average,
exponential smoothing, and seasonal decomposition.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class ForecastMethod(Enum):
    """Available forecasting methods."""
    SIMPLE_MOVING_AVERAGE = "sma"
    WEIGHTED_MOVING_AVERAGE = "wma"
    EXPONENTIAL_SMOOTHING = "exp"
    DOUBLE_EXPONENTIAL = "double_exp"
    HOLT_WINTERS = "holt_winters"


@dataclass
class ForecastResult:
    """Container for forecast results."""
    forecast: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    method: ForecastMethod
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error


class DemandForecaster:
    """
    Demand forecasting engine for inventory optimization.
    
    Provides multiple methods for forecasting future demand
    based on historical data.
    """
    
    def __init__(self, historical_demand: List[float]):
        """
        Initialize the forecaster with historical demand data.
        
        Args:
            historical_demand: List of historical demand values
        """
        if len(historical_demand) < 3:
            raise ValueError("At least 3 historical data points required")
        
        self.data = np.array(historical_demand)
        self._cached_seasonal_factors: Optional[np.ndarray] = None
    
    def simple_moving_average(self, periods: int, forecast_horizon: int = 1) -> ForecastResult:
        """
        Calculate Simple Moving Average forecast.
        
        Args:
            periods: Number of periods for averaging
            forecast_horizon: Number of periods to forecast ahead
            
        Returns:
            ForecastResult with forecasts and error metrics
        """
        if periods > len(self.data):
            periods = len(self.data)
        
        # Calculate SMA
        sma = np.mean(self.data[-periods:])
        forecast = [sma] * forecast_horizon
        
        # Calculate historical error for confidence intervals
        errors = []
        for i in range(periods, len(self.data)):
            pred = np.mean(self.data[i-periods:i])
            errors.append(abs(self.data[i] - pred))
        
        std_error = np.std(errors) if errors else np.std(self.data)
        
        lower = [max(0, f - 1.96 * std_error) for f in forecast]
        upper = [f + 1.96 * std_error for f in forecast]
        
        mape = self._calculate_mape(periods, ForecastMethod.SIMPLE_MOVING_AVERAGE)
        rmse = self._calculate_rmse(periods, ForecastMethod.SIMPLE_MOVING_AVERAGE)
        
        return ForecastResult(
            forecast=forecast,
            confidence_lower=lower,
            confidence_upper=upper,
            method=ForecastMethod.SIMPLE_MOVING_AVERAGE,
            mape=mape,
            rmse=rmse
        )
    
    def weighted_moving_average(
        self, 
        weights: List[float], 
        forecast_horizon: int = 1
    ) -> ForecastResult:
        """
        Calculate Weighted Moving Average forecast.
        
        Args:
            weights: Weights for each period (most recent last)
            forecast_horizon: Number of periods to forecast ahead
            
        Returns:
            ForecastResult with forecasts and error metrics
        """
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        n_weights = len(weights)
        if n_weights > len(self.data):
            raise ValueError("More weights than data points")
        
        # Calculate WMA
        wma = np.sum(self.data[-n_weights:] * weights)
        forecast = [wma] * forecast_horizon
        
        # Error estimation
        std_error = np.std(self.data[-n_weights:])
        lower = [max(0, f - 1.96 * std_error) for f in forecast]
        upper = [f + 1.96 * std_error for f in forecast]
        
        return ForecastResult(
            forecast=forecast,
            confidence_lower=lower,
            confidence_upper=upper,
            method=ForecastMethod.WEIGHTED_MOVING_AVERAGE,
            mape=self._estimate_mape(forecast[0]),
            rmse=self._estimate_rmse(forecast[0])
        )
    
    def exponential_smoothing(
        self, 
        alpha: float = 0.3, 
        forecast_horizon: int = 1
    ) -> ForecastResult:
        """
        Calculate Single Exponential Smoothing forecast.
        
        Args:
            alpha: Smoothing factor (0 < alpha < 1)
            forecast_horizon: Number of periods to forecast ahead
            
        Returns:
            ForecastResult with forecasts and error metrics
        """
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        # Calculate exponentially smoothed values
        smoothed = [self.data[0]]
        for i in range(1, len(self.data)):
            smoothed.append(alpha * self.data[i] + (1 - alpha) * smoothed[-1])
        
        forecast_value = smoothed[-1]
        forecast = [forecast_value] * forecast_horizon
        
        # Calculate forecast errors
        errors = [abs(self.data[i] - smoothed[i-1]) for i in range(1, len(self.data))]
        std_error = np.std(errors) if errors else np.std(self.data)
        
        lower = [max(0, f - 1.96 * std_error) for f in forecast]
        upper = [f + 1.96 * std_error for f in forecast]
        
        mape = np.mean([abs((self.data[i] - smoothed[i-1]) / self.data[i]) * 100 
                        for i in range(1, len(self.data)) if self.data[i] != 0])
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        
        return ForecastResult(
            forecast=forecast,
            confidence_lower=lower,
            confidence_upper=upper,
            method=ForecastMethod.EXPONENTIAL_SMOOTHING,
            mape=mape,
            rmse=rmse
        )
    
    def double_exponential_smoothing(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        forecast_horizon: int = 1
    ) -> ForecastResult:
        """
        Calculate Double Exponential Smoothing (Holt's Method).
        
        Handles data with trend component.
        
        Args:
            alpha: Level smoothing factor
            beta: Trend smoothing factor
            forecast_horizon: Number of periods to forecast ahead
            
        Returns:
            ForecastResult with forecasts and error metrics
        """
        if not (0 < alpha < 1 and 0 < beta < 1):
            raise ValueError("Alpha and beta must be between 0 and 1")
        
        # Initialize
        level = [self.data[0]]
        trend = [self.data[1] - self.data[0]]
        
        for i in range(1, len(self.data)):
            level.append(alpha * self.data[i] + (1 - alpha) * (level[-1] + trend[-1]))
            trend.append(beta * (level[-1] - level[-2]) + (1 - beta) * trend[-1])
        
        # Generate forecasts
        forecast = [level[-1] + (i + 1) * trend[-1] for i in range(forecast_horizon)]
        
        # Error estimation
        std_error = np.std(self.data)
        lower = [max(0, f - 1.96 * std_error) for f in forecast]
        upper = [f + 1.96 * std_error for f in forecast]
        
        return ForecastResult(
            forecast=forecast,
            confidence_lower=lower,
            confidence_upper=upper,
            method=ForecastMethod.DOUBLE_EXPONENTIAL,
            mape=self._estimate_mape(forecast[0]),
            rmse=self._estimate_rmse(forecast[0])
        )
    
    def detect_seasonality(self, period: int = 12) -> Dict[str, any]:
        """
        Detect seasonal patterns in the data.
        
        Args:
            period: Expected seasonal period (e.g., 12 for monthly)
            
        Returns:
            Dictionary with seasonality analysis results
        """
        if len(self.data) < 2 * period:
            return {'seasonal': False, 'reason': 'Insufficient data'}
        
        # Calculate seasonal indices
        n_complete_periods = len(self.data) // period
        seasonal_data = self.data[:n_complete_periods * period].reshape(-1, period)
        seasonal_means = np.mean(seasonal_data, axis=0)
        overall_mean = np.mean(self.data)
        
        seasonal_indices = seasonal_means / overall_mean
        self._cached_seasonal_factors = seasonal_indices
        
        # Test for seasonality using coefficient of variation
        cv = np.std(seasonal_indices) / np.mean(seasonal_indices)
        is_seasonal = cv > 0.1  # Threshold for seasonal significance
        
        return {
            'seasonal': is_seasonal,
            'seasonal_indices': seasonal_indices.tolist(),
            'coefficient_of_variation': cv,
            'period': period,
            'peak_period': int(np.argmax(seasonal_indices)),
            'trough_period': int(np.argmin(seasonal_indices))
        }
    
    def auto_select_method(self) -> Tuple[ForecastMethod, ForecastResult]:
        """
        Automatically select the best forecasting method.
        
        Evaluates multiple methods and returns the one with lowest MAPE.
        
        Returns:
            Tuple of (best_method, forecast_result)
        """
        results = {}
        
        # Test each method
        try:
            results[ForecastMethod.SIMPLE_MOVING_AVERAGE] = self.simple_moving_average(3)
        except Exception:
            pass
        
        try:
            results[ForecastMethod.EXPONENTIAL_SMOOTHING] = self.exponential_smoothing(0.3)
        except Exception:
            pass
        
        try:
            results[ForecastMethod.DOUBLE_EXPONENTIAL] = self.double_exponential_smoothing(0.3, 0.1)
        except Exception:
            pass
        
        # Select best based on MAPE
        best_method = min(results.keys(), key=lambda m: results[m].mape)
        return best_method, results[best_method]
    
    def _calculate_mape(self, periods: int, method: ForecastMethod) -> float:
        """Calculate Mean Absolute Percentage Error."""
        errors = []
        for i in range(periods, len(self.data)):
            if method == ForecastMethod.SIMPLE_MOVING_AVERAGE:
                pred = np.mean(self.data[i-periods:i])
            else:
                pred = self.data[i-1]
            
            if self.data[i] != 0:
                errors.append(abs((self.data[i] - pred) / self.data[i]) * 100)
        
        return np.mean(errors) if errors else 0.0
    
    def _calculate_rmse(self, periods: int, method: ForecastMethod) -> float:
        """Calculate Root Mean Square Error."""
        errors = []
        for i in range(periods, len(self.data)):
            if method == ForecastMethod.SIMPLE_MOVING_AVERAGE:
                pred = np.mean(self.data[i-periods:i])
            else:
                pred = self.data[i-1]
            errors.append((self.data[i] - pred) ** 2)
        
        return np.sqrt(np.mean(errors)) if errors else 0.0
    
    def _estimate_mape(self, forecast: float) -> float:
        """Estimate MAPE based on data variance."""
        avg = np.mean(self.data)
        return abs(avg - forecast) / avg * 100 if avg != 0 else 0
    
    def _estimate_rmse(self, forecast: float) -> float:
        """Estimate RMSE based on data standard deviation."""
        return np.std(self.data)
