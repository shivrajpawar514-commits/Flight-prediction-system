"""
Uncertainty Quantification & Monte Carlo Volatility Simulation Engine.
Computes empirical confidence intervals (80% & 95%) and simulates
price probability distributions under market fluctuations.
"""

import numpy as np
from typing import Dict, Any, List

class UncertaintyEngine:
    """
    Quantifies prediction uncertainty and executes Monte Carlo price simulations.
    """

    def __init__(self, uncertainty_params: Dict[str, float]):
        self.std_res = uncertainty_params.get("std_residual", 1800.0)
        self.res_80_lower = uncertainty_params.get("res_80_lower", -1500.0)
        self.res_80_upper = uncertainty_params.get("res_80_upper", 1500.0)
        self.res_95_lower = uncertainty_params.get("res_95_lower", -3000.0)
        self.res_95_upper = uncertainty_params.get("res_95_upper", 3000.0)

    def compute_prediction_intervals(self, predicted_price: float, volatility_multiplier: float = 1.0) -> Dict[str, Any]:
        """
        Calculates 80% and 95% confidence/prediction intervals around the predicted price.
        """
        # Scale residuals based on volatility (e.g. peak season, long haul routes)
        v_scale = max(0.5, volatility_multiplier)
        
        lower_80 = max(1000.0, predicted_price + (self.res_80_lower * v_scale))
        upper_80 = max(lower_80 + 500.0, predicted_price + (self.res_80_upper * v_scale))

        lower_95 = max(800.0, predicted_price + (self.res_95_lower * v_scale))
        upper_95 = max(lower_95 + 1000.0, predicted_price + (self.res_95_upper * v_scale))

        # Margin of error as percentage
        margin_80_pct = round(((upper_80 - lower_80) / (2 * max(1.0, predicted_price))) * 100, 1)

        return {
            "predicted_price": round(predicted_price, 2),
            "ci_80": {
                "lower": round(lower_80, 2),
                "upper": round(upper_80, 2),
                "margin_pct": margin_80_pct
            },
            "ci_95": {
                "lower": round(lower_95, 2),
                "upper": round(upper_95, 2)
            },
            "std_error": round(self.std_res * v_scale, 2)
        }

    def simulate_monte_carlo(self, base_price: float, n_simulations: int = 1000,
                             volatility: float = 0.12) -> Dict[str, Any]:
        """
        Runs Monte Carlo simulation of 1000 pricing paths sampling geometric Brownian
        variations, demand shocks, and residual noise.
        """
        np.random.seed(42)
        # Log-normal distribution to avoid negative prices
        sigma = volatility
        mu = np.log(max(100.0, base_price)) - 0.5 * sigma**2
        simulated_prices = np.random.lognormal(mean=mu, sigma=sigma, size=n_simulations)

        # Add empirical noise
        noise = np.random.normal(0, self.std_res * 0.3, size=n_simulations)
        final_prices = np.maximum(1200.0, simulated_prices + noise)

        # Compute distribution statistics
        p10, p25, p50, p75, p90 = np.percentile(final_prices, [10, 25, 50, 75, 90])
        
        # Build histogram bins (20 bins)
        counts, bin_edges = np.histogram(final_prices, bins=20)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        histogram_data = [
            {"price_bin": int(bin_centers[i]), "frequency": int(counts[i])}
            for i in range(len(counts))
        ]

        return {
            "n_simulations": n_simulations,
            "mean_price": round(float(np.mean(final_prices)), 2),
            "median_price": round(float(p50), 2),
            "std_dev": round(float(np.std(final_prices)), 2),
            "min_price": round(float(np.min(final_prices)), 2),
            "max_price": round(float(np.max(final_prices)), 2),
            "percentiles": {
                "p10": round(float(p10), 2),
                "p25": round(float(p25), 2),
                "p50": round(float(p50), 2),
                "p75": round(float(p75), 2),
                "p90": round(float(p90), 2)
            },
            "distribution": histogram_data
        }
