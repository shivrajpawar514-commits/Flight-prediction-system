"""
AI Fare Timing Advisor, Deal Classifier & Price Trend Engine.
Provides smart 'Buy Now vs. Wait' recommendations, anomaly/deal detection,
and 14-day price trajectory projections.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

class FareAdvisorEngine:
    """
    AI Travel Advisor analyzing fare anomalies and timing recommendations.
    """

    def __init__(self, raw_df: pd.DataFrame):
        self.df = raw_df.copy()
        # Compute route-level statistics
        self.route_stats = {}
        for (src, dst), group in self.df.groupby(['Source', 'Destination']):
            prices = group['Price'].values
            self.route_stats[(src, dst)] = {
                "mean": float(np.mean(prices)),
                "median": float(np.median(prices)),
                "std": float(np.std(prices)),
                "p15": float(np.percentile(prices, 15)),
                "p35": float(np.percentile(prices, 35)),
                "p70": float(np.percentile(prices, 70)),
                "p90": float(np.percentile(prices, 90)),
                "min": float(np.min(prices)),
                "max": float(np.max(prices)),
                "count": len(prices)
            }

        # Global stats fallback
        all_prices = self.df['Price'].values
        self.global_stats = {
            "mean": float(np.mean(all_prices)),
            "median": float(np.median(all_prices)),
            "std": float(np.std(all_prices)),
            "p15": float(np.percentile(all_prices, 15)),
            "p35": float(np.percentile(all_prices, 35)),
            "p70": float(np.percentile(all_prices, 70)),
            "p90": float(np.percentile(all_prices, 90)),
            "min": float(np.min(all_prices)),
            "max": float(np.max(all_prices)),
            "count": len(all_prices)
        }

    def evaluate_fare(self, predicted_price: float, source: str, destination: str,
                      departure_date: str) -> Dict[str, Any]:
        """
        Evaluates a predicted fare against route benchmarks and outputs Buy/Wait advice.
        """
        stats = self.route_stats.get((source, destination), self.global_stats)

        p15 = stats["p15"]
        p35 = stats["p35"]
        p70 = stats["p70"]
        p90 = stats["p90"]
        median_price = stats["median"]

        # Calculate percentile score of predicted price
        diff_from_median = predicted_price - median_price
        diff_pct = round((diff_from_median / max(1.0, median_price)) * 100.0, 1)

        # Classification & Timing Advice
        if predicted_price <= p15:
            deal_rating = "Rare Bargain 🔥"
            deal_badge = "bargain"
            action = "BUY NOW"
            action_desc = f"Fare is in the lowest 15% historically for {source} → {destination}. Exceptional value!"
            confidence_score = 94
            risk_of_rise = "Very High (88%)"
        elif predicted_price <= p35:
            deal_rating = "Great Deal ✨"
            deal_badge = "good"
            action = "BUY SOON"
            action_desc = f"Price is {abs(diff_pct)}% below typical median (₹{int(median_price)}). Favorable window."
            confidence_score = 82
            risk_of_rise = "High (72%)"
        elif predicted_price <= p70:
            deal_rating = "Fair Market Price ⚖️"
            deal_badge = "fair"
            action = "MONITOR OR BOOK"
            action_desc = f"Standard market rate for this route. Prices may fluctuate moderately."
            confidence_score = 65
            risk_of_rise = "Moderate (50%)"
        else:
            deal_rating = "Surge / High Demand ⚠️"
            deal_badge = "surge"
            action = "CONSIDER WAITING"
            action_desc = f"Price is {diff_pct}% higher than median due to peak demand or tight seat capacity."
            confidence_score = 75
            risk_of_rise = "Low / Price Peak (30%)"

        # 14-day price trajectory simulation
        trajectory = self._generate_trajectory(predicted_price)

        return {
            "action": action,
            "action_desc": action_desc,
            "deal_rating": deal_rating,
            "deal_badge": deal_badge,
            "confidence_score": confidence_score,
            "risk_of_rise": risk_of_rise,
            "route_median": round(median_price, 2),
            "diff_from_median": round(diff_from_median, 2),
            "diff_pct": diff_pct,
            "route_min": round(stats["min"], 2),
            "route_max": round(stats["max"], 2),
            "trajectory": trajectory
        }

    def _generate_trajectory(self, current_price: float) -> List[Dict[str, Any]]:
        """
        Generates 14-day forward expected pricing curve based on airline dynamic pricing curve.
        """
        trajectory = []
        days_ahead = list(range(14, 0, -1))  # 14 days before flight down to 1 day before flight

        for d in days_ahead:
            # Yield management curve: prices rise as departure approaches
            if d > 10:
                mult = 0.96 + (14 - d) * 0.008
            elif d > 5:
                mult = 1.00 + (10 - d) * 0.025
            elif d > 2:
                mult = 1.12 + (5 - d) * 0.05
            else:
                mult = 1.28 + (2 - d) * 0.10

            expected_p = current_price * mult
            trajectory.append({
                "days_before_dep": d,
                "projected_price": round(expected_p, 2),
                "trend": "rising" if mult > 1.0 else "stable"
            })

        return trajectory
