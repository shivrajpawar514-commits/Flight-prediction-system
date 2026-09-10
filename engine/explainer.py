"""
Explainable AI (XAI) Engine using SHAP (SHapley Additive exPlanations) & PDP.
Computes local feature attributions (in ₹), waterfall breakdowns,
global feature importance, and partial dependence curves.
"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

FEATURE_FRIENDLY_NAMES = {
    "Total_Stops": "Total Number of Stops",
    "Journey_Day": "Day of Month",
    "Journey_Month": "Month of Travel",
    "Day_of_Week": "Day of Week",
    "Is_Weekend": "Weekend Travel Flag",
    "Dep_Hour": "Departure Hour",
    "Dep_Min": "Departure Minute",
    "Arrival_Hour": "Arrival Hour",
    "Arrival_Min": "Arrival Minute",
    "Duration_Hours": "Flight Duration (Hours)",
    "Duration_Mins": "Flight Duration (Minutes)",
    "Duration_Total_Mins": "Total Duration (Minutes)",
    "Route_Distance_KM": "Geodesic Route Distance (km)",
    "Estimated_Speed_KMH": "Estimated Flight Velocity (km/h)",
    "Stops_x_Duration": "Stop-Duration Congestion Factor",
    "Distance_per_Stop": "Distance per Segment (km)",
    "Is_Morning_Flight": "Morning Departure Window",
    "Is_Evening_Flight": "Evening Departure Window",
    "Is_RedEye_Flight": "Red-Eye / Night Flight",
    "Airline_Jet Airways": "Airline: Jet Airways",
    "Airline_IndiGo": "Airline: IndiGo",
    "Airline_Air India": "Airline: Air India",
    "Airline_Multiple carriers": "Airline: Multiple Carriers",
    "Airline_SpiceJet": "Airline: SpiceJet",
    "Airline_Vistara": "Airline: Vistara",
    "Airline_Air Asia": "Airline: Air Asia",
    "Airline_GoAir": "Airline: GoAir",
    "Airline_Jet Airways Business": "Airline: Jet Airways Business",
    "Source_Delhi": "Origin: Delhi",
    "Source_Kolkata": "Origin: Kolkata",
    "Source_Banglore": "Origin: Banglore",
    "Source_Mumbai": "Origin: Mumbai",
    "Source_Chennai": "Origin: Chennai",
    "Destination_Cochin": "Destination: Cochin",
    "Destination_Delhi": "Destination: Delhi",
    "Destination_Banglore": "Destination: Banglore",
    "Destination_Hyderabad": "Destination: Hyderabad",
    "Destination_Kolkata": "Destination: Kolkata",
    "Destination_New Delhi": "Destination: New Delhi"
}


class ModelExplainer:
    """
    SHAP-based Explainability engine for local and global model interpretability.
    """

    def __init__(self, model, feature_names: List[str], base_expected_value: float):
        self.model = model
        self.feature_names = feature_names
        self.base_expected_value = float(base_expected_value)
        self.tree_explainer = shap.TreeExplainer(self.model)

    def explain_prediction(self, X_input: pd.DataFrame, top_k: int = 8) -> Dict[str, Any]:
        """
        Computes exact local SHAP feature attributions for a single flight.
        Returns waterfall components sorted by absolute contribution.
        """
        # Compute shap values
        shap_vals = self.tree_explainer.shap_values(X_input)
        if isinstance(shap_vals, list):
            sv = shap_vals[0]
        else:
            sv = shap_vals

        if sv.ndim > 1:
            sv = sv[0]

        predicted_val = float(self.model.predict(X_input)[0])
        base_val = self.base_expected_value

        attributions = []
        for idx, col in enumerate(self.feature_names):
            shap_impact = float(sv[idx])
            val = float(X_input.iloc[0][col])
            
            # Skip zero impact or unactivated dummy features
            if abs(shap_impact) < 5.0 and (col.startswith("Airline_") or col.startswith("Source_") or col.startswith("Destination_")) and val == 0:
                continue

            friendly_name = FEATURE_FRIENDLY_NAMES.get(col, col.replace("_", " "))
            
            attributions.append({
                "feature_code": col,
                "feature_name": friendly_name,
                "input_value": val,
                "shap_impact": round(shap_impact, 2),
                "abs_impact": abs(shap_impact),
                "direction": "increases_price" if shap_impact >= 0 else "decreases_price"
            })

        # Sort by absolute impact
        attributions.sort(key=lambda item: item["abs_impact"], reverse=True)
        top_attributions = attributions[:top_k]

        # Group remaining as 'Other Factors'
        other_sum = sum(item["shap_impact"] for item in attributions[top_k:])
        if abs(other_sum) > 1.0:
            top_attributions.append({
                "feature_code": "other",
                "feature_name": "Other Combined Features",
                "input_value": 0,
                "shap_impact": round(other_sum, 2),
                "abs_impact": abs(other_sum),
                "direction": "increases_price" if other_sum >= 0 else "decreases_price"
            })

        return {
            "base_price": round(base_val, 2),
            "predicted_price": round(predicted_val, 2),
            "net_shift": round(predicted_val - base_val, 2),
            "net_shap_shift": round(predicted_val - base_val, 2),
            "attributions": top_attributions
        }

    def compute_partial_dependence(self, X_sample: pd.DataFrame, feature_col: str, grid_values: List[float]) -> List[Dict[str, float]]:
        """
        Computes 1D Partial Dependence values across a variable grid.
        """
        pdp_results = []
        if feature_col not in self.feature_names:
            return pdp_results

        df_copy = X_sample.copy().iloc[:50]  # sample subset for fast interactive computation
        for val in grid_values:
            df_copy[feature_col] = val
            preds = self.model.predict(df_copy)
            avg_pred = float(np.mean(preds))
            pdp_results.append({
                "value": val,
                "partial_price": round(avg_pred, 2)
            })

        return pdp_results
