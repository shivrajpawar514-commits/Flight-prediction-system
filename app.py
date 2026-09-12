"""
AeroPulse AI - Enterprise Flight Price Prediction & Market Intelligence Platform.
Flask Application serving modern interactive UI and RESTful AI/ML endpoints.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for

from engine.preprocessor import FlightDataPreprocessor, AIRPORT_COORDINATES, AIRPORT_CODES, haversine_distance
from engine.explainer import ModelExplainer
from engine.uncertainty import UncertaintyEngine
from engine.advisor import FareAdvisorEngine
from engine.recommender import MultiCriteriaRecommender
from engine.analytics import FlightAnalyticsEngine

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ---------------------------------------------------------
# Load Dataset, Serialized Models & AI Engines on Startup
# ---------------------------------------------------------
MODEL_PATH = "model/flight_ensemble_models.pkl"
BENCHMARK_PATH = "model/model_benchmark.json"
DATA_PATH = "flight_data.csv"

print("[*] Initializing AeroPulse AI Server...")
df_raw = pd.read_csv(DATA_PATH)

package = joblib.load(MODEL_PATH)

models = package["models"]
preprocessor: FlightDataPreprocessor = package["preprocessor"]
feature_names = package["feature_names"]
uncertainty_params = package["uncertainty"]
base_expected_value = package["base_expected_value"]
feature_importance = package.get("feature_importance", [])

with open(BENCHMARK_PATH, "r") as f:
    benchmark_metrics = json.load(f)

# Instantiate Core Engines
explainer = ModelExplainer(models["Random Forest"], feature_names, base_expected_value)
uncertainty_engine = UncertaintyEngine(uncertainty_params)
advisor_engine = FareAdvisorEngine(df_raw)
recommender_engine = MultiCriteriaRecommender(df_raw)
analytics_engine = FlightAnalyticsEngine(df_raw)

print(f"[✓] AeroPulse AI loaded {len(models)} models and dataset with {len(df_raw)} records.")


# ---------------------------------------------------------
# Web Page Views
# ---------------------------------------------------------

@app.route("/")
def home():
    """Main Forecaster, Explainable AI & What-If Simulator Hub."""
    airlines = preprocessor.airline_list
    sources = preprocessor.source_list
    destinations = preprocessor.dest_list
    model_names = list(models.keys())
    kpis = analytics_engine.get_summary_kpis()

    return render_template(
        "home.html",
        airlines=airlines,
        sources=sources,
        destinations=destinations,
        model_names=model_names,
        kpis=kpis
    )


@app.route("/dashboard")
def dashboard():
    """Market Intelligence, Interactive Route Map & Pricing Power."""
    kpis = analytics_engine.get_summary_kpis()
    routes = analytics_engine.get_route_intelligence()
    airlines = analytics_engine.get_airline_analytics()
    temporal = analytics_engine.get_temporal_analytics()
    airport_nodes = analytics_engine.get_airport_nodes()

    return render_template(
        "dashboard.html",
        kpis=kpis,
        routes=routes,
        airlines=airlines,
        temporal=temporal,
        airport_nodes=airport_nodes
    )


@app.route("/recommend")
def recommend():
    """TOPSIS & Pareto Frontier Multi-Criteria Flight Explorer."""
    sources = ["All"] + preprocessor.source_list
    destinations = ["All"] + preprocessor.dest_list
    initial_recs = recommender_engine.rank_flights()

    return render_template(
        "recommend.html",
        sources=sources,
        destinations=destinations,
        recommendations=initial_recs["flights"],
        pareto_frontier=initial_recs["pareto_frontier"]
    )


@app.route("/benchmark")
def benchmark():
    """ML Model Benchmark Leaderboard, Cross-Validation & Global Feature Importance."""
    top_features = feature_importance[:20] if feature_importance else []
    return render_template(
        "benchmark.html",
        benchmark_metrics=benchmark_metrics,
        top_features=top_features
    )


@app.route("/docs")
def docs():
    """Interactive REST API Documentation & Live Sandbox Console."""
    return render_template("docs.html")


# ---------------------------------------------------------
# REST API Endpoints (JSON)
# ---------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Computes price prediction, confidence intervals, SHAP feature attributions, and Buy/Wait advice.
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()

        dep_time = data.get("Dep_Time", "2026-06-15T09:30")
        arr_time = data.get("Arrival_Time", "2026-06-15T12:45")
        source = data.get("Source", "Delhi")
        dest = data.get("Destination", "Cochin")
        stops = int(data.get("stops", 0))
        airline = data.get("airline", "IndiGo")
        model_name = data.get("model_name", "Stacking Ensemble")

        # Select model
        selected_model = models.get(model_name, models["Stacking Ensemble"])

        # Preprocess input
        X_input = preprocessor.create_single_input(
            dep_time=dep_time,
            arr_time=arr_time,
            source=source,
            destination=dest,
            stops=stops,
            airline=airline
        )

        # Predict
        predicted_price = float(selected_model.predict(X_input)[0])
        predicted_price = max(1200.0, predicted_price)

        # Uncertainty intervals
        intervals = uncertainty_engine.compute_prediction_intervals(predicted_price)

        # SHAP Explainability
        shap_explanation = explainer.explain_prediction(X_input)

        # Fare Advisor & Timing
        fare_advice = advisor_engine.evaluate_fare(
            predicted_price=predicted_price,
            source=source,
            destination=dest,
            departure_date=dep_time
        )

        response = {
            "status": "success",
            "model_used": model_name,
            "prediction": {
                "price": int(round(predicted_price)),
                "formatted_price": f"₹{int(round(predicted_price)):,}",
                "confidence_interval_80": intervals["ci_80"],
                "confidence_interval_95": intervals["ci_95"],
                "std_error": intervals["std_error"]
            },
            "explainability": {
                "base_price": int(round(shap_explanation["base_price"])),
                "net_shift": int(round(shap_explanation["net_shift"])),
                "attributions": shap_explanation["attributions"]
            },
            "advisor": fare_advice,
            "input_metadata": {
                "source": source,
                "destination": dest,
                "airline": airline,
                "stops": stops,
                "departure_time": dep_time,
                "arrival_time": arr_time
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    """
    POST /api/whatif
    Real-time sensitivity analysis comparing alternate airlines, stops, and departure times.
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()

        dep_time = data.get("Dep_Time", "2026-06-15T09:30")
        arr_time = data.get("Arrival_Time", "2026-06-15T12:45")
        source = data.get("Source", "Delhi")
        dest = data.get("Destination", "Cochin")
        current_stops = int(data.get("stops", 0))
        current_airline = data.get("airline", "IndiGo")
        model_name = data.get("model_name", "Stacking Ensemble")

        model = models.get(model_name, models["Stacking Ensemble"])

        # Base prediction
        base_X = preprocessor.create_single_input(dep_time, arr_time, source, dest, current_stops, current_airline)
        base_price = float(model.predict(base_X)[0])

        # 1. Compare across all airlines
        airline_comparisons = []
        for a in preprocessor.airline_list:
            if a in ["Jet Airways Business", "Vistara Premium economy", "Multiple carriers Premium economy", "Trujet"]:
                continue
            alt_X = preprocessor.create_single_input(dep_time, arr_time, source, dest, current_stops, a)
            alt_price = float(model.predict(alt_X)[0])
            diff = alt_price - base_price
            airline_comparisons.append({
                "airline": a,
                "price": int(round(alt_price)),
                "difference": int(round(diff)),
                "is_current": (a == current_airline)
            })
        airline_comparisons.sort(key=lambda x: x['price'])

        # 2. Compare across stop counts (0, 1, 2)
        stops_comparisons = []
        for s in [0, 1, 2]:
            alt_X = preprocessor.create_single_input(dep_time, arr_time, source, dest, s, current_airline)
            alt_price = float(model.predict(alt_X)[0])
            diff = alt_price - base_price
            stops_comparisons.append({
                "stops": s,
                "label": "Non-Stop" if s == 0 else f"{s} Stop{'s' if s>1 else ''}",
                "price": int(round(alt_price)),
                "difference": int(round(diff)),
                "is_current": (s == current_stops)
            })

        # 3. Compare departure time shifts (-3h, -1h, +2h, +5h)
        dep_dt = pd.to_datetime(dep_time)
        arr_dt = pd.to_datetime(arr_time)
        duration = arr_dt - dep_dt

        time_shifts = [-4, -2, 2, 4, 6]
        time_comparisons = []
        for shift in time_shifts:
            new_dep = dep_dt + pd.Timedelta(hours=shift)
            new_arr = new_dep + duration
            alt_X = preprocessor.create_single_input(
                new_dep.strftime("%Y-%m-%dT%H:%M"),
                new_arr.strftime("%Y-%m-%dT%H:%M"),
                source, dest, current_stops, current_airline
            )
            alt_price = float(model.predict(alt_X)[0])
            diff = alt_price - base_price
            time_comparisons.append({
                "shift_label": f"{'+' if shift>0 else ''}{shift} Hours ({new_dep.strftime('%H:%M')})",
                "price": int(round(alt_price)),
                "difference": int(round(diff))
            })

        return jsonify({
            "status": "success",
            "base_price": int(round(base_price)),
            "airline_matrix": airline_comparisons,
            "stops_matrix": stops_comparisons,
            "time_shift_matrix": time_comparisons
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    POST /api/simulate
    Runs 1,000-iteration Monte Carlo price distribution simulation.
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
        price = float(data.get("price", 8500.0))
        volatility = float(data.get("volatility", 0.12))
        sim_res = uncertainty_engine.simulate_monte_carlo(price, n_simulations=1000, volatility=volatility)
        return jsonify({"status": "success", "simulation": sim_res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    POST /api/recommend
    Multi-Criteria Decision Making (TOPSIS) flight ranking with customizable weights.
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
        
        weights = {
            "price": float(data.get("weight_price", 0.40)),
            "duration": float(data.get("weight_duration", 0.25)),
            "reliability": float(data.get("weight_reliability", 0.15)),
            "stops": float(data.get("weight_stops", 0.10)),
            "time_of_day": float(data.get("weight_time", 0.10))
        }
        
        source = data.get("source", "All")
        destination = data.get("destination", "All")

        result = recommender_engine.rank_flights(
            user_weights=weights,
            filter_source=source,
            filter_dest=destination,
            limit=30
        )
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/benchmark", methods=["GET"])
def api_benchmark():
    """GET /api/benchmark - Returns ML model evaluation metrics & leaderboard."""
    return jsonify({
        "status": "success",
        "benchmark": benchmark_metrics,
        "feature_importance": feature_importance
    })


@app.route("/api/analytics/routes", methods=["GET"])
def api_analytics_routes():
    """GET /api/analytics/routes - Returns route distances, coordinates, and market intelligence."""
    routes = analytics_engine.get_route_intelligence()
    airports = analytics_engine.get_airport_nodes()
    return jsonify({
        "status": "success",
        "routes": routes,
        "airports": airports
    })


@app.route("/api/features/importance", methods=["GET"])
def api_feature_importance():
    """GET /api/features/importance - Returns top global feature importance."""
    return jsonify({
        "status": "success",
        "features": feature_importance[:25]
    })


# ---------------------------------------------------------
# Application Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=4000)
