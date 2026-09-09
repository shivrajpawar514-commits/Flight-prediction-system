"""
Feature Engineering and Preprocessing Pipeline for AeroPulse AI.
Includes cyclical time encoding, geodesic airport distances, interaction features,
and scikit-learn compatible transformation.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# Geocoordinates of major Indian airports in the dataset
AIRPORT_COORDINATES: Dict[str, Tuple[float, float]] = {
    "Delhi": (28.5562, 77.1000),      # DEL - Indira Gandhi Int'l
    "New Delhi": (28.5562, 77.1000),  # DEL
    "Banglore": (13.1986, 77.7066),   # BLR - Kempegowda Int'l
    "Mumbai": (19.0896, 72.8656),     # BOM - Chhatrapati Shivaji Int'l
    "Kolkata": (22.6547, 88.4467),    # CCU - Netaji Subhash Chandra Bose
    "Chennai": (12.9941, 80.1709),    # MAA - Chennai Int'l
    "Cochin": (10.1518, 76.4019),     # COK - Cochin Int'l
    "Hyderabad": (17.2403, 78.4294),  # HYD - Rajiv Gandhi Int'l
}

AIRPORT_CODES: Dict[str, str] = {
    "Delhi": "DEL",
    "New Delhi": "DEL",
    "Banglore": "BLR",
    "Mumbai": "BOM",
    "Kolkata": "CCU",
    "Chennai": "MAA",
    "Cochin": "COK",
    "Hyderabad": "HYD",
}

def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate the great-circle distance between two coordinates in kilometers."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    r = 6371.0  # Earth's radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(r * c, 2)

def parse_duration_to_mins(duration_str: str) -> int:
    """Parses duration string like '2h 50m', '5h', '30m' into total minutes."""
    if pd.isna(duration_str) or not isinstance(duration_str, str):
        return 0
    tokens = duration_str.strip().split()
    hours = 0
    mins = 0
    for token in tokens:
        if 'h' in token:
            try:
                hours = int(token.replace('h', ''))
            except ValueError:
                pass
        elif 'm' in token:
            try:
                mins = int(token.replace('m', ''))
            except ValueError:
                pass
    return hours * 60 + mins

def parse_stops(stops_val: Any) -> int:
    """Converts stops text/int to numeric count (0 for non-stop, 1 for 1 stop, etc)."""
    if pd.isna(stops_val):
        return 0
    if isinstance(stops_val, (int, float)):
        return int(stops_val)
    val_str = str(stops_val).strip().lower()
    if 'non-stop' in val_str or 'non stop' in val_str or val_str == '0':
        return 0
    elif '1 stop' in val_str:
        return 1
    elif '2 stop' in val_str:
        return 2
    elif '3 stop' in val_str:
        return 3
    elif '4 stop' in val_str:
        return 4
    try:
        return int(val_str[0])
    except Exception:
        return 0

class FlightDataPreprocessor:
    """
    End-to-end Feature Preprocessor for Flight Price Modeling.
    Extracts cyclical temporal features, geospatial geodesic distances,
    interaction features, and one-hot encoding matrices.
    """

    def __init__(self):
        self.feature_names: List[str] = []
        self.airline_list: List[str] = [
            "Air Asia", "Air India", "GoAir", "IndiGo", "Jet Airways",
            "Jet Airways Business", "Multiple carriers",
            "Multiple carriers Premium economy", "SpiceJet", "Trujet",
            "Vistara", "Vistara Premium economy"
        ]
        self.source_list: List[str] = ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai"]
        self.dest_list: List[str] = ["Banglore", "Cochin", "Delhi", "Hyderabad", "Kolkata", "New Delhi"]
        self.is_fitted: bool = False

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw flight dataframe into rich engineered feature matrix."""
        df_out = df.copy()

        # 1. Clean Destination (Map 'New Delhi' to 'Delhi' if standardizing, or keep standard)
        # We retain standard names
        
        # 2. Date of Journey decomposition
        if 'Date_of_Journey' in df_out.columns:
            journey_dt = pd.to_datetime(df_out['Date_of_Journey'], format='%d/%m/%Y', errors='coerce')
            df_out['Journey_Day'] = journey_dt.dt.day.fillna(1).astype(int)
            df_out['Journey_Month'] = journey_dt.dt.month.fillna(3).astype(int)
            df_out['Day_of_Week'] = journey_dt.dt.dayofweek.fillna(0).astype(int)
            df_out['Is_Weekend'] = df_out['Day_of_Week'].isin([5, 6]).astype(int)
        
        # 3. Departure Time
        if 'Dep_Time' in df_out.columns:
            dep_dt = pd.to_datetime(df_out['Dep_Time'], errors='coerce')
            df_out['Dep_Hour'] = dep_dt.dt.hour.fillna(10).astype(int)
            df_out['Dep_Min'] = dep_dt.dt.minute.fillna(0).astype(int)
        
        # 4. Arrival Time
        if 'Arrival_Time' in df_out.columns:
            # Handle formats like '19:00' or '22 Mar 01:20'
            arr_time_clean = df_out['Arrival_Time'].astype(str).str.split().str[-1]
            arr_dt = pd.to_datetime(arr_time_clean, format='%H:%M', errors='coerce')
            df_out['Arrival_Hour'] = arr_dt.dt.hour.fillna(12).astype(int)
            df_out['Arrival_Min'] = arr_dt.dt.minute.fillna(0).astype(int)

        # 5. Duration parsing
        if 'Duration' in df_out.columns:
            df_out['Duration_Total_Mins'] = df_out['Duration'].apply(parse_duration_to_mins)
            df_out['Duration_Hours'] = df_out['Duration_Total_Mins'] // 60
            df_out['Duration_Mins'] = df_out['Duration_Total_Mins'] % 60
        elif 'Duration_Hours' in df_out.columns and 'Duration_Mins' in df_out.columns:
            df_out['Duration_Total_Mins'] = df_out['Duration_Hours'] * 60 + df_out['Duration_Mins']
        else:
            # Fallback calculate from dep and arr
            df_out['Duration_Total_Mins'] = (df_out.get('Arrival_Hour', 12) - df_out.get('Dep_Hour', 10)) * 60
            df_out['Duration_Total_Mins'] = df_out['Duration_Total_Mins'].apply(lambda m: m if m > 0 else m + 1440)
            df_out['Duration_Hours'] = df_out['Duration_Total_Mins'] // 60
            df_out['Duration_Mins'] = df_out['Duration_Total_Mins'] % 60

        # 6. Total Stops
        if 'Total_Stops' in df_out.columns:
            df_out['Total_Stops'] = df_out['Total_Stops'].apply(parse_stops)

        # 7. Cyclical Temporal Features (Sine & Cosine Transforms for Smooth Periodic Transitions)
        # Hours (0-23) -> period 24
        df_out['Dep_Hour_sin'] = np.sin(2 * np.pi * df_out['Dep_Hour'] / 24.0)
        df_out['Dep_Hour_cos'] = np.cos(2 * np.pi * df_out['Dep_Hour'] / 24.0)
        df_out['Arr_Hour_sin'] = np.sin(2 * np.pi * df_out['Arrival_Hour'] / 24.0)
        df_out['Arr_Hour_cos'] = np.cos(2 * np.pi * df_out['Arrival_Hour'] / 24.0)
        
        # Months (1-12) -> period 12
        month_val = df_out.get('Journey_Month', 3)
        df_out['Month_sin'] = np.sin(2 * np.pi * (month_val - 1) / 12.0)
        df_out['Month_cos'] = np.cos(2 * np.pi * (month_val - 1) / 12.0)
        
        # Day of Week (0-6) -> period 7
        dow_val = df_out.get('Day_of_Week', 0)
        df_out['DOW_sin'] = np.sin(2 * np.pi * dow_val / 7.0)
        df_out['DOW_cos'] = np.cos(2 * np.pi * dow_val / 7.0)

        # 8. Time of Day Categorization (Early Morning: 0-6, Morning: 6-12, Afternoon: 12-17, Evening: 17-21, Night: 21-24)
        df_out['Is_Morning_Flight'] = ((df_out['Dep_Hour'] >= 6) & (df_out['Dep_Hour'] < 12)).astype(int)
        df_out['Is_Evening_Flight'] = ((df_out['Dep_Hour'] >= 17) & (df_out['Dep_Hour'] < 22)).astype(int)
        df_out['Is_RedEye_Flight'] = ((df_out['Dep_Hour'] >= 22) | (df_out['Dep_Hour'] < 5)).astype(int)

        # 9. Geospatial Geodesic Distance (km) & Estimated Velocity (km/h)
        def get_route_dist(row):
            src = str(row.get('Source', 'Delhi')).strip()
            dst = str(row.get('Destination', 'Cochin')).strip()
            c1 = AIRPORT_COORDINATES.get(src, (28.5562, 77.1000))
            c2 = AIRPORT_COORDINATES.get(dst, (10.1518, 76.4019))
            return haversine_distance(c1, c2)

        df_out['Route_Distance_KM'] = df_out.apply(get_route_dist, axis=1)
        
        # Flight Speed Proxy (Distance / Total Hours, clipped to realistic bounds 100-900 km/h)
        total_hours = np.maximum(df_out['Duration_Total_Mins'] / 60.0, 0.5)
        df_out['Estimated_Speed_KMH'] = np.clip(df_out['Route_Distance_KM'] / total_hours, 80.0, 950.0)

        # 10. Interaction Features
        df_out['Stops_x_Duration'] = df_out['Total_Stops'] * df_out['Duration_Total_Mins']
        df_out['Distance_per_Stop'] = df_out['Route_Distance_KM'] / (df_out['Total_Stops'] + 1)

        # 11. One-Hot Categorical Encoding
        for airline in self.airline_list:
            col_name = f"Airline_{airline}"
            df_out[col_name] = (df_out.get('Airline', '') == airline).astype(int)

        for source in self.source_list:
            col_name = f"Source_{source}"
            df_out[col_name] = (df_out.get('Source', '') == source).astype(int)

        for dest in self.dest_list:
            col_name = f"Destination_{dest}"
            df_out[col_name] = (df_out.get('Destination', '') == dest).astype(int)

        # Build feature list if not set
        numeric_cols = [
            'Total_Stops', 'Journey_Day', 'Journey_Month', 'Day_of_Week', 'Is_Weekend',
            'Dep_Hour', 'Dep_Min', 'Arrival_Hour', 'Arrival_Min',
            'Duration_Hours', 'Duration_Mins', 'Duration_Total_Mins',
            'Dep_Hour_sin', 'Dep_Hour_cos', 'Arr_Hour_sin', 'Arr_Hour_cos',
            'Month_sin', 'Month_cos', 'DOW_sin', 'DOW_cos',
            'Is_Morning_Flight', 'Is_Evening_Flight', 'Is_RedEye_Flight',
            'Route_Distance_KM', 'Estimated_Speed_KMH',
            'Stops_x_Duration', 'Distance_per_Stop'
        ]
        
        ohe_cols = (
            [f"Airline_{a}" for a in self.airline_list] +
            [f"Source_{s}" for s in self.source_list] +
            [f"Destination_{d}" for d in self.dest_list]
        )

        all_cols = numeric_cols + ohe_cols
        if not self.is_fitted:
            self.feature_names = all_cols
            self.is_fitted = True

        return df_out[self.feature_names]

    def create_single_input(self, dep_time: str, arr_time: str, source: str,
                            destination: str, stops: int, airline: str) -> pd.DataFrame:
        """Converts user form inputs into exact feature matrix row."""
        dep_dt = pd.to_datetime(dep_time)
        arr_dt = pd.to_datetime(arr_time)

        # Compute duration in minutes (handle overnight flights)
        diff_mins = int((arr_dt - dep_dt).total_seconds() // 60)
        if diff_mins <= 0:
            diff_mins = diff_mins + 24 * 60  # Next day arrival

        dur_hours = diff_mins // 60
        dur_mins = diff_mins % 60
        duration_str = f"{dur_hours}h {dur_mins}m"
        date_str = dep_dt.strftime('%d/%m/%Y')
        dep_str = dep_dt.strftime('%H:%M')
        arr_str = arr_dt.strftime('%H:%M')

        input_df = pd.DataFrame([{
            'Airline': airline,
            'Date_of_Journey': date_str,
            'Source': source,
            'Destination': destination,
            'Dep_Time': dep_str,
            'Arrival_Time': arr_str,
            'Duration': duration_str,
            'Total_Stops': stops
        }])

        return self.extract_features(input_df)
