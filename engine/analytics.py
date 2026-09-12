"""
Market Intelligence, Route Analytics & Airport Connectivity Engine.
Aggregates route distances, airline pricing power indices, seasonal trends,
and geospatial airport network graph data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from engine.preprocessor import AIRPORT_COORDINATES, AIRPORT_CODES, haversine_distance, parse_duration_to_mins, parse_stops


class FlightAnalyticsEngine:
    """
    Analytics engine extracting deep route, airline, and geospatial market intelligence.
    """

    def __init__(self, raw_df: pd.DataFrame):
        self.df = raw_df.copy()
        if 'duration_mins' not in self.df.columns:
            self.df['duration_mins'] = self.df['Duration'].apply(parse_duration_to_mins)
        if 'stops_num' not in self.df.columns:
            self.df['stops_num'] = self.df['Total_Stops'].apply(parse_stops)

    def get_summary_kpis(self) -> Dict[str, Any]:
        """Calculates overarching executive KPI summary metrics."""
        prices = self.df['Price']
        total_flights = len(self.df)
        avg_price = int(prices.mean())
        median_price = int(prices.median())
        min_price = int(prices.min())
        max_price = int(prices.max())
        std_price = int(prices.std())
        
        airline_counts = self.df['Airline'].value_counts()
        most_popular_airline = airline_counts.index[0]
        top_airline_share = round((airline_counts.iloc[0] / total_flights) * 100, 1)

        # Unique routes
        routes = self.df[['Source', 'Destination']].drop_duplicates()
        total_routes = len(routes)

        return {
            "total_flights": total_flights,
            "avg_price": avg_price,
            "median_price": median_price,
            "min_price": min_price,
            "max_price": max_price,
            "std_price": std_price,
            "most_popular_airline": most_popular_airline,
            "top_airline_share": top_airline_share,
            "total_routes": total_routes
        }

    def get_route_intelligence(self) -> List[Dict[str, Any]]:
        """Calculates deep route-level analytics including geodesic distances and airline market shares."""
        route_list = []
        for (src, dst), group in self.df.groupby(['Source', 'Destination']):
            coord1 = AIRPORT_COORDINATES.get(src, (28.5562, 77.1000))
            coord2 = AIRPORT_COORDINATES.get(dst, (10.1518, 76.4019))
            dist_km = haversine_distance(coord1, coord2)
            
            prices = group['Price'].values
            durations = group['duration_mins'].values
            
            # Top airline on this route
            top_airline = group['Airline'].mode()[0]
            top_airline_count = (group['Airline'] == top_airline).sum()
            top_airline_share = round((top_airline_count / len(group)) * 100, 1)

            # Direct vs multi-stop ratio
            direct_pct = round(((group['stops_num'] == 0).sum() / len(group)) * 100, 1)

            route_list.append({
                "source": src,
                "source_code": AIRPORT_CODES.get(src, src[:3].upper()),
                "destination": dst,
                "dest_code": AIRPORT_CODES.get(dst, dst[:3].upper()),
                "route_label": f"{src} → {dst}",
                "distance_km": dist_km,
                "flight_count": len(group),
                "avg_price": int(np.mean(prices)),
                "min_price": int(np.min(prices)),
                "max_price": int(np.max(prices)),
                "median_price": int(np.median(prices)),
                "avg_duration_mins": int(np.mean(durations)),
                "avg_duration_formatted": f"{int(np.mean(durations)//60)}h {int(np.mean(durations)%60)}m",
                "direct_pct": direct_pct,
                "top_airline": top_airline,
                "top_airline_share": top_airline_share,
                "src_coords": list(coord1),
                "dst_coords": list(coord2)
            })

        # Sort routes by flight traffic volume descending
        route_list.sort(key=lambda r: r['flight_count'], reverse=True)
        return route_list

    def get_airline_analytics(self) -> List[Dict[str, Any]]:
        """Calculates market share, average fare, price volatility, and pricing power by airline."""
        total_flights = len(self.df)
        airline_list = []
        
        for airline, group in self.df.groupby('Airline'):
            if len(group) < 3:
                continue
            prices = group['Price'].values
            durations = group['duration_mins'].values
            
            airline_list.append({
                "airline": airline,
                "flight_count": len(group),
                "market_share_pct": round((len(group) / total_flights) * 100, 2),
                "avg_price": int(np.mean(prices)),
                "median_price": int(np.median(prices)),
                "min_price": int(np.min(prices)),
                "max_price": int(np.max(prices)),
                "std_price": int(np.std(prices)),
                "avg_duration_mins": int(np.mean(durations)),
                "direct_flight_pct": round(((group['stops_num'] == 0).sum() / len(group)) * 100, 1)
            })

        # Sort by market share descending
        airline_list.sort(key=lambda a: a['flight_count'], reverse=True)
        return airline_list

    def get_temporal_analytics(self) -> Dict[str, Any]:
        """Calculates day-of-week, hour-of-day, and month trends."""
        df_t = self.df.copy()
        
        # Parse dates
        dt_journey = pd.to_datetime(df_t['Date_of_Journey'], format='%d/%m/%Y', errors='coerce')
        df_t['dow'] = dt_journey.dt.day_name()
        df_t['month'] = dt_journey.dt.month_name()
        
        # Day of week price averages
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_summary = []
        for d in dow_order:
            subset = df_t[df_t['dow'] == d]
            if len(subset) > 0:
                dow_summary.append({
                    "day": d,
                    "avg_price": int(subset['Price'].mean()),
                    "flight_count": len(subset)
                })

        # Stops vs Average Price
        stops_summary = []
        for stop_val in sorted(df_t['stops_num'].unique()):
            sub = df_t[df_t['stops_num'] == stop_val]
            stops_summary.append({
                "stops": int(stop_val),
                "label": "Non-Stop" if stop_val == 0 else f"{int(stop_val)} Stop{'s' if stop_val>1 else ''}",
                "avg_price": int(sub['Price'].mean()),
                "median_price": int(sub['Price'].median()),
                "count": len(sub)
            })

        return {
            "day_of_week": dow_summary,
            "stops_breakdown": stops_summary
        }

    def get_airport_nodes(self) -> List[Dict[str, Any]]:
        """Returns geospatial airport nodes with traffic and location metadata."""
        nodes = []
        for name, coords in AIRPORT_COORDINATES.items():
            if name == "New Delhi":
                continue
            src_traffic = (self.df['Source'] == name).sum()
            dst_traffic = (self.df['Destination'] == name).sum()
            total_traffic = int(src_traffic + dst_traffic)

            nodes.append({
                "city": name,
                "code": AIRPORT_CODES.get(name, name[:3].upper()),
                "lat": coords[0],
                "lng": coords[1],
                "departures": int(src_traffic),
                "arrivals": int(dst_traffic),
                "total_traffic": total_traffic
            })

        return nodes
