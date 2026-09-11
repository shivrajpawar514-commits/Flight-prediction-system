# recommender.py
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------
# Utilities
# -------------------------
def minutes_between(start_iso, end_iso):
    s = pd.to_datetime(start_iso)
    e = pd.to_datetime(end_iso)
    return int((e - s).total_seconds() // 60)

# Normalize helper
def normalize_series(s, reverse=False):
    arr = s.astype(float).to_numpy()
    if arr.max() == arr.min():
        return np.ones_like(arr)
    if reverse:
        # lower is better -> invert
        return (arr.max() - arr) / (arr.max() - arr.min())
    else:
        return (arr - arr.min()) / (arr.max() - arr.min())

# -------------------------
# Weighted scoring method
# -------------------------
def weighted_score(df, weights=None):
    """
    df: DataFrame with columns price, duration_mins, on_time_pct, stops, departure_dt
    weights: dict with keys 'price','duration','reliability','stops','time_of_day'
    returns df with 'score_weighted' column (higher better)
    """
    if weights is None:
        weights = {'price':0.4, 'duration':0.2, 'reliability':0.2, 'stops':0.1, 'time_of_day':0.1}

    # Feature transforms
    df = df.copy()
    df['price_n'] = normalize_series(df['price'], reverse=True)          # lower price better
    df['duration_n'] = normalize_series(df['duration_mins'], reverse=True)
    df['reliability_n'] = normalize_series(df['on_time_pct'], reverse=False)
    df['stops_n'] = normalize_series(df['stops'], reverse=True)          # fewer stops better -> reverse
    # time_of_day preference: morning/night (user configurable). For default, prefer morning (6-11)
    def tod_score(dt):
        hr = pd.to_datetime(dt).hour
        # example: morning (6-11) => 1, day (12-17) => 0.7, evening (18-23)=>0.4, night(0-5)=>0.2
        if 6 <= hr <= 11:
            return 1.0
        if 12 <= hr <= 17:
            return 0.7
        if 18 <= hr <= 23:
            return 0.4
        return 0.2
    df['time_pref_n'] = df['departure_dt'].apply(tod_score)
    # combine
    df['score_weighted'] = (
        df['price_n'] * weights['price'] +
        df['duration_n'] * weights['duration'] +
        df['reliability_n'] * weights['reliability'] +
        df['stops_n'] * weights['stops'] +
        df['time_pref_n'] * weights['time_of_day']
    )
    # normalize into 0..1
    df['score_weighted'] = (df['score_weighted'] - df['score_weighted'].min()) / max(1e-9, (df['score_weighted'].max() - df['score_weighted'].min()))
    return df.sort_values('score_weighted', ascending=False)

# -------------------------
# Simple Fuzzy logic (handcrafted membership functions)
# -------------------------
def fuzzy_score(df, rules=None):
    """
    Returns df with 'score_fuzzy' (0-1).
    We implement simple triangular/trapezoid membership for price/duration/on_time/stops.
    """
    df = df.copy()
    price = df['price'].astype(float)
    dur = df['duration_mins'].astype(float)
    rel = df['on_time_pct'].astype(float)
    stops = df['stops'].astype(float)

    # membership functions (these cutoffs are heuristics; tune to your market)
    def good_price(x):
        # piecewise: price <= p25 -> 1, between p25-p75 -> linear, >p75 -> 0
        p25, p75 = np.percentile(price, [25,75])
        return np.clip((p75 - x) / max(1e-9, (p75 - p25)), 0, 1)

    def short_duration(x):
        d25, d75 = np.percentile(dur, [25,75])
        return np.clip((d75 - x) / max(1e-9, (d75 - d25)), 0, 1)

    def reliable(x):
        # on_time >= 85 => 1, 60-85 linear, <60 -> 0
        return np.clip((x - 60) / 25, 0, 1)

    def few_stops(x):
        # 0 stops => 1, 1=>0.6, 2=>0.2, >2=>0
        return np.where(x==0, 1.0, np.where(x==1, 0.6, np.where(x==2, 0.2, 0.0)))

    # vectorized
    price_m = good_price(price)
    dur_m = short_duration(dur)
    rel_m = reliable(rel)
    stops_m = few_stops(stops)

    # rules — example: prefer price & reliability most
    # final membership: weighted average of memberships
    df['score_fuzzy'] = (0.45 * price_m + 0.25 * rel_m + 0.2 * dur_m + 0.1 * stops_m)
    # normalize
    df['score_fuzzy'] = (df['score_fuzzy'] - df['score_fuzzy'].min()) / max(1e-9, (df['score_fuzzy'].max() - df['score_fuzzy'].min()))
    return df.sort_values('score_fuzzy', ascending=False)

# -------------------------
# TOPSIS method (multi-criteria decision making)
# -------------------------
def topsis(df, criteria_weights=None):
    """
    Implements a simple TOPSIS over criteria:
      - price (min) -> cost
      - duration_mins (min) -> cost
      - on_time_pct (max) -> benefit
      - stops (min) -> cost
    criteria_weights is dict mapping to weights that sum to 1.
    """
    df = df.copy()
    if criteria_weights is None:
        criteria_weights = {'price':0.4,'duration':0.25,'reliability':0.25,'stops':0.1}

    # build decision matrix
    M = df[['price','duration_mins','on_time_pct','stops']].astype(float).to_numpy()
    # normalize (vector normalization)
    norm = np.linalg.norm(M, axis=0)
    norm[norm==0] = 1.0
    R = M / norm
    # apply weights
    w = np.array([criteria_weights['price'], criteria_weights['duration'], criteria_weights['reliability'], criteria_weights['stops']])
    V = R * w
    # ideal best/worst: for price,duration,stops -> min is best (cost criteria)
    # for on_time_pct -> max is best
    ideal_best = np.array([V[:,0].min(), V[:,1].min(), V[:,2].max(), V[:,3].min()])
    ideal_worst = np.array([V[:,0].max(), V[:,1].max(), V[:,2].min(), V[:,3].max()])
    # distances
    dist_best = np.linalg.norm(V - ideal_best, axis=1)
    dist_worst = np.linalg.norm(V - ideal_worst, axis=1)
    score = dist_worst / (dist_best + dist_worst + 1e-9)
    df['score_topsis'] = score
    df['score_topsis'] = (df['score_topsis'] - df['score_topsis'].min()) / max(1e-9, (df['score_topsis'].max() - df['score_topsis'].min()))
    return df.sort_values('score_topsis', ascending=False)


# -------------------------
# Single API to compute combined recommendations
# -------------------------
def recommend(df, method='combined', user_weights=None):
    """
    method: 'weighted', 'fuzzy', 'topsis', 'combined'
    combined -> ensemble of normalized scores (average)
    user_weights -> weights for weighted scoring if method == 'weighted'
    """
    outputs = {}
    if method in ['weighted','combined']:
        wdf = weighted_score(df, weights=user_weights)
        outputs['weighted'] = wdf
    if method in ['fuzzy','combined']:
        fdf = fuzzy_score(df)
        outputs['fuzzy'] = fdf
    if method in ['topsis','combined']:
        tdf = topsis(df)
        outputs['topsis'] = tdf

    if method == 'combined':
        # Align indices and compute ensemble score
        base = df.copy().reset_index(drop=True)
        base = base.astype(object)
        # merge normalized scores by flight id (we expect index alignment)
        base['s_weighted'] = outputs['weighted']['score_weighted'].values if 'weighted' in outputs else 0
        base['s_fuzzy'] = outputs['fuzzy']['score_fuzzy'].values if 'fuzzy' in outputs else 0
        base['s_topsis'] = outputs['topsis']['score_topsis'].values if 'topsis' in outputs else 0
        base['score_ensemble'] = (base['s_weighted'] + base['s_fuzzy'] + base['s_topsis']) / ( (1 if 'weighted' in outputs else 0) + (1 if 'fuzzy' in outputs else 0) + (1 if 'topsis' in outputs else 0))
        base['score_ensemble'] = (base['score_ensemble'] - base['score_ensemble'].min()) / max(1e-9, (base['score_ensemble'].max() - base['score_ensemble'].min()))
        return base.sort_values('score_ensemble', ascending=False)

    # otherwise return the df corresponding to method
    return outputs[method]
