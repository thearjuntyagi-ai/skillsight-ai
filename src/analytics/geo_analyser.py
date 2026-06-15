import pandas as pd

CITY_COORDINATES = {
    "Bengaluru":     {"lat": 12.9716, "lon": 77.5946,   "state": "Karnataka"},
    "Mumbai":        {"lat": 19.0760, "lon": 72.8777,   "state": "Maharashtra"},
    "Hyderabad":     {"lat": 17.3850, "lon": 78.4867,   "state": "Telangana"},
    "Delhi":         {"lat": 28.6139, "lon": 77.2090,   "state": "Delhi"},
    "Pune":          {"lat": 18.5204, "lon": 73.8567,   "state": "Maharashtra"},
    "Chennai":       {"lat": 13.0827, "lon": 80.2707,   "state": "Tamil Nadu"},
    "Gurugram":      {"lat": 28.4595, "lon": 77.0266,   "state": "Haryana"},
    "Noida":         {"lat": 28.5355, "lon": 77.3910,   "state": "Uttar Pradesh"},
    "Kolkata":       {"lat": 22.5726, "lon": 88.3639,   "state": "West Bengal"},
    "Ahmedabad":     {"lat": 23.0225, "lon": 72.5714,   "state": "Gujarat"},
    "New York":      {"lat": 40.7128, "lon": -74.0060,  "state": "New York"},
    "San Francisco": {"lat": 37.7749, "lon": -122.4194, "state": "California"},
    "Chicago":       {"lat": 41.8781, "lon": -87.6298,  "state": "Illinois"},
    "Austin":        {"lat": 30.2672, "lon": -97.7431,  "state": "Texas"},
    "Seattle":       {"lat": 47.6062, "lon": -122.3321, "state": "Washington"},
    "Boston":        {"lat": 42.3601, "lon": -71.0589,  "state": "Massachusetts"},
    "Los Angeles":   {"lat": 34.0522, "lon": -118.2437, "state": "California"},
    "Atlanta":       {"lat": 33.7490, "lon": -84.3880,  "state": "Georgia"},
    "Dallas":        {"lat": 32.7767, "lon": -96.7970,  "state": "Texas"},
    "Denver":        {"lat": 39.7392, "lon": -104.9903, "state": "Colorado"},
    "Washington":    {"lat": 38.9072, "lon": -77.0369,  "state": "DC"},
    "San Jose":      {"lat": 37.3382, "lon": -121.8863, "state": "California"},
    "San Diego":     {"lat": 32.7157, "lon": -117.1611, "state": "California"},
    "Minneapolis":   {"lat": 44.9778, "lon": -93.2650,  "state": "Minnesota"},
    "Phoenix":       {"lat": 33.4484, "lon": -112.0740, "state": "Arizona"},
}


class GeoAnalyser:

    def get_city_demand(self, df: pd.DataFrame,
                        skill_filter: str = None) -> pd.DataFrame:
        try:
            filtered = df.copy()

            if skill_filter and skill_filter != "All Skills":
                filtered = filtered[
                    filtered["skills"].apply(
                        lambda s: skill_filter in (s if isinstance(s, list) else [])
                    )
                ]

            if filtered.empty:
                return self._fallback_geo()

            # Get city counts from the city column
            all_cities = []
            for col in ["city", "location_raw"]:
                if col in filtered.columns:
                    vals = filtered[col].dropna().tolist()
                    all_cities.extend(vals)
                    break

            if not all_cities:
                return self._fallback_geo()

            # Count matches against known cities
            rows = []
            for city, coords in CITY_COORDINATES.items():
                count = sum(
                    1 for c in all_cities
                    if city.lower() in str(c).lower()
                )
                if count > 0:
                    rows.append({
                        "city":      city,
                        "job_count": count,
                        "lat":       coords["lat"],
                        "lon":       coords["lon"],
                        "state":     coords["state"],
                    })

            if not rows:
                return self._fallback_geo()

            result = pd.DataFrame(rows)
            result = result.sort_values("job_count", ascending=False)
            return result.reset_index(drop=True)

        except Exception as e:
            return self._fallback_geo()

    def get_skill_by_city(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            rows = []
            for city in list(CITY_COORDINATES.keys())[:10]:
                city_df = df[
                    df["city"].str.contains(city, case=False, na=False)
                ] if "city" in df.columns else pd.DataFrame()

                if len(city_df) == 0:
                    continue

                top_skill = city_df["skills"].explode().value_counts()
                if len(top_skill) > 0:
                    rows.append({
                        "city":      city,
                        "top_skill": top_skill.index[0],
                        "job_count": len(city_df),
                    })

            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _fallback_geo(self) -> pd.DataFrame:
        """Return sample geo data when real data can't be parsed."""
        data = [
            {"city": "San Francisco", "job_count": 245, "lat": 37.7749,
             "lon": -122.4194, "state": "California"},
            {"city": "New York",      "job_count": 312, "lat": 40.7128,
             "lon": -74.0060,  "state": "New York"},
            {"city": "Seattle",       "job_count": 178, "lat": 47.6062,
             "lon": -122.3321, "state": "Washington"},
            {"city": "Austin",        "job_count": 134, "lat": 30.2672,
             "lon": -97.7431,  "state": "Texas"},
            {"city": "Chicago",       "job_count": 156, "lat": 41.8781,
             "lon": -87.6298,  "state": "Illinois"},
            {"city": "Boston",        "job_count": 143, "lat": 42.3601,
             "lon": -71.0589,  "state": "Massachusetts"},
            {"city": "Los Angeles",   "job_count": 167, "lat": 34.0522,
             "lon": -118.2437, "state": "California"},
            {"city": "Denver",        "job_count": 98,  "lat": 39.7392,
             "lon": -104.9903, "state": "Colorado"},
        ]
        return pd.DataFrame(data)