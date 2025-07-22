import pandas as pd


class WeatherService:
    def __init__(self, weather_csv_path: str):
        df = pd.read_csv(weather_csv_path)
        # Convert 'datetime' column to datetime objects and normalize to the hour start
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")
        # Set 'datetime' as index and convert 'weather_code' to a dictionary
        self.weather_data = df.set_index("datetime")["weather_code"].to_dict()

    def get_weather_code(self, dt) -> int:
        """Get weather code for the hour containing datetime dt"""
        hour_key = dt.replace(minute=0, second=0)
        # Default: 1 (sunny)
        return self.weather_data.get(hour_key, 1)


# weather_service = WeatherService(weather_csv_path="./data/weather.csv")
