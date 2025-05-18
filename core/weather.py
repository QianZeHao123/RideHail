# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime

class WeatherService:
    def __init__(self, weather_csv_path: str):
        df = pd.read_csv(weather_csv_path)
        self.weather_data = {}
        
        for _, row in df.iterrows():
            dt = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S.%f')
            hour_key = dt.replace(minute=0, second=0)
            self.weather_data[hour_key] = row['weather_code']
    
    def get_weather_code(self, dt) -> int:
        """Get weather code for the hour containing datetime dt"""
        hour_key = dt.replace(minute=0, second=0)
        return self.weather_data.get(hour_key, 1)  # Default: 1 (sunny)

