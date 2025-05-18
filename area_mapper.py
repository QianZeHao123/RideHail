# -*- coding: utf-8 -*-
import pandas as pd
from geopy.distance import great_circle

class AreaMapper:
    def __init__(self, mapping_csv_path):
        self.df = pd.read_csv(mapping_csv_path)
    
    def get_area(self, lat: float, lon: float) -> int:
        """Find nearest area code for given coordinates"""
        distances = self.df.apply(
            lambda row: great_circle((lat, lon), (row['lat1'], row['lon1'])).km,
            axis=1
        )
        return self.df.iloc[distances.idxmin()]['area_code']

