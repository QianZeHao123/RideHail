# -*- coding: utf-8 -*-
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import math
import joblib
from pathlib import Path


@dataclass
class Driver:
    driver_id: int
    current_lat: float
    current_lon: float
    current_area: int
    work_time_minutes: float
    available: bool = True
    accepted_order: bool = False
    # model_path: str = str(Path(__file__).parent / 'acceptance_model_ensemble.pkl')
    model = None

    # def __post_init__(self):
    #     self.model = joblib.load(self.model_path)

    def distance_to(self, lat: float, lon: float) -> float:
        """Calculate Euclidean distance in kilometers (approx)."""
        return (
            math.sqrt((self.current_lat - lat) ** 2 + (self.current_lon - lon) ** 2)
            * 111
        ) * 1000

    def calculate_accept_prob(self, order, weather_code: int) -> float:
        """Predict acceptance probability using logistic regression."""
        if self.model is None:
            raise ValueError(
                "Driver model not initialized! Must be set in DeliverySimulator."
            )

        features = {
            "commission": [order.driver_commission],
            "driver_distance": [self.distance_to(order.pickup_lat, order.pickup_lon)],
            "hour": [order.hour_of_day],
            "weather_code": [weather_code],
            "work_time_minutes": [self.work_time_minutes],
        }

        # print("\nDEBUG - Acceptance Decision Features:")
        # for key, val in features.items():
        #    print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

        return self.model.predict_proba(pd.DataFrame(features))[0][1]

    def decide_acceptance(self, order, weather_code: int) -> bool:
        """Make acceptance decision based on probability."""
        if not self.available:
            return False

        random_value = np.random.random()
        prob = self.calculate_accept_prob(order, weather_code)
        # print(f"Random Value: {random_value:.2f}, Acceptance Probability: {prob:.2f}")
        # return random_value < prob

        accepted = random_value < prob
        if accepted:
            self.accepted_order = True 
        return accepted

    def update_location(self, lat: float, lon: float, new_area: int):
        """Update location only if the driver has taken an order."""
        if self.accepted_order:
            self.current_lat = lat
            self.current_lon = lon
            self.current_area = new_area
