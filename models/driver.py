import pandas as pd
import numpy as np
from .order import Order
from geopy.distance import geodesic


class Driver:
    def __init__(
        self,
        driver_id: int,
        current_lat: float,
        current_lon: float,
        current_area: int,
        work_time_minutes: float,
        available: bool = True,
        accepted_order: bool = False,
        # Model still needs to be set externally
        model=None
    ):
        """
        Initializes a Driver object.

        Args:
            driver_id (int): Unique identifier for the driver.
            current_lat (float): Current latitude coordinate of the driver's location.
            current_lon (float): Current longitude coordinate of the driver's location.
            current_area (int): Identifier for the driver's current geographical area.
            work_time_minutes (float): Total minutes the driver has worked.
            available (bool, optional): True if the driver is available for new orders, False otherwise. Defaults to True.
            accepted_order (bool, optional): True if the driver has accepted an order and is en route, False otherwise. Defaults to False.
        """
        self.driver_id = driver_id
        self.current_lat = current_lat
        self.current_lon = current_lon
        self.current_area = current_area
        self.work_time_minutes = work_time_minutes
        self.available = available
        self.accepted_order = accepted_order
        # Model still needs to be set externally
        self.model = model

        print(
            f"Driver {self.driver_id} is initialized with location ({self.current_lat}, {self.current_lon})"
        )

    def distance_to(self, order: Order) -> float:
        """Calculate Euclidean distance in kilometers (approx)."""
        # original_distance = (
        #     math.sqrt(
        #         (self.current_lat - order.pickup_lat) ** 2
        #         + (self.current_lon - order.pickup_lon) ** 2
        #     )
        #     * 111
        # ) * 1000

        # print(f"The distance calculated by traditional method is {original_distance}")
        """Geodesic Distance (Calculate the geodesic distance in meters from the driver's current location to the pickup location of a given order.)"""
        point_current = (self.current_lat, self.current_lon)
        point_pickup = (order.pickup_lat, order.pickup_lon)
        distance = geodesic(point_current, point_pickup).m

        print(f"The distance calculated by geodesic is {distance}")

        return distance

    def calculate_accept_prob(self, order: Order,
                              #   weather_code: int,
                              ) -> float:
        """Predict acceptance probability using logistic regression."""
        if self.model is None:
            raise ValueError(
                "Driver model not initialized! Must be set in DeliverySimulator."
            )

        features = {
            "commission": [order.driver_commission],
            # "distance": [self.distance_to(order)],
            "driver_distance": [self.distance_to(order)],
            "hour": [order.hour_of_day],
            # "weather_code": [weather_code],
            "weather_code": [order.weather_code],
            "work_time_minutes": [self.work_time_minutes],
        }
        print("Features input to the model for prediction:")
        print(features)
        return self.model.predict_proba(pd.DataFrame(features))[0][1]

    def decide_acceptance(
        self,
        order: Order,
        # weather_code: int,
        schedule_data: pd.DataFrame,
        threshold: float = np.random.random(),
    ) -> bool:
        """Make acceptance decision based on probability."""

        # check if the driver is avaliable:
        order_data_ymd = order.datetime.date()
        order_hour = order.hour_of_day
        driver_schedule_data: pd.DataFrame = schedule_data[
            (schedule_data['driver_id'] == self.driver_id) &
            (pd.to_datetime(schedule_data['date']).dt.date == order_data_ymd) &
            (schedule_data['hour'] == order_hour)
        ]

        if driver_schedule_data.empty:
            self.available = False

        if not self.available:
            print(
                f"Driver {self.driver_id} is not scheduled to work at {order_data_ymd} {order_hour:02d}:00.")
            return False
        else:
            print(
                f"Driver {self.driver_id} can work at {order_data_ymd} {order_hour:02d}:00.")

        # random_value = np.random.random()
        random_value = threshold
        # print(f"Random Value: {random_value:.2f}")
        # prob = self.calculate_accept_prob(order, weather_code)
        prob = self.calculate_accept_prob(order)

        accepted = bool(random_value < prob)
        if accepted:
            print(
                f"Driver {self.driver_id} accept the order with probability of {prob} and threshold {threshold}"
            )
            self.accepted_order = True
        else:
            print(
                f"Driver {self.driver_id} did not accept the order with probability of {prob} and threshold {threshold}"
            )
        return accepted

    # def update_location(self, order: Order):
    #     """Update location only if the driver has taken an order."""
    #     if self.accepted_order:
    #         self.current_lat = order.dropoff_lat
    #         self.current_lon = order.dropoff_lon
    #         self.current_area = order.dropoff_area
    #         print(
    #             f"Driver {self.driver_id} location moves to ({self.current_lat}, {self.current_lon})"
    #         )
    #     else:
    #         print(f"Driver {self.driver_id} keeps the same location")
