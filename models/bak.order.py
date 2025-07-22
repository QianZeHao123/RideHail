# -*- coding: utf-8 -*-
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Order:
    order_id: int
    datetime: str
    pickup_area: int
    dropoff_area: int
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    customer_price: float
    commissionPercent: float  # New column from order.csv

    def __post_init__(self):
        self.datetime = datetime.strptime(self.datetime, '%Y-%m-%d %H:%M:%S.%f')
        self.driver_commission = self.customer_price * (1 - self.commissionPercent) 
        self.platform_revenue = self.customer_price * self.commissionPercent  
        self.hour_of_day = self.datetime.hour
