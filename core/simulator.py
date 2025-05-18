# -*- coding: utf-8 -*-

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib  # Added Joblib import to load the model
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from models import Order, Driver

class DeliverySimulator:
    def __init__(self, orders, driver_data, schedule_data, weather_service):
        self.orders = sorted(orders, key=lambda o: o.datetime)
        self.weather = weather_service
        self.driver_schedule = self._load_driver_schedule(schedule_data)
        self.driver_attempts = self._load_driver_attempts(driver_data)
        self.drivers_by_id = {}  # Cache all drivers by ID
        self.area_drivers = self._group_drivers_by_area()
        self.drivers_with_updated_location = set()  # Track drivers with updated location
        self.metrics = {
            'assignments': [],
            'unassigned_orders': [],
            'rejected_orders': [],
            'total_distance_km': 0.0,
            'platform_revenue': 0.0,
            'driver_earnings': defaultdict(float)
        }

        # Load model once globally instead of within each driver instance
        if not hasattr(DeliverySimulator, "shared_model"):
            DeliverySimulator.shared_model = joblib.load("models/acceptance_model_ensemble.pkl")
    
    
    # normalize states to make RL training more stable        
    def _normalize_state(self, order):
        """Normalize key order attributes for RL training."""
        return {
            "pickup_area": int(order.pickup_area),  
            "dropoff_area": int(order.dropoff_area),  
            "hour_of_day": int(order.hour_of_day),
            "weather_code": int(self.weather.get_weather_code(order.datetime)), 
            "customer_price": order.customer_price / 10100000.0,  #  Normalized by max price threshold
            "commissionPercent": order.commissionPercent / 100.0  # Already between 0-1 (no changes needed)
        }

    def _load_driver_schedule(self, schedule_data):
        """Loads driver work schedules from a CSV file into a dictionary."""
        schedule = defaultdict(set)
        for _, row in schedule_data.iterrows():
            driver_id = row['driver_id']
            date = row['date']
            hour = row['hour']
            schedule[(driver_id, date)].add(hour)
        return schedule

    def _load_driver_attempts(self, driver_data):
        """Loads driver assignment attempts, tracking all instances a driver receives an order."""
        attempts = defaultdict(list)
        for _, row in driver_data.iterrows():
            order_id = row['order_id']
            driver_id = row['driver_id']
            datetime = row['datetime']
            lat, lon, area = row['driver_lat'], row['driver_lon'], row['driver_area']
            work_time_minutes = row['work_time_minutes']
            attempts[order_id].append((driver_id, datetime, lat, lon, area, work_time_minutes))
        return attempts

    def _group_drivers_by_area(self):
        """Groups drivers by their current area for efficient order assignment."""
        area_drivers = defaultdict(list)
        for order_id, driver_attempts in self.driver_attempts.items():
            for driver_id, datetime, lat, lon, area, work_time_minutes in driver_attempts:
                if driver_id not in self.drivers_by_id:
                    driver = Driver(driver_id=driver_id, current_lat=lat, current_lon=lon, current_area=area, work_time_minutes=work_time_minutes)
                    driver.model = DeliverySimulator.shared_model
                    self.drivers_by_id[driver_id] = driver
                    area_drivers[area].append(driver)
        return area_drivers

    def _is_driver_available_by_time(self, driver, order):
        """Checks if a driver is working at the given order's datetime."""
        date = order.datetime.strftime('%Y-%m-%d')
        hour = order.datetime.hour
        available = hour in self.driver_schedule.get((driver.driver_id, date), set())
        return available

    def _assign_order(self, order):
        """Assigns an order to the best available driver (one at a time)."""
        attempts = self.driver_attempts.get(order.order_id, [])
        weather_code = self.weather.get_weather_code(order.datetime)
        pickup_area = order.pickup_area
        seen_drivers = set()

        # Collect drivers from driver_data (original CSV)
        combined_attempts = list(attempts)

        # Add dynamic drivers currently in the pickup area (who've accepted earlier orders)
        for driver in self.area_drivers.get(pickup_area, []):
            if driver.driver_id not in [a[0] for a in attempts]:
                # Insert a synthetic "attempt" with current known location
                combined_attempts.append((
                    driver.driver_id,
                    order.datetime,
                    driver.current_lat,
                    driver.current_lon,
                    driver.current_area,
                    driver.work_time_minutes
                ))

        for driver_id, datetime, lat, lon, area, work_time_minutes in combined_attempts:
            if driver_id in seen_drivers:
                continue
            seen_drivers.add(driver_id)

            driver = self.drivers_by_id.get(driver_id)
            if not driver:
                continue

            # Skip if not scheduled to work at order time
            if not self._is_driver_available_by_time(driver, order):
                continue

            # If driver has accepted an earlier order, verify their internal current location
            if driver_id in self.drivers_with_updated_location:
                if driver.current_area != pickup_area:
                    continue  # Driver has moved to another area, skip
            else:
                # First-time assignment, update driver's location from driver_data
                driver.current_lat = lat
                driver.current_lon = lon
                driver.current_area = area

            # Acceptance probability check
            prob = driver.calculate_accept_prob(order, weather_code)
            if np.random.random() < prob:
                # Mark driver as updated after accepting
                self.drivers_with_updated_location.add(driver_id)

                distance = driver.distance_to(order.dropoff_lat, order.dropoff_lon)
                normalized_state = self._normalize_state(order)

                #order.driver_commission = normalized_state['customer_price'] * (1 - normalized_state['commissionPercent'])
                #order.platform_revenue = normalized_state['customer_price'] * normalized_state['commissionPercent']

                order.driver_commission = order.customer_price * (1 - normalized_state['commissionPercent'])
                order.platform_revenue = order.customer_price * normalized_state['commissionPercent']

                self._record_assignment(order, driver, distance, normalized_state)

                # Update driver's current location
                driver.update_location(order.dropoff_lat, order.dropoff_lon, order.dropoff_area)

                # Move driver to new area
                if driver in self.area_drivers.get(pickup_area, []):
                    self.area_drivers[pickup_area].remove(driver)
                self.area_drivers[order.dropoff_area].append(driver)

                return True
            else:
                self.metrics['rejected_orders'].append({
                    'order_id': order.order_id,
                    'driver_id': driver_id
                })

        # If no driver accepts
        self.metrics['unassigned_orders'].append(order)
        return False

    def _record_assignment(self, order, driver, distance, normalized_state):
        """Records assignment details, including revenue and travel distance."""
        self.metrics['assignments'].append({
            'order_id': order.order_id,
            'driver_id': driver.driver_id,
            'distance_km': distance,
            'pickup_area': normalized_state["pickup_area"],  # Stored normalized state
            'dropoff_area': normalized_state["dropoff_area"],
            'hour_of_day': normalized_state["hour_of_day"],
            'weather_code': normalized_state["weather_code"],
            'commission': order.driver_commission,  
            'revenue': order.platform_revenue
        })
        self.metrics['total_distance_km'] += distance
        self.metrics['platform_revenue'] += order.platform_revenue
        self.metrics['driver_earnings'][driver.driver_id] += order.driver_commission

    def run(self, adjust_commission=None):
        """Process orders & allow RL model to adjust commission dynamically."""
        for order in self.orders:
            if adjust_commission:
                order.commissionPercent = adjust_commission(order)  # **Allow RL to modify commission rate**
            self._assign_order(order)
