# -*- coding: utf-8 -*-

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import Order, Driver
from core import DeliverySimulator, WeatherService
import gym
from gym import spaces
import numpy as np
import pandas as pd
from collections import defaultdict

class DeliveryEnv(gym.Env):
    def __init__(self, orders, driver_data, schedule_data, weather_service):
        super(DeliveryEnv, self).__init__()
        self.simulator = DeliverySimulator(orders, driver_data, schedule_data, weather_service)

        # Define action space (continuous commission rate between 0 and 1)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define state space (order + specific driver attributes)
        self.observation_space = spaces.Dict({
            'customer_price': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'pickup_area': spaces.Discrete(62),
            'dropoff_area': spaces.Discrete(62),
            'hour_of_day': spaces.Discrete(24),
            'day_of_week': spaces.Discrete(7),
            'weather': spaces.Discrete(4),
            'driver_area': spaces.Discrete(62),
            'working_status': spaces.Discrete(2)
        })
        
        self.orders_by_day = defaultdict(list)
        for order in self.simulator.orders:
            order_day = order.datetime.date()
            self.orders_by_day[order_day].append(order)

        # Tracking variables
        self.current_day_index = 0  # Tracks training epoch (day index)
        self.current_order_index = 0  # Tracks current order within the day
        self.current_day = None  # Current date being trained
        self.updated_drivers = set()  # Track drivers who have accepted at least one order

    def reset(self):
        """Resets environment at the start of each operational day (8 AM)."""
        if self.current_day_index >= len(self.orders_by_day):  # Stop after training all days
            print("DEBUG -> Training complete: `reset()` returning None!")  # DEBUG
            return None

        # Select orders for current day
        self.current_day = list(self.orders_by_day.keys())[self.current_day_index]
        self.current_orders = self.orders_by_day[self.current_day]

        self.current_order_index = 0
        self.updated_drivers.clear()
        
        # Count order-driver pairs correctly
        self.current_day_order_driver_pairs = sum(len(self._get_driver_pool(order)) for order in self.current_orders)
        print(f"DEBUG: Resetting for day {self.current_day}, Orders: {len(self.current_orders)}, Order-Driver Pairs: {self.current_day_order_driver_pairs}")
        
        # DEBUG
        obs = self._get_observation()
        if obs is None:
            print(f"Warning: `reset()` returned None unexpectedly on day {self.current_day_index}.")
        
        return obs

    def step(self, action):
        """Processes an order-driver interaction with a given commission rate."""
        commission_rate = action[0]
        
        # Check if we have run out of orders for the day
        if self.current_order_index >= len(self.current_orders):
            print(f"DEBUG: Out of orders for day {self.current_day}, triggering `_is_done()`")
            return None, 0, self._is_done(self.current_day), {}
        
        order = self.current_orders[self.current_order_index]
        

        # Get dynamically updated driver pool (historical + internal updates)
        driver_pool = self._get_driver_pool(order)
        
        print(f"DEBUG: Processing Order {order.order_id}, Drivers Available: {len(driver_pool)}")

        for driver_id, datetime, lat, lon, area, work_time_minutes in driver_pool:
            driver = self.simulator.drivers_by_id.get(driver_id)
            if not driver:
                continue
            
            # Ensure drivers haven't been skipped
            print(f"DEBUG: Checking Driver {driver_id} for Order {order.order_id}")

            # Use CSV location only if driver hasn't accepted any order yet
            if driver_id not in self.updated_drivers:
                driver.current_lat = lat
                driver.current_lon = lon
                driver.current_area = area
                driver.work_time_minutes = work_time_minutes

            # Check if driver is working at the time of this order
            if not self._is_driver_working(driver, order.datetime):
                continue

            # Driver is working → proceed with offer
            order.commissionPercent = commission_rate
            weather_code = self.simulator.weather.get_weather_code(order.datetime)
            accepted = driver.decide_acceptance(order, weather_code)

            if accepted:
                revenue = order.customer_price * commission_rate
                reward = revenue

                # Track old area before moving
                old_area = driver.current_area

                # Update driver's location
                driver.update_location(order.dropoff_lat, order.dropoff_lon, order.dropoff_area)
                self.updated_drivers.add(driver_id)
                
                print(f"DEBUG: Driver {driver_id} accepted order {order.order_id}, moving to {order.dropoff_area}")

                # Update area_drivers mapping
                if driver in self.simulator.area_drivers[old_area]:
                    self.simulator.area_drivers[old_area].remove(driver)
                self.simulator.area_drivers[driver.current_area].append(driver)

                # Update working status for next order
                driver.available = self._is_driver_working(driver, self._get_next_order_time())

                self.current_order_index += 1
                return self._get_observation(), reward, self._is_done(order.datetime), {}

            else:
                # Rejected → update working status for next order
                driver.available = self._is_driver_working(driver, self._get_next_order_time())

        # No driver accepted → move to next order
        print(f"DEBUG: No driver accepted Order {order.order_id}, skipping to next order")
        self.current_order_index += 1
        return self._get_observation(), 0, self._is_done(order.datetime), {}

    def _get_driver_pool(self, order):
        """Retrieve the pool of drivers (historical + internal updates) while filtering out moved drivers."""
        combined_attempts = list(self.simulator.driver_attempts.get(order.order_id, []))
        valid_drivers = []

        # Include dynamic drivers from updated locations
        for driver in self.simulator.area_drivers.get(order.pickup_area, []):
            if driver.driver_id not in [a[0] for a in combined_attempts]:  # Avoid duplicates
                combined_attempts.append((
                    driver.driver_id, order.datetime, driver.current_lat, driver.current_lon, driver.current_area, driver.work_time_minutes
                ))

        # Filter drivers to ensure they are actually in the pickup area
        for driver_id, datetime, lat, lon, area, work_time_minutes in combined_attempts:
            driver = self.simulator.drivers_by_id.get(driver_id)

            if not driver:
                continue

            # If the driver has moved, **skip them** unless they are actually in the correct pickup area
            if driver_id in self.simulator.drivers_with_updated_location:
                if driver.current_area != order.pickup_area:
                    continue  # Driver moved to another area, so exclude them
                # Otherwise, use updated location
                lat, lon, area = driver.current_lat, driver.current_lon, driver.current_area

            valid_drivers.append((driver_id, datetime, lat, lon, area, work_time_minutes))
            
        print(f"Order {order.order_id}: Driver pool size = {len(valid_drivers)}")

        return valid_drivers

    def _get_observation(self):
        """Extracts order & driver state for RL input."""
        if self.current_order_index >= len(self.current_orders):
            print("DEBUG -> Warning: `_get_observation()` is returning None!") # DEBUG
            return None

        order = self.current_orders[self.current_order_index]
        normalized_state = self.simulator._normalize_state(order)

        # Use first driver in attempt list who is still in the pickup area
        driver_pool = self._get_driver_pool(order)
        driver = None
        for driver_id, _, _, _, _, _ in driver_pool:
            d = self.simulator.drivers_by_id.get(driver_id)
            if d and d.current_area == order.pickup_area:
                driver = d
                break

        if driver:
            driver_area = driver.current_area
            working_status = 1 if driver.available else 0
        else:
            driver_area = 0
            working_status = 0

        return {
            'customer_price': np.array([normalized_state['customer_price']], dtype=np.float32),
            'pickup_area': normalized_state['pickup_area'],
            'dropoff_area': normalized_state['dropoff_area'],
            'hour_of_day': normalized_state['hour_of_day'],
            'day_of_week': order.datetime.weekday(),
            'weather': normalized_state['weather_code'],
            'driver_area': int(driver_area),
            'working_status': int(working_status)
        }

    def _is_done(self, current_datetime):
        """Terminates an episode at the end of the day."""
        if self.current_order_index >= len(self.current_orders):
            self.current_day_index += 1  # Move to next day
            if self.current_day_index >= len(self.orders_by_day):  # End after 14 days
                print("All training epochs completed!")  # DEBUG
                return True
            
            print(f"End of day {self.current_day}, moving to next.")  # DEBUG
            return True  # End the current day and reset

        return False

    def _is_driver_working(self, driver, datetime):
        """Checks if a driver is scheduled to work at a given time."""
        date = datetime.date()
        hour = datetime.hour
        return hour in self.simulator.driver_schedule.get((driver.driver_id, date), set())
    
    def _get_next_order_time(self):
        """Retrieve the timestamp for the next order."""
        if self.current_order_index + 1 < len(self.current_orders):
            return self.current_orders[self.current_order_index + 1].datetime
        return self.current_orders[self.current_order_index].datetime

    def render(self, mode='human'):
        """Prints current order details for monitoring."""
        print(f"Processing Order {self.current_order_index} → Hour: {self.simulator.orders[self.current_order_index].datetime.hour}, Weather Code: {self.simulator.weather.get_weather_code(self.simulator.orders[self.current_order_index].datetime)}")
