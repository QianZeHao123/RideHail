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
import pdb
from collections import defaultdict

class DeliveryEnv(gym.Env):
    def __init__(self, orders, driver_data, schedule_data, weather_service):
        super(DeliveryEnv, self).__init__()
        self.simulator = DeliverySimulator(orders, driver_data, schedule_data, weather_service)

        # Define action space (continuous commission rate between 0 and 1)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define state space (order + specific driver attributes)
        self.observation_space = spaces.Dict({
            'customer_price': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32), # normalize
            'pickup_area': spaces.Discrete(62),
            'dropoff_area': spaces.Discrete(62),
            'hour_of_day': spaces.Discrete(24),
            'day_of_week': spaces.Discrete(7),
            'weather': spaces.Discrete(4),
            'driver_area': spaces.Discrete(62),
            'working_status': spaces.Discrete(2)
        })
        
        #self.observation_space = spaces.Box(
        #    low=0.0, 
        #    high=1.0, 
        #    shape=(224,),  # 1 + 62 + 62 + 24 + 7 + 4 + 62 + 2 = 224
        #    dtype=np.float32
        #)
        
        self.orders_by_day = defaultdict(list)
        for order in self.simulator.orders:
            order_day = order.datetime.date()
            self.orders_by_day[order_day].append(order)

        # Tracking variables
        self.assigned_order = 0 # Tracks # of unassigned orders
        self.current_day_index = 0  # Tracks training epoch (day index)
        self.current_order_index = 0  # Tracks current order within the day
        self.current_driver_index = 0
        self.current_day = None  # Current date being trained
        self.updated_drivers = set()  # Track drivers who have accepted at least one order
        self.next_order = False
        self.episode_rewards = 0
        self.episode_steps = 0
        self.total_driver_commission = 0.0
        self.max_steps = 30000

    def reset(self):
        """Resets environment at the start of each operational day (8 AM)."""
        # if self.current_day_index >= len(self.orders_by_day):  # end of all training days
        
            #print("DEBUG -> Training complete: `reset()` returning None!")  # DEBUG
            #return None

        # Select orders for current day
        self.current_day = list(self.orders_by_day.keys())[self.current_day_index]
        self.current_orders = self.orders_by_day[self.current_day]
        
        # if self.current_order_index >= len(self.current_orders):
        #     print("DEBUG -> Training complete: `reset()` returning None!")  # DEBUG
        #     return None

        self.current_order_index = 0
        self.current_driver_index = 0
        self.updated_drivers.clear()
        
        self.episode_rewards = 0
        self.episode_steps = 0
        self.assigned_order = 0
        self.total_driver_commission = 0.0
        
        # Count order-driver pairs correctly
        self.current_day_order_driver_pairs = sum(len(self._get_driver_pool(order)) for order in self.current_orders)
        print(f"DEBUG: Resetting for day {self.current_day}, Orders: {len(self.current_orders)}, Order-Driver Pairs: {self.current_day_order_driver_pairs}")
        
        if len(self.current_orders) > 0:
            self.current_order_driver_pool = self._get_driver_pool(self.current_orders[self.current_order_index])
            # print('DEBUG: # of current order - driver pairs ', len(self.current_order_driver_pool))
        else:
            self.current_order_driver_pool = [] 
            print('EBUG: no order at all!')
        
        obs = self._get_observation()
        
        return obs

    def step(self, action):
        """Processes one order-driver pair, ensuring driver availability updates for the next order."""
        self.next_order = False
        order = self.current_orders[self.current_order_index]
        
        driver = self.current_order_driver_pool[self.current_driver_index]


        # Offer commission rate
        order.commissionPercent = np.clip(action[0], 0.0, 1.0)
        order.driver_commission = order.customer_price * (1 - order.commissionPercent)
        self.total_driver_commission += order.driver_commission
        weather_code = self.simulator.weather.get_weather_code(order.datetime)

        accepted = driver.decide_acceptance(order, weather_code)
        # pdb.set_trace()

        # Update working status for next order no matter what
        driver.available = self._is_driver_working(driver, self._get_next_order_time())

        if accepted:
            self.assigned_order += 1
            reward = order.customer_price * order.commissionPercent
            
            # Track old area before moving
            old_area = driver.current_area

            # Track movement and availability updates
            driver.update_location(order.dropoff_lat, order.dropoff_lon, order.dropoff_area)
            self.updated_drivers.add(driver.driver_id)
            
            # Update area_drivers mapping
            if driver in self.simulator.area_drivers[old_area]:
                self.simulator.area_drivers[old_area].remove(driver)
            self.simulator.area_drivers[driver.current_area].append(driver)

            # Move to next order
            self.current_order_index += 1
            self.next_order = True
            self.current_driver_index = 0  

        else:
            # Move to next driver for the same order
            reward = 0
            self.current_driver_index += 1
            if self.current_driver_index >= len(self.current_order_driver_pool):
                self.current_order_index += 1
                self.next_order = True  
                # self.unassigned_order += 1
                self.current_driver_index = 0
                
        self.episode_steps += 1
        self.episode_rewards += reward

        done = self._is_done(order.datetime)
        
        
        info = {}
        
        if done:
            info['episode'] = {
                'r': self.episode_rewards,  # Report FINAL totals
                'l': self.episode_steps,
                'a': self.assigned_order,
                'o': len(self.current_orders),
                'c': self.total_driver_commission
            }
            
            
        obs = self._get_observation()

        return obs, reward, done, info

    def _get_driver_pool(self, order):
        """Retrieve the pool of drivers (historical + internal updates) while filtering out moved drivers."""
        combined_attempts = list(self.simulator.driver_attempts.get(order.order_id, []))
        valid_drivers = []

        # Include dynamic drivers from updated locations
        for driver in self.simulator.area_drivers.get(order.pickup_area, []):
            if driver.driver_id not in [a[0] for a in combined_attempts]:  # Avoid duplicates
                if order.datetime.hour in self.simulator.driver_schedule.get((driver.driver_id, order.datetime.date()), set()):
                    combined_attempts.append(driver)

        # Filter drivers to ensure they are actually in the pickup area
        for attempt in combined_attempts:
            driver_id, datetime, lat, lon, area, work_time_minutes = attempt
            driver = self.simulator.drivers_by_id.get(driver_id)

            if not driver:
                continue

            # If the driver has moved, **skip them** unless they are actually in the correct pickup area
            if driver_id in self.simulator.drivers_with_updated_location:
                if driver.current_area != order.pickup_area:
                    continue  # Driver moved to another area, so exclude them
                # Otherwise, use updated location
                driver.current_lat, driver.current_lon, driver.current_area = lat, lon, area  # Update location

            valid_drivers.append(driver)
            
        if len(valid_drivers) == 0:
            print(f"WARNING: No available drivers for Order {order.order_id} in pickup area {order.pickup_area}!")
            
        # print(f"Order {order.order_id}: Driver pool size = {len(valid_drivers)}")

        return valid_drivers

    def _get_observation(self):
        """Extracts order & driver state for RL input."""
        if self.current_order_index >= len(self.current_orders):
            return None
            # return np.zeros(self.observation_space.shape, dtype=np.float32) # return dummy observations

        order = self.current_orders[self.current_order_index]
        normalized_state = self.simulator._normalize_state(order)
        
        if self.next_order:
            self.current_order_driver_pool = self._get_driver_pool(order)
            self.current_driver_index = 0

        driver = self.current_order_driver_pool[self.current_driver_index]

        obs_dict = {
            'customer_price': np.array([normalized_state['customer_price']], dtype=np.float32),
            'pickup_area': normalized_state['pickup_area'],
            'dropoff_area': normalized_state['dropoff_area'],
            'hour_of_day': normalized_state['hour_of_day'],
            'day_of_week': order.datetime.weekday(),
            'weather': normalized_state['weather_code'],
            #'driver_id': driver.driver_id,
            'driver_area': driver.current_area,
            'working_status': 1 if driver.available else 0
        }
        
        return obs_dict
        
        #flattened = [
        #obs_dict['customer_price'][0],  # Already normalized to [0,1]
        #*np.eye(62)[obs_dict['pickup_area']],        # 62-dim one-hot
        #*np.eye(62)[obs_dict['dropoff_area']],       # 62-dim
        #*np.eye(24)[obs_dict['hour_of_day']],        # 24-dim
        #*np.eye(7)[obs_dict['day_of_week']],         # 7-dim
        #*np.eye(4)[obs_dict['weather']],             # 4-dim
        #*np.eye(62)[obs_dict['driver_area']],        # 62-dim
        #*np.eye(2)[obs_dict['working_status']]       # 2-dim
    #]
        
        # print('flattened obs ', flattened)
        # return np.array(flattened, dtype=np.float32).reshape(-1)


    def _is_done(self, current_datetime):
        """Terminates an episode at the end of the day."""
        if self.current_order_index >= len(self.current_orders):
            #self.current_day_index += 1  # Move to next day
            #if self.current_day_index >= len(self.orders_by_day):  # End after all training days
            #    print("All training days completed! Start over!")  
                
            #    return True
            
            print("Current training day completed!")  
            print("# of assigned orders ", self.assigned_order)
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