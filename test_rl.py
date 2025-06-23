import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from delivery_env import DeliveryEnv  # Import the current version of the environment
from models import Order, Driver
from core import DeliverySimulator, WeatherService
import os
import joblib
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/testing.log"),
        logging.StreamHandler()
    ]
)

class TestLogger:
    def __init__(self):
        self.test_results = []
        
    def log_step(self, order_info, driver_id, action, reward):
        """Record all relevant information for each step"""
        self.test_results.append({
            'order_id': order_info.get('order_id', ''),
            'datetime': order_info.get('datetime', ''),
            'pickup_area': order_info.get('pickup_area', ''),
            'dropoff_area': order_info.get('dropoff_area', ''),
            'customer_price': order_info.get('customer_price', 0),
            'driver_id': driver_id,
            'action': np.clip(action, 0.0, 1.0)[0],  # Clip and extract scalar
            'reward': reward
        })
    
    def save_results(self):
        """Save results to CSV with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(self.test_results)
        
        # Reorder columns for better readability
        results_df = results_df[[
            'order_id', 'datetime', 'driver_id',
            'pickup_area', 'dropoff_area', 
            'customer_price', 'action', 'reward'
        ]]
        
        csv_path = f"test_results/test_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        logging.info(f"Saved detailed test results to {csv_path}")

def load_test_data():
    """Load and preprocess test data with additional order features"""
    orders_df = pd.read_csv('data/order.csv')
    driver_data = pd.read_csv('data/driver_update.csv')
    schedule_data = pd.read_csv('data/driver_schedule.csv')
    
    # Filter orders (different date range than training)
    orders_df = orders_df.loc[
        (orders_df['outside'] == 0) & 
        (orders_df['pickup_area'].notnull())
    ]
    orders_df = orders_df[
        (pd.to_datetime(orders_df['datetime']).dt.hour >= 8) & 
        (pd.to_datetime(orders_df['datetime']).dt.hour < 24)
    ]
    orders_df['date'] = pd.to_datetime(orders_df['date']).dt.date
    orders_df = orders_df[orders_df['date'] > pd.to_datetime("2025-04-16").date()]
    
    orders_df = orders_df[['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    
    orders = [Order(**row) for _, row in orders_df.iterrows()]
    return orders, driver_data, schedule_data

def run_testing(model_path="models/ppo_delivery_latest"):
    """Evaluate model and save detailed results"""
    try:
        # Load pre-trained model
        logging.info(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        
        # Initialize test environment
        test_orders, driver_data, schedule_data = load_test_data()
        test_env = DeliveryEnv(
            test_orders,
            driver_data,
            schedule_data,
            WeatherService('data/weather.csv')
        )
        
        # Initialize logger
        test_logger = TestLogger()
        logging.info("Starting testing with detailed logging...")
        
        # Run evaluation
        obs = test_env.reset()
        episode_count = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            # Extract order and driver info from the environment
            current_order = getattr(test_env, 'current_order', None)
            current_driver = getattr(test_env, 'current_driver', None)
            
            if current_order and current_driver:
                test_logger.log_step(
                    order_info={
                        'order_id': current_order.order_id,
                        'datetime': getattr(current_order, 'datetime', ''),
                        'pickup_area': current_order.pickup_area,
                        'dropoff_area': current_order.dropoff_area,
                        'customer_price': current_order.customer_price
                    },
                    driver_id=current_driver.driver_id,
                    action=action,
                    reward=reward
                )
            
            if done:
                episode_count += 1
                obs = test_env.reset()
                
                # Stop condition
                if episode_count >= 100 or not test_orders:
                    break
        
        test_logger.save_results()
        logging.info("Testing completed successfully!")
        
    except Exception as e:
        logging.error(f"Testing failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/ppo_delivery_latest"
    run_testing(model_path)