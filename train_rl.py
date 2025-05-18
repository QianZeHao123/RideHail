# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # Set correct import path

import gym
import numpy as np
import logging
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from tqdm import tqdm
from delivery_env import DeliveryEnv  # Import the current version of the environment
from models import Order, Driver
from core import DeliverySimulator, WeatherService
import joblib
import csv

# Initialize logging
logging.basicConfig(filename="logs/ppo_training.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def load_data():
    """Load necessary data for the environment."""
    orders_df = pd.read_csv('data/order.csv')
    driver_data = pd.read_csv('data/driver_update.csv')
    schedule_data = pd.read_csv('data/driver_schedule.csv')
    
    orders_df = orders_df.loc[(orders_df['outside'] == 0) & (orders_df['pickup_area'].notnull())]
    orders_df = orders_df[(pd.to_datetime(orders_df['datetime']).dt.hour >= 8) & (pd.to_datetime(orders_df['datetime']).dt.hour < 24)]
    orders_df['date'] = pd.to_datetime(orders_df['date']).dt.date
    orders_df = orders_df[orders_df['date'] <= pd.to_datetime("2025-04-16").date()] # first 10 days as training data
    
    
    orders_df = orders_df[['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    
    orders = [Order(**row) for _, row in orders_df.iterrows()]
    return orders, driver_data, schedule_data

# Ensure global model is loaded before initializing DeliverySimulator
if not hasattr(DeliverySimulator, "shared_model"):
    logging.info("Loading shared_model globally in train_rl.py...")
    DeliverySimulator.shared_model = joblib.load("models/acceptance_model_ensemble.pkl")
    logging.info("shared_model successfully loaded!")

# Load data & initialize environment
orders, driver_data, schedule_data = load_data()
weather_service = WeatherService('data/weather.csv')
env = DeliveryEnv(orders, driver_data, schedule_data, weather_service)

# Initialize PPO model
model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=1e-4, batch_size=512, gamma=0.95)
log_dir = "logs/ppo/"
model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

# Training parameters
total_timesteps = 200000
num_iterations = 10

logging.info(f"Starting PPO training for {num_iterations} iterations...")

# Train PPO with proper episode resets
with tqdm(total=total_timesteps) as progress_bar:
    for i in range(num_iterations):
        obs = env.reset()  # ✅ Reset environment before each training iteration
        print('initial observation is ', obs)
        if obs is None:  # Stop training if all days have been processed
            logging.info("All training epochs completed!")
            break
        
        # print('number of order-driver pairs ', env.current_day_order_driver_pairs)
    
        # for _ in range(total_timesteps // num_iterations):
        # loss = []
        # results = []
        for _ in range(total_timesteps // num_iterations):
            action, _ = model.predict(obs, deterministic=True)
            print('action is ', action)
            obs, reward, done, _ = env.step(action)
            # print('rewward is ', reward)
            print('next observation is ', obs)

            # model.learn(env.current_day_order_driver_pairs, log_interval=10)
            model.learn(1)
            
            #loss_info = model.logger.get_log_dict()  # Get training logs
            #if "loss" in loss_info:
            #    loss.append(loss_info["loss"])  # Save loss over time
                
            # results.append([env.current_orders[env.current_order_index].order_id, action[0], reward])

            if done:
                print('is done? ', done)
                obs = env.reset()  # ✅ Automatically reset when reaching terminal state (end of day)
                print('observation after done ', obs)
                if obs is None:
                    break
                    

        progress_bar.update(total_timesteps // num_iterations)
        logging.info(f"Iteration {i}: Training update complete.")

# Save trained model
model.save("trained_model/ppo_commission_optimizer")
logging.info("PPO training complete!")

# Save simulation results to `logs/simulation_results.csv`
results_file = "logs/simulation_results.csv"
with open(results_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["order_id", "commission", "reward","loss"])  # Add headers
    
    for i, row in enumerate(results):
        loss_value = loss[i] if i < len(loss) else np.nan  # Handle cases where loss isn't available
        writer.writerow(row + [loss_value])  # ✅ Append loss to the row




