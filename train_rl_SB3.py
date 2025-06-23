# -*- coding: utf-8 -*-

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from delivery_env import DeliveryEnv  # Import the current version of the environment
from models import Order, Driver
from core import DeliverySimulator, WeatherService
import os
import joblib
from datetime import datetime
import pdb
import random
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
)


class MetricsCallback(BaseCallback):
    """Callback to collect all metrics and save to single CSV."""

    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.metrics = {
            "timesteps": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_accepted_order": [],
            "episode_total_orders": [],
            "episode_total_driver_commission": [],
            "clipped_actions": [],
            "losses": [],
            "entropy": [],
        }
        self.current_episode_rewards = []
        self.stop_training = False

    def _on_step(self) -> bool:
        # Clip and store actions
        if "actions" in self.locals:
            clipped_actions = np.clip(self.locals["actions"], 0.0, 1.0)
            self.metrics["clipped_actions"].extend(clipped_actions.tolist()[0])

        # Track episode completion
        for info in self.locals.get("infos", []):
            if "episode" in info:
                # print('DEBUG: info contains episode:', info['episode'])
                # current_ep = len(self.metrics['episode_rewards']) + 1
                # logging.info(
                #    f"Episode {current_ep} completed: "
                #    f"Reward={info['episode']['r']:.2f}, "
                #    f"Steps={info['episode']['l']}"
                # )

                self.metrics["episode_rewards"].append(info["episode"]["r"])
                self.metrics["episode_lengths"].append(info["episode"]["l"])
                # self.metrics['episode_accepted_order'].append(accepted_orders)
                # self.metrics['episode_total_orders'].append(total_orders)
                # self.metrics['episode_total_driver_commission'].append(total_commission)
                self.metrics["timesteps"].append(self.num_timesteps)

                self.stop_training = True

        if self.stop_training:
            logging.info("Training stopped after 1 episode")
            return False

            # if current_ep == 10:  # Expected total episodes
            #    logging.info("All 10 episodes recorded successfully")

        return True

    def _on_rollout_end(self) -> None:
        # Collect training metrics
        if self.logger is not None:
            log_dict = self.logger.name_to_value
            self.metrics["losses"].append(log_dict.get("train/loss", np.nan))
            self.metrics["entropy"].append(log_dict.get("train/entropy_loss", np.nan))

    def save_to_csv(self):
        """Save all metrics to a single CSV file."""
        try:
            # Create DataFrame from collected metrics
            max_len = max(len(v) for v in self.metrics.values())
            metrics_df = pd.DataFrame(
                {k: v + [np.nan] * (max_len - len(v)) for k, v in self.metrics.items()}
            )

            # Save to CSV with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"logs/training_metrics_{timestamp}.csv"
            metrics_df.to_csv(csv_path, index=False)
            logging.info(f"Metrics saved to {csv_path}")
        except Exception as e:
            logging.error(f"Failed to save metrics: {e}")


def load_data():
    """Load and preprocess environment data."""
    orders_df = pd.read_csv("data/order.csv")
    driver_data = pd.read_csv("data/driver_update.csv")
    schedule_data = pd.read_csv("data/driver_schedule.csv")

    orders_df = orders_df.loc[
        (orders_df["outside"] == 0) & (orders_df["pickup_area"].notnull())
    ]
    orders_df = orders_df[
        (pd.to_datetime(orders_df["datetime"]).dt.hour >= 8)
        & (pd.to_datetime(orders_df["datetime"]).dt.hour < 24)
    ]

    orders_df["date"] = pd.to_datetime(orders_df["date"]).dt.date
    orders_df = orders_df[orders_df["date"] <= pd.to_datetime("2025-04-16").date()]
    valid_days = orders_df["date"].unique().tolist()

    # orders_df['revenue'] = orders_df.loc[orders_df['status'] == 5]['customer_price'] * (orders_df.loc[orders_df['status'] == 5]['commissionPercent'] / 100)
    # print('total revenue ', orders_df['revenue'].sum())
    # pdb.set_trace()

    orders_df = orders_df[
        [
            "order_id",
            "datetime",
            "pickup_area",
            "dropoff_area",
            "pickup_lat",
            "pickup_lon",
            "dropoff_lat",
            "dropoff_lon",
            "customer_price",
            "commissionPercent",
            "date",
        ]
    ]

    # orders = [Order(**row) for _, row in orders_df.iterrows()]
    return orders_df, driver_data, schedule_data, valid_days


# Ensure global model is loaded before initializing DeliverySimulator
if not hasattr(DeliverySimulator, "shared_model"):
    logging.info("Loading shared_model globally in train_rl.py...")
    DeliverySimulator.shared_model = joblib.load("models/acceptance_model_ensemble.pkl")
    logging.info("shared_model successfully loaded!")


# Load data & initialize environment
orders_df, driver_data, schedule_data, valid_days = load_data()
weather_service = WeatherService("data/weather.csv")

# Initialize PPO model
dummy_env = DeliveryEnv([], driver_data, schedule_data, weather_service)
model = PPO(
    "MultiInputPolicy",
    dummy_env,
    verbose=0,
    learning_rate=1e-4,
    batch_size=512,
    gamma=0.95,
)
total_timesteps = 27000
num_iterations = 5

callback = MetricsCallback()

with tqdm(total=num_iterations, desc="Training Progress") as progress_bar:
    for i in range(num_iterations):
        selected_day = random.choice(valid_days)  # Randomly pick a training day
        logging.info(f"Iteration {i+1}: Training on day {selected_day}")

        # Filter orders for selected day
        daily_orders = orders_df[orders_df["date"] == selected_day]
        daily_orders = daily_orders[
            [
                "order_id",
                "datetime",
                "pickup_area",
                "dropoff_area",
                "pickup_lat",
                "pickup_lon",
                "dropoff_lat",
                "dropoff_lon",
                "customer_price",
                "commissionPercent",
            ]
        ]
        orders = [Order(**row) for _, row in daily_orders.iterrows()]

        # Reinitialize environment with filtered orders
        env = DeliveryEnv(orders, driver_data, schedule_data, weather_service)
        # env = Monitor(env, filename = None, info_keywords=("a","o","c"))
        model.set_env(env)  # Update PPO model with new environment

        # obs = env.reset()
        # if obs is None:
        #     logging.info(f"Skipping iteration {i+1} as no data for day {selected_day}")
        #     continue

        # Train for one episode (one full day)
        # try:
        model.learn(total_timesteps, reset_num_timesteps=False, callback=callback)
        # except Exception as e:
        #     print(f"DEBUG -> Training stopped: {e}")

        progress_bar.update(1)

callback.save_to_csv()

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"trained_models/ppo_delivery_{timestamp}"
model.save(model_path)
# model.save("models/ppo_delivery_latest")
logging.info(f"Models saved to {model_path} and models/ppo_delivery")
