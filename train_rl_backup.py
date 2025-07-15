# -*- coding: utf-8 -*-

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
import pdb

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
            "clipped_actions": [],
            "losses": [],
            "entropy": [],
        }
        self.current_episode_rewards = []

    def _on_step(self) -> bool:
        # Clip and store actions
        if "actions" in self.locals:
            clipped_actions = np.clip(self.locals["actions"], 0.0, 1.0)
            self.metrics["clipped_actions"].extend(clipped_actions.tolist()[0])

        # Track episode completion
        for info in self.locals.get("infos", []):
            if "episode" in info:
                current_ep = len(self.metrics["episode_rewards"]) + 1
                logging.info(
                    f"Episode {current_ep} completed: "
                    f"Reward={info['episode']['r']:.2f}, "
                    f"Steps={info['episode']['l']}"
                )
                self.metrics["episode_rewards"].append(info["episode"]["r"])
                self.metrics["episode_lengths"].append(info["episode"]["l"])
                self.metrics["timesteps"].append(self.num_timesteps)

                if current_ep == 10:  # Expected total episodes
                    logging.info("All 10 episodes recorded successfully")

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
        ]
    ]

    orders = [Order(**row) for _, row in orders_df.iterrows()]
    return orders, driver_data, schedule_data


# Ensure global model is loaded before initializing DeliverySimulator
if not hasattr(DeliverySimulator, "shared_model"):
    logging.info("Loading shared_model globally in train_rl.py...")
    DeliverySimulator.shared_model = joblib.load("models/acceptance_model_ensemble.pkl")
    logging.info("shared_model successfully loaded!")


# Load data & initialize environment
orders, driver_data, schedule_data = load_data()
weather_service = WeatherService("data/weather.csv")
env = DeliveryEnv(orders, driver_data, schedule_data, weather_service)

# Initialize PPO model
model = PPO(
    "MultiInputPolicy", env, verbose=1, learning_rate=1e-4, batch_size=512, gamma=0.95
)

total_timesteps = 300000
num_iterations = 10

callback = MetricsCallback()
try:
    model.learn(total_timesteps, reset_num_timesteps=False, callback=callback)
except Exception as e:
    print(f"DEBUG -> Training stopped: {e}")

callback.save_to_csv()

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"trained_models/ppo_delivery_{timestamp}"
model.save(model_path)
# model.save("models/ppo_delivery_latest")
logging.info(f"Models saved to {model_path} and models/ppo_delivery")
