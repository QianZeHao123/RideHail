# -*- coding: utf-8 -*-

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import joblib  # Import Joblib to load the model
import pandas as pd
from models import Order, Driver
from core import DeliverySimulator, WeatherService


def load_data():
    orders_df = pd.read_csv("data/order.csv")
    driver_data = pd.read_csv("data/driver_update.csv")
    schedule_data = pd.read_csv("data/driver_schedule.csv")

    # order has to be within Esfahan & no missing pickup location (remove 145 orders)
    orders_df = orders_df.loc[
        (orders_df["outside"] == 0) & (orders_df["pickup_area"].notnull())
    ]
    # order has to be within 8am and 11:59pm
    orders_df = orders_df[
        (pd.to_datetime(orders_df["datetime"]).dt.hour >= 8)
        & (pd.to_datetime(orders_df["datetime"]).dt.hour < 24)
    ]

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


def print_results(sim):
    print("\n=== Simulation Results ===")
    print(f"Assigned Orders: {len(sim.metrics['assignments'])}")
    print(f"Rejected Orders: {len(sim.metrics['rejected_orders'])}")
    print(f"Unassigned Orders: {len(sim.metrics['unassigned_orders'])}")
    print(f"\nTotal Platform Revenue: ${sim.metrics['platform_revenue']:.2f}")
    print(f"Total Distance Traveled: {sim.metrics['total_distance_km']:.2f} km")

    print("\nTop 3 Assignments:")
    for assignment in sim.metrics["assignments"][:3]:
        print(
            f"Order {assignment['order_id']} → Driver {assignment['driver_id']} "
            f"(Distance: {assignment['distance_km']:.1f} km, "
            f"Commission: ${assignment['commission']:.2f})"
        )


if __name__ == "__main__":
    #  Load the model once globally before initializing DeliverySimulator
    if not hasattr(DeliverySimulator, "shared_model"):
        DeliverySimulator.shared_model = joblib.load(
            "models/acceptance_model_ensemble.pkl"
        )

    # Load and prepare data
    orders, driver_data, schedule_data = load_data()
    weather_service = WeatherService("data/weather.csv")

    # Run simulation
    sim = DeliverySimulator(orders, driver_data, schedule_data, weather_service)
    sim.run()

    # Print results
    print_results(sim)
