# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import re

# Load PPO training log
log_file = "logs/ppo_training.log"

# Extract only the last loss value per iteration
loss_dict_per_iteration = {}

with open(log_file, "r") as file:
    for line in file:
        match = re.search(r"Iteration (\d+): Loss Values - defaultdict.*?({.*})", line)
        if match:
            iteration = int(match.group(1))  # Extract iteration number
            loss_dict = eval(match.group(2))  # Convert dictionary string to Python dict
            
            if "train/loss" in loss_dict:  
                loss_dict_per_iteration[iteration] = loss_dict["train/loss"]  # Stores only last value per iteration

# Convert dictionary to DataFrame for plotting
df_loss = pd.DataFrame({"Iteration": list(loss_dict_per_iteration.keys()), "Loss": list(loss_dict_per_iteration.values())})
df_loss.set_index("Iteration", inplace=True)

# Load simulation results to visualize commission rates and revenue
simulation_results = pd.read_csv("logs/simulation_results.csv")

# Plot Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(df_loss["Loss"], label="Total PPO Loss", marker="o", linestyle="-", color="b")
plt.xlabel("Iteration")
plt.ylabel("Loss Value")
plt.title("PPO Training Loss Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot Commission Rate Over Time
plt.figure(figsize=(10, 5))
plt.plot(simulation_results["order_id"], simulation_results["commission"], label="Optimal Commission Rate", marker="o", linestyle="-", color="r")
plt.xlabel("Order ID")
plt.ylabel("Commission Rate")
plt.title("Optimized Commission Rate Over Orders")
plt.legend()
plt.grid()
plt.show()

# Plot Revenue Trend Over Time
plt.figure(figsize=(10, 5))
plt.plot(simulation_results["order_id"], simulation_results["revenue"], label="Platform Revenue", marker="o", linestyle="-", color="g")
plt.xlabel("Order ID")
plt.ylabel("Revenue")
plt.title("Revenue Generated Over Orders")
plt.legend()
plt.grid()
plt.show()



