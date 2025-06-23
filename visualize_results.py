# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

def visualize_metrics():
    """Load and visualize training metrics from CSV."""
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("logs/training_metrics_20250521_192201.csv")
    df['order acceptance rate'] = df['episode assigned orders'] / df['episode total orders']
    df['episode avg driver commission'] = df['episode total driver commission'] / df['episode total orders']
    
    plt.scatter(df['order acceptance rate'], df['episode avg driver commission'])
    
    # plt.figure(figsize=(12, 6))
    plt.figure(figsize=(16, 12))
    plt.suptitle(f"Training Metrics - {datetime.now().strftime('%Y-%m-%d')}", y=1.02)
    
    # 1. Training Loss
    plt.subplot(2, 2, 1)
    df['loss'].dropna().plot(color='tab:blue')
    plt.title("Training Loss Over Update Steps")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    # 2. Episode Rewards
    plt.subplot(2, 2, 2)
    df['episode reward'].dropna().plot(color='tab:green')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    
    # 3. Episode Lengths
    plt.subplot(2, 2, 3)
    df['episode length'].dropna().plot(color='tab:orange')
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True, alpha=0.3)
    
    # 4. Action Distribution
    plt.subplot(2, 2, 4)
    pd.Series(df['action'].dropna()).hist(bins=50, color='tab:red')
    plt.title("Action Value Distribution")
    plt.xlabel("Clipped Action Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with timestamp
    plot_path = f"plots/training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Generated visualization: {plot_path}")
        

if __name__ == "__main__":
    visualize_metrics()


