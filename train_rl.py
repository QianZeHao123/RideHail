# -*- coding: utf-8 -*-

import gym
import logging
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
from collections import defaultdict
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
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)

class DictRolloutBuffer:
    def __init__(self):
        self.observations = defaultdict(list)
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.returns = []
        self.advantages = []

    def add(self, obs, action, reward, done, value, log_prob):
        # Store each observation component separately
        for key, value in obs.items():
            self.observations[key].append(value)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns_advantages(self, next_value, gamma=0.99, gae_lambda=0.95):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones + [False])
        
        deltas = rewards + gamma * values[1:] * (1 - dones[1:]) - values[:-1]
        
        advantages = np.zeros_like(rewards)
        advantage = 0
        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + gamma * gae_lambda * (1 - dones[t+1]) * advantage
            advantages[t] = advantage
            
        self.returns = (advantages + values[:-1]).tolist()
        self.advantages = ((advantages - advantages.mean()) / 
                          (advantages.std() + 1e-8)).tolist()

    def clear(self):
        for key in self.observations:
            self.observations[key].clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.returns.clear()
        self.advantages.clear()

class PPOAgent(nn.Module):
    def __init__(self, obs_spaces, act_dim):
        super().__init__()
        # Create feature extractors for each observation space component
        self.feature_extractors = nn.ModuleDict()
        total_features = 0
        self.obs_spaces = obs_spaces
        
        for key, space in self.obs_spaces.items():
            # Handle different space types
            if isinstance(space, gym.spaces.Box):
                # Continuous features
                input_dim = space.shape[0]
            elif isinstance(space, gym.spaces.Discrete):
                # Discrete features need one-hot encoding
                input_dim = space.n
            else:
                raise ValueError(f"Unsupported space type: {type(space)}")

            self.feature_extractors[key] = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.Tanh()
            )
            total_features += 32
            
        print('DEBUG: total features is ', total_features)
            
        # Actor and Critic networks
        self.actor = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
            nn.Sigmoid() # within 0 and 1
        )
        self.critic = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Learnable standard deviation for exploration
        # self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0)) # Initial smaller std ≈ 0.37
        
    def _process_observation(self, obs):
        """Convert raw observation to tensor with proper encoding"""
        processed = {}
        for key, value in obs.items():
            space = self.obs_spaces[key]
            
            if isinstance(space, gym.spaces.Box):
                # Already continuous, just convert to tensor
                processed[key] = torch.FloatTensor(value)
            elif isinstance(space, gym.spaces.Discrete):
                # Convert to one-hot encoding
                one_hot = torch.zeros(space.n)
                one_hot[int(value)] = 1.0
                processed[key] = one_hot
                
        return processed

    def get_action(self, obs):
        # Process dictionary observation through feature extractors
        
        processed_obs = self._process_observation(obs)
        
        features = []
        for key, value in processed_obs.items():
            features.append(self.feature_extractors[key](value))
            
        concat_features = torch.cat(features)
        action_mean = self.actor(concat_features)
        std = torch.exp(self.log_std)
        
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        value = self.critic(concat_features).squeeze()
        
        return action.detach().numpy(), log_prob.detach().item(), value.detach().item()
    
    def get_value(self, obs):
        """Get state value for a single observation (no gradient)"""
        with torch.no_grad():
            # Process observation through the same preprocessing
            processed_obs = self._process_observation(obs)
            
            features = []
            for key, value in processed_obs.items():
                # Already processed to appropriate encoding
                tensor = value.unsqueeze(0)  # Add batch dimension
                features.append(self.feature_extractors[key](tensor))
                
            concat_features = torch.cat(features, dim=1)  # Concatenate features
            value = self.critic(concat_features).squeeze()
            return value.item()

    def evaluate(self, obs_batch, action_batch):
        # Process batched dictionary observations
        processed_batch = []
                
        for key in obs_batch.keys():
            space = self.obs_spaces[key]
            
            if isinstance(space, gym.spaces.Discrete):
                batch_size = obs_batch[key].size(0)
                one_hot = torch.zeros(batch_size, space.n)
                one_hot.scatter_(1, obs_batch[key].long().unsqueeze(-1), 1)
                processed = self.feature_extractors[key](one_hot)
            else:
                processed = self.feature_extractors[key](obs_batch[key].float())
                
            processed_batch.append(processed)
            
        #concat_features = torch.cat(features, dim=1)
        concat_features = torch.cat(processed_batch, dim=1)
        action_mean = self.actor(concat_features)
        std = torch.exp(self.log_std)
        
        dist = torch.distributions.Normal(action_mean, std)
        log_probs = dist.log_prob(action_batch).sum(-1)
        values = self.critic(concat_features).squeeze()
        
        return log_probs, values
    
class PPOTrainer:
    def __init__(self, days_data, obs_space, action_dim):
        self.days_data = days_data
        self.agent = PPOAgent(obs_space, action_dim)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=3e-4)
        
        # Training parameters
        self.num_epochs = 100
        self.update_epochs = 1
        self.batch_size = 64
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2

    def train(self): 
        rewards = []
        steps = []
        assigned_order =[]
        total_order = []
        total_driver_commission = []
        actions = []
        losses = []
        
        for epoch in range(self.num_epochs):
            # Randomly select a day's data
            day_data = random.choice(self.days_data)
            
            orders = [Order(**row) for _, row in day_data.iterrows()]
            
            # Create environment for this day
            env = DeliveryEnv(orders, driver_data, schedule_data, weather_service)
            
            # Collect full day's experience
            buffer, episode_reward, step, episode_total_order, episode_total_assigned_order, episode_total_driver_commission, episode_action = self._collect_episode(env)
            
            rewards.append(episode_reward)
            steps.append(step)
            assigned_order.append(episode_total_assigned_order)
            total_order.append(episode_total_order)
            total_driver_commission.append(episode_total_driver_commission)
            actions.append(episode_action)
            
            # Update policy
            update_loss = self._update_policy(buffer)
            
            losses.append(update_loss)
            
            # Cleanup
            del env
            print(f"Epoch {epoch+1}/{self.num_epochs} completed")
            
        return rewards, steps, assigned_order, total_order, total_driver_commission, actions, losses

    def _collect_episode(self, env):
        buffer = DictRolloutBuffer()
        obs = env.reset()
        done = False
        
        episode_reward = 0
        step = 0
        episode_action = []
        
        while not done:
            with torch.no_grad():
                action, log_prob, value = self.agent.get_action(obs)
                episode_action.append(action[0])
            
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            buffer.add(obs, action, reward, done, value, log_prob)
            
            if done:
                episode_total_assigned_order = info['episode']['a']
                episode_total_order = info['episode']['o']
                episode_total_driver_commission = info['episode']['c']
            else:
                obs = next_obs
        
        # Compute final value for advantage calculation
        with torch.no_grad():
            final_value = self.agent.get_value(obs)
            
        buffer.compute_returns_advantages(final_value, self.gamma, self.gae_lambda)
        return buffer, episode_reward, step, episode_total_order, episode_total_assigned_order, episode_total_driver_commission, episode_action

    def _update_policy(self, buffer):
        # Convert buffer data to tensors
        obs_tensors = {}
        for key in buffer.observations.keys():
            # Convert list to numpy array first to handle different dtypes
            arr = np.array(buffer.observations[key])
            
            # Handle discrete observations (already one-hot encoded)
            if arr.dtype == np.int64 or arr.dtype == np.int32:
                obs_tensors[key] = torch.LongTensor(arr)
            else:
                obs_tensors[key] = torch.FloatTensor(arr)
        
        act_tensor = torch.FloatTensor(np.array(buffer.actions))
        ret_tensor = torch.FloatTensor(np.array(buffer.returns))
        adv_tensor = torch.FloatTensor(np.array(buffer.advantages))
        old_log_probs = torch.FloatTensor(np.array(buffer.log_probs))

        # Policy optimization
        update_loss = []
        for _ in range(self.update_epochs):
            indices = torch.randperm(len(buffer.actions))
            for start in range(0, len(buffer.actions), self.batch_size):
                idx = indices[start:start+self.batch_size]

                batch_obs = {k: v[idx] for k, v in obs_tensors.items()}
                batch_act = act_tensor[idx]
                batch_ret = ret_tensor[idx]
                batch_adv = adv_tensor[idx]
                batch_old_log_probs = old_log_probs[idx]

                # Calculate losses
                log_probs, values = self.agent.evaluate(batch_obs, batch_act)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * batch_adv
                policy_loss = -(torch.min(ratio * batch_adv, clip_adv)).mean()
                value_loss = 0.5 * (values - batch_ret).pow(2).mean()
                loss = policy_loss + value_loss
                
                update_loss.append(loss.detach().numpy())

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()
                
        return update_loss
    
    def _save_model(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"trained_models/ppo_model_{timestamp}.pt"
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")
                


def load_data():
    """Load and preprocess environment data."""
    orders_df = pd.read_csv('data/order.csv')
    driver_data = pd.read_csv('data/driver_update.csv')
    schedule_data = pd.read_csv('data/driver_schedule.csv')
    
    orders_df = orders_df.loc[(orders_df['outside'] == 0) & (orders_df['pickup_area'].notnull())]
    orders_df = orders_df[(pd.to_datetime(orders_df['datetime']).dt.hour >= 8) & 
                         (pd.to_datetime(orders_df['datetime']).dt.hour < 24)]
    
    orders_df['date'] = pd.to_datetime(orders_df['date']).dt.date
    orders_df = orders_df[orders_df['date'] <= pd.to_datetime("2025-04-16").date()]
    
    order_day1 = orders_df.loc[orders_df['date'] == pd.to_datetime("2025-04-07")][['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    order_day2 = orders_df.loc[orders_df['date'] == pd.to_datetime("2025-04-08")][['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    order_day3 = orders_df.loc[orders_df['date'] == pd.to_datetime("2025-04-09")][['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    order_day4 = orders_df.loc[orders_df['date'] == pd.to_datetime("2025-04-10")][['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    order_day5 = orders_df.loc[orders_df['date'] == pd.to_datetime("2025-04-11")][['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    order_day6 = orders_df.loc[orders_df['date'] == pd.to_datetime("2025-04-12")][['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    order_day7 = orders_df.loc[orders_df['date'] == pd.to_datetime("2025-04-13")][['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    order_day8 = orders_df.loc[orders_df['date'] == pd.to_datetime("2025-04-14")][['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    order_day9 = orders_df.loc[orders_df['date'] == pd.to_datetime("2025-04-15")][['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    order_day10 = orders_df.loc[orders_df['date'] == pd.to_datetime("2025-04-16")][['order_id', 'datetime', 'pickup_area', 'dropoff_area', 'pickup_lat',
                           'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'customer_price', 'commissionPercent']]
    
    # orders = [Order(**row) for _, row in orders_df.iterrows()]
    return order_day1, order_day2, order_day3, order_day4, order_day5, order_day6, order_day7, order_day8, order_day9, order_day10, driver_data, schedule_data


# Ensure global model is loaded before initializing DeliverySimulator
if not hasattr(DeliverySimulator, "shared_model"):
    logging.info("Loading shared_model globally in train_rl.py...")
    DeliverySimulator.shared_model = joblib.load("models/acceptance_model_ensemble.pkl")
    logging.info("shared_model successfully loaded!")


# Load data & initialize environment
order_day1, order_day2, order_day3, order_day4, order_day5, order_day6, order_day7, order_day8, order_day9, order_day10, driver_data, schedule_data = load_data()
weather_service = WeatherService('data/weather.csv')

# Initialize PPO model
if __name__ == "__main__":
    # Prepare your 10 days' data
    days_data = [
       order_day1, order_day2, order_day3, order_day4, order_day5, order_day6, order_day7, order_day8, order_day9, order_day10
    ]
    
    orders = [Order(**row) for _, row in days_data[0].iterrows()]
    
    # Get observation space shape from your environment
    sample_env = DeliveryEnv(orders,  driver_data, schedule_data, weather_service)
    obs_space= sample_env.observation_space.spaces
    action_dim = sample_env.action_space.shape[0]
    del sample_env

    # Initialize and train
    trainer = PPOTrainer(days_data, obs_space, action_dim)
    rewards, steps, assigned_order, total_order, total_driver_commission, actions, losses = trainer.train()
    
    actions = [item for sublist in actions for item in sublist]
    losses = [item for sublist in losses for item in sublist]
    
    df = pd.DataFrame({'episode reward': pd.Series(rewards), 'episode length': pd.Series(steps), 'episode assigned orders': pd.Series(assigned_order), 'episode total orders': pd.Series(total_order),
                       'episode total driver commission': pd.Series(total_driver_commission), 'action': pd.Series(actions),'loss': pd.Series(losses)})
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"logs/training_metrics_{timestamp}.csv"
    df.to_csv(results_path, index = False)
