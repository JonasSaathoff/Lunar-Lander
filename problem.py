'''
Requirements: 
    pip install "gymnasium[box2d]" numpy matplotlib
'''

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class GymProblem:
    def __init__(self, env_name: str = "LunarLander-v3", continuous: bool = False, gravity=-10.0, enable_wind: bool = False, wind_power: float = 0, turbulence_power: float = 0):
        assert env_name in gym.registry
        
        self.simulation_params = {
            "continuous": continuous,
            "gravity": gravity,
            "enable_wind": enable_wind,
            "wind_power": wind_power,
            "turbulence_power": turbulence_power
        }

        self.env_spec = gym.registry[env_name]
        self.env = self.env_spec.make(**self.simulation_params)
        self.state_size = self.env.observation_space.shape[0]
        
        self.continuous = continuous

        if not continuous:
            self.activation = np.argmax
            self.n_outputs = self.env.action_space.n
        else:
            self.activation = np.tanh
            self.n_outputs = self.env.action_space.shape[0]
            
        self.n_variables = self.state_size * self.n_outputs
        self.M = np.zeros((self.state_size, self.n_outputs))

    def play_episode(self, x: np.ndarray, **env_kwargs) -> float:
        self.M = x.reshape(self.state_size, self.n_outputs)
        
        env = self.env_spec.make(**env_kwargs)
        observation, *_ = env.reset()
        
        returns = 0
        rewards = []
        for _ in range(self.env_spec.max_episode_steps):
            action = self.activation(self.M.T @ observation)
            observation, reward, terminated, truncated, info = env.step(action)
            returns += reward
            rewards.append(reward)
            if terminated or truncated:
                break
        
        env.close()
        return returns, rewards

    def __call__(self, x: np.ndarray):
        return self.play_episode(x, **self.simulation_params)

    def show(self, x: np.ndarray) -> float:
        return self.play_episode(x, render_mode="human", **self.simulation_params)
    
    def sample(self):
        return np.random.uniform(-1, 1, size=self.n_variables)

