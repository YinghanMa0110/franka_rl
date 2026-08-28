import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import panda_gym

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def forward(self, x):
        return self.actor_mean(x)

env = gym.make('PandaReach-v3', reward_type='dense')
env = gym.wrappers.FlattenObservation(env)
env = gym.wrappers.ClipAction(env)
# env = gym.wrappers.NormalizeObservation(env)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = Agent(obs_dim, action_dim)
checkpoint = torch.load('runs/PandaReach-v3__ppo_continuous_action__1__1787871884/ppo_continuous_action.cleanrl_model', map_location='cpu')
agent.load_state_dict(checkpoint, strict=False)
agent.eval()

successes = 0
n_episodes = 10

for i in range(n_episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        obs_tensor = torch.Tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = agent(obs_tensor).numpy()[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    success = info.get('is_success', False)
    successes += int(success)
    print(f"Episode {i+1}: success={success}")

print(f"\nSuccess rate: {successes}/{n_episodes} = {successes/n_episodes*100:.0f}%")
