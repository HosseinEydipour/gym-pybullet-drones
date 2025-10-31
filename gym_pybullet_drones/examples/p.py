import time
import pybullet as p
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_pybullet_drones.utils.utils import sync  # For real-time sync
import gym_pybullet_drones.envs.navigation_env

# Create env with GUI and recording
env = gym.make('navigation-aviary-v0', gui=True, record=True)

# Start video logging
logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "best_policy_video.mp4", physicsClientId=env.unwrapped.CLIENT)

# Load model and normalizer
model = PPO.load("results/best_model.zip")
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize.load("results/vec_normalize.pkl", vec_env)
vec_env.training = False
vec_env.norm_reward = False

obs = vec_env.reset()
done = [False]  # VecEnv uses list for dones
info = {}  # Placeholder
start = time.time()
i = 0
max_steps = 2400  # ~10 seconds at 240 Hz; adjust as needed

while not any(done) and i < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    
    # Sync for real-time visualization (keeps GUI open longer)
    sync(i, start, env.unwrapped.TIMESTEP)
    i += 1

# Stop logging
p.stopStateLogging(logging_id)

vec_env.close()