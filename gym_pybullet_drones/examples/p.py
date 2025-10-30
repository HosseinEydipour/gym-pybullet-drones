import pybullet as p
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Load env with GUI and recording
env = gym.make('navigation-aviary-v0', gui=True, record=True)  # GUI for visualization

# Start video logging
logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "best_policy_video.mp4", physicsClientId=env.getPyBulletClient())

# Load model
model = PPO.load("results/best_model.zip")
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize.load("results/vec_normalize.pkl", vec_env)  # Load normalization if used

obs = vec_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    p.stepSimulation(physicsClientId=env.getPyBulletClient())  # Step for GUI

# Stop logging
p.stopStateLogging(logging_id)

vec_env.close()