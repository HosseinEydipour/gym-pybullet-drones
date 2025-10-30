import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import gym_pybullet_drones.envs.navigation_env  # Add this line to register the custom env

# Make env
def make_env():
    return gym.make('navigation-aviary-v0', gui=False, record=False)

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Train PPO
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, batch_size=256, ent_coef=0.01)
eval_callback = EvalCallback(env, best_model_save_path='./results/', log_path='./logs/', eval_freq=10000, n_eval_episodes=10)
model.learn(total_timesteps=1000000, callback=eval_callback)  # Train for 1M steps; adjust as needed

model.save("results/best_model.zip")
env.close()