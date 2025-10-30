import numpy as np
import pybullet as p
from gymnasium.envs.registration import register
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TrackingAviary(BaseAviary):
    """Multi-drone environment for tracking a moving target."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 aggregate: bool = False  # False for true multi-agent (dict obs/rewards)
                 ):
        """Initialization of a multi-agent RL environment.

        Parameters as in BaseAviary, plus 'aggregate' for shared vs. per-agent spaces.
        """
        self.NUM_DRONES = num_drones
        self.aggregate = aggregate
        self.target_pos = np.array([0.0, 0.0, 0.5])  # Initial target position
        self.target_vel = np.array([0.1, 0.0, 0.0])  # Constant velocity (moving in x-direction)
        self.target_id = None

        # Base obs space per drone: kinematic info (pos, quat, rpy, vel, ang vel) = 20 values
        single_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,))  # +3 for rel_pos to target

        if aggregate:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_DRONES * 23,))
        else:
            self.observation_space = spaces.Dict({i: single_obs_space for i in range(self.NUM_DRONES)})

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    def _addObstacles(self):
        """Load the moving target as an 'obstacle' (but non-colliding)."""
        super()._addObstacles()
        # Use a simple URDF (replace with car.urdf if available; duck is a PyBullet default)
        self.target_id = p.loadURDF("duck.urdf", self.target_pos, p.getQuaternionFromEuler([0, 0, 0]))

    def _computeObs(self):
        """Augment base obs with relative position to target."""
        obs = super()._computeObs()  # Dict {0: array(20,), 1: array(20,)} or flat if aggregate

        if self.aggregate:
            # Flatten all drone states first
            drone_obs = np.hstack([obs[i * 20:(i + 1) * 20] for i in range(self.NUM_DRONES)])
            rel_pos_all = np.hstack([ (self.target_pos - self.pos[i]) / 10.0 for i in range(self.NUM_DRONES) ])  # Normalize
            return np.hstack([drone_obs, rel_pos_all])
        else:
            for i in range(self.NUM_DRONES):
                rel_pos = (self.target_pos - self.pos[i]) / 10.0  # Normalize for stability
                obs[i] = np.hstack([obs[i], rel_pos])
            return obs

    def _computeReward(self):
        """Negative squared distance to target for estimation accuracy."""
        rewards = {}
        for i in range(self.NUM_DRONES):
            dist = np.linalg.norm(self.pos[i] - self.target_pos)
            rewards[i] = -dist ** 2  # Encourage close proximity for better 'estimation'
        if self.aggregate:
            return sum(rewards.values()) / self.NUM_DRONES
        return rewards

    def _computeTerminated(self):
        """Terminate if any drone crashes or drifts too far."""
        for i in range(self.NUM_DRONES):
            dist = np.linalg.norm(self.pos[i] - self.target_pos)
            if dist > 10.0:  # Arbitrary threshold for 'lost target'
                return True
        return super()._computeTerminated()

    def _computeTruncated(self):
        return False  # No truncation

    def _computeInfo(self):
        return {i: {"estimation_error": np.linalg.norm(self.pos[i] - self.target_pos)} for i in range(self.NUM_DRONES)}

    def step(self, action):
        """Step the env, updating target first."""
        # Update target position before drone physics
        dt = 1 / self.ctrl_freq  # Timestep
        self.target_pos += self.target_vel * dt
        p.resetBasePositionAndOrientation(self.target_id, self.target_pos, p.getQuaternionFromEuler([0, 0, 0]))

        return super().step(action)

    def reset(self, seed=None, options=None):
        """Reset target position on episode start."""
        self.target_pos = np.array([0.0, 0.0, 0.5])
        return super().reset(seed=seed, options=options), {}
    

register(
    id="tracking-aviary-v0",
    entry_point="gym_pybullet_drones.envs:TrackingAviary",
    kwargs={
        "num_drones": 2,
        "aggregate": False,  # True multi-agent
        "obs": ObservationType.KIN,
        "act": ActionType.RPM
    }
)