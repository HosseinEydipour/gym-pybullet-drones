import os
import gymnasium as gym
import numpy as np
import pybullet as p  # For pybullet access
from gymnasium import spaces  # For action/obs spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # Corrected import
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool

class NavigationAviary(BaseAviary):
    def __init__(self, drone_model: DroneModel=DroneModel.CF2X, num_drones=1, initial_xyzs=None, initial_rpys=None, gui=False, record=False, obstacles=True):
        super().__init__(drone_model=drone_model, num_drones=num_drones, initial_xyzs=initial_xyzs or np.array([[0.0, 0.0, 1.0]]), initial_rpys=initial_rpys or np.array([[0.0, 0.0, 0.0]]), physics=Physics.PYB, pyb_freq=240, ctrl_freq=240, gui=gui, record=record, obstacles=obstacles)
        self.AGGR_PHY_STEPS = self.PYB_FREQ // self.CTRL_FREQ  # Explicitly set (1 in this case)
        self.goal = np.array([5.0, 5.0, 1.0])  # Target position
        self.obstacle_positions = [np.array([1.5, 1.5, 1.0]), np.array([2.5, 2.5, 1.0]), np.array([3.5, 3.5, 1.0])]  # Static cubes
        self.obstacle_ids = []
        for pos in self.obstacle_positions:
            self._addObstacle(pos)

    def _addObstacle(self, pos):
        col_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], physicsClientId=self.CLIENT)
        vis_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1], physicsClientId=self.CLIENT)
        obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape_id, baseVisualShapeIndex=vis_shape_id, basePosition=pos, physicsClientId=self.CLIENT)
        self.obstacle_ids.append(obstacle_id)

    def _actionSpace(self):
        act_lower_bound = np.array([0.] * 4)
        act_upper_bound = np.array([self.MAX_RPM] * 4)
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        # Normalized state (20) + relative goal (3) = 23 dims
        return spaces.Box(low=-np.ones(23), high=np.ones(23), dtype=np.float32)

    def _preprocessAction(self, action):
        return np.clip(action, 0, self.MAX_RPM)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        norm_state = self._clipAndNormalizeState(state)
        # Relative goal (normalized)
        rel_goal = (self.goal - self.pos[0]) / np.array([10.0, 10.0, 2.0])  # MAX_XY=10, MAX_Z=2
        return np.hstack([norm_state, rel_goal]).astype(np.float32)

    def _clipAndNormalizeState(self, state):
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 2
        MAX_ANG_VEL = 12 * np.pi  # Added for ang_vel clipping
        MAX_XY = 10
        MAX_Z = 2
        MAX_PITCH_ROLL = np.pi

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        clipped_ang_vel = np.clip(state[13:16], -MAX_ANG_VEL, MAX_ANG_VEL)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # Yaw
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = clipped_ang_vel / MAX_ANG_VEL

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],  # Quat
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20] / self.MAX_RPM
                                      ])
        return norm_and_clipped

    def _computeInfo(self):
        return {}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        state = self._getDroneStateVector(0)
        self.pos[0] = state[0:3]  # Fixed: Use slice assignment
        self.prev_dist = np.linalg.norm(self.pos[0] - self.goal)
        return obs, info

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        self.pos[0] = state[0:3]  # Fixed: Use slice assignment
        dist_to_goal = np.linalg.norm(self.pos[0] - self.goal)
        reward = -dist_to_goal
        
        reward += 5 * (self.prev_dist - dist_to_goal)
        self.prev_dist = dist_to_goal
        
        vel = state[10:13]
        dir_to_goal = (self.goal - self.pos[0]) / (dist_to_goal + 1e-6)
        if not self._obstacleAhead():
            reward += 2 * np.dot(vel, dir_to_goal)
        
        reward -= 0.005
        if self._checkCollision(): reward -= 500
        if dist_to_goal < 0.3: reward += 200
        return reward

    def _computeTerminated(self):
        if np.linalg.norm(self.pos[0] - self.goal) < 0.3: return True
        if self._checkCollision() or self.pos[0][2] < 0.1: return True  # Fixed: self.pos[0][2]
        return False

    def _computeTruncated(self):
        return (self.step_counter * self.AGGR_PHY_STEPS) / self.PYB_FREQ > 10.0

    def _obstacleAhead(self):
        ray_from = self.pos[0]
        ray_to = self.pos[0] + 2.0 * (self.goal - self.pos[0]) / (np.linalg.norm(self.goal - self.pos[0]) + 1e-6)
        ray_info = p.rayTest(ray_from, ray_to, physicsClientId=self.CLIENT)
        if ray_info[0][0] != -1:
            return True
        return False

    def _checkCollision(self):
        for i in range(p.getNumJoints(self.DRONE_IDS[0], physicsClientId=self.CLIENT)):
            contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], linkIndexA=i, physicsClientId=self.CLIENT)
            for contact in contacts:
                if contact[2] != self.GROUND_PLANE_ID:
                    return True
        return False

# Register as Gym env
gym.register(id='navigation-aviary-v0', entry_point='gym_pybullet_drones.envs.navigation_env:NavigationAviary')