import os
import gymnasium as gym
import numpy as np
import pybullet as p  # Add this for pybullet access

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel  # Corrected import
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool

class NavigationAviary(BaseAviary):
    def __init__(self, drone_model: DroneModel=DroneModel.CF2X, num_drones=1, initial_xyzs=None, gui=False, record=False, obstacles=True):
        super().__init__(drone_model=drone_model, num_drones=num_drones, initial_xyzs=initial_xyzs or np.array([[0.0, 0.0, 1.0]]), freq=240, gui=gui, record=record, obstacles=obstacles)
        self.goal = np.array([5.0, 5.0, 1.0])  # Target position
        self.obstacle_positions = [np.array([1.5, 1.5, 1.0]), np.array([2.5, 2.5, 1.0]), np.array([3.5, 3.5, 1.0])]  # Static cubes
        self.obstacle_ids = []
        for pos in self.obstacle_positions:
            self._addObstacle(pos)

    def _addObstacle(self, pos):
        # Use a primitive box shape instead of URDF for simplicity (no need for external files)
        col_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], physicsClientId=self.CLIENT)
        vis_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1], physicsClientId=self.CLIENT)
        obstacle_id = p.createMultiBody(baseMass=0,  # Static
                                        baseCollisionShapeIndex=col_shape_id,
                                        baseVisualShapeIndex=vis_shape_id,
                                        basePosition=pos,
                                        physicsClientId=self.CLIENT)
        self.obstacle_ids.append(obstacle_id)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        state = self._getDroneStateVector(0)  # For drone 0
        self.pos = state[0:3]  # XYZ position
        self.prev_dist = np.linalg.norm(self.pos - self.goal)
        return obs, info

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        self.pos = state[0:3]
        dist_to_goal = np.linalg.norm(self.pos - self.goal)
        reward = -dist_to_goal  # Distance penalty
        
        # Potential-based shaping
        reward += 5 * (self.prev_dist - dist_to_goal)
        self.prev_dist = dist_to_goal
        
        # Velocity alignment bonus (if no obstacle ahead)
        vel = state[10:13]  # Linear velocity
        dir_to_goal = (self.goal - self.pos) / (dist_to_goal + 1e-6)
        if not self._obstacleAhead():
            reward += 2 * np.dot(vel, dir_to_goal)
        
        # Penalties
        reward -= 0.005  # Time penalty
        if self._checkCollision(): reward -= 500  # Crash
        if dist_to_goal < 0.3: reward += 200  # Goal reached
        return reward

    def _computeTerminated(self):
        if np.linalg.norm(self.pos - self.goal) < 0.3: return True  # Goal
        if self._checkCollision() or self.pos[2] < 0.1: return True  # Crash
        return False

    def _computeTruncated(self):
        return self.step_counter / self.SIM_FREQ > 10.0  # e.g., 10 seconds max (adjust based on freq=240)

    def _obstacleAhead(self):
        # Simple raycast forward (along velocity or to goal)
        ray_from = self.pos
        ray_to = self.pos + 2.0 * (self.goal - self.pos) / (np.linalg.norm(self.goal - self.pos) + 1e-6)  # 2m ahead toward goal
        ray_info = p.rayTest(ray_from, ray_to, physicsClientId=self.CLIENT)
        if ray_info[0][0] != -1:  # Hit something
            return True
        return False

    def _checkCollision(self):
        # Check contacts with any body except ground (bodyB != -1 or self.GROUND_PLANE_ID)
        for i in range(p.getNumJoints(self.DRONE_IDS[0], physicsClientId=self.CLIENT)):
            contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], linkIndexA=i, physicsClientId=self.CLIENT)
            for contact in contacts:
                if contact[2] != self.GROUND_PLANE_ID:  # Not ground
                    return True
        return False

# Register as Gym env
gym.register(id='navigation-aviary-v0', entry_point='gym_pybullet_drones.envs.navigation_env:NavigationAviary')