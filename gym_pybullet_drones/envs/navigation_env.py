import gymnasium as gym
import numpy as np
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.DroneModel import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool

class NavigationAviary(BaseAviary):
    def __init__(self, drone_model: DroneModel=DroneModel.CF2X, num_drones=1, initial_xyzs=None, gui=False, record=False, obstacles=True):
        super().__init__(drone_model=drone_model, num_drones=num_drones, initial_xyzs=initial_xyzs, gui=gui, record=record, obstacles=obstacles)
        self.goal = np.array([5.0, 5.0, 1.0])  # Target position
        self.obstacle_positions = [np.array([1.5, 1.5, 1.0]), np.array([2.5, 2.5, 1.0]), np.array([3.5, 3.5, 1.0])]  # Static cubes
        for pos in self.obstacle_positions:
            self._addObstacle(pos)  # Custom method to add obstacles (extend BaseAviary if needed)

    def _addObstacle(self, pos):
        # Add a cube obstacle (extend this with PyBullet's createVisualShape/createCollisionShape)
        p.loadURDF("cube.urdf", pos, physicsClientId=self.CLIENT)

    def reset(self, seed=None, options=None):
        obs = super().reset()
        self.pos = self.getPyBulletClient().getBasePositionAndOrientation(0)[0]  # Drone position
        return obs, {}

    def _computeReward(self):
        dist_to_goal = np.linalg.norm(self.pos - self.goal)
        reward = -dist_to_goal  # Distance penalty
        
        # Potential-based shaping
        prev_dist = getattr(self, 'prev_dist', dist_to_goal)
        reward += 5 * (prev_dist - dist_to_goal)
        self.prev_dist = dist_to_goal
        
        # Velocity alignment bonus (if no obstacle ahead)
        vel = self.getPyBulletClient().getBaseVelocity(0)[0]
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
        return self.step_counter > 1000  # Max steps

    def _obstacleAhead(self):
        # Raycast check for obstacles (implement with PyBullet rayTest)
        return False  # Placeholder; add rayTest logic

    def _checkCollision(self):
        # Check contacts with obstacles
        return len(self.getPyBulletClient().getContactPoints(bodyA=0)) > 0  # Drone ID 0

# Register as Gym env
gym.register(id='navigation-aviary-v0', entry_point='navigation_env:NavigationAviary')