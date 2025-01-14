import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

class HopperEnvForw(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 250,
    }

    def __init__(self, episode_len=1500, healthy_reward = 0.5, control_cost_weight = 0.1,_forward_reward_weight = 1.0, z_healthy = -0.95, x_lim = 0.2, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        MujocoEnv.__init__(
            self,
            os.path.abspath("assets/hopper_2.xml"),
            4,
            observation_space=observation_space,
            camera_name= "main",
            **kwargs
        )
        self.step_number = 0
        self.episode_len = episode_len
        self.init_q = np.array([0.0, 0.0, -0.72, 1.02])
        self._forward_reward_weight = _forward_reward_weight
        self.healthy_reward = healthy_reward
        self.control_cost_weight = control_cost_weight
        self.z_helthy = z_healthy
        self.x_lim = x_lim
    
    def control_cost(self, action):
        control_cost = self.control_cost_weight * np.sum(np.square(action))
        return control_cost
    
    def is_healthy(self, z_pos, x_pos):
        is_healthy = np.all(np.logical_and(self.z_helthy < z_pos, x_pos < self.x_lim))
        return is_healthy


    def terminated(self, z_pos, x_pos):
        terminated = not self.is_healthy(z_pos, x_pos)
        return terminated


    def step(self, a):
        x_position_before = self.data.qpos[1]
        self.do_simulation(a, self.frame_skip)
        z_position_after = self.data.qpos[0]
        x_position_after = self.data.qpos[1]
        self.step_number += 1
        x_velocity = (-x_position_after + x_position_before) / self.dt

        ctrl_cost = self.control_cost(a)
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = float(self.is_healthy(z_position_after, x_position_after))*self.healthy_reward

        rewards = forward_reward + healthy_reward
        reward = rewards - ctrl_cost

        info = {
            
            "upper_motor": (self.data.joint("upper_joint").qfrc_smooth + self.data.joint("upper_joint").qfrc_constraint),
            "lower_motor": (self.data.joint("lower_joint").qfrc_smooth + self.data.joint("lower_joint").qfrc_constraint),
            "z_coordinate": z_position_after,
            "x_coordinate": x_position_after

        }

        obs = self._get_obs()
        terminated = self.terminated(z_position_after, x_position_after)
        truncated = self.step_number > self.episode_len
        return obs, reward, terminated, truncated, info


    def reset_model(self):
        self.step_number = 0

        qpos = self.init_q + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):

        obs = np.concatenate((np.array(self.data.joint("main_body_z").qpos, dtype = np.float32),
                              np.array(self.data.joint("main_body_z").qvel, dtype = np.float32),
                              np.array(self.data.joint("main_body_x").qpos, dtype = np.float32),
                              np.array(self.data.joint("main_body_x").qvel, dtype = np.float32),
                              np.array(self.data.joint("upper_joint").qpos, dtype = np.float32),
                              np.array(self.data.joint("upper_joint").qvel,dtype = np.float32),
                              np.array(self.data.joint("lower_joint").qpos,dtype = np.float32),
                              np.array(self.data.joint("lower_joint").qvel,dtype = np.float32)), axis=0)
        return obs
