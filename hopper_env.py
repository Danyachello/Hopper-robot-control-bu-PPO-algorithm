import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os


# you can completely modify this class for your MuJoCo environment by following the directions
class HopperEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 250,
    }

    # set default episode_len for truncate episodes
    def __init__(self, episode_len=1000, healthy_reward = 1, control_cost_weight = 0.004,_forward_reward_weight = 1.0, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # change shape of observation to your observation space size
        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        # load your MJCF model with env and choose frames count between actions
        MujocoEnv.__init__(
            self,
            os.path.abspath("assets/my_hopper.xml"),
            4,
            observation_space=observation_space,
            **kwargs
        )
        self.step_number = 0
        self.episode_len = episode_len
        self.init_q = np.array([0, -0.72, 1.02])
        self._forward_reward_weight = _forward_reward_weight
        self.healthy_reward = healthy_reward
        self.control_cost_weight = control_cost_weight
    
    def control_cost(self, action):
        control_cost = self.control_cost_weight * np.sum(np.square(action))
        return control_cost
    
    def is_healthy(self, z_pos):
        is_healthy = z_pos > -1.0
        return is_healthy


    def terminated(self, z_pos):
        terminated = not self.is_healthy(z_pos)
        return terminated



    # determine the reward depending on observation or other properties of the simulation
    def step(self, a):
        z_position_before = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        z_position_after = self.data.qpos[0]
        self.step_number += 1
        z_velocity = (z_position_after - z_position_before) / self.dt

        ctrl_cost = self.control_cost(a)
        forward_reward = self._forward_reward_weight * z_velocity
        healthy_reward = float(self.is_healthy(z_position_after)*self.healthy_reward)

        rewards = forward_reward + healthy_reward
        reward = rewards - ctrl_cost

        info = {
            "z_position": z_position_after,
            "z_velocity": z_velocity
        }

        obs = self._get_obs()
        terminated = self.terminated(z_position_after)
        truncated = self.step_number > self.episode_len
        return obs, reward, terminated, truncated, info

    # define what should happen when the model is reset (at the beginning of each episode)

    def reset_model(self):
        self.step_number = 0

        # for example, noise is added to positions and velocities
        qpos = self.init_q 
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    # determine what should be added to the observation
    # for example, the velocities and positions of various joints can be obtained through their names, as stated here
    def _get_obs(self):

        obs = np.concatenate((np.array(self.data.joint("main_body").qpos),
                              np.array(self.data.joint("main_body").qvel),
                              np.array(self.data.joint("upper_joint").qpos),
                              np.array(self.data.joint("upper_joint").qvel),
                              np.array(self.data.joint("lower_joint").qpos),
                              np.array(self.data.joint("lower_joint").qvel)), axis=0)
        return obs
