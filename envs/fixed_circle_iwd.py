import os
import sys
import gym
import torch
import numpy as np
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../xcar-simulation"))
from gpu_vectorized_car_env import GPUVectorizedCarEnv

class FixedCircleIWDEnv(GPUVectorizedCarEnv):
    """Example 2IWD task of CCW circling around (0, 1) with radius 1 and sideslip angle -1."""
    def __init__(self, preset_name, n, device, **kwargs):
        super().__init__(preset_name, n, device=device, drivetrain="2iwd", **kwargs)
        self.max_steps = 1000

        self.num_states = 10
        self.num_actions = 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
        self.action_space = gym.spaces.Box(low=np.array([-0.46, 1., 1.]), high=np.array([0.46, 7., 7.]), shape=(3,))
        self.state_space = self.observation_space

        self.need_reset = False
        self.recent_obs = [None, None]  # most recent first
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)   # 0 = not done, 1 = failed, 2 = succeeded

        self.train = kwargs.get("train", False)
        self.radius = 1.

    def update_recent_obs(self, obs):
        self.recent_obs[1] = self.recent_obs[0]
        self.recent_obs[0] = obs

    def obs(self):
        x = self.s[:, 0]
        y = self.s[:, 1]
        psi = self.s[:, 2]
        xd = self.s[:, 3]
        yd = self.s[:, 4]
        psid = self.s[:, 5]
        center_dist = torch.hypot(x, y - self.radius)
        phase = torch.atan2(y - self.radius, x)
        tangent_dir = phase + torch.pi / 2
        tangent_dir = torch.atan2(torch.sin(tangent_dir), torch.cos(tangent_dir))
        veldir = torch.atan2(yd, xd)
        dir_diff = tangent_dir - veldir
        dir_diff = torch.atan2(torch.sin(dir_diff), torch.cos(dir_diff))

        r = self.es[:, 2]
        beta = self.es[:, 7]
        v = self.es[:, 6]
        vfx = (self.es[:, 8] + self.es[:, 9]) / 2

        failed = (torch.abs(dir_diff) > 0.7) | (torch.abs(center_dist - 1) > 0.5)

        # Use random stopping to increase diversity in batch in this env
        succeeded = ((self.step_count >= self.max_steps) * torch.bernoulli(0.002 * torch.ones_like(self.step_count)).to(dtype=torch.long)).to(dtype=torch.bool)

        self.is_done[failed] = 1
        self.is_done[succeeded] = 2

        last_delta = self.u[:, 0]
        last_omegaf = self.u[:, 1]
        last_omegar = self.u[:, 2]
        center_dist, dir_diff, r, beta, v, vfx, last_delta, last_omegaf, last_omegar, is_failed = map(lambda t: torch.unsqueeze(t, 1), [center_dist, dir_diff, r, beta, v, vfx, last_delta, last_omegaf, last_omegar, failed.to(dtype=torch.float32)])
        obs = torch.cat([center_dist, dir_diff, r, beta, v, vfx, last_delta, last_omegaf, last_omegar, is_failed], 1)

        self.update_recent_obs(obs)

        return obs

    def reward(self):
        obs = self.recent_obs[0]
        last_obs = self.recent_obs[1] if self.recent_obs[1] is not None else torch.zeros_like(obs)

        # Extracting values from obs
        center_dist = obs[:, 0]
        dir_diff = obs[:, 1]
        r = obs[:, 2]
        beta = obs[:, 3]
        v = obs[:, 4]
        vfx = obs[:, 5]
        delta = obs[:, 6]
        omegaf = obs[:, 7]
        omegar = obs[:, 8]
        is_failed = obs[:, 9]

        # Last timestep values
        last_delta = last_obs[:, 6]
        last_omegaf = last_obs[:, 7]
        last_omegar = last_obs[:, 8]

        # Reward components
        rew_beta = -(beta + 1.) ** 2
        rew_center_dist = -(center_dist - self.radius) ** 2 if self.recent_obs[1] is not None else torch.zeros_like(rew_beta)
        rew_dir_diff = -dir_diff ** 2 if self.recent_obs[1] is not None else torch.zeros_like(rew_beta)
        rew_smooth = -((delta - last_delta) ** 2 + 1e-2 * ((omegaf - last_omegaf) ** 2 + (omegar - last_omegar) ** 2)) if self.recent_obs[1] is not None else torch.zeros_like(rew_beta)
        rew_sf = -(vfx - omegaf) ** 2 if self.recent_obs[1] is not None else torch.zeros_like(rew_beta)

        # Penalize low speeds
        rew_lowspeed = torch.clamp(v, 0., 0.5) - 0.5
        rew_final = -is_failed

        # Total reward
        rew = 2. + 1.4 * rew_beta + 0.02 * rew_smooth + 0.1 * rew_lowspeed + 0.1 * rew_sf + 2.5 * rew_center_dist + 0.6 * rew_dir_diff + 50. * rew_final

        return rew

    def done(self):
        return self.is_done

    def reset(self):
        super().reset()
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        return self.obs()

    def info(self):
        return {
            "time_outs": (self.is_done == 2)
        }

    def reset_done_envs(self):
        """Only reset envs that are already done."""
        is_done = self.is_done.bool()
        size = torch.sum(is_done)
        self.step_count[is_done] = 0
        self.s[is_done, :] = 0
        self.u[is_done, :] = 0
        def gen_random_state(size):
            x = torch.zeros((size,1), device=self.device)
            y = self.radius * (torch.rand((size,1), device=self.device) - 0.5)
            psi = 0.2 * torch.randn((size,1), device=self.device)

            random_indices = torch.bernoulli(0.2 * torch.ones((size,1), device=self.device))
            random_r = (2 * torch.rand((size,1), device=self.device) + 1) * random_indices  
            random_beta = (-0.5 * torch.rand((size,1), device=self.device) - 1) * random_indices
            random_V = (torch.rand((size,1), device=self.device) + 1) * random_indices

            xd = random_V * torch.cos(random_beta + psi)
            yd = random_V * torch.sin(random_beta + psi)
            psid = random_r

            return torch.cat([x, y, psi, xd, yd, psid], 1)
        self.s[is_done, :] = gen_random_state(size)
        self.is_done[:] = 0

    def step(self, action, **kwargs):
        self.reset_done_envs()
        obs, reward, done, info = super().step(action, **kwargs)
        return obs, reward, done, info
