import os
import sys
import gym
import torch
import numpy as np
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../xcar-simulation"))
from gpu_vectorized_car_env import GPUVectorizedCarEnv

class EightDriftIWDEnv(GPUVectorizedCarEnv):
    def __init__(self, preset_name, n, device, **kwargs):
        super().__init__(preset_name, n, device=device, drivetrain="2iwd", **kwargs)
        self.max_steps = 2000

        self.num_states = 26
        self.num_actions = 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
        self.action_space = gym.spaces.Box(low=np.array([-0.46, 1., 1.]), high=np.array([0.46, 7., 7.]), shape=(self.num_actions,))
        self.state_space = self.observation_space

        dummy_obs = torch.zeros((self.n, self.num_states), device=self.device)
        self.recent_obs = [dummy_obs, dummy_obs]  # most recent first

        self.train = kwargs.get("train", False)
        self.quiet = kwargs.get("quiet", False)

        num_waypoints = 2400      # one waypoint per 0.005m
        radius = 1.
        waypoint_x = radius * torch.sin(torch.arange(num_waypoints) / num_waypoints * 4 * np.pi)
        waypoint_y = torch.cat((
            radius - radius * torch.cos(torch.arange(num_waypoints / 2) / num_waypoints * 4 * np.pi),
            -radius + radius * torch.cos(torch.arange(num_waypoints / 2) / num_waypoints * 4 * np.pi),
        ))
        waypoint_veldir = torch.cat((
            torch.arange(num_waypoints / 2) / num_waypoints * 2,
            num_waypoints / 2 - torch.arange(num_waypoints / 2) / num_waypoints * 2,
        )) * 2 * np.pi
        waypoint_veldir = torch.atan2(torch.sin(waypoint_veldir), torch.cos(waypoint_veldir))
        waypoint_betaref = torch.cat((
            -torch.arange(num_waypoints / 8) / num_waypoints * 8,
            -torch.ones((num_waypoints // 4, )),
            torch.arange(num_waypoints / 4) / num_waypoints * 8 - 1,
            torch.ones((num_waypoints // 4, )),
            1 - torch.arange(num_waypoints / 8) / num_waypoints * 8,
        ))
        self.waypoints = torch.stack((waypoint_x, waypoint_y, waypoint_veldir, waypoint_betaref), dim=1).to(self.device)

        self.progress = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.step_progress = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)   # 0 = not done, 1 = failed, 2 = succeeded
        self.no_progress_count = torch.zeros(self.n, dtype=torch.uint8, device=self.device)

    def update_recent_obs(self, obs):
        self.recent_obs[1] = self.recent_obs[0]
        self.recent_obs[0] = obs

    def find_nearest_waypoint(self, x, y, veldir, beta):
        # Look ahead 20 steps
        ind = torch.stack([self.progress + i for i in range(20)], dim=1) % len(self.waypoints)
        waypoints_filtered = self.waypoints[ind]
        current = torch.stack((x, y, veldir, beta), dim=1)
        diff = current.unsqueeze(1) - waypoints_filtered
        # wrap angle diff to (-pi, pi)
        angle_diff = torch.atan2(torch.sin(diff[:, :, 2]), torch.cos(diff[:, :, 2]))
        pos_diff = torch.hypot(diff[:, :, 0], diff[:, :, 1])
        beta_diff = torch.atan2(torch.sin(diff[:, :, 3]), torch.cos(diff[:, :, 3]))
        # total_diff = torch.abs(angle_diff) + pos_diff
        self.step_progress = torch.argmin(pos_diff, 1)
        self.progress = (self.progress + self.step_progress) % len(self.waypoints)
        self.no_progress_count[self.step_progress != 0] = 0
        self.no_progress_count[self.step_progress == 0] += 1

        # u = vector from nearest waypoint to current pos; v = velocity direction at nearest waypoint
        # theta = angle from u to v
        # theta > 0 means current pos is to the right of waypoint
        best = self.step_progress.unsqueeze(1)
        ux = diff[:, :, 0].gather(1, best).squeeze(1)
        uy = diff[:, :, 1].gather(1, best).squeeze(1)
        u_angle = torch.atan2(uy, ux)
        v_angle = waypoints_filtered[:, :, 2].gather(1, best).squeeze(1)
        theta = v_angle - u_angle
        theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        pos_diff_sign = torch.sign(theta)
        self.angle_diff = angle_diff.gather(1, best).squeeze(1)
        self.signed_pos_diff = pos_diff_sign * pos_diff.gather(1, best).squeeze(1)
        self.beta_diff = beta_diff.gather(1, best).squeeze(1)

    def obs(self):
        x = self.s[:, 0]
        y = self.s[:, 1]
        psi = self.s[:, 2]
        xd = self.s[:, 3]
        yd = self.s[:, 4]
        veldir = torch.atan2(yd, xd)
        r = self.es[:, 2]
        beta = self.es[:, 7]
        v = self.es[:, 6]
        vfx = (self.es[:, 8] + self.es[:, 9]) / 2
        last_delta = self.u[:, 0]
        last_omegaf = self.u[:, 1]
        last_omegar = self.u[:, 2]

        self.find_nearest_waypoint(x, y, veldir, beta)
        angle_diff = self.angle_diff
        signed_pos_diff = self.signed_pos_diff
        beta_diff = self.beta_diff

        # look ahead 3m
        la = [self.waypoints[(self.progress + i) % len(self.waypoints)] for i in (200, 400, 600)]
        def convert(waypoints):
            """Cast into current reference frame"""
            x1 = waypoints[:, 0]
            y1 = waypoints[:, 1]
            veldir1 = waypoints[:, 2]
            beta1 = waypoints[:, 3]
            theta = torch.atan2(y1 - y, x1 - x) - veldir
            l = torch.hypot(y1 - y, x1 - x)
            x2 = l * torch.cos(theta)
            y2 = l * torch.sin(theta)
            veldir2 = veldir1 - veldir
            cv2 = torch.cos(veldir2)
            sv2 = torch.sin(veldir2)
            beta2 = beta1 - beta
            beta2 = torch.atan2(torch.sin(beta2), torch.cos(beta2))
            x2, y2, cv2, sv2, beta2 = map(lambda t: torch.unsqueeze(t, 1), [x2, y2, cv2, sv2, beta2])
            return torch.cat([x2, y2, cv2, sv2, beta2], 1)
        la_converted = [convert(waypoints) for waypoints in la]

        failed = (self.no_progress_count >= 20) | (torch.abs(angle_diff) > 1.) | (torch.abs(signed_pos_diff) > 1.)
        succeeded = (self.step_count >= self.max_steps)
        self.is_done[failed] = 1
        self.is_done[succeeded] = 2

        r, beta, v, vfx, last_delta, last_omegaf, last_omegar, angle_diff, signed_pos_diff, beta_diff, prog, is_failed = map(lambda t: torch.unsqueeze(t, 1), [r, beta, v, vfx, last_delta, last_omegaf, last_omegar, angle_diff, signed_pos_diff, beta_diff, self.step_progress, failed.to(dtype=torch.float32)])

        obs_list = [r, beta, v, vfx, last_delta, last_omegaf, last_omegar, angle_diff, signed_pos_diff, beta_diff, prog]
        obs_list.extend(la_converted)
        obs = torch.cat(obs_list, 1)

        return obs

    def reward(self):
        obs = self.recent_obs[0]
        last_obs = self.recent_obs[1]

        v = obs[:, 2]
        vfx = obs[:, 3]
        delta = obs[:,4]
        omegaf = obs[:,5]
        omegar = obs[:, 6]
        last_delta = last_obs[:, 4]
        last_omegaf = last_obs[:, 5]
        last_omegar = last_obs[:, 6]
        angle_diff = obs[:, 7]
        signed_pos_diff = obs[:, 8]
        beta_diff = obs[:, 9]
        prog = obs[:, 10]

        rew_pos = - signed_pos_diff ** 2
        rew_dir = - angle_diff ** 2
        rew_prog = torch.clamp(prog, 0.0, 5.0)
        rew_smooth = - ((delta - last_delta) ** 2 + 1e-2 * ((omegaf - last_omegaf) ** 2 + (omegar - last_omegar) ** 2))
        rew_beta = - beta_diff ** 2
        rew_lowspeed = torch.clamp(v, 0., 0.5) - 0.5
        rew_sf = - (vfx - omegaf) ** 2
        rew = 2.5 * rew_pos + 0.5 * rew_dir + 0.2 * rew_prog + 0.02 * rew_smooth + 1 * rew_beta + 0.1 * rew_lowspeed + 0.01 * rew_sf

        return rew

    def done(self):
        return self.is_done

    def reset(self):
        super().reset()
        self.progress = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.step_progress = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self.no_progress_count = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        return self.obs()

    def info(self):
        info_dict = {
            "time_outs": (self.is_done == 2)
        }
        return info_dict

    def reset_done_envs(self):
        """Only reset envs that are already done."""
        is_done = self.is_done.bool()

        size = torch.sum(is_done)
        self.step_count[is_done] = 0
        self.progress[is_done] = torch.randint(self.waypoints.shape[0], (size,), device=self.device)
        self.step_progress[is_done] = 0
        self.no_progress_count[is_done] = 0
        def gen_random_state(prog):
            x = self.waypoints[prog, 0]
            y = self.waypoints[prog, 1]
            veldir = self.waypoints[prog, 2]
            beta = self.waypoints[prog, 3]
            v = torch.rand(prog.shape, device=self.device) * 3
            xd = v * torch.cos(veldir)
            yd = v * torch.sin(veldir)
            psi = veldir - beta
            psid = -torch.sign(beta) * (1 + torch.rand(prog.shape, device=self.device) * 2)

            x += 0.1 * torch.randn_like(x)
            y += 0.1 * torch.randn_like(y)
            psi += 0.1 * torch.randn_like(psi)

            random_beta = beta + 0.1 * torch.randn_like(beta)
            random_V = v

            xd = random_V * torch.cos(random_beta + psi)
            yd = random_V * torch.sin(random_beta + psi)

            x, y, psi, xd, yd, psid = map(lambda t: torch.unsqueeze(t, 1), [x, y, psi, xd, yd, psid])
            return torch.cat([x, y, psi, xd, yd, psid], 1)

        self.s[is_done, :] = gen_random_state(self.progress[is_done])
        self.u[is_done, :] = 0
        self.randomize(is_done)
        self.is_done[:] = 0

    def step(self, action, **kwargs):
        if self.train:
            self.reset_done_envs()
        obs, reward, done, info = super().step(action, **kwargs)
        self.update_recent_obs(obs)
        return obs, reward, done, info
