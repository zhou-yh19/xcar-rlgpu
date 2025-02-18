import os
import sys
import gym
import torch
import time
import math
import csv
from pathlib import Path
import numpy as np
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../xcar-simulation"))
sys.path.append(os.path.join(file_path, "../utils"))
from gpu_vectorized_car_env import GPUVectorizedCarEnv
from generate_segment_racetrack import TrajGen

class ContinuousDriftIWDEnv(GPUVectorizedCarEnv):
    def __init__(self, preset_name, n, device, ref_mode="hybrid", initial_state=None, **kwargs):
        # Initialize the parent class with specific parameters
        super().__init__(preset_name, n, device=device, drivetrain="iwd", initial_state=initial_state, **kwargs)

        # Set environment parameters
        self.max_steps = 3000
        self.num_states = 31
        self.quiet = kwargs.get("quiet", False)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
        self.num_actions = 5
        self.action_space = gym.spaces.Box(low=np.array([-0.46, 1., 1., 1., 1.]), high=np.array([0.46, 7., 7., 7., 7.]), shape=(self.num_actions,))
        self.state_space = self.observation_space

        dummy_obs = torch.zeros((self.n, self.num_states), device=self.device)
        self.recent_obs = [dummy_obs, dummy_obs]  # most recent first

        self.train = kwargs.get("train", False)

        if ref_mode == "hybrid":
            self.num_waypoints = 10000
            path_length = 50
            self.trajectory_generator = TrajGen(self.num_waypoints, path_length, device=device)
            self.waypoints = self.trajectory_generator.generate_random_waypoints(self.n)
        # circle trajectory
        elif ref_mode == "circle":
            self.num_waypoints = 2512      # one waypoint per 0.005m
            radius = 1.
            waypoint_x = radius * torch.sin(torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi)
            waypoint_y = radius - radius * torch.cos(torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi)
            waypoint_veldir = torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi
            waypoint_veldir = torch.atan2(torch.sin(waypoint_veldir), torch.cos(waypoint_veldir))
            waypoint_betaref = -torch.ones((self.num_waypoints, ))
            waypoint_kappa = torch.ones((self.num_waypoints, )) / radius
            self.waypoints = torch.stack((waypoint_x, waypoint_y, waypoint_veldir, waypoint_betaref, waypoint_kappa), dim=1).to(self.device)
            self.waypoints = self.waypoints.unsqueeze(0).expand(self.n, -1, -1)
        # eight-shaped trajectory
        elif ref_mode == "eight":
            self.num_waypoints = 2512      # one waypoint per 0.005m
            radius = 1.
            waypoint_x = radius * torch.sin(torch.arange(self.num_waypoints) / self.num_waypoints * 4 * np.pi)
            waypoint_y = torch.cat((
                radius - radius * torch.cos(torch.arange(self.num_waypoints // 2) / self.num_waypoints * 4 * np.pi),
                -radius + radius * torch.cos(torch.arange(self.num_waypoints // 2) / self.num_waypoints * 4 * np.pi),
            ))
            waypoint_veldir = torch.cat((
                torch.arange(self.num_waypoints // 2) / self.num_waypoints * 2,
                self.num_waypoints // 2 - torch.arange(self.num_waypoints // 2) / self.num_waypoints * 2,
            )) * 2 * np.pi
            waypoint_veldir = torch.atan2(torch.sin(waypoint_veldir), torch.cos(waypoint_veldir))
            waypoint_betaref = torch.cat((
                -torch.ones((self.num_waypoints // 2, )),
                torch.ones((self.num_waypoints // 2, )),
            ))
            waypoint_kappa = torch.cat((
                torch.ones((self.num_waypoints // 2, )) / radius,
                -torch.ones((self.num_waypoints // 2, )) / radius,
            ))
            self.waypoints = torch.stack((waypoint_x, waypoint_y, waypoint_veldir, waypoint_betaref, waypoint_kappa), dim=1).to(self.device)
            self.waypoints = self.waypoints.unsqueeze(0).expand(self.n, -1, -1)
        elif ref_mode == "three":
            path_length = 12 * torch.pi
            self.num_waypoints = int(path_length / 0.005)

            traj_gen = TrajGen(num_waypoints=self.num_waypoints, path_length=path_length, device=self.device)

            segment_length = torch.tensor([(1 + 1 / 12) * 2 * torch.pi, (1 + 5 / 6) * 2 * torch.pi, (1 + 1 / 6) * 2 * torch.pi, (1 + 11 / 12) * 2 * torch.pi], device=self.device)

            curvature = torch.tensor([1, -1, 1, -1], device=self.device)

            self.waypoints = traj_gen.generate_custom_waypoints(segment_length, curvature)
        elif ref_mode == "olympic":
            path_length = 18 * torch.pi
            self.num_waypoints = int(path_length / 0.005)

            traj_gen = TrajGen(num_waypoints=self.num_waypoints, path_length=path_length, device=self.device)

            segment_length = torch.tensor([(1 + 1 / 4) * 2 * torch.pi, (1 + 5 / 6) * 2 * torch.pi, (1 + 1 / 3) * 2 * torch.pi, (1 + 2 / 3) * 2 * torch.pi, (1 + 1 / 6) * 2 * torch.pi, (1 + 3 / 4) * 2 * torch.pi], device=self.device)

            curvature = torch.tensor([1, -1, 1, -1, 1, -1], device=self.device)

            self.waypoints = traj_gen.generate_custom_waypoints(segment_length, curvature)
        elif ref_mode == "variable_curvature":
            circle_length_small_start = 4.5 * torch.pi
            circle_length_small_mid = 5 * torch.pi
            circle_length_small_end = 4.5 * torch.pi
            sin_length = math.asin(1 / 2) * 8
            circle_length_large = (torch.pi - (math.asin(1 / 2) + math.sqrt(3) / 2 - 1) * 8)  * 2 * 2

            path_length = circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length + circle_length_small_end
            self.num_waypoints = int(path_length / 0.005)

            traj_gen = TrajGen(num_waypoints=self.num_waypoints, path_length=path_length, device=self.device)
            self.waypoints = traj_gen.generate_variable_curvature_waypoints()
            self.waypoints = self.waypoints.expand(self.n, -1, -1)
        else:
            raise ValueError("Invalid ref_mode")

        if not self.train:
            basename = time.strftime("%Y%m%d-%H%M%S")
            Path("data").mkdir(parents=True, exist_ok=True)
            with open(os.path.join("data", "traj-" + basename + ".csv"), 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for i in range(self.num_waypoints):
                    x, y, psi = self.waypoints[0, i, 0].item(), self.waypoints[0, i, 1].item(), self.waypoints[0, i, 2].item()
                    writer.writerow([x, y, psi])

        self.progress = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.step_progress = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)   # 0 = not done, 1 = failed, 2 = succeeded
        self.no_progress_count = torch.zeros(self.n, dtype=torch.uint8, device=self.device)

    def update_recent_obs(self, obs):
        self.recent_obs[1] = self.recent_obs[0]
        self.recent_obs[0] = obs

    def find_nearest_waypoint(self, x, y, veldir, beta, kappa):
        # Look ahead 20 steps
        if self.train:
            ind = torch.stack([self.progress + i for i in range(20)], dim=1)
            ind[ind >= self.num_waypoints] = self.num_waypoints - 1
        else:
            ind = torch.stack([self.progress + i for i in range(20)], dim=1) % self.num_waypoints
        expanded_index = ind.unsqueeze(-1).expand(-1, -1, 5)
        waypoints_filtered = torch.gather(self.waypoints, 1, expanded_index)
        current = torch.stack((x, y, veldir, beta, kappa), dim=1)
        diff = current.unsqueeze(1) - waypoints_filtered
        # wrap angle diff to (-pi, pi)
        angle_diff = torch.atan2(torch.sin(diff[:, :, 2]), torch.cos(diff[:, :, 2]))
        pos_diff = torch.hypot(diff[:, :, 0], diff[:, :, 1])
        beta_diff = torch.atan2(torch.sin(diff[:, :, 3]), torch.cos(diff[:, :, 3]))
        kappa_diff = diff[:, :, 4]
        self.step_progress = torch.argmin(pos_diff, 1)
        self.progress = (self.progress + self.step_progress)
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
        self.beta_ref = waypoints_filtered[:, :, 3].gather(1, best).squeeze(1)
        self.kappa_diff = kappa_diff.gather(1, best).squeeze(1)
        self.kappa_ref = waypoints_filtered[:, :, 4].gather(1, best).squeeze(1)

    def obs(self):
        x = self.s[:, 0]
        y = self.s[:, 1]
        psi = self.s[:, 2]
        psi = torch.atan2(torch.sin(psi), torch.cos(psi))
        r = self.es[:, 2]
        beta = self.es[:, 7]
        v = self.es[:, 6]
        vfrx = self.es[:, 8]
        vflx = self.es[:, 9]
        last_delta = self.u[:, 0]
        last_omegafr = self.u[:, 1]
        last_omegafl = self.u[:, 2]
        last_omegarr = self.u[:, 3]
        last_omegarl = self.u[:, 4]
        kappa = r / (v + 1e-6)

        xd = self.s[:, 3]
        yd = self.s[:, 4]
        veldir = torch.atan2(yd, xd)
        veldir[torch.hypot(xd, yd) < 0.01] = psi[torch.hypot(xd, yd) < 0.01]

        self.find_nearest_waypoint(x, y, veldir, beta, kappa)
        angle_diff = self.angle_diff
        signed_pos_diff = self.signed_pos_diff
        beta_diff = self.beta_diff
        beta_ref = self.beta_ref
        kappa_diff = self.kappa_diff
        kappa_ref = self.kappa_ref

        # look ahead 3m
        if self.train:
            la_inds = [(self.progress + i).unsqueeze(-1)for i in (200, 400, 600)]
            for la_ind in la_inds:
                la_ind[la_ind >= self.num_waypoints] = self.num_waypoints - 1
        else:
            la_inds = [(self.progress + i).unsqueeze(-1) % self.num_waypoints for i in (200, 400, 600)]
        la = [torch.gather(self.waypoints, 1, la_ind.unsqueeze(-1).expand(-1, -1, 5)).squeeze(1) for la_ind in la_inds]
        def convert(waypoints):
            """Cast into current reference frame"""
            x1 = waypoints[:, 0]
            y1 = waypoints[:, 1]
            veldir1 = waypoints[:, 2]
            beta1 = waypoints[:, 3]
            kappa1 = waypoints[:, 4]
            theta = torch.atan2(y1 - y, x1 - x) - veldir
            l = torch.hypot(y1 - y, x1 - x)
            x2 = l * torch.cos(theta)
            y2 = l * torch.sin(theta)
            veldir2 = veldir1 - veldir
            veldir2 = torch.atan2(torch.sin(veldir2), torch.cos(veldir2))
            beta2 = beta1 - beta
            beta2 = torch.atan2(torch.sin(beta2), torch.cos(beta2))
            x2, y2, veldir2, beta2, kappa1 = map(lambda t: torch.unsqueeze(t, 1), [x2, y2, veldir2, beta2, kappa1])
            return torch.cat([x2, y2, veldir2, beta2, kappa1], 1)
        la_converted = [convert(waypoints) for waypoints in la]

        failed = (self.progress >= 5) & ((self.no_progress_count >= 20) | (torch.abs(angle_diff) > 1.) | (torch.abs(signed_pos_diff) > 1.))
        if self.train:
            succeeded = (self.step_count >= self.max_steps) | (self.progress >= self.num_waypoints - 100)
        else:
            succeeded = (self.step_count >= self.max_steps)
        self.is_done[failed] = 1
        self.is_done[succeeded] = 2

        r, beta, v, kappa, vfrx, vflx, last_delta, last_omegafr, last_omegafl, last_omegarr, last_omegarl, angle_diff, signed_pos_diff, beta_diff, kappa_ref, prog, is_failed = map(lambda t: torch.unsqueeze(t, 1), [r, beta, v, kappa, vfrx, vflx, last_delta, last_omegafr, last_omegafl, last_omegarr, last_omegarl, angle_diff, signed_pos_diff, beta_diff, kappa_ref, self.step_progress, failed.to(dtype=torch.float32)])

        obs_list = [r, beta, v, kappa, vfrx, vflx, last_delta,  last_omegafr, last_omegafl, last_omegarr, last_omegarl, angle_diff, signed_pos_diff, beta_diff, kappa_ref, prog]
        obs_list.extend(la_converted)
        obs = torch.cat(obs_list, 1)

        return obs

    def reward(self):
        obs = self.recent_obs[0]
        last_obs = self.recent_obs[1]

        beta = obs[:, 1]
        v = obs[:, 2]
        kappa = obs[:, 3]
        vfrx = obs[:, 4]
        vflx = obs[:, 5]
        delta = obs[:, 6]
        omegafr = obs[:, 7]
        omegafl = obs[:, 8]
        omegarr = obs[:, 9]
        omegarl = obs[:, 10]
        last_delta = last_obs[:, 6]
        last_omegafr = last_obs[:, 7]
        last_omegafl = last_obs[:, 8]
        last_omegarr = last_obs[:, 9]
        last_omegarl = last_obs[:, 10]
        angle_diff = obs[:, 11]
        signed_pos_diff = obs[:, 12]
        beta_diff = obs[:, 13]
        kappa_ref = obs[:, 14]
        prog = obs[:, 15]
        is_failed = obs[:, 16]

        rew_pos = - signed_pos_diff ** 2
        rew_dir = - angle_diff ** 2
        rew_prog = torch.clamp(prog, 0.0, 5.0)
        rew_smooth = - ((delta - last_delta) ** 2 + 1e-2 * ((omegafr - last_omegafr) ** 2 + (omegafl - last_omegafl) ** 2 + (omegarr - last_omegarr) ** 2 + (omegarl - last_omegarl) ** 2))
        rew_beta = - beta_diff ** 2
        rew_lowspeed = torch.clamp(v, 0., 0.5) - 0.5
        rew_sf = - (vfrx - omegafr) ** 2 - (vflx - omegafl) ** 2
        rew_kappa = -(kappa - kappa_ref) ** 2
        rew_final = -is_failed

        rew = 2.4 * rew_pos + 0.5 * rew_dir + 0.2 * rew_prog + 0.015 * rew_smooth + 1.6 * rew_beta + 0.1 * rew_lowspeed + 0.005 * rew_sf + 0.15 * rew_kappa
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
        self.waypoints[is_done] = self.trajectory_generator.generate_random_waypoints(is_done.sum().item())
        self.progress[is_done] = torch.randint(self.num_waypoints // 2, (size,), device=self.device)
        self.step_progress[is_done] = 0
        self.no_progress_count[is_done] = 0

        def gen_random_state(indices):
            random_indices = torch.bernoulli(0.8 * torch.ones((len(indices),), device=self.device))
            self.progress[indices] = self.progress[indices] * random_indices.bool()
            prog = self.progress[indices]
            waypoints = self.waypoints[indices, prog, :]

            x = waypoints[:, 0]
            y = waypoints[:, 1]
            veldir = waypoints[:, 2]
            beta = waypoints[:, 3]
            kappa = waypoints[:, 4]

            x += 0.1 * torch.randn_like(x)
            y += 0.1 * torch.randn_like(y)

            random_V = torch.rand(prog.shape, device=self.device) * 3
            random_beta = torch.sign(beta) * torch.rand(prog.shape, device=self.device)

            psi = veldir - random_beta
            psi += 0.1 * torch.randn_like(psi)

            xd = random_V * torch.cos(random_beta + psi)
            yd = random_V * torch.sin(random_beta + psi)
            psid = -torch.sign(beta) * (1 + torch.rand(prog.shape, device=self.device) * 2)

            x, y, psi, xd, yd, psid = map(lambda t: torch.unsqueeze(t, 1), [x, y, psi, xd, yd, psid])
            return torch.cat([x, y, psi, xd, yd, psid], 1) * random_indices.unsqueeze(1)

        self.s[is_done, :] = gen_random_state(is_done.nonzero().squeeze(-1))
        self.u[is_done, :] = 0
        self.randomize(is_done)
        self.is_done[:] = 0

    def step(self, action, **kwargs):
        if self.train:
            self.reset_done_envs()
        obs, reward, done, info = super().step(action, **kwargs)
        self.update_recent_obs(obs)
        return obs, reward, done, info
