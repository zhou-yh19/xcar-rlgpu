import numpy as np
import os
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import yaml
import gym
import time
import random
import csv
from pathlib import Path
from IWDCarDynamics import IWDCarDynamics

class GPUVectorizedCarEnv:
    def __init__(self,
        preset_name,
        n,
        dt=0.01,
        solver="euler",
        device="cuda:0",
        drivetrain="iwd",
        disturbance_param=None,
        randomize_param={},
        random_seed=None,
        initial_state=None,
        **kwargs,
    ):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.num_states = 6
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.state_space = self.observation_space
        self.drivetrain = drivetrain
        if drivetrain == "4wd":
            self.num_actions = 2
            self.action_space = gym.spaces.Box(low=np.array([-0.45, 0.]), high=np.array([0.45, 7.]), shape=(2,))
            self.cast_action = lambda u: torch.cat(list(map(lambda v: torch.unsqueeze(v, 1), [u[:, 0], u[:, 1], u[:, 1], u[:, 1], u[:, 1]])), 1)
        elif drivetrain == "2iwd":
            self.num_actions = 3
            self.action_space = gym.spaces.Box(low=np.array([-0.45, 0., 0.]), high=np.array([0.45, 7., 7.]), shape=(3,))
            self.cast_action = lambda u: torch.cat(list(map(lambda v: torch.unsqueeze(v, 1), [u[:, 0], u[:, 1], u[:, 1], u[:, 2], u[:, 2]])), 1)
        elif drivetrain == "iwd":
            self.num_actions = 5
            self.action_space = gym.spaces.Box(low=np.array([-0.45, 0., 0., 0., 0.]), high=np.array([0.45, 7., 7., 7., 7.]), shape=(5,))
            self.cast_action = lambda u: u

        self.preset_name = preset_name
        self.n = n
        self.dt = dt
        self.solver = solver
        self.device = torch.device(device)
        file_path = os.path.dirname(__file__)
        with open(os.path.join(file_path, "presets.yaml")) as f:
            presets = yaml.safe_load(f)
            params = presets[preset_name]["parameters"]
        self.p_body = torch.zeros((n, 8), device=self.device)
        self.p_body[:, 0] = params["lF"]
        self.p_body[:, 1] = params["lR"]
        self.p_body[:, 2] = params["m"]
        self.p_body[:, 3] = params["h"]
        self.p_body[:, 4] = params["g"]
        self.p_body[:, 5] = params["Iz"]
        self.p_body[:, 6] = params["T"]
        self.p_tyre = torch.zeros((n, 4), device=self.device)
        self.p_tyre[:, 0] = params["B"]
        self.p_tyre[:, 1] = params["C"]
        self.p_tyre[:, 2] = params["D"]
        self.p_tyre[:, 3] = params["E"]
        self.randomize_param = randomize_param
        self.initial_state = initial_state
        self.s = self.initial_state if initial_state is not None else torch.zeros((n, 6), device=self.device)
        self.current_obs = None
        self.dynamics = None
        self.disturbance_param = disturbance_param
        if disturbance_param is not None:
            self.disturbance = torch.zeros((self.n, 8), device=self.device)
        self.step_count = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.total_step_count = 0
        self.saved_data = []
        self.train = kwargs.get("train", False)

    def randomize_item_(self, env_mask, override, key, target):
        num = int(torch.sum(env_mask).item())
        if key in override:
            target[env_mask] = override[key]
        elif key in self.randomize_param:
            lo, hi = self.randomize_param[key]
            target[env_mask] = lo + (hi - lo) * torch.rand(num, device=self.device)

    def randomize_items_(self, env_mask, override):
        rand_item_ = lambda key, target: self.randomize_item_(env_mask, override, key, target)
        rand_item_("B", self.p_tyre[:, 0])
        rand_item_("C", self.p_tyre[:, 1])
        rand_item_("D", self.p_tyre[:, 2])

    def randomize(self, env_mask=None, override={}):
        if env_mask is None:
            env_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.randomize_items_(env_mask, override)

    def obs(self):
        return self.s

    def reward(self):
        return torch.zeros(self.n, device=self.device)

    def done(self):
        return torch.zeros(self.n, device=self.device)

    def info(self):
        return {}

    def get_number_of_agents(self):
        return self.n

    def disturbed_dynamics(self):
        a, w = self.disturbance_param
        self.disturbance = a * self.disturbance + w * torch.randn((self.n, 8), device=self.device)
        return IWDCarDynamics(self.cast_action(self.u), self.p_body, self.p_tyre, disturbance=self.disturbance)

    def reset(self):
        self.randomize()
        self.s = self.initial_state if self.initial_state is not None else torch.zeros((self.n, 6), device=self.device)
        self.u = torch.zeros((self.n, self.num_actions), device=self.device)
        self.dynamics = IWDCarDynamics(self.cast_action(self.u), self.p_body, self.p_tyre)
        self.es = self.dynamics.compute_extended_state(self.s)
        self.step_count = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.total_step_count = 0
        self.saved_data = []
        obs = self.obs()
        self.current_obs = obs
        return obs

    def step(self, u, override_s=None):
        self.u = u

        self.dynamics = IWDCarDynamics(self.cast_action(u), self.p_body, self.p_tyre) if self.disturbance_param is None else self.disturbed_dynamics()
        
        if override_s is None:
            self.s = odeint(self.dynamics, self.s, torch.tensor([0., self.dt], device=self.device), method=self.solver)[1, :, :]
        else:
            self.s[:] = torch.tensor(override_s).unsqueeze(0)
        
        self.es = self.dynamics.compute_extended_state(self.s)
        self.step_count += 1
        self.total_step_count += 1
        obs = self.obs()
        if self.train:
            reward = self.reward()
        else:
            reward = torch.zeros(self.n, device=self.device)
        done = self.done()
        info = self.info()
        self.current_obs = obs
        if torch.all(done) and self.saved_data:
            basename = time.strftime("%Y%m%d-%H%M%S")
            Path("data").mkdir(parents=True, exist_ok=True)
            with open(os.path.join("data", basename + ".csv"), 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for i in range(len(self.saved_data)):
                    s = self.saved_data[i][0]
                    u = self.cast_action(self.saved_data[i][1].unsqueeze(0)).squeeze(0)
                    es = self.saved_data[i][2]
                    x, y, psi = s[0].item(), s[1].item(), s[2].item()
                    x_dot, y_dot, psi_dot = s[3].item(), s[4].item(), s[5].item()
                    delta, omega_fr, omega_fl, omega_rr, omega_rl = u[0].item(), u[1].item(), u[2].item(), u[3].item(), u[4].item()
                    r = es[2].item()
                    beta = es[7].item()
                    v = es[6].item()
                    writer.writerow([x, y, psi, x_dot, y_dot, psi_dot, delta, omega_fr, omega_fl, omega_rr, omega_rl, r, beta, v])
            torch.save(self.saved_data, os.path.join("data", basename + ".pth"))
            print("Total steps:", self.total_step_count)
            exit(0)
        return obs, reward, done, info

    def render(self, **kwargs):
        """Save rollout data to emulate rendering."""
        self.saved_data.append((self.s[0, :].cpu(), self.u[0, :].cpu(), self.es[0, :].cpu(), self.current_obs[0, :].cpu()))

    def detach(self):
        """Clear the gradient stored in the current state of the environment."""
        self.s = self.s.detach()
        self.u = self.u.detach()
        self.es = self.es.detach()

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    import time

    # Compare solvers
    env_euler = GPUVectorizedCarEnv("xcar", 1, solver="euler", drivetrain="iwd")
    env_rk = GPUVectorizedCarEnv("xcar", 1, solver="dopri5", drivetrain="iwd")
    
    traj_euler = [env_euler.reset().cpu().numpy()]
    traj_rk = [env_rk.reset().cpu().numpy()]
    
    # Test trajectory with varying inputs
    for i in range(500):
        if i < 100:
            u = [0., 2., 2., 2., 2.]  # Straight acceleration
        elif i < 200:
            u = [0.4, 4., 3., 4., 3.]  # Left turn with differential speeds
        elif i < 300:
            u = [-0.4, 3., 4., 3., 4.]  # Right turn with differential speeds
        elif i < 400:
            u = [0., 4., -4., 4., -4.]  # Spin in place
        else:
            u = [0.2, 3., 3., 3., 3.]  # Gentle right turn
            
        s_euler, _, _, _ = env_euler.step(torch.tensor([u], device=torch.device("cuda:0")))
        s_rk, _, _, _ = env_rk.step(torch.tensor([u], device=torch.device("cuda:0")))
        traj_euler.append(s_euler.cpu().numpy())
        traj_rk.append(s_rk.cpu().numpy())
        print(f"Step {i} done")

    # Plot trajectories
    plt.figure(dpi=300)
    plt.plot([s[0][0] for s in traj_euler], [s[0][1] for s in traj_euler], label="Euler")
    plt.plot([s[0][0] for s in traj_rk], [s[0][1] for s in traj_rk], label="RK5")
    plt.legend()
    plt.axis("equal")
    plt.title("Half-DOF Car Model Trajectories")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.savefig("half_dof_car_trajectories.png")

    plt.figure(dpi=300)
    plt.plot([s[0][2] for s in traj_euler], label="Euler")
    plt.plot([s[0][0] for s in traj_rk], [s[0][1] for s in traj_rk], label="RK5")
    plt.legend()
    plt.title("Half-DOF Car Model Trajectories")
    plt.xlabel("Time step")
    plt.ylabel("Psi")
    plt.savefig("half_dof_car_psi.png")
    