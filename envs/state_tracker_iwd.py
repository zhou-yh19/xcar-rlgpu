import os
import sys
import gym
import torch
import numpy as np
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../xcar-simulation"))
from gpu_vectorized_car_env import GPUVectorizedCarEnv
from icecream import ic

class StateTrackerIWDEnv(GPUVectorizedCarEnv):

    def __init__(self, preset_name, n, device, ref_mode, **kwargs):
        super().__init__(preset_name, n, drivetrain="2iwd", device=device, **kwargs)
        self.min_steps = 1500

        self.num_states = 15
        self.num_actions = 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
        self.action_space = gym.spaces.Box(low=np.array([-0.46, 1., 1.]), high=np.array([0.46, 7., 7.]), shape=(3,))
        self.state_space = self.observation_space
        self.need_reset = False

        self.recent_obs = [None, None]  # most recent first
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)   # 0 = not done, 1 = failed, 2 = succeeded

        # Reference for r, beta, V
        self.ref = torch.zeros((self.n, 3), dtype=torch.float32, device=self.device)

        # Mode of reference generation: 0 = fixed (counterclockwise), 1 = fixed (clockwise), 2 = switching between clockwise and counter clockwise, 3 = ramp, 4 = sinusoidal
        self.current_ref_mode = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        ref_mode_to_int = {
            "fixed_circle_ccw": 0,
            "fixed_circle_cw": 1,
            "eight_drift": 2,
        }
        if ref_mode != "hybrid":
            self.current_ref_mode[:] = ref_mode_to_int[ref_mode]
        self.mode_probabilities = torch.tensor([0.2, 0.0, 0.6, 0.1, 0.1], dtype=torch.float32, device=self.device)

        # Parameters of reference generation; below are nominal values, but they can be randomized if mode is hybrid
        self.r_magnitude = 2. * torch.ones(self.n, dtype=torch.float32, device=self.device)
        self.beta_magnitude = torch.ones(self.n, dtype=torch.float32, device=self.device)
        self.v_magnitude = 2. * torch.ones(self.n, dtype=torch.float32, device=self.device)
        self.to_reverse_phase = torch.zeros(self.n, dtype=torch.float32, device=self.device)   # Reference generated in CCW first, then CW if this is zero; CW first, then CCW if this is 1; randomized at reset for balanced data in two directions
        self.period = 7. * torch.ones(self.n, dtype=torch.float32, device=self.device)   # period in seconds

        # Mode for scheduling how the the mode of reference generation is chosen:
        # - fixed_circle_ccw = fixed at mode 0 and value is nominal,
        # - fixed_circle_cw = fixed at mode 1 and normal is nominal,
        # - eight_drift = fixed at mode 2 and value is nominal,
        # - hybrid = mode is chosen randomly for every simulation instance and the value is randomized.
        # current_ref_mode is initialized using this parameter (see blocks above), and is randomized at reset if ref_mode is "hybrid".
        # Use "fixed_circle_ccw", "fixed_circle_cw" or "eight_drift" during test, but "hybrid" during training for better diversity of data
        self.ref_mode = ref_mode
        self.train = kwargs.get("train", False)

        # Record of reward, used for reward shaping
        self.last_reward = {
        }
        # self.gamma = gamma

        # Other parameters
        self.v_min = 0.5    # Minimum reference velocity and minimum velocity used for curvature computation
        self.quiet = kwargs.get("quiet", False)

        self.reset()

    def update_recent_obs(self, obs):
        self.recent_obs[1] = self.recent_obs[0]
        self.recent_obs[0] = obs

    def obs(self):
        # Current state
        r = self.es[:, 2]
        beta = self.es[:, 7]
        v = self.es[:, 6]
        kappa = r / v.clamp(self.v_min, None)

        # Reference
        r_ref = self.ref[:, 0]
        beta_ref = self.ref[:, 1]
        v_ref = self.ref[:, 2]
        kappa_ref = r_ref / v_ref

        # Last action (for smoothness)
        last_delta = self.u[:, 0]
        last_omega_f = self.u[:, 1]
        last_omega_r = self.u[:, 2]
        
        # vfx (for penalizing front slip)
        vfx = (self.es[:, 8] + self.es[:, 9]) / 2

        # Whether the current reference is piecewise constant; useful for deciding whether to minimize steady-state error
        is_steady = (self.current_ref_mode == 0) | (self.current_ref_mode == 1) | (self.current_ref_mode == 2)
        is_steady = is_steady.to(dtype=torch.float32)

        # Update done status: 1 for failure, 2 for timeout
        failed = (self.step_count >= 20) * ((torch.abs(beta - beta_ref) > 2.8) | (torch.abs(v - v_ref) > 2) | (torch.abs(r - r_ref) > 5.6))
        timeout = ((self.step_count >= self.min_steps) * torch.bernoulli(0.002 * torch.ones_like(self.step_count)).to(dtype=torch.long)).to(dtype=torch.bool)
        self.is_done[failed] = 1
        self.is_done[timeout] = 2

        # Add last dimension to list of observed variables
        r, beta, v, kappa, r_ref, beta_ref, v_ref, kappa_ref, last_delta, last_omega_f, last_omega_r, vfx, is_steady, is_failed, is_timeout = \
            map(lambda x: x.unsqueeze(1), [r, beta, v, kappa, r_ref, beta_ref, v_ref, kappa_ref, last_delta, last_omega_f, last_omega_r, vfx, is_steady, failed.to(dtype=torch.float32), timeout.to(dtype=torch.float32)])

        # Concatenate observed variables
        obs = torch.cat([r, beta, v, kappa, r_ref, beta_ref, v_ref, kappa_ref, last_delta, last_omega_f, last_omega_r, vfx, is_steady, is_failed, is_timeout], dim=1)

        self.update_recent_obs(obs)

        return obs

    def reward(self):
        # Read variables
        obs = self.recent_obs[0]
        r = obs[:, 0]
        beta = obs[:, 1]
        v = obs[:, 2]
        kappa = obs[:, 3]
        r_ref = obs[:, 4]
        beta_ref = obs[:, 5]
        v_ref = obs[:, 6]
        kappa_ref = obs[:, 7]
        is_steady = obs[:, 12]

        # Tracking error penalty of r, beta, v
        rew_r = -(r - r_ref) ** 2
        rew_beta = -(beta - beta_ref) ** 2
        rew_v = -(v - v_ref) ** 2

        # Penalize small steady-state beta and curvature error
        rew_beta_steady = is_steady * (-(beta - beta_ref).abs().clamp(0, 0.3))
        rew_kappa = is_steady * (-(kappa - kappa_ref).abs().clamp(0, 0.5))

        # Penalize front slip to encourage approximation of RWD behavior
        vfx = obs[:, 11]
        omega_f = self.u[:, 1]
        rew_sf = - (vfx - omega_f) ** 2

        # Penalty for stopping at a place
        rew_low_speed = torch.clamp(v, 0., self.v_min) - self.v_min

        # Penalize for oscillating action
        last_obs = self.recent_obs[1]
        last_obs_is_not_done = (last_obs[:, -1] == 0.) & (last_obs[:, -2] == 0.)
        delta = self.u[:, 0]
        omega_r = self.u[:, 2]
        last_delta = last_obs[:, 8]
        last_omega_f = last_obs[:, 9]
        last_omega_r = last_obs[:, 10]
        rew_smooth = last_obs_is_not_done * (-((delta - last_delta) ** 2 + 1e-2 * ((omega_f - last_omega_f) ** 2 + (omega_r - last_omega_r) ** 2)))

        # Penalty for failure
        is_failed = obs[:, -2]
        rew_final = -is_failed

        # Define coefficients and compute weighted sum
        # name -> (coefficient, value)
        reward_dict = {
            "rew_alive": (3., torch.ones(self.n, dtype=torch.float32, device=self.device)),
            "rew_r": (1., rew_r),
            "rew_beta": (2., rew_beta),
            "rew_v": (1., rew_v),
            "rew_beta_steady": (2., rew_beta_steady),
            "rew_kappa": (1., rew_kappa),
            "rew_sf": (0.1, rew_sf),
            "rew_smooth": (0.01, rew_smooth),
            "rew_low_speed": (0.1, rew_low_speed),
            "rew_final": (50., rew_final),
        }

        rew_total = sum([coeff * value for (coeff, value) in reward_dict.values()])

        if not self.quiet:
            reward = {f"avg_weighted_{name}": (coeff * value).mean().item() for (name, (coeff, value)) in reward_dict.items()}
            reward["avg_total"] = rew_total.mean().item()
            ic(reward)

        return rew_total

    def done(self):
        return self.is_done

    def reset(self):
        super().reset()
        # Set all simulations to timeout to reset
        self.is_done = 2 * torch.ones(self.n, dtype=torch.uint8, device=self.device)
        self.reset_done_envs()
        return self.obs()

    def info(self):
        return {
            "time_outs": (self.is_done == 2)
        }

    def randomize_current_ref_mode(self, indices):
        """
        Randomize the current reference mode for the simulation instances of given indices.

        Args:
            indices (torch.Tensor): Indices of simulation instances to randomize.
        """
        if self.ref_mode != "hybrid":
            # Reference mode is fixed, no need to randomize
            return

        size = len(indices)
        if size == 0:
            return

        # Randomize current reference mode
        self.current_ref_mode[indices] = torch.multinomial(self.mode_probabilities, size, replacement=True).to(dtype=torch.uint8)

        # Randomize magnitude of r between 1.5 and 2.5
        self.r_magnitude[indices] = 1.5 + torch.rand((size,), device=self.device)
        # Randomize magnitude of beta between 0.8 and 1.2
        self.beta_magnitude[indices] = 0.8 + 0.4 * torch.rand((size,), device=self.device)
        # Randomize magnitude of v between 1.5 and 2.5
        self.v_magnitude[indices] = 1.5 + torch.rand((size,), device=self.device)
        # Randomize period between 5s and 9s
        self.period[indices] = 5. + 4. * torch.rand((size,), device=self.device)
        # Randomize in which direction to start the periodic reference
        self.to_reverse_phase[indices] = torch.bernoulli(0.5 * torch.ones((size,), device=self.device))

    def randomize_initial_speed(self, indices):
        """
        Randomize the initial speed for the simulation instances of given indices.

        Args:
            indices (torch.Tensor): Indices of simulation instances to randomize.
        """
        # Pose is fixed at (0, 0, 0) in initialization, so only need to randomize \dot{x}
        # Randomize 80% samples in range [0, 2], and 20% samples at 0 (to train starting)
        random_indices = torch.bernoulli(0.5 * torch.ones((len(indices),), device=self.device))
        random_r = (2 * torch.rand((len(indices),), device=self.device) + 1) * random_indices
        random_beta = (-0.5 * torch.rand((len(indices),), device=self.device) - 1) * random_indices
        random_V = (torch.rand((len(indices),), device=self.device) + 1) * random_indices
        self.s[indices, 3] = random_V * torch.cos(random_beta)
        self.s[indices, 4] = random_V * torch.sin(random_beta)
        self.s[indices, 5] = random_r

        reverse_indices = (self.to_reverse_phase[indices] == 1)
        self.s[indices[reverse_indices], 4] = -self.s[indices[reverse_indices], 4]
        self.s[indices[reverse_indices], 5] = -self.s[indices[reverse_indices], 5]

    def reset_done_envs(self):
        """Only reset envs that are already done."""
        is_done = self.is_done.bool()
        size = torch.sum(is_done)
        self.step_count[is_done] = 0
        self.s[is_done, :] = 0.
        self.u[is_done, :] = 0.
        indices_to_randomize = torch.arange(self.n, device=self.device)[is_done]
        self.randomize_current_ref_mode(indices_to_randomize)
        if self.train:
            self.randomize_initial_speed(is_done.nonzero().squeeze(-1))
        self.is_done[:] = 0
        self.update_reference()

    @staticmethod
    def generate_square(low, high, phase):
        """
        Generates a square wave signal with high value in first half period, and low value in second half period, based on the phase, in batch.
        
        Args:
            high (torch.Tensor): The highest value of the signal.
            low (torch.Tensor): The lowest value of the signal.
            phase (torch.Tensor): A tensor representing the phase in [0, 2 * pi].
        
        Returns:
            torch.Tensor: A tensor of the same shape as input tensors, containing the signal values.
        """
        normalized_phase = phase / (2 * np.pi)
        signal = torch.where(
            normalized_phase < 0.5,
            high,
            low
        )
        return signal

    @staticmethod
    def generate_ramp(low, high, phase):
        """
        Generates a ramp signal aligned similarly to a sine wave based on the phase, in batch.
        
        Args:
            high (torch.Tensor): The highest value of the signal.
            low (torch.Tensor): The lowest value of the signal.
            phase (torch.Tensor): A tensor representing the phase in [0, 2 * pi].
        
        Returns:
            torch.Tensor: A tensor of the same shape as input tensors, containing the signal values.
        """
        mid = (high + low) / 2
        amplitude = (high - low) / 2
        normalized_phase = phase / (2 * np.pi)
        signal = torch.where(
            normalized_phase < 0.25,
            mid + 4 * amplitude * normalized_phase,
            torch.where(
                normalized_phase < 0.75,
                high - 4 * amplitude * (normalized_phase - 0.25),
                low + 4 * amplitude * (normalized_phase - 0.75),
            )
        )
        return signal

    @staticmethod
    def generate_sine(low, high, phase):
        """
        Generates a sine wave signal based on the phase, in batch.
        
        Args:
            high (torch.Tensor): The highest value of the signal.
            low (torch.Tensor): The lowest value of the signal.
            phase (torch.Tensor): A tensor representing the phase in [0, 2 * pi].
        
        Returns:
            torch.Tensor: A tensor of the same shape as input tensors, containing the signal values.
        """
        mid = (high + low) / 2
        amplitude = (high - low) / 2
        signal = mid + amplitude * torch.sin(phase)
        return signal

    def update_reference(self):
        """
        Update the current reference according to the current reference mode.
        """

        # Fixed circle counterclockwise: set r to +magnitude, beta to -magnitude, v to +magnitude
        mode_fixed_ccw_indices = (self.current_ref_mode == 0)
        self.ref[mode_fixed_ccw_indices, 0] = self.r_magnitude[mode_fixed_ccw_indices]
        self.ref[mode_fixed_ccw_indices, 1] = -self.beta_magnitude[mode_fixed_ccw_indices]
        self.ref[mode_fixed_ccw_indices, 2] = self.v_magnitude[mode_fixed_ccw_indices]

        # Fixed circle clockwise: set r to -magnitude, beta to +magnitude, v to +magnitude
        mode_fixed_cw_indices = (self.current_ref_mode == 1)
        self.ref[mode_fixed_cw_indices, 0] = -self.r_magnitude[mode_fixed_cw_indices]
        self.ref[mode_fixed_cw_indices, 1] = self.beta_magnitude[mode_fixed_cw_indices]
        self.ref[mode_fixed_cw_indices, 2] = self.v_magnitude[mode_fixed_cw_indices]

        # Generate reference for the periodic modes
        def generate_periodic_ref(indices, function, always_max_v=False):
            """
            Generate reference for the periodic modes.

            Args:
                indices (torch.Tensor): Indices of simulation instances to generate reference for.
                function (function): A function that takes in low, high and phase, and outputs a tensor of the same shape as low, high and phase.
                always_max_v (bool, optional): Whether to always set v to max. Defaults to False.
            """
            period = self.period[indices]
            raw_phase = torch.fmod(self.step_count[indices] * self.dt, period) * (2 * np.pi / period)   # Phase in period according to current time
            phase = torch.where(self.to_reverse_phase[indices] == 0, raw_phase, 2 * np.pi - raw_phase)   # Shift phase by 180 degrees if to_reverse_phase is 1
            beta_phase = torch.fmod(phase + np.pi, 2 * np.pi)   # Phase of beta is shifted by 180 degrees
            v_phase = torch.fmod(2 * phase + 1.5 * np.pi, 2 * np.pi)   # Period of v is half of r and beta, and phase starts at 270 degrees
            r_mag = self.r_magnitude[indices]
            beta_mag = self.beta_magnitude[indices]
            v_mag = self.v_magnitude[indices]
            self.ref[indices, 0] = function(-r_mag, r_mag, phase)
            self.ref[indices, 1] = function(-beta_mag, beta_mag, beta_phase)
            if always_max_v:
                self.ref[indices, 2] = v_mag
            else:
                self.ref[indices, 2] = function(self.v_min, v_mag, v_phase)

        # Switching between clockwise and counterclockwise (eight drift)
        mode_switching_indices = (self.current_ref_mode == 2)
        generate_periodic_ref(mode_switching_indices, self.generate_square, always_max_v=True)

        # Ramp
        mode_ramp_indices = (self.current_ref_mode == 3)
        generate_periodic_ref(mode_ramp_indices, self.generate_ramp)

        # Sinusoidal
        mode_sinusoidal_indices = (self.current_ref_mode == 4)
        generate_periodic_ref(mode_sinusoidal_indices, self.generate_sine)

    def step(self, action, switch=False, **kwargs):
        self.reset_done_envs()
        self.update_reference()
        obs, reward, done, info = super().step(action, **kwargs)
        return obs, reward, done, info

