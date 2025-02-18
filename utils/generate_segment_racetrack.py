import torch
import math

class TrajGen:
    def __init__(self, num_waypoints: int = 4800, path_length: int = 24, device: str = "cuda"):
        self.num_waypoints = num_waypoints
        self.path_length = path_length
        self.device = device

    def generate_random_waypoints(self, n: int, theta0: float = 0.) -> torch.Tensor:
        # Waypoints along the path
        s = torch.linspace(0, self.path_length, self.num_waypoints, device=self.device)
        ds = s[1] - s[0]  # Step size

        num_scenes = n
        # Random segment lengths and curvature
        segment_length = torch.rand((num_scenes, self.path_length // 3), device=self.device) * 5.0 + 3.0

        threshold = torch.cumsum(segment_length, dim=1)  # (n, path_length)
        
        curvature = (torch.rand_like(threshold, device=self.device) * 0.36 + 0.7) * torch.sign(torch.randn_like(threshold))
        
        # We need to loop through the trajectories since torch.bucketize expects 1D boundaries per sample
        kappa = torch.zeros((num_scenes, self.num_waypoints), device=self.device)
        for i in range(num_scenes):
            kappa[i] = curvature[i][torch.bucketize(s, threshold[i])]
        
        # Integrate curvature to get heading angle theta
        theta = torch.cumsum(kappa * ds, dim=1)  # Numeric integration of kappa to get angle
        
        # Starting conditions
        x0, y0, theta0 = 0, 0, theta0
        
        # Compute x, y positions by integrating the heading angle
        x = torch.cumsum(torch.cos(theta + theta0) * ds, dim=1) + x0
        y = torch.cumsum(torch.sin(theta + theta0) * ds, dim=1) + y0
        
        # Velocity direction (theta) and sign of curvature (beta)
        vel_dir = torch.atan2(torch.sin(theta + theta0), torch.cos(theta + theta0))
        beta = - torch.sign(kappa)
        
        # Stack the output: [x, y, velocity direction, sign of curvature]
        waypoints = torch.stack((x, y, vel_dir, beta, kappa), dim=2)
        
        return waypoints
    
    def generate_custom_waypoints(self, segment_length: torch.Tensor, curvature: torch.Tensor, theta0: float = 0.) -> torch.Tensor:
        assert segment_length.shape[0] == curvature.shape[0], "segment_length and curvature must have the same number of scenes"

        # Waypoints along the path
        s = torch.linspace(0, self.path_length, self.num_waypoints, device=self.device)
        ds = s[1] - s[0] # Step size

        num_scenes = 1
        threshold = torch.cumsum(segment_length.unsqueeze(0), dim=1)
        curvature = curvature.unsqueeze(0)

        kappa = torch.zeros((num_scenes, self.num_waypoints), device=self.device)
        kappa[0] = curvature[0][torch.bucketize(s, threshold[0])]

        # Integrate curvature to get heading angle theta
        theta = torch.cumsum(kappa * ds, dim=1)  # Numeric integration of kappa to get angle

        # Starting conditions
        x0, y0, theta0 = 0, 0, theta0

        # Compute x, y positions by integrating the heading angle
        x = torch.cumsum(torch.cos(theta + theta0) * ds, dim=1) + x0
        y = torch.cumsum(torch.sin(theta + theta0) * ds, dim=1) + y0

        # Velocity direction (theta) and sign of curvature (beta)
        vel_dir = torch.atan2(torch.sin(theta + theta0), torch.cos(theta + theta0))
        beta = - torch.sign(kappa)

        # Stack the output: [x, y, velocity direction, sign of curvature]
        waypoints = torch.stack((x, y, vel_dir, beta, kappa), dim=2)

        return waypoints
    
    def generate_variable_curvature_waypoints(self, theta0: float = 0.) -> torch.Tensor:
        # Define the boundaries for each segment
        s = torch.linspace(0, self.path_length, self.num_waypoints, device=self.device)
        ds = s[1] - s[0]  # Step size

        # length of each segment
        circle_length_small_start = 4.5 * torch.pi
        circle_length_small_mid = 5 * torch.pi
        circle_length_small_end = 4.5 * torch.pi
        sin_length = math.asin(1 / 2) * 8
        circle_length_large = (torch.pi - (math.asin(1 / 2) + math.sqrt(3) / 2 - 1) * 8)  * 2 * 2
        k = 8
        constant_curvature = 1 / 2

        # Define curvature as a function of s
        curvature = torch.zeros_like(s, device=self.device)

        # Segment 1: curvature = 1 for s in [0, 4.5pi]
        curvature[s <= circle_length_small_start] = 1

        # Segment 2: curvature = 1 - sin((s - 4.5pi) / k) for s in (4.5pi, 4.5pi + sin_length]
        curvature[(s > circle_length_small_start) & (s <= circle_length_small_start + sin_length)] = 1 - torch.sin((s[(s > circle_length_small_start) & (s <= circle_length_small_start + sin_length)] - circle_length_small_start) / k)

        # Segment 3: curvature = constant_curvature for s in (4.5pi + sin_length, 4.5pi + sin_length + circle_length_large]
        curvature[(s > circle_length_small_start + sin_length) & (s <= circle_length_small_start + sin_length + circle_length_large)] = constant_curvature

        # Segment 4: curvature = 1 + sin((s - (4.5pi + sin_length + circle_length_large + sin_length)) / k) for s in (4.5pi + sin_length + circle_length_large, 4.5pi + sin_length + circle_length_large + sin_length]
        curvature[(s > circle_length_small_start + sin_length + circle_length_large) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length)] = 1 + torch.sin((s[(s > circle_length_small_start + sin_length + circle_length_large) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length)] - (circle_length_small_start + sin_length + circle_length_large + sin_length)) / k)

        # Segment 5: curvature = 1 for s in (4.5pi + sin_length + circle_length_large + sin_length, 9.5pi + sin_length + circle_length_large + sin_length]
        curvature[(s > circle_length_small_start + sin_length + circle_length_large + sin_length) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid)] = 1

        # Segment 6: curvature = 1 - sin((s - 9pi - sin_length - circle_length_large - sin_length) / k) for s in (9pi + sin_length + circle_length_large + sin_length, 9pi + sin_length + circle_length_large + sin_length + sin_length]
        curvature[(s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length)] = 1 - torch.sin((s[(s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length)] - (circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid)) / k)

        # Segment 7: curvature = constant_curvature for s in (9pi + sin_length + circle_length_large + sin_length + sin_length, 9pi + sin_length + circle_length_large + sin_length + sin_length + circle_length_large]
        curvature[(s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large)] = constant_curvature

        # Segment 8: curvature = 1 + sin((s - 9pi - sin_length - circle_length_large - sin_length - circle_length_large) / k) for s in (9pi + sin_length + circle_length_large + sin_length + sin_length + circle_length_large, 9pi + sin_length + circle_length_large + sin_length + sin_length + circle_length_large + sin_length]
        curvature[(s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length)] = 1 + torch.sin((s[(s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large) & (s <= circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length)] - (circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length)) / k)

        # Segment 9: curvature = 1 for s in (9pi + sin_length + circle_length_large + sin_length + sin_length + circle_length_large + sin_length, 13.5pi + sin_length + circle_length_large + sin_length + sin_length + circle_length_large + sin_length]
        curvature[s > circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length] = 1

        # Initialize kappa (curvature values at waypoints)
        kappa = curvature.unsqueeze(0)

        # Numeric integration of curvature to get heading angle
        theta = torch.cumsum(kappa * ds, dim=1)  # Integrate curvature to compute heading

        # Initial position and angle
        x0, y0, theta0 = 0, 0, theta0

        # Compute x, y positions by integrating the heading angle
        x = torch.cumsum(torch.cos(theta + theta0) * ds, dim=1) + x0
        y = torch.cumsum(torch.sin(theta + theta0) * ds, dim=1) + y0

        # Velocity direction (theta) and sign of curvature (beta)
        vel_dir = torch.atan2(torch.sin(theta + theta0), torch.cos(theta + theta0))
        beta = - torch.sign(kappa)

        # Stack the output: [x, y, velocity direction, sign of curvature]
        waypoints = torch.stack((x, y, vel_dir, beta, kappa), dim=2)

        return waypoints

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    ## three circles
    path_length = 12 * torch.pi
    num_waypoints = int(path_length / 0.005)

    traj_gen = TrajGen(num_waypoints=num_waypoints, path_length=path_length)

    segment_length = torch.tensor([(1 + 1 / 12) * 2 * torch.pi, (1 + 5 / 6) * 2 * torch.pi, (1 + 1 / 6) * 2 * torch.pi, (1 + 11 / 12) * 2 * torch.pi], device='cuda')

    curvature = torch.tensor([1, -1, 1, -1], device='cuda')

    waypoints = traj_gen.generate_custom_waypoints(segment_length, curvature)

    ## olympic rings
    path_length = 18 * torch.pi
    num_waypoints = int(path_length / 0.005)

    traj_gen = TrajGen(num_waypoints=num_waypoints, path_length=path_length)

    segment_length = torch.tensor([(1 + 1 / 4) * 2 * torch.pi, (1 + 5 / 6) * 2 * torch.pi, (1 + 1 / 3) * 2 * torch.pi, (1 + 2 / 3) * 2 * torch.pi, (1 + 1 / 6) * 2 * torch.pi, (1 + 3 / 4) * 2 * torch.pi], device='cuda')

    curvature = torch.tensor([1, -1, 1, -1, 1, -1], device='cuda')

    waypoints = traj_gen.generate_custom_waypoints(segment_length, curvature, theta0=-0.5 * torch.pi)

    # random racetrack
    traj_gen = TrajGen(num_waypoints=2400, path_length=24)
    waypoints = traj_gen.generate_random_waypoints(1)

    # variable curvature racetrack
    circle_length_small_start = 4.5 * torch.pi
    circle_length_small_mid = 5 * torch.pi
    circle_length_small_end = 4.5 * torch.pi
    sin_length = math.asin(1 / 2) * 8
    circle_length_large = (torch.pi - (math.asin(1 / 2) + math.sqrt(3) / 2 - 1) * 8)  * 2 * 2
    traj_gen = TrajGen(num_waypoints=int((circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length + circle_length_small_end) / 0.005), path_length=(circle_length_small_start + sin_length + circle_length_large + sin_length + circle_length_small_mid + sin_length + circle_length_large + sin_length + circle_length_small_end))
    waypoints = traj_gen.generate_variable_curvature_waypoints()
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(waypoints[0, :, 0].cpu().numpy(), waypoints[0, :, 1].cpu().numpy())
    ax.set_aspect('equal')
    plt.savefig("racetrack.png")