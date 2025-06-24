import torch
import torch.nn as nn

class IWDCarDynamics(nn.Module):
    def __init__(self, u, p_body, p_tyre, differentiable=False, disturbance=None, drivetrain="iwd"):
        """
        u: [batch_size, 5] - (delta, omega_fr, omega_fl, omega_rr, omega_rl)
        p_body: [batch_size, 8] - (lf, lr, m, h, g, Iz, T, wheel_radius)
        p_tyre: [batch_size, 4] - (B, C, D, E) for Pacejka formula
        """
        super().__init__()
        to_param = nn.Parameter if differentiable else lambda x: x
        self.u = to_param(u)
        self.p_body = to_param(p_body)
        self.p_tyre = to_param(p_tyre)
        self.disturbance = disturbance
        self.drivetrain = drivetrain

    def compute_extended_state(self, s):
        """
        s: [batch_size, 6] - (x, y, psi, vx, vy, psid)
        Returns: Extended state including velocities, forces, and accelerations
        """
        # Decoding state and parameters
        x, y, psi = s[:, 0], s[:, 1], s[:, 2]
        vx, vy, psid = s[:, 3], s[:, 4], s[:, 5]
        
        delta = self.u[:, 0]
        omega_fr = self.u[:, 1]
        omega_fl = self.u[:, 2] 
        omega_rr = self.u[:, 3]
        omega_rl = self.u[:, 4]
        
        lf = self.p_body[:, 0]
        lr = self.p_body[:, 1]
        m = self.p_body[:, 2]
        h = self.p_body[:, 3]
        g = self.p_body[:, 4]
        Iz = self.p_body[:, 5]
        T = self.p_body[:, 6]

        B = self.p_tyre[:, 0]
        C = self.p_tyre[:, 1]
        D = self.p_tyre[:, 2]
        E = self.p_tyre[:, 3]

        eps = 1e-8

        # Transform velocities from world to body frame [batch_size]
        c_psi = torch.cos(psi)
        s_psi = torch.sin(psi)
        vx_body = c_psi * vx + s_psi * vy
        vy_body = -s_psi * vx + c_psi * vy

        # Calculate vehicle states
        v = torch.hypot(vx_body, vy_body)
        beta = torch.atan2(vy, vx + eps) - psi
        beta = torch.atan2(torch.sin(beta), torch.cos(beta))

        # Wheel positions relative to CoG [batch_size, 4]
        corner_x = torch.stack([lf, lf, -lr, -lr], dim=1)  # fr, fl, rr, rl
        corner_y = torch.stack([-T/2, T/2, -T/2, T/2], dim=1)  # fr, fl, rr, rl

        # Wheel velocities [batch_size, 2, 4]
        v_base = torch.stack([
            vx_body.unsqueeze(1).expand(-1, 4),
            vy_body.unsqueeze(1).expand(-1, 4)
        ], dim=1)

        delta_v = torch.stack([-corner_y, corner_x], dim=1) * psid.unsqueeze(1).unsqueeze(2)
        v_wheel = v_base + delta_v

        # Apply steering to front wheels [batch_size, 2, 2]
        c_delta = torch.cos(delta)
        s_delta = torch.sin(delta)
        R_steer = torch.stack([
            torch.stack([c_delta, s_delta], dim=1),
            torch.stack([-s_delta, c_delta], dim=1)
        ], dim=1)
        
        # Transform front wheel velocities [batch_size, 2, 4]
        v_wheel_local = v_wheel.clone()
        v_wheel_local[:, :, :2] = torch.bmm(R_steer, v_wheel[:, :, :2])

        # Reset front wheel velocities when drivetrain is "rwd"
        if self.drivetrain == "rwd":
            omega_fr = v_wheel_local[:, 0, 0]
            omega_fl = v_wheel_local[:, 0, 1]
        
        # Wheel angular velocities [batch_size, 4]
        omega = torch.stack([omega_fr, omega_fl, omega_rr, omega_rl], dim=1)
        
        # Calculate slip ratios [batch_size, 4]
        vx_local = v_wheel_local[:, 0, :]  # [batch_size, 4]
        vy_local = v_wheel_local[:, 1, :]  # [batch_size, 4]
        
        sx = (vx_local - omega) / (torch.abs(omega) + eps)
        sy = vy_local / (torch.abs(omega) + eps)
        
        # Combined slip
        slip = torch.hypot(sx, sy)
        
        # Pacejka magic formula [batch_size, 4]
        mu = D.unsqueeze(1) * torch.sin(
            C.unsqueeze(1) * torch.atan(
                B.unsqueeze(1) * slip - E.unsqueeze(1) * (B.unsqueeze(1) * slip - torch.atan(B.unsqueeze(1) * slip))
            )
        )

        # Slip angles [batch_size, 4]
        alpha = torch.atan2(sy, sx + eps)

        # Friction components [batch_size, 4]
        mux = -torch.cos(alpha) * mu
        muy = -torch.sin(alpha) * mu
        
        # Load transfer
        G = m * g
        l = lf + lr
        
        # Longitudinal load transfer [batch_size]
        fz_front = (lr * G - h * G * mux[:, 2:].mean(dim=1)) / (
            l + h * (mux[:, :2].mean(dim=1) * c_delta - muy[:, :2].mean(dim=1) * s_delta - mux[:, 2:].mean(dim=1))
        )
        fz_rear = G - fz_front

        # Individual wheel loads [batch_size, 4]
        fz = torch.cat([
            fz_front.unsqueeze(1).expand(-1, 2) / 2,
            fz_rear.unsqueeze(1).expand(-1, 2) / 2
        ], dim=1)
        
        # Tire forces in wheel frame [batch_size, 4]
        fx_local = mux * fz
        fy_local = muy * fz
        
        # Transform forces to body frame
        fx = fx_local.clone()
        fy = fy_local.clone()
        fx[:, :2] = fx_local[:, :2] * c_delta.unsqueeze(1) - fy_local[:, :2] * s_delta.unsqueeze(1)
        fy[:, :2] = fx_local[:, :2] * s_delta.unsqueeze(1) + fy_local[:, :2] * c_delta.unsqueeze(1)
        
        if self.disturbance is not None:
            fx = fx + self.disturbance[:, :4]
            fy = fy + self.disturbance[:, 4:]
        
        # Body accelerations [batch_size]
        ax = torch.sum(fx, dim=1) / m  # Longitudinal acceleration from forces
        ay = torch.sum(fy, dim=1) / m  # Update lateral acceleration with force contribution
        
        # Yaw acceleration [batch_size]
        yaw_moment = (fy[:, :2].sum(dim=1) * lf) - (fy[:, 2:].sum(dim=1) * lr) + \
                    (fx[:, [0,2]].sum(dim=1) * (T/2)) - (fx[:, [1,3]].sum(dim=1) * (T/2))
        psid_dot = yaw_moment / Iz
       
        # Transform accelerations back to world frame [batch_size]
        ax_world = c_psi * ax - s_psi * ay
        ay_world = s_psi * ax + c_psi * ay

        # Combine all extended states
        extended_states = [
            vx, vy, psid,  # Original velocities [3]
            ax_world, ay_world, psid_dot,  # Accelerations [3]
            v, beta,  # Vehicle states [2]
            v_wheel_local[:, 0, :], v_wheel_local[:, 1, :],  # Wheel velocities [8]
            sx, sy, slip,  # Slip ratios [12]
            mu, alpha,  # Friction states [8]
            mux, muy,  # Friction components [8]
            fz, fx, fy  # Forces [12]
        ]

        return torch.cat([t.unsqueeze(1) if t.dim() == 1 else t for t in extended_states], 1)
    
    def forward(self, t, s):
        es = self.compute_extended_state(s)
        return es[:, :6]