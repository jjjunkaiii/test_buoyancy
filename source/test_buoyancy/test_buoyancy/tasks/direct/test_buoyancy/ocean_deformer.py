import torch
import numpy as np
import omni.graph.core as og

class OceanDeformer:
    def __init__(self, num_env, num_points, profile_res=8192, profile_data_num=1000, direction_count=128, gravity=9.80665, device="cuda"):
        self.node = og.get_node_by_path("/World/water_surface/PushGraph/ocean_deformer")
        self.profile_res = profile_res
        self.profile_data_num = profile_data_num
        self.direction_count = direction_count
        self.device = device
        self.profile_extent = 410.0
        self.gravity = gravity
        self.num_env = num_env
        self.num_points = num_points
        # init attributes
        self._init_attr()
        # init variables
        self._init_variables()
        self._init_buffers()
        self.profile = torch.zeros(self.profile_res, 3, dtype=torch.float16, device=self.device)

    def _init_attr(self):
        self.node.get_attribute("inputs:waveAmplitude").set(0.2)
        # get info from node
        self.inputs_antiAlias = self.node.get_attribute("inputs:antiAlias").get()
        self.inputs_cameraPos = self.node.get_attribute("inputs:cameraPos").get()
        self.inputs_direction = self.node.get_attribute("inputs:direction").get()
        self.inputs_directionality = self.node.get_attribute("inputs:directionality").get()
        self.inputs_scale = self.node.get_attribute("inputs:scale").get()
        self.inputs_waterDepth = self.node.get_attribute("inputs:waterDepth").get()
        self.inputs_waveAmplitude = self.node.get_attribute("inputs:waveAmplitude").get()
        self.inputs_windSpeed = self.node.get_attribute("inputs:windSpeed").get()

        # process info (torch tensors)
        self.amplitude = torch.clamp(torch.tensor(float(self.inputs_waveAmplitude), device=self.device, dtype=torch.float64), 0.0001, 1000.0)
        self.minWavelength = 0.1
        self.maxWavelength = 250.0
        self.direction = torch.tensor(float(self.inputs_direction) % (2 * np.pi), device=self.device, dtype=torch.float64)
        self.directionality = torch.clamp(torch.tensor(0.02 * float(self.inputs_directionality), device=self.device, dtype=torch.float64), 0.0, 1.0)
        self.windspeed = torch.clamp(torch.tensor(float(self.inputs_windSpeed), device=self.device, dtype=torch.float64), 0.0, 30.0)
        self.waterdepth = torch.clamp(torch.tensor(float(self.inputs_waterDepth), device=self.device, dtype=torch.float64), 1.0, 1000.0)
        self.scale = torch.clamp(torch.tensor(float(self.inputs_scale), device=self.device, dtype=torch.float16), 0.001, 10000.0)
        self.antiAlias = int(bool(self.inputs_antiAlias))
        self.campos = torch.tensor(self.inputs_cameraPos, device=self.device, dtype=torch.float64)

        # update time attribute
        self.update_time()
    
    @torch.no_grad()
    def _init_variables(self):
        # Generate shared random arrays (fixed seed to match Warp)
        np.random.seed(7)
        self.rand_arr_profile = torch.tensor(
            np.random.rand(self.profile_data_num), dtype=torch.float16, device=self.device
        )
        self.rand_arr_points = torch.tensor(
            np.random.rand(self.direction_count), dtype=torch.float16, device=self.device
        )        
        # Compute omega range
        omega0 = np.sqrt(2.0 * np.pi * self.gravity / self.minWavelength)
        omega1 = np.sqrt(2.0 * np.pi * self.gravity / self.maxWavelength)
        self.omega = omega0 + (omega1 - omega0) * torch.arange(self.profile_data_num, device=self.device, dtype=torch.float64) / self.profile_data_num
        self.omega_delta = (omega1 - omega0) / self.profile_data_num
        self.k = self.omega ** 2 / self.gravity   
        self.amp = 10000.0 * torch.sqrt(torch.abs(2.0 * self.omega_delta * self._TMA_spectrum(self.omega, 100.0, 3.3)))
        # Spatial sample positions
        self.x_idx = torch.arange(self.profile_res, device=self.device, dtype=torch.float64)
        self.space_pos = self.profile_extent * self.x_idx / self.profile_res  # (res,)
        # Blending coefficients
        s = self.x_idx / self.profile_res  # (res,)
        self.c1 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        self.c2 = -2.0 * s**3 + 3.0 * s**2
        # Directions
        d = torch.arange(self.direction_count, device=self.device, dtype=torch.float64)
        r = d * 2.0 * np.pi / self.direction_count + 0.02
        dir_x = torch.cos(r)
        dir_y = torch.sin(r)
        self.dirs = torch.stack([dir_x, dir_y], dim=1)  # shape: (D, 2)
        # Dirctional amplitude
        direction_exp = self.direction.unsqueeze(-1)  # shape: (D, 1)
        r_exp = r.unsqueeze(0)                        # shape: (1, D)
        t = torch.abs(direction_exp - r_exp)
        t = torch.where(t > np.pi, 2 * np.pi - t, t)
        t = t ** 1.2
        directionality_exp = self.directionality.unsqueeze(-1)  # shape: (D, 1)
        dir_amp = (2 * t**3 - 3 * t**2 + 1) + (-2 * t**3 + 3 * t**2) * (1.0 - directionality_exp)
        self.dir_amp = dir_amp / (1.0 + 10.0 * directionality_exp)

        # reduce bits
        self.omega = self.omega.to(torch.float16)
        self.k = self.k.to(torch.float16)
        self.amp = self.amp.to(torch.float16)
        self.c1 = self.c1.to(torch.float16)
        self.c2 = self.c2.to(torch.float16)
        self.dirs = self.dirs.to(torch.float16)
        self.dir_amp = self.dir_amp.to(torch.float16)
        self.space_pos = self.space_pos.to(torch.float16)
        
    def _init_buffers(self):
        B, N, D = self.num_env, self.num_points, self.direction_count
        self.points_xy = torch.empty((B, N, 2), dtype=torch.float16, device=self.device)
        self.dots = torch.empty((B, N, D), dtype=torch.float16, device=self.device)
        self.x_proj = torch.empty((B, N, D), dtype=torch.float16, device=self.device)
        self.x_scaled = torch.empty((B, N, D), dtype=torch.float16, device=self.device)
        self.pos0 = torch.empty((B, N, D), dtype=torch.int32, device=self.device)
        self.pos1 = torch.empty((B, N, D), dtype=torch.int32, device=self.device)
        self.w = torch.empty((B, N, D), dtype=torch.float16, device=self.device)
        self.p0 = torch.empty((B, N, D), dtype=torch.float16, device=self.device)
        self.p1 = torch.empty((B, N, D), dtype=torch.float16, device=self.device)
        self.disp_z = torch.empty((B, N), dtype=torch.float16, device=self.device)
        self.submerged_mask = torch.empty((B, N), dtype=torch.bool, device=self.device)

    def update(self):
        self.update_time()
        self.update_profile()

    def compute(self, points, water_level):
        assert points.shape == (self.num_env, self.num_points, 3), \
            f"Expected input shape ({self.num_env}, {self.num_points}, 3), but got {points.shape}"
        disp, mask = self._update_points(points, water_level)
        return disp, mask
    
    @torch.no_grad()
    def update_time(self):
        self.inputs_time = self.node.get_attribute("inputs:time").get()
        self.time = torch.tensor(float(self.inputs_time), device=self.device, dtype=torch.float16)

    @torch.no_grad()
    def update_profile(self):
        # Wavenumber and phase            # (N,)
        phase = -self.time * self.omega + self.rand_arr_profile * 2.0 * np.pi  # (N,)
        # Amplitudes from spectrum
        amp = self.amp
        k = self.k    
        # Helper to compute displacement at given spatial position
        def compute_disp_at(pos):  # pos: (res,)
            angle = k.unsqueeze(1) * pos.unsqueeze(0) + phase.unsqueeze(1)  # (N, res)
            sin_phase = torch.sin(angle)
            cos_phase = torch.cos(angle)
            disp_x = (amp.unsqueeze(1) * sin_phase).sum(dim=0)  # (res,)
            disp_y = -(amp.unsqueeze(1) * cos_phase).sum(dim=0)  # (res,)
            return torch.stack([disp_x, disp_y, torch.zeros_like(disp_x)], dim=1) / self.profile_data_num  # (res, 3)

        # Compute displacements at three spatial positions
        disp1 = compute_disp_at(self.space_pos)                          # space_pos_1
        disp2 = compute_disp_at(self.space_pos + self.profile_extent)    # space_pos_2
        disp3 = compute_disp_at(self.space_pos - self.profile_extent)    # space_pos_3

        # Final profile
        self.profile = disp1 + self.c1.unsqueeze(1) * disp2 + self.c2.unsqueeze(1) * disp3  # (res, 3)

    @torch.no_grad()
    def _update_points(self, points, water_level=0.0):
        # points: (B, N, 3)
        B, N, _ = points.shape
        # Project to horizontal plane (Isaac: x, -y)
        self.points_xy[:] = torch.stack([points[:, :, 0], -points[:, :, 1]], dim=2).to(torch.float16)
        torch.matmul(self.points_xy, self.dirs.mT, out=self.dots)
        self.x_proj[:] = self.dots / (self.profile_extent * self.scale) + self.rand_arr_points
        self.x_scaled[:] = self.x_proj * self.profile_res
        self.pos0[:] = torch.floor(self.x_scaled).to(torch.int32)
        self.pos0[self.pos0 < 0] += self.profile_res - 1
        self.pos0.remainder_(self.profile_res)
        torch.remainder(self.pos0 + 1, self.profile_res, out=self.pos1)
        self.w[:] = self.x_scaled - torch.floor(self.x_scaled)
        self.p0[:] = self.profile[self.pos0, 1]
        self.p1[:] = self.profile[self.pos1, 1]
        temp = (1.0 - self.w) * self.p0 + self.w * self.p1  # (B, N, D)
        disp = torch.matmul(temp, self.dir_amp.mT)
        disp = disp.sum(dim=-1)  # (B, N)
        self.disp_z[:] = disp * (self.amplitude / self.direction_count) + water_level
        self.submerged_mask[:] = self.disp_z > points[:, :, 2]
        return self.disp_z, self.submerged_mask

    def _TMA_spectrum(self, omega, fetch_km, gamma):
        fetch = 1000.0 * fetch_km
        alpha = 0.076 * (self.windspeed ** 2 / (self.gravity * fetch)) ** 0.22
        peak_omega = 22.0 * torch.abs((self.gravity ** 2 / (self.windspeed * fetch))) ** (1.0 / 3.0)

        sigma = torch.where(omega > peak_omega, 0.09, 0.07)
        exponent = -0.5 * ((omega - peak_omega) / (sigma * peak_omega)) ** 2
        jonswap_sharp = gamma ** torch.exp(exponent)

        alpha_term = alpha * self.gravity**2 / omega**5
        beta = 1.25
        shape_term = torch.exp(-beta * (peak_omega / omega) ** 4)
        spectrum = jonswap_sharp * alpha_term * shape_term

        omegaH = omega * torch.sqrt(self.waterdepth / self.gravity)
        omegaH = torch.clamp(omegaH, 0.0, 2.2)
        phi = torch.where(omegaH <= 1.0,
                        0.5 * omegaH**2,
                        1.0 - 0.5 * (2.0 - omegaH)**2)

        return phi * spectrum