import omni.graph.core as og
import torch
import numpy as np

class OceanDeformer:
    def __init__(self, profile_res=8192, profile_data_num=1000, direction_count=128, gravity=9.80665, device="cuda"):
        self.node = og.get_node_by_path("/World/water_surface/PushGraph/ocean_deformer")
        self.profile_res = profile_res
        self.profile_data_num = profile_data_num
        self.direction_count = direction_count
        self.device = device
        self.profile_extent = 410.0
        self.gravity = gravity
        # Generate shared random arrays (fixed seed to match Warp)
        np.random.seed(7)
        self.rand_arr_profile = torch.tensor(
            np.random.rand(self.profile_data_num), dtype=torch.float32, device=device
        )
        self.rand_arr_points = torch.tensor(
            np.random.rand(self.direction_count), dtype=torch.float32, device=device
        )

        self.update_attr()

        self.profile = torch.zeros(self.profile_res, 3, dtype=torch.float32, device=self.device)

    def update_attr(self):
        # get info from node
        self.inputs_antiAlias = self.node.get_attribute("inputs:antiAlias")
        self.inputs_cameraPos = self.node.get_attribute("inputs:cameraPos")
        self.inputs_direction = self.node.get_attribute("inputs:direction")
        self.inputs_directionality = self.node.get_attribute("inputs:directionality")
        self.inputs_scale = self.node.get_attribute("inputs:scale")
        self.inputs_waterDepth = self.node.get_attribute("inputs:waterDepth")
        self.inputs_waveAmplitude = self.node.get_attribute("inputs:waveAmplitude")
        self.inputs_windSpeed = self.node.get_attribute("inputs:windSpeed")

        # process info (torch tensors)
        self.amplitude = torch.clamp(torch.tensor(float(self.inputs_waveAmplitude), device=self.device), 0.0001, 1000.0)
        self.minWavelength = 0.1
        self.maxWavelength = 250.0
        self.direction = torch.tensor(float(self.inputs_direction) % (2 * np.pi), device=self.device)
        self.directionality = torch.clamp(torch.tensor(0.02 * float(self.inputs_directionality), device=self.device), 0.0, 1.0)
        self.windspeed = torch.clamp(torch.tensor(float(self.inputs_windSpeed), device=self.device), 0.0, 30.0)
        self.waterdepth = torch.clamp(torch.tensor(float(self.inputs_waterDepth), device=self.device), 1.0, 1000.0)
        self.scale = torch.clamp(torch.tensor(float(self.inputs_scale), device=self.device), 0.001, 10000.0)
        self.antiAlias = int(bool(self.inputs_antiAlias))
        self.campos = torch.tensor(self.inputs_cameraPos, device=self.device, dtype=torch.float32)

        # update time attribute
        self.update_time_attr()

    def update_time_attr(self):
        self.inputs_time = self.node.get_attribute("inputs:time")
        self.time = torch.tensor(float(self.inputs_time), device=self.device)

    def compute(self, points):
        self.update_time_attr()
        self.update_profile()
        disp, mask = self.update_points(points)

        return disp, mask

    def is_submerged(self, disp):
        return 
        
    def update_profile(self):
        omega0 = torch.sqrt(2.0 * np.pi * self.gravity / self.minWavelength)
        omega1 = torch.sqrt(2.0 * np.pi * self.gravity / self.maxWavelength)
        omega = torch.linspace(omega0, omega1, self.profile_data_num, device=self.device)
        omega_delta = (omega1 - omega0) / self.profile_data_num

        x_idx = torch.arange(self.profile_res, device=self.device)
        space_pos = self.profile_extent * x_idx / self.profile_res

        disp = torch.zeros((self.profile_res, 3), device=self.device)

        for i in range(self.profile_data_num):
            k = omega[i] ** 2 / self.gravity
            phase = -self.time * omega[i] + self.rand_arr_profile[i] * 2.0 * np.pi
            amp = 10000.0 * torch.sqrt(torch.abs(2.0 * omega_delta * self.TMA_spectrum(omega[i], 100.0, 3.3)))
            angle = space_pos * k + phase
            sin_phase = torch.sin(angle)
            cos_phase = torch.cos(angle)

            disp[:, 0] += amp * sin_phase
            disp[:, 1] -= amp * cos_phase

        disp /= self.profile_data_num

        # Spatial blending
        s = x_idx / self.profile_res
        c1 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        c2 = -2.0 * s**3 + 3.0 * s**2

        disp1 = torch.roll(disp, shifts=self.profile_res // 3, dims=0)
        disp2 = torch.roll(disp, shifts=-(self.profile_res // 3), dims=0)
        self.profile = disp + c1.unsqueeze(1) * disp1 + c2.unsqueeze(1) * disp2

        # return self.profile  # shape (profile_res, 3)

    def update_points(self, points):
        num_points = points.shape[0]
        disp_y = torch.zeros(num_points, device=self.device)        

        for d in range(self.direction_count):
            r = float(d) * 2.0 * np.pi / self.direction_count + 0.02
            dir_x = np.cos(r)
            dir_y = np.sin(r)

            t = torch.abs(self.direction - r)
            t = torch.where(t > np.pi, 2 * np.pi - t, t)
            t = t ** 1.2

            dir_amp = (2 * t ** 3 - 3 * t ** 2 + 1) + (-2 * t ** 3 + 3 * t ** 2) * (1.0 - self.directionality)
            dir_amp = dir_amp / (1.0 + 10.0 * self.directionality)

            rand_phase = self.rand_arr_points[d]
            x_proj = (points[:, 0] * dir_x + points[:, 1] * dir_y) / (self.profile_extent * self.scale) + rand_phase

            x_scaled = x_proj * self.profile_res
            # pos0 = torch.floor(x_scaled).long() % self.profile_res
            pos0_raw = torch.floor(x_scaled).long()
            pos0 = pos0_raw.remainder(self.profile_res)
            pos1 = (pos0 + 1) % self.profile_res
            w = x_scaled - pos0.float()

            p0 = self.profile[pos0, 0]  # use x component of profile
            p1 = self.profile[pos1, 0]

            prof_height = dir_amp * ((1.0 - w) * p0 + w * p1)
            disp_y += dir_y * prof_height

        # Apply amplitude and averaging
        disp_y = self.amplitude * disp_y / self.direction_count
        submerged_mask = (disp_y<=0).unsqueeze(1)
        return disp_y, submerged_mask   # shape (N, 1)
    
    def TMA_spectrum(self, omega, fetch_km, gamma):
        # jonswap_spectrum
        # https://wikiwaves.org/Ocean-Wave_Spectra#JONSWAP_Spectrum
        fetch = 1000.0 * fetch_km
        alpha = 0.076 * (self.windspeed ** 2 / (self.gravity * fetch)) ** 0.22
        peak_omega = 22.0 * torch.abs((self.gravity ** 2 / (self.windspeed * fetch))) ** (1.0 / 3.0)
        # jonswap_peak_sharpening
        sigma = torch.where(omega > peak_omega, 0.09, 0.07)
        exponent = -0.5 * ((omega - peak_omega) / (sigma * peak_omega)) ** 2
        jonswap_sharp = gamma ** torch.exp(exponent)
        # alpha_beta_spectrum
        alpha_term = alpha * self.gravity**2 / omega**5
        beta = 1.25
        shape_term = torch.exp(-beta * (peak_omega / omega) ** 4)
        spectrum = jonswap_sharp * alpha_term * shape_term
        # TMA_spectrum 
        # https://dl.acm.org/doi/10.1145/2791261.2791267
        omegaH = omega * torch.sqrt(self.waterdepth / self.gravity)
        omegaH = torch.clamp(omegaH, 0.0, 2.2)
        phi = torch.where(omegaH <= 1.0,
                          0.5 * omegaH**2,
                          1.0 - 0.5 * (2.0 - omegaH)**2)

        return phi * spectrum