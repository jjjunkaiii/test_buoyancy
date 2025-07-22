import numpy as np
import torch
import math


class OceanDeformer:
    def __init__(self, profile_res=8192, profile_data_num=1000, direction_count=128, device="cuda"):
        self.profile_res = profile_res
        self.profile_data_num = profile_data_num
        self.direction_count = direction_count
        self.device = device
        self.profile_extent = 410.0
        self.gravity = 9.80665

        # Generate shared random arrays (fixed seed to match Warp)
        np.random.seed(7)
        self.rand_arr_profile = torch.tensor(
            np.random.rand(self.profile_data_num), dtype=torch.float32, device=device
        )
        self.rand_arr_points = torch.tensor(
            np.random.rand(self.direction_count), dtype=torch.float32, device=device
        )



    def TMA_spectrum(self, omega, windspeed, fetch_km, gamma, waterdepth):
        # jonswap_spectrum
        # https://wikiwaves.org/Ocean-Wave_Spectra#JONSWAP_Spectrum
        fetch = 1000.0 * fetch_km
        alpha = 0.076 * (windspeed ** 2 / (self.gravity * fetch)) ** 0.22
        peak_omega = 22.0 * torch.abs((self.gravity ** 2 / (windspeed * fetch))) ** (1.0 / 3.0)
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
        omegaH = omega * torch.sqrt(waterdepth / self.gravity)
        omegaH = torch.clamp(omegaH, 0.0, 2.2)
        phi = torch.where(omegaH <= 1.0,
                          0.5 * omegaH**2,
                          1.0 - 0.5 * (2.0 - omegaH)**2)

        return phi * spectrum

    def update_profile(self, time, windspeed, waterdepth):
        omega0 = torch.sqrt(2.0 * math.pi * self.gravity / 0.1)
        omega1 = torch.sqrt(2.0 * math.pi * self.gravity / 250.0)
        omega = torch.linspace(omega0, omega1, self.profile_data_num, device=self.device)
        omega_delta = (omega1 - omega0) / self.profile_data_num

        x_idx = torch.arange(self.profile_res, device=self.device)
        space_pos = self.profile_extent * x_idx / self.profile_res

        disp = torch.zeros((self.profile_res, 3), device=self.device)

        for i in range(self.profile_data_num):
            k = omega[i] ** 2 / self.gravity
            phase = -time * omega[i] + self.rand_arr_profile[i] * 2.0 * math.pi
            amp = 10000.0 * torch.sqrt(torch.abs(2.0 * omega_delta * self.TMA_spectrum(
                omega[i], windspeed, 100.0, 3.3, waterdepth)))
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
        final_disp = disp + c1.unsqueeze(1) * disp1 + c2.unsqueeze(1) * disp2

        return final_disp  # shape (profile_res, 3)

    def get_water_heights(self, points: torch.Tensor, time: float, profile_cache=None,
                          amplitude=1.0, direction=0.0, directionality=0.0,
                          windspeed=5.0, waterdepth=100.0, scale=1.0, camera_pos=(0, 0, 0)) -> torch.Tensor:

        # points: (N, 3)
        points = points.to(self.device)
        N = points.shape[0]

        if profile_cache is None:
            profile = self.update_profile(time, windspeed, waterdepth)
        else:
            profile = profile_cache.to(self.device)

        disp_y = torch.zeros(N, device=self.device)

        for d in range(128):
            r = float(d) * 2.0 * math.pi / 128.0 + 0.02
            dir_x = math.cos(r)
            dir_y = math.sin(r)
            t = abs(direction - r)
            if t > math.pi:
                t = 2 * math.pi - t
            t = t ** 1.2
            dir_amp = (2 * t ** 3 - 3 * t ** 2 + 1) + (-2 * t ** 3 + 3 * t ** 2) * (1 - directionality)
            dir_amp = dir_amp / (1 + 10 * directionality)

            rand_phase = self.rand_arr_points[d]
            x_crd = (points[:, 0] * dir_x + points[:, 2] * dir_y) / (self.profile_extent * scale) + rand_phase
            x_crd_scaled = x_crd * self.profile_res

            pos0 = torch.floor(x_crd_scaled).to(torch.int64) % self.profile_res
            pos1 = (pos0 + 1) % self.profile_res
            w = x_crd_scaled - pos0.float()

            p0 = profile[pos0, 0]
            p1 = profile[pos1, 0]
            height = dir_amp * ((1 - w) * p0 + w * p1)

            disp_y += dir_y * height

        disp_y = amplitude * disp_y / 128.0
        return points[:, 1] + disp_y  # final water surface height

    def update_points_torch(self, points: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        """
        points:    (N, 3) torch tensor on GPU
        profile:   (profile_res, 3) torch tensor from update_profile
        returns:   (N, 3) displaced points with Y updated (vertical water level)
        """
        N = points.shape[0]
        profile_res = self.profile_res
        extent = self.profile_extent * self.scale
        direction_count = 128
        disp_y = torch.zeros(N, device=self.device)

        for d in range(direction_count):
            r = float(d) * 2.0 * math.pi / direction_count + 0.02
            dir_x = math.cos(r)
            dir_y = math.sin(r)

            t = torch.abs(self.direction - r)
            t = torch.where(t > math.pi, 2 * math.pi - t, t)
            t = t ** 1.2

            dir_amp = (2 * t ** 3 - 3 * t ** 2 + 1) + (-2 * t ** 3 + 3 * t ** 2) * (1.0 - self.directionality)
            dir_amp = dir_amp / (1.0 + 10.0 * self.directionality)

            rand_phase = self.rand_arr_points[d]
            x_proj = (points[:, 0] * dir_x + points[:, 2] * dir_y) / extent + rand_phase

            x_scaled = x_proj * profile_res
            pos0 = torch.floor(x_scaled).long() % profile_res
            pos1 = (pos0 + 1) % profile_res
            w = x_scaled - pos0.float()

            p0 = profile[pos0, 0]  # use x component of profile
            p1 = profile[pos1, 0]

            prof_height = dir_amp * ((1.0 - w) * p0 + w * p1)
            disp_y += dir_y * prof_height

        # Apply amplitude and averaging
        disp_y = self.amplitude * disp_y / direction_count

        displaced = points.clone()
        displaced[:, 1] = displaced[:, 1] + disp_y

        return displaced  # shape (N, 3)