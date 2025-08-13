# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Expand No.1–7 to include more ships.

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, matrix_from_quat

from .test_buoyancy_env_cfg import TestBuoyancyEnvCfg

from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import MdlFileCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import numpy as np
import trimesh
import time
import omni.graph.core as og
import os

'''
for test env, the shapes of actions, observations, rewards, dones are:
Out of bounds shape: torch.Size([4096]), Time out shape: torch.Size([4096])
Total reward shape: [4096]
Rewards shape: torch.Size([4096])
Observations shape: torch.Size([4096, 4])
'''

'''
test in terminal:
python scripts/random_agent.py --task=Template-Test-Buoyancy-Direct-v0
'''

extention_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
water_path = "asset/mesh/water/omni.ocean-0.4.1/data/ocean_small.usd"
barge_mesh_path = "asset/mesh/URDF2/barge.SLDASM/meshes/barge.STL"
tugboat_mesh_path = "asset/mesh/URDF3/TUGBOAT.SLDASM/meshes/tugboat.STL"

class TestBuoyancyEnv(DirectRLEnv):
    cfg: TestBuoyancyEnvCfg

    def __init__(self, cfg: TestBuoyancyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Define robot names and mesh paths
        self.robot_names = ["barge", "tugboat1", "tugboat2"]
        self.mesh_paths = [
            barge_mesh_path,
            tugboat_mesh_path,
            tugboat_mesh_path,
        ]

        # Init robot buffers and voxel data
        self.robots = []
        self.voxel_data = []
        self.num_voxels = []
        for name, mesh_path in zip(self.robot_names, self.mesh_paths):
            self._init_robot_buffers(name)
            self._load_voxel(mesh_path=mesh_path, suffix=name)
            self.robots.append(getattr(self, name))
            valid_voxels = getattr(self, f"valid_voxels_{name}")
            self.num_voxels.append(valid_voxels.shape[0])
            self.voxel_data.append(valid_voxels)

        self.debug_visualisation = False

        if self.debug_visualisation:
            self.debug_vis = []
            for name in self.robot_names:
                self._define_voxel_markers(name)
                self.debug_vis.append(getattr(self, f"marker_{name}"))
            self._define_voxel_markers("buoy")

        # Batchify all voxel positions for all robots
        self.total_voxels = sum(self.num_voxels)
        self.voxel_robot_slices = []
        start = 0
        for n in self.num_voxels:
            self.voxel_robot_slices.append(slice(start, start + n))
            start += n

        # Preallocate batch voxel positions (num_env, total_voxels, 3)
        self.voxel_pos_w = torch.zeros((self.num_envs, self.total_voxels, 3), dtype=torch.float32, device=self.device)

        # Instantiate batch OceanDeformer
        self.oceandeformer = OceanDeformer(
            num_env=self.num_envs,
            num_points=self.total_voxels,
            device=self.device,
        )

        # For debug visualisation (buoys), use a separate OceanDeformer
        if self.debug_visualisation:
            self.num_buoys = 72 * 30 + 1
            self.water_surface_target_pos = torch.zeros((self.num_buoys, 3), dtype=torch.float32, device=self.device)
            for angle in range(72):
                for r in range(30):
                    x = (r+1) * math.cos(angle * 2 * math.pi / 72) * (100 / 30)
                    y = (r+1) * math.sin(angle * 2 * math.pi / 72) * (100 / 30)
                    z = 0.0
                    self.water_surface_target_pos[angle * 30 + r] = torch.tensor([x, y, z], device=self.device)
            self.water_surface_target_pos[-1] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            self.oceandeformer_buoy = OceanDeformer(num_env=1, num_points=self.num_buoys, device=self.device)
        # time.sleep(10) # TODO:remove in the future

        # self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)

        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel
    
    def _init_robot_buffers(self, suffix: str): 
        setattr(self, f"rot_mats_{suffix}", torch.zeros((self.num_envs, 4, 4), device=self.device))
        setattr(self, f"R_{suffix}", torch.zeros((self.num_envs, 3, 3), device=self.device))
        setattr(self, f"buoyancy_force_w_{suffix}", torch.zeros((self.num_envs, 3), device=self.device))
        setattr(self, f"buoyancy_force_b_{suffix}", torch.zeros((self.num_envs, 3), device=self.device))
        setattr(self, f"buoyancy_torque_w_{suffix}", torch.zeros((self.num_envs, 3), device=self.device))
        setattr(self, f"buoyancy_torque_b_{suffix}", torch.zeros((self.num_envs, 3), device=self.device))
        setattr(self, f"buoyancy_centre_w_{suffix}", torch.zeros((self.num_envs, 3), device=self.device))
        setattr(self, f"buoyancy_centre_b_{suffix}", torch.zeros((self.num_envs, 3), device=self.device))

    def _setup_scene(self):
        # No.4
        self.barge = RigidObject(self.cfg.bargeCfg)
        self.tugboat1 = RigidObject(self.cfg.tugCfg1)
        self.tugboat2 = RigidObject(self.cfg.tugCfg2)

        water_surface_cfg = RigidObjectCfg(
            prim_path="/World/water_surface",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(extention_path, water_path),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(0.02, 0.08, 0.25), 
                    emissive_color=(0.0, 0.0, 0.0),   
                    roughness=0.1,                    
                    metallic=0.0,                     
                    opacity=0.9                       
                ),
                rigid_props=None,
                collision_props=None,
                visible=True,
                scale = [1, 1, 1],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.cfg.water_level),
                rot=(0.7071, 0.7071, 0.0, 0.0),
            ),
        )
        water_surface = RigidObject(water_surface_cfg)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # # No.5
        self.scene.rigid_objects["barge"] = self.barge
        self.scene.rigid_objects["tugboat1"] = self.tugboat1
        self.scene.rigid_objects["tugboat2"] = self.tugboat2


        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.oceandeformer.update()
        if self.debug_visualisation:
            self.oceandeformer_buoy.update()
        self.apply_buoyancy()

    def _apply_action(self) -> None:
        # self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)
        pass # test env, TODO: remove

    def _get_observations(self) -> dict:
        # obs = torch.cat(
        #     (
        #         self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )
        obs = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device) # test env, TODO: remove
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # total_reward = compute_rewards(
        #     self.cfg.rew_scale_alive,
        #     self.cfg.rew_scale_terminated,
        #     self.cfg.rew_scale_pole_pos,
        #     self.cfg.rew_scale_cart_vel,
        #     self.cfg.rew_scale_pole_vel,
        #     self.joint_pos[:, self._pole_dof_idx[0]],
        #     self.joint_vel[:, self._pole_dof_idx[0]],
        #     self.joint_pos[:, self._cart_dof_idx[0]],
        #     self.joint_vel[:, self._cart_dof_idx[0]],
        #     self.reset_terminated,
        # )
        total_reward = torch.ones((self.num_envs,), device=self.device)  # test env, TODO: remove 
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        out_of_bounds = time_out # test env, TODO: remove
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # joint_pos = self.robot.data.default_joint_pos[env_ids]
        # joint_pos[:, self._pole_dof_idx] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        # joint_vel = self.robot.data.default_joint_vel[env_ids]

        # default_root_state = self.robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # self.joint_pos[env_ids] = joint_pos
        # self.joint_vel[env_ids] = joint_vel

        # self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def step(self, action):
        super().step(action)
        if self.debug_visualisation:
            for name in self.robot_names:
                self._visualise_voxels(name)
            self._visualise_buoys()

# ----------- buoyancy related functions -----------
    def _load_voxel(self, mesh_path: str, suffix: str):
        voxel_res = 32
        mesh = trimesh.load(os.path.join(extention_path, mesh_path))
        bounds = mesh.bounds
        min_bound, max_bound = bounds[0], bounds[1]
        interval = (max_bound - min_bound) / voxel_res

        xs = np.arange(min_bound[0], max_bound[0], interval[0])
        ys = np.arange(min_bound[1], max_bound[1], interval[1])
        zs = np.arange(min_bound[2], max_bound[2], interval[2])

        xv, yv, zv = np.meshgrid(xs, ys, zs, indexing='ij')
        voxel_centers = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
        t_start = time.time()
        inside_mask = mesh.contains(voxel_centers)
        t_end = time.time()
        print(f"[{suffix}] Mask generated, time used: {t_end - t_start:.4f} sec")

        valid_voxels = voxel_centers[inside_mask]
        voxel_volume = interval[0] * interval[1] * interval[2]

        setattr(self, f"voxel_l_{suffix}", interval[0])
        setattr(self, f"voxel_w_{suffix}", interval[1])
        setattr(self, f"voxel_h_{suffix}", interval[2])
        setattr(self, f"inside_mask_{suffix}", torch.tensor(inside_mask, dtype=torch.bool, device=self.device))
        setattr(self, f"valid_voxels_{suffix}", torch.tensor(valid_voxels, dtype=torch.float32, device=self.device))
        setattr(self, f"voxel_volume_{suffix}", torch.tensor(voxel_volume, dtype=torch.float32, device=self.device))

        voxel_pos_w = torch.zeros((self.num_envs, valid_voxels.shape[0], 3), dtype=torch.float32, device=self.device)
        setattr(self, f"voxel_pos_w_{suffix}", voxel_pos_w)

        setattr(self, f"submerged_mask_{suffix}", torch.zeros((self.num_envs, valid_voxels.shape[0]), dtype=torch.bool))

    def get_pose_mat(self, suffix: str):
        robot = getattr(self, suffix)
        rigid_body_states = robot.data.root_link_pose_w  # (num_envs, 7)
        rot_mats = getattr(self, f"rot_mats_{suffix}")   # (num_envs, 4, 4)
        R = getattr(self, f"R_{suffix}")                 # (num_envs, 3, 3)

        # Compute rotation matrices for all envs at once
        R[:] = matrix_from_quat(rigid_body_states[:, 3:7])  # (num_envs, 3, 3)
        rot_mats.zero_()
        rot_mats[:, :3, :3] = R
        rot_mats[:, :3, 3] = rigid_body_states[:, :3]
        rot_mats[:, 3, 3] = 1.0

    def get_voxel_pos_w(self):
        # Batch update all voxel positions for all robots
        for idx, name in enumerate(self.robot_names):
            self.get_pose_mat(suffix=name)
            rot_mats = getattr(self, f"rot_mats_{name}")      # (num_envs, 4, 4)
            valid_voxels = getattr(self, f"valid_voxels_{name}")  # (num_voxels, 3)
            N = self.num_envs
            M = valid_voxels.shape[0]
            # Expand valid_voxels for all envs: (num_envs, num_voxels, 3)
            voxels = valid_voxels.unsqueeze(0).expand(N, -1, -1).contiguous()
            # Apply rotation and translation in batch
            R = rot_mats[:, :3, :3]  # (num_envs, 3, 3)
            t = rot_mats[:, :3, 3]   # (num_envs, 3)
            # (num_envs, num_voxels, 3) = bmm((num_envs, num_voxels, 3), (num_envs, 3, 3).T)
            pos = torch.bmm(voxels, R.transpose(1, 2)) + t.unsqueeze(1)
            self.voxel_pos_w[:, self.voxel_robot_slices[idx]] = pos

    def apply_buoyancy(self, water_level=0.0, water_density=1500.0):
        """
        Apply buoyancy force to all rigid objects in batch.
        """
        self.get_voxel_pos_w()
        # Compute water heights and masks for all voxels in all envs
        disp, mask = self.oceandeformer.compute(self.voxel_pos_w, self.cfg.water_level)
        # For each robot, slice out its voxels and compute buoyancy
        for idx, name in enumerate(self.robot_names):
            voxel_slice = self.voxel_robot_slices[idx]
            voxel_volume = getattr(self, f"voxel_volume_{name}")
            buoyancy_force_w = getattr(self, f"buoyancy_force_w_{name}")
            buoyancy_centre_w = getattr(self, f"buoyancy_centre_w_{name}")
            buoyancy_force_b = getattr(self, f"buoyancy_force_b_{name}")
            buoyancy_torque_b = getattr(self, f"buoyancy_torque_b_{name}")
            robot = getattr(self, name)
            mask_robot = mask[:, voxel_slice]  # (num_envs, num_voxels)
            voxel_pos_robot = self.voxel_pos_w[:, voxel_slice]  # (num_envs, num_voxels, 3)

            # Batch: get number of submerged voxels per env
            submerged_mask = mask_robot
            num_submerged_voxels = submerged_mask.sum(dim=1)  # (num_envs,)
            submerged_volume = num_submerged_voxels * voxel_volume  # (num_envs,)

            # Batch: set buoyancy force (only z component nonzero)
            buoyancy_force_w.zero_()
            buoyancy_force_w[:, 2] = submerged_volume * water_density * 9.81

            # Batch: compute buoyancy centre (mean of submerged voxels)
            buoyancy_centre_w.zero_()
            submerged_voxels = voxel_pos_robot * submerged_mask.unsqueeze(-1)
            sum_submerged = submerged_voxels.sum(dim=1)  # (num_envs, 3)
            divisor = num_submerged_voxels.clamp(min=1).unsqueeze(-1)
            buoyancy_centre_w[:] = sum_submerged / divisor

            # Batch: calculate wrenches
            self.calculate_buoyancy_wrench_batch(name)

            # Batch: set forces and torques
            zero_mask = num_submerged_voxels == 0
            buoyancy_force_b[zero_mask] = 0
            buoyancy_torque_b[zero_mask] = 0

            robot.set_external_force_and_torque(
                forces=buoyancy_force_b.unsqueeze(1), torques=-buoyancy_torque_b.unsqueeze(1), env_ids=torch.arange(self.num_envs, device=self.device)
            )

    def calculate_buoyancy_wrench_batch(self, suffix: str):
        robot = getattr(self, suffix)
        R = getattr(self, f"R_{suffix}")  # (num_envs, 3, 3)
        buoyancy_force_w = getattr(self, f"buoyancy_force_w_{suffix}")  # (num_envs, 3)
        buoyancy_force_b = getattr(self, f"buoyancy_force_b_{suffix}")  # (num_envs, 3)
        buoyancy_centre_w = getattr(self, f"buoyancy_centre_w_{suffix}")  # (num_envs, 3)
        buoyancy_torque_w = getattr(self, f"buoyancy_torque_w_{suffix}")  # (num_envs, 3)
        buoyancy_torque_b = getattr(self, f"buoyancy_torque_b_{suffix}")  # (num_envs, 3)
        c_m_w = robot.data.body_com_pos_w[:, 0, :]  # (num_envs, 3)
        c_b_w = buoyancy_centre_w  # (num_envs, 3)
        c_delta_w = c_m_w - c_b_w  # (num_envs, 3)
        buoyancy_torque_w[:] = torch.cross(c_delta_w, buoyancy_force_w, dim=1)
        buoyancy_force_b[:] = torch.bmm(R.transpose(1, 2), buoyancy_force_w.unsqueeze(-1)).squeeze(-1)
        buoyancy_torque_b[:] = torch.bmm(R.transpose(1, 2), buoyancy_torque_w.unsqueeze(-1)).squeeze(-1)

# ----------- debug visualisations -----------
    def _define_voxel_markers(self, suffix: str) -> VisualizationMarkers:
        """Define markers with various different shapes."""
        if suffix != 'buoy':
            voxel_l = getattr(self, f"voxel_l_{suffix}")
            voxel_w = getattr(self, f"voxel_w_{suffix}")
            voxel_h = getattr(self, f"voxel_h_{suffix}")
            markers={
                "unsubmerged": sim_utils.CuboidCfg(
                size=(voxel_l*0.8, voxel_w*0.8, voxel_h*0.8),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0),),
                visible=False,
                ),
                "submerged": sim_utils.CuboidCfg(
                size=(voxel_l*0.8, voxel_w*0.8, voxel_h*0.8),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                visible=False,
                ),
            }
            voxel_pos_w = getattr(self, f"voxel_pos_w_{suffix}")
            num_voxels = voxel_pos_w.shape[1]
            setattr(self, f"marker_indices_{suffix}", torch.zeros(num_voxels, dtype=torch.float32, device=self.device))
            setattr(self, f"marker_translations_{suffix}", torch.zeros(num_voxels, 3, dtype=torch.float32, device=self.device))
            setattr(self, f"marker_orientations_{suffix}", torch.zeros(num_voxels, 4, dtype=torch.float32, device=self.device))
        elif suffix == 'buoy':
            markers={
                "buoy": sim_utils.SphereCfg(
                    radius=0.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    visible=True,
                ),   
            }
            angles = 72
            radius = 30
            self.num_buoys = angles * radius + 1
            self.water_surface_target_pos = torch.zeros((self.num_buoys, 3), dtype=torch.float32, device=self.device)
            for angle in range(angles):
                for r in range(radius):
                    x = (r+1) * math.cos(angle * 2 * math.pi / angles) * (100 / radius)
                    y = (r+1) * math.sin(angle * 2 * math.pi / angles) * (100 / radius)
                    z = 0.0
                    self.water_surface_target_pos[angle * radius + r] = torch.tensor([x, y, z], device=self.device)
            self.water_surface_target_pos[-1] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        marker_cfg = VisualizationMarkersCfg(
            prim_path=f"/DebugVis/Markers_{suffix}",
            markers=markers
        )
        setattr(self, f"marker_{suffix}", VisualizationMarkers(marker_cfg))

    def _visualise_voxels(self, name: str):
        idx = self.robot_names.index(name)
        voxel_slice = self.voxel_robot_slices[idx]
        marker = getattr(self, f"marker_{name}")
        marker_indices = getattr(self, f"marker_indices_{name}")
        marker_translations = getattr(self, f"marker_translations_{name}")
        marker_orientations = getattr(self, f"marker_orientations_{name}")
        robot = getattr(self, name)
        rigid_body_states = robot.data.root_link_pose_w

        # Use the first env for visualization
        mask = self.oceandeformer.submerged_mask[0, voxel_slice]
        marker_indices[:] = mask.int()
        marker_translations[:] = self.voxel_pos_w[0, voxel_slice]
        marker_orientations[:] = rigid_body_states[0][3:7]
        marker.visualize(
            translations=marker_translations,
            orientations=marker_orientations,
            marker_indices=marker_indices
        )

    def _visualise_buoys(self):
        self.buoy_indices = torch.full((self.num_buoys,), 0, device=self.device)
        self.buoy_translations = torch.clone(self.water_surface_target_pos)
        disp, _ = self.oceandeformer_buoy.compute(self.water_surface_target_pos.unsqueeze(0), self.cfg.water_level)
        self.buoy_translations[:, 2] = disp[0]
        self.buoy_orientations = torch.tensor([0.7071,0,-0.7071,0], device=self.device).repeat(self.num_buoys,1)
        self.marker_buoy.visualize(translations=self.buoy_translations, orientations=self.buoy_orientations, marker_indices=self.buoy_indices,)  



# @torch.jit.script
# def compute_rewards(
#     rew_scale_alive: float,
#     rew_scale_terminated: float,
#     rew_scale_pole_pos: float,
#     rew_scale_cart_vel: float,
#     rew_scale_pole_vel: float,
#     pole_pos: torch.Tensor,
#     pole_vel: torch.Tensor,
#     cart_pos: torch.Tensor,
#     cart_vel: torch.Tensor,
#     reset_terminated: torch.Tensor,
# ):
#     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
#     rew_termination = rew_scale_terminated * reset_terminated.float()
#     rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
#     rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
#     rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
#     total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
#     return total_reward


# Ocean Deformer
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