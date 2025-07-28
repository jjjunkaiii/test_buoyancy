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
barge_mesh_path = "asset/URDF2/barge.SLDASM/meshes/base_link.STL"
tugboat_mesh_path = "asset/URDF3/TUGBOAT.SLDASM/meshes/tugboat.STL"

class TestBuoyancyEnv(DirectRLEnv):
    cfg: TestBuoyancyEnvCfg

    def __init__(self, cfg: TestBuoyancyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # init buoyancy related buffers
        # No.1
        self._init_robot_buffers("barge")
        self._init_robot_buffers("tugboat1")
        self._init_robot_buffers("tugboat2")

        # No.2
        self._load_voxel(mesh_path=barge_mesh_path, suffix="barge")
        self._load_voxel(mesh_path=tugboat_mesh_path, suffix="tugboat1")
        self._load_voxel(mesh_path=tugboat_mesh_path, suffix="tugboat2")


        self.debug_visualisation = True
        # No.3
        if self.debug_visualisation:
            self._define_markers("barge")
            self._define_markers("tugboat1")
            self._define_markers("tugboat2")


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
                scale = [1, 0.1, 1],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
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
        # No.6
        self.apply_buoyancy(suffix="barge")
        self.apply_buoyancy(suffix="tugboat1")
        self.apply_buoyancy(suffix="tugboat2")


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
        # No.7
        if self.debug_visualisation:
            self._visualise_markers("barge")
            self._visualise_markers("tugboat1")
            self._visualise_markers("tugboat2")


# ----------- buoyancy related functions -----------
    def _load_voxel(self, mesh_path: str, suffix: str):
        voxel_res = 64
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
         # only displays the approximated value, torch will use the precise value for calculation
        setattr(self, f"voxel_volume_{suffix}", torch.tensor(voxel_volume, dtype=torch.float32, device=self.device))

        voxel_pos_w = torch.zeros((self.num_envs, valid_voxels.shape[0], 3), dtype=torch.float32, device=self.device)
        setattr(self, f"voxel_pos_w_{suffix}", voxel_pos_w)
     

    def get_pose_mat(self, suffix: str):

        robot = getattr(self, suffix)
        rigid_body_states = robot.data.root_link_pose_w 
        rot_mats = getattr(self, f"rot_mats_{suffix}")
        R = getattr(self, f"R_{suffix}")
        
        for i in range(self.num_envs):

            R[i] = matrix_from_quat(rigid_body_states[i][3:7])

            rot_mats[i][:3, :3] = R[i]

            rot_mats[i][:3, 3] = rigid_body_states[i][:3].T

            rot_mats[i][3, 3] = 1.0
        
    def get_voxel_pos_w(self,suffix: str):

        self.get_pose_mat(suffix=suffix)
        rot_mats = getattr(self, f"rot_mats_{suffix}")
        valid_voxels = getattr(self, f"valid_voxels_{suffix}")
        voxel_pos_w = getattr(self, f"voxel_pos_w_{suffix}")
        for i in range(self.num_envs):
            voxel_pos_w[i] = (rot_mats[i][:3, :3] @ valid_voxels.T + rot_mats[i][:3, 3].unsqueeze(1)).T

    def apply_buoyancy(self, water_level=0.0, water_density=1500.0, suffix: str = "barge"):
        """
        Apply buoyancy force to the rigid object based on the voxel positions and water level.
        """
        self.get_voxel_pos_w(suffix=suffix)

        voxel_pos_w = getattr(self, f"voxel_pos_w_{suffix}")
        voxel_volume = getattr(self, f"voxel_volume_{suffix}")
        buoyancy_force_w = getattr(self, f"buoyancy_force_w_{suffix}")
        buoyancy_centre_w = getattr(self, f"buoyancy_centre_w_{suffix}")
        buoyancy_force_b = getattr(self, f"buoyancy_force_b_{suffix}")
        buoyancy_torque_b = getattr(self, f"buoyancy_torque_b_{suffix}")
        robot = getattr(self, suffix)
        
        # Calculate the buoyancy force for each voxel
        
        for i in range(self.num_envs):

            submerged_voxels = voxel_pos_w[i][voxel_pos_w[i][:, 2] < water_level]
            num_submerged_voxels = submerged_voxels.shape[0]
            if num_submerged_voxels != 0:

                submerged_volume = num_submerged_voxels * voxel_volume.item()

                buoyancy_force_w[i, 2] = submerged_volume * water_density * 9.81

                buoyancy_centre_w[i] = submerged_voxels.mean(dim=0)

                self.calculate_buoyancy_wrench(i, suffix=suffix)

                robot.set_external_force_and_torque(forces=buoyancy_force_b[i], torques=-buoyancy_torque_b[i], env_ids=i)
            else:

                robot.set_external_force_and_torque(forces=torch.zeros(3), torques=torch.zeros(3), env_ids=i)

    def calculate_buoyancy_wrench(self, env, suffix: str):
        robot = getattr(self, suffix)
        R = getattr(self, f"R_{suffix}")
        buoyancy_force_w = getattr(self, f"buoyancy_force_w_{suffix}")
        buoyancy_force_b = getattr(self, f"buoyancy_force_b_{suffix}")
        buoyancy_centre_w = getattr(self, f"buoyancy_centre_w_{suffix}")
        buoyancy_torque_w = getattr(self, f"buoyancy_torque_w_{suffix}")
        buoyancy_torque_b = getattr(self, f"buoyancy_torque_b_{suffix}")

        c_m_w = robot.data.body_com_pos_w[env][0]

        c_b_w = buoyancy_centre_w[env]

        c_delta_w = c_m_w - c_b_w

        buoyancy_torque_w[env] = torch.linalg.cross(c_delta_w, buoyancy_force_w[env])

        buoyancy_force_b[env] = torch.matmul(R[env].T, buoyancy_force_w[env])

        buoyancy_torque_b[env] = torch.matmul(R[env].T, buoyancy_torque_w[env])

# ----------- debug visualisations -----------
    def _define_markers(self, suffix: str) -> VisualizationMarkers:
        """Define markers with various different shapes."""
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
        voxel_l = getattr(self, f"voxel_l_{suffix}")
        voxel_w = getattr(self, f"voxel_w_{suffix}")
        voxel_h = getattr(self, f"voxel_h_{suffix}")

        marker_cfg = VisualizationMarkersCfg(
            prim_path=f"/Markers_{suffix}",
            markers={
                # "frame": sim_utils.UsdFileCfg(
                #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                #     scale=(1, 1, 1),
                # ),
                # "arrow_x": sim_utils.UsdFileCfg(
                #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                #     scale=(1.0, 0.5, 0.5),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                # ),
                "cube_red": sim_utils.CuboidCfg(
                size=(voxel_l*0.8, voxel_w*0.8, voxel_h*0.8),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0),),
                visible=False,
                ),
                "cube_blue": sim_utils.CuboidCfg(
                size=(voxel_l*0.8, voxel_w*0.8, voxel_h*0.8),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                visible=False,
                ),
            },
        )

        setattr(self, f"marker_{suffix}", VisualizationMarkers(marker_cfg))
        voxel_pos_w = getattr(self, f"voxel_pos_w_{suffix}")
        num_voxels = voxel_pos_w.shape[1]
        setattr(self, f"marker_indices_{suffix}", torch.zeros(num_voxels, dtype=torch.float32, device=self.device))
        setattr(self, f"marker_translations_{suffix}", torch.zeros(num_voxels, 3, dtype=torch.float32, device=self.device))
        setattr(self, f"marker_orientations_{suffix}", torch.zeros(num_voxels, 4, dtype=torch.float32, device=self.device))


    def _visualise_markers(self, suffix: str):
        robot = getattr(self, suffix)
        voxel_pos_w = getattr(self, f"voxel_pos_w_{suffix}")
        marker = getattr(self, f"marker_{suffix}")
        marker_indices = getattr(self, f"marker_indices_{suffix}")
        marker_translations = getattr(self, f"marker_translations_{suffix}")
        marker_orientations = getattr(self, f"marker_orientations_{suffix}")

        rigid_body_states = robot.data.root_link_pose_w

        mask = voxel_pos_w[0, :, 2] < 0

        marker_indices[:] = mask.int()

        marker_translations[:] = voxel_pos_w[0]

        marker_orientations[:] = rigid_body_states[0][3:7]

        marker.visualize(translations=marker_translations, orientations=marker_orientations, marker_indices=marker_indices)  

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