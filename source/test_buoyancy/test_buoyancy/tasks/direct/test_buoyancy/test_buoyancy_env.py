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
barge_mesh_path = "asset/mesh/URDF2/barge.SLDASM/meshes/base_link.STL"
tugboat_mesh_path = "asset/mesh/URDF3/TUGBOAT.SLDASM/meshes/tugboat.STL"

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

        self.oceandeformer = OceanDeformer(device=self.device)
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
        setattr(self, f"external_force_b_{suffix}", torch.zeros((self.num_envs, 3), device=self.device))
        setattr(self, f"external_torque_b_{suffix}", torch.zeros((self.num_envs, 3), device=self.device))
        setattr(self, f"teleop_force_b_{suffix}", torch.zeros((self.num_envs, 3), device=self.device))
        setattr(self, f"teleop_torque_b_{suffix}", torch.zeros((self.num_envs, 3), device=self.device))

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
        self.reset_external_force_and_torque(suffix="barge")
        self.reset_external_force_and_torque(suffix="tugboat1")
        self.reset_external_force_and_torque(suffix="tugboat2")

        self.apply_buoyancy(suffix="barge")
        self.apply_buoyancy(suffix="tugboat1")
        self.apply_buoyancy(suffix="tugboat2")

        self.apply_teleop_force_and_torque(suffix="barge")
        self.apply_teleop_force_and_torque(suffix="tugboat1")
        self.apply_teleop_force_and_torque(suffix="tugboat2")

        self.apply_external_force_and_torque(suffix="barge")
        self.apply_external_force_and_torque(suffix="tugboat1")
        self.apply_external_force_and_torque(suffix="tugboat2")

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
         # only displays the approximated value, torch will use the precise value for calculation
        setattr(self, f"voxel_volume_{suffix}", torch.tensor(voxel_volume, dtype=torch.float32, device=self.device))

        voxel_pos_w = torch.zeros((self.num_envs, valid_voxels.shape[0], 3), dtype=torch.float32, device=self.device)
        setattr(self, f"voxel_pos_w_{suffix}", voxel_pos_w)

        setattr(self, f"submerged_mask_{suffix}", torch.zeros((self.num_envs, valid_voxels.shape[0]), dtype=torch.bool))
     

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
        external_force_b = getattr(self, f"external_force_b_{suffix}")
        external_torque_b = getattr(self, f"external_torque_b_{suffix}")
        
        # Calculate the buoyancy force for each voxel
        mask = getattr(self, f"submerged_mask_{suffix}")

        for i in range(self.num_envs):
            water_height, submerged_mask = self.oceandeformer.compute(voxel_pos_w[i], self.cfg.water_level)

            mask[i] = submerged_mask

            submerged_voxels = voxel_pos_w[i][submerged_mask]
            num_submerged_voxels = submerged_voxels.shape[0] #TODO: pre-allocate memory to speed up in the future
            if num_submerged_voxels != 0:

                submerged_volume = num_submerged_voxels * voxel_volume.item()

                buoyancy_force_w[i, 2] = submerged_volume * water_density * 9.81

                buoyancy_centre_w[i] = submerged_voxels.mean(dim=0)

                self.calculate_buoyancy_wrench(i, suffix=suffix)

            else:
                buoyancy_force_w[i].zero_()
                buoyancy_centre_w[i].zero_()
                buoyancy_torque_b[i].zero_()
                buoyancy_force_b[i].zero_()
            external_force_b[i] += buoyancy_force_b[i]
            external_torque_b[i] += -buoyancy_torque_b[i]

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

    def apply_external_force_and_torque(self, suffix: str):
        robot = getattr(self, suffix)
        external_force_b = getattr(self, f"external_force_b_{suffix}")
        external_torque_b = getattr(self, f"external_torque_b_{suffix}")
        for i in range(self.num_envs):
            robot.set_external_force_and_torque(forces=external_force_b[i], torques=external_torque_b[i], env_ids=i)

    def reset_external_force_and_torque(self, suffix: str):
        external_force_b = getattr(self, f"external_force_b_{suffix}")
        external_torque_b = getattr(self, f"external_torque_b_{suffix}")
        external_force_b.zero_()
        external_torque_b.zero_()

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
                "buoy": sim_utils.SphereCfg(
                    radius=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    visible=False,
                ),   
                # "arrow_z": sim_utils.UsdFileCfg(
                #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                #     scale=(1, 1, 5),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                # ),

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
        mask_all = getattr(self, f"submerged_mask_{suffix}")
        rigid_body_states = robot.data.root_link_pose_w

        # mask = voxel_pos_w[0, :, 2] < 5.0# TODO
        mask = mask_all[0]


        marker_indices[:] = mask.int()

        marker_translations[:] = voxel_pos_w[0]

        marker_orientations[:] = rigid_body_states[0][3:7]

        marker.visualize(translations=marker_translations, orientations=marker_orientations, marker_indices=marker_indices)  

    # def _visualise_markers(self):
    #     rigid_body_states = self.robot.data.root_link_pose_w
    #     # mask = self.voxel_pos_w[0, :, 2] < self.cfg.water_level
    #     self.marker_indices[:self.voxel_pos_w.shape[1]] = self.submerged_mask
    #     self.marker_translations[:self.voxel_pos_w.shape[1]] = self.voxel_pos_w[0]
    #     self.marker_orientations[:self.voxel_pos_w.shape[1]] = rigid_body_states[0][3:7]

    #     # water related markers
    #     self.water_surface_target_pos = torch.tensor([[50,0,0],
    #                                                   [0,50,0],
    #                                                   [-50,0,0],
    #                                                   [0,-50,0],
    #                                                   [25,0,0],
    #                                                   [0,25,0],
    #                                                   [-25,0,0],
    #                                                   [0,-25,0],
    #                                                   [35.35/2,35.35/2,0],
    #                                                   [-35.35/2,35.35/2,0],
    #                                                   [35.35/2,-35.35/2,0],
    #                                                   [-35.35/2,-35.35/2,0],
    #                                                   [0,0,0],], device=self.device)
    #     # sample_point = np.array([  [ -25.22   ,    0.     , -499.365  ],
    #     #                         [ 242.335  ,    0.     , -347.63   ],
    #     #                         [ 207.19499,    0.     , -251.9    ],
    #     #                         [ 287.47498,    0.     , -168.3    ],
    #     #                         [-253.91998,    0.     ,  -88.265  ],
    #     #                         [  25.425  ,    0.     ,  -11.29   ],
    #     #                         [ 170.26999,    0.     ,   64.31   ],
    #     #                         [ 109.64   ,    0.     ,  142.92   ],
    #     #                         [ 136.72   ,    0.     ,  224.76999],
    #     #                         [-248.455  ,    0.     ,  316.755  ],
    #     #                         [  66.65   ,    0.     ,  431.295  ]], dtype=np.float32)
    #     # sample_point = sample_point[:, [0,2,1]]
    #     # self.water_surface_target_pos = torch.tensor(sample_point, device=self.device)
    #     self.marker_indices[-13:] = torch.full((13,), 2, device=self.device)
    #     self.marker_translations[-13:] = self.water_surface_target_pos
    #     disp, _ = self.oceandeformer.compute(self.water_surface_target_pos, self.cfg.water_level)
    #     self.marker_translations[-13:, 2] = disp
    #     self.marker_orientations[-13:] = torch.tensor([0.7071,0,-0.7071,0], device=self.device).repeat(13,1)

    #     self.marker.visualize(translations=self.marker_translations, orientations=self.marker_orientations, marker_indices=self.marker_indices,)  

# ----------- teleoperation -----------
    def read_teleop_force_and_torque(self, fx, fy, tz, suffix):
        teleop_force_b = getattr(self, f"teleop_force_b_{suffix}")
        teleop_torque_b = getattr(self, f"teleop_torque_b_{suffix}")
        teleop_force_b.zero_()
        teleop_torque_b.zero_()
        teleop_force_b[0, 2] = -fx
        teleop_force_b[0, 0] = fy # only the first env
        teleop_torque_b[0, 1] = -tz

    def read_teleop_vel(self, vx, vy, oz, suffix):
        teleop_force_b = getattr(self, f"teleop_force_b_{suffix}")
        teleop_torque_b = getattr(self, f"teleop_torque_b_{suffix}")
        teleop_force_b.zero_()
        teleop_torque_b.zero_()

        teleop_force_b[0, 2] = -vx
        teleop_force_b[0, 0] = vy # only the first env
        teleop_torque_b[0, 1] = -oz

    def apply_teleop_force_and_torque(self, suffix: str):
        teleop_force_b = getattr(self, f"teleop_force_b_{suffix}")
        teleop_torque_b = getattr(self, f"teleop_torque_b_{suffix}")
        external_force_b = getattr(self, f"external_force_b_{suffix}")
        external_torque_b = getattr(self, f"external_torque_b_{suffix}")
        for i in range(self.num_envs):
            external_force_b[i] += teleop_force_b[i]
            external_torque_b[i] += teleop_torque_b[i]

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
        # self.rand_arr_profile = torch.tensor(
        #     np.load('/home/marmot/isaacsim/exts/omni.ocean-0.4.1/omni/ocean/nodes/rand_arr_profile.npy')
        #     , dtype=torch.float32, device=device
        # )
        # self.rand_arr_points = torch.tensor(
        #     np.load('/home/marmot/isaacsim/exts/omni.ocean-0.4.1/omni/ocean/nodes/rand_arr_points.npy')
        #     , dtype=torch.float32, device=device
        # )

        self.update_attr()

        self.profile = torch.zeros(self.profile_res, 3, dtype=torch.float32, device=self.device)

    def update_attr(self):
        self.inputs_waveAmplitude = self.node.get_attribute("inputs:waveAmplitude").set(0.2)
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
        self.inputs_time = self.node.get_attribute("inputs:time").get()
        self.time = torch.tensor(float(self.inputs_time), device=self.device)

    def compute(self, points, water_level):
        time1 = time.time()
        self.update_time_attr()
        # self.time = 123.456
        time2 = time.time()
        self.update_profile()
        time3 = time.time()
        disp, mask = self.update_points(points, water_level)
        time4 = time.time()

        # print("time for update_time", time2-time1)
        # print("time for update_profile", time3-time2)
        # print("time for update_points", time4-time3)

        return disp, mask

    def update_profile(self):
        # Compute omega range
        # print(self.profile_res, self.profile_data_num, self.minWavelength, self.maxWavelength, self.profile_extent, self.time, self.windspeed, self.waterdepth)

        omega0 = np.sqrt(2.0 * np.pi * self.gravity / self.minWavelength)
        omega1 = np.sqrt(2.0 * np.pi * self.gravity / self.maxWavelength)
        omega = omega0 + (omega1 - omega0) * torch.arange(self.profile_data_num, device=self.device) / self.profile_data_num
        omega_delta = (omega1 - omega0) / self.profile_data_num

        # Spatial sample positions
        x_idx = torch.arange(self.profile_res, device=self.device)
        space_pos = self.profile_extent * x_idx / self.profile_res  # (res,)

        # Wavenumber and phase
        k = omega ** 2 / self.gravity                     # (N,)
        phase = -self.time * omega + self.rand_arr_profile * 2.0 * np.pi  # (N,)

        # Amplitudes from spectrum
        amp = 10000.0 * torch.sqrt(torch.abs(2.0 * omega_delta * self.TMA_spectrum(omega, 100.0, 3.3)))  # (N,)

        # Helper to compute displacement at given spatial position
        def compute_disp_at(pos):  # pos: (res,)
            angle = k.unsqueeze(1) * pos.unsqueeze(0) + phase.unsqueeze(1)  # (N, res)
            sin_phase = torch.sin(angle)
            cos_phase = torch.cos(angle)
            disp_x = (amp.unsqueeze(1) * sin_phase).sum(dim=0)  # (res,)
            disp_y = -(amp.unsqueeze(1) * cos_phase).sum(dim=0)  # (res,)
            return torch.stack([disp_x, disp_y, torch.zeros_like(disp_x)], dim=1) / self.profile_data_num  # (res, 3)

        # Compute displacements at three spatial positions
        disp1 = compute_disp_at(space_pos)                          # space_pos_1
        disp2 = compute_disp_at(space_pos + self.profile_extent)    # space_pos_2
        disp3 = compute_disp_at(space_pos - self.profile_extent)    # space_pos_3

        # Blending coefficients
        s = x_idx / self.profile_res  # (res,)
        c1 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        c2 = -2.0 * s**3 + 3.0 * s**2

        # Final profile
        self.profile = disp1 + c1.unsqueeze(1) * disp2 + c2.unsqueeze(1) * disp3  # (res, 3)

    def update_points(self, points, water_level=0.0):

        d = torch.arange(self.direction_count, device=self.device, dtype=torch.float32)
        r = d * 2.0 * np.pi / self.direction_count + 0.02
        dir_x = torch.cos(r)
        dir_y = torch.sin(r)

        points_xy = torch.stack([points[:, 0], -points[:, 1]], dim=1).float()
        dirs = torch.stack([dir_x, dir_y], dim=1)  # shape: (D, 2)
        dots = torch.matmul(points_xy, dirs.T)     # shape: (N, D)

        x_proj = dots / (self.profile_extent * self.scale) + self.rand_arr_points  # shape: (N, D)
        x_scaled = x_proj * self.profile_res

        pos0_raw = torch.floor(x_scaled).long()
        pos0 = pos0_raw.clone()
        pos0[pos0_raw < 0] = pos0[pos0_raw < 0] + self.profile_res - 1
        pos0 = pos0.remainder(self.profile_res)
        pos1 = (pos0 + 1) % self.profile_res

        w = x_scaled - torch.floor(x_scaled)           # shape: (N, D)
        p0 = self.profile[pos0, 1]                # (N, D)
        p1 = self.profile[pos1, 1]                # (N, D)

        direction_exp = self.direction.unsqueeze(-1)  # shape: (D, 1)
        r_exp = r.unsqueeze(0)                        # shape: (1, D)
        t = torch.abs(direction_exp - r_exp)
        t = torch.where(t > np.pi, 2 * np.pi - t, t)
        t = t ** 1.2

        directionality_exp = self.directionality.unsqueeze(-1)  # shape: (D, 1)
        dir_amp = (2 * t**3 - 3 * t**2 + 1) + (-2 * t**3 + 3 * t**2) * (1.0 - directionality_exp)
        dir_amp = dir_amp / (1.0 + 10.0 * directionality_exp)

        # dir_amp_diag = torch.diagonal(dir_amp, 0)  # shape: (D,)

        # prof_height = dir_amp_diag.unsqueeze(0) * ((1.0 - w) * p0 + w * p1)  # shape: (N, D)
        
        # disp_z = torch.sum(dir_y.unsqueeze(0) * prof_height, dim=1)         # shape: (N,)

        disp_z = torch.matmul(((1.0 - w) * p0 + w * p1), dir_amp.mT).squeeze(1)  # shape: (N,)

        disp_z = self.amplitude * disp_z / self.direction_count
        disp_z += water_level

        submerged_mask = disp_z > points[:, 2]  # IsaacLab: Z is up

        return disp_z, submerged_mask

    def TMA_spectrum(self, omega, fetch_km, gamma):
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