# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from isaaclab.assets import RigidObject, RigidObjectCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

import os
extention_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))

barge_path = "asset/mesh/URDF2/barge.SLDASM/urdf/barge.usd"
tugboat_path = "asset/mesh/URDF3/TUGBOAT.SLDASM/urdf/tugboat.usd"


@configclass
class TestBuoyancyEnvCfg(DirectRLEnvCfg):
    water_level = 5.0
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    # robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    bargeCfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/barge",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=os.path.join(extention_path,barge_path),
                    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                        articulation_enabled=False,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        max_linear_velocity=1000.0,
                        max_angular_velocity=1000.0,
                        linear_damping=2,
                        angular_damping=3,
                        enable_gyroscopic_forces=True,
                        ),
                    visible = True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 20.0),
                    rot=(0, 0, 0.7071, -0.7071),
                    ),
                debug_vis = True

    )

    tugCfg1 = RigidObjectCfg(
                prim_path="/World/envs/env_.*/tugboat1",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=os.path.join(extention_path,tugboat_path),
                    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                        articulation_enabled=False,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        max_linear_velocity=1000.0,
                        max_angular_velocity=1000.0,
                        linear_damping=2,
                        angular_damping=3,
                        enable_gyroscopic_forces=True,
                        ),
                    visible = True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    # pos=(0, 0, 20.0),
                    pos=(20.0, 20.0, 20.0),
                    rot=(-0.7071, 0.7071, 0, 0),
                    ),
                debug_vis = True

    )

    tugCfg2 = RigidObjectCfg(
                prim_path="/World/envs/env_.*/tugboat2",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=os.path.join(extention_path,tugboat_path),
                    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                        articulation_enabled=False,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        max_linear_velocity=1000.0,
                        max_angular_velocity=1000.0,
                        linear_damping=2,
                        angular_damping=3,
                        enable_gyroscopic_forces=True,
                        ),
                    visible = True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-20.0, 20.0, 20.0),
                    rot=(-0.7071, 0.7071, 0, 0),
                    ),
                debug_vis = True

    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    # cart_dof_name = "slider_to_cart"
    # pole_dof_name = "cart_to_pole"
    # - action scale
    # action_scale = 100.0  # [N]
    # # - reward scales
    # rew_scale_alive = 1.0
    # rew_scale_terminated = -2.0
    # rew_scale_pole_pos = -1.0
    # rew_scale_cart_vel = -0.01
    # rew_scale_pole_vel = -0.005
    # # - reset states/conditions
    # initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    # max_cart_pos = 3.0  # reset if cart exceeds this position [m]