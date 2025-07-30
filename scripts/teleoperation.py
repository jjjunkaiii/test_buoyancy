# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.task = 'Template-Test-Buoyancy-Direct-v0'
args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import test_buoyancy.tasks  # noqa: F401

import pygame
# Xbox Series X Controller
# Xbox One S Controller
def process_joystick(controller):
    x1 = - 2500000 * controller.get_axis(1)
    y1 = - 2500000 * controller.get_axis(0)
    if abs(x1) < 2.5:
        x1 = 0
    if abs(y1) < 2.5:
        y1 = 0
    x2 = - 25000000 * controller.get_axis(3) * 2 * np.pi / 360
    if abs(x2) < - 25000000 * 2 * np.pi / 3600:
        x2 = 0
    # speed = np.sqrt(x1 * x1 + y1 * y1)
    # course = np.rad2deg(np.arctan2(y1, x1))
    return x1, y1, x2 # fx, fy, tz

def main():

    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # init joysticks
    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for joystick in joysticks:
        if joystick.get_name() == 'Xbox One S Controller':
            gamepadS = pygame.joystick.Joystick(joystick.get_id())
        elif joystick.get_name() == 'Xbox Series X Controller': 
            gamepadX = pygame.joystick.Joystick(joystick.get_id())
        else:
            pass
    # reset environment
    env.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            pygame.event.pump()
            # sample actions from -1 to 1
            fxS, fyS, tzS = process_joystick(gamepadS)
            fxX, fyX, tzX = process_joystick(gamepadX)
            env.unwrapped.read_teleop_force_and_torque(fxS, fyS, tzS, 'tugboat1')
            env.unwrapped.read_teleop_force_and_torque(fxX, fyX, tzX, 'tugboat2')

            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
