# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Reach-WX250s-Vision-v0",
    entry_point="omni.isaac.lab.envs:DirectRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.widowx_camera_env_cfg:WidowXVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:wx250sReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-WX250s-Vision-v0",
    entry_point="omni.isaac.lab.envs:DirectRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.widowx_camera_env_cfg:WidowXVisionEnvPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:wx250sReachPPORunnerCfg",
    },
)
