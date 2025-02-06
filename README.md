Modified from Nvidia IsaacLab, rsl_rl

Dependencies:

IsaacSim 4.5.0, IsaacLab

Usage:

in conda environment with isaaclab, isaacgym:

python rsl_rl/train.py --task Isaac-Reach-wx250s-v0 --num_envs 16 +HYDRA_FULL_ERROR=1 (--headless for faster)