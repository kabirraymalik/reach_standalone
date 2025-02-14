import os
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ActuatorNetMLPCfg, ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

GO1_ACTUATOR_CFG = ActuatorNetMLPCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/Unitree/unitree_go1.pt",
    pos_scale=-1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="pos_vel",
    input_idx=[0, 1, 2],
    effort_limit=23.7,  # taken from spec sheet
    velocity_limit=30.0,  # taken from spec sheet
    saturation_effort=23.7,  # same as effort limit
)
"""Configuration of Go1 actuators using MLP model.

Actuator specifications: https://shop.unitree.com/products/go1-motor

This model is taken from: https://github.com/Improbable-AI/walk-these-ways
"""

wx250s_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(os.getcwd()+"/config/widowx_go1_direct/assets/wx250s.usd"),
        activate_contact_sensors = False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # lin_vel=(100.0, 100.0, 100.0),
        # # set a random yaw velocity within 0.05 rad/s
        # ang_vel=(0.0, 0.0, random.uniform(-0.05, 0.05)),
        joint_pos={
            "waist": 0.0,
            "shoulder": 0.0,
            "elbow": 0.0,
            "forearm_roll": 0.0,
            "wrist_angle": 0.0,
            "wrist_rotate": 0.0,
            "left_finger": 0.037,
            "right_finger": 0.37,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "left_finger", "right_finger"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=400.0,
            damping=10.0,
        ),
    },
)
"""Configuration of wx250s arm using implicit actuator models."""
"""Paramters adopted from https://github.com/haosulab/ManiSkill/blob/e2954732ba37c7f38816b3d038a0aa41da4275b3/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/base_env.py#L59"""

WIDOW_GO1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(os.getcwd()+"config/widowx_go1_direct/assets/widowGo1.usd"),
        # usd_path="/home/zhou/visual_wholebody/third_party/isaacgym/assets/urdf/widowGo1_new/urdf/widowGo1/widowGo1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0,
            fix_root_link=False
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35), # initial position of the robot base height
        joint_pos={
            # dog leg
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            # arm
            "widow_waist": 0.0,
            "widow_shoulder": 0.0,
            "widow_elbow": 0.0,
            "widow_forearm_roll": 0.0,
            "widow_wrist_angle": 0.0,
            "widow_wrist_rotate": 0.0,
            "widow_left_finger": 0.037,
            "widow_right_finger": -0.037,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": GO1_ACTUATOR_CFG,
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["widow_waist", "widow_shoulder", "widow_elbow", "widow_forearm_roll", 
                              "widow_wrist_angle", "widow_wrist_rotate", "widow_left_finger", "widow_right_finger"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness={
                "widow_waist": 1169.7891719504198,
                "widow_shoulder": 730.0,
                "widow_elbow": 808.4601346394447,
                "widow_forearm_roll": 1229.1299089624076,
                "widow_wrist_angle": 1272.2760456418862,
                "widow_wrist_rotate": 1056.3326605132252,
                "widow_left_finger": 10000,
                "widow_right_finger": 10000,
            },
            damping= {
                "widow_waist": 330.0,
                "widow_shoulder": 180.0,
                "widow_elbow": 152.12036565582588,
                "widow_forearm_roll": 309.6215302722146,
                "widow_wrist_angle": 201.04998711007383,
                "widow_wrist_rotate": 269.51458932695414,
                "widow_left_finger": 200,
                "widow_right_finger": 200,
            }
        ),
    },
)