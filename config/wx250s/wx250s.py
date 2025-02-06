import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

import os

wx250s_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.getcwd()+"/config/wx250s/wx250s.usd",
        activate_contact_sensors = False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "waist": 0.0,
            "shoulder": 0.0,
            "elbow": 0.0,
            "forearm_roll": 0.0,
            "wrist_angle": 0.0,
            "wrist_rotate": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=5.0,
            damping=0.5,
        ),
    },
)