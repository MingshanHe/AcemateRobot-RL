# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from acemate.tasks.manager_based.tennis_bouncing.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
# from isaaclab_assets import KINOVA_GEN3_N7_CFG  # isort: skip

##
# Robot configuration
##
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

ACEMATEROBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/root/IsaacLab/scripts/acemate/robot_description/usd/acemate.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),   
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        activate_contact_sensors=False,
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
        },
        pos=(0.0, 0.0, 0.2),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint1","joint2","joint3","joint4","joint5"],
            effort_limit={
                "joint1": 1000.0,
                "joint2": 1000.0,
                "joint3": 1000.0,
                "joint4": 1000.0,
                "joint5": 1000.0,
            },
            stiffness={
                "joint1": 50.0,
                "joint2": 50.0,
                "joint3": 50.0,
                "joint4": 50.0,
                "joint5": 50.0,
            },
            damping={
                "joint1": 1.0,
                "joint2": 1.0,
                "joint3": 1.0,
                "joint4": 1.0,
                "joint5": 1.0,
            },
        ),
    },
)

##
# Tennis Ball configuration
##
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
BALL_CONFIGURATION = RigidObjectCfg(
    prim_path="/World/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.033,  # 网球半径
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.9, 0.2)), # 网球绿
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.057),
        # 关键：物理材质
        physics_material=sim_utils.RigidBodyMaterialCfg(
            restitution=0.85,  # 弹性系数 (0-1)，越高越弹
            static_friction=0.5,
            dynamic_friction=0.5,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(), # 启用碰撞
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8), # 初始高度 2米
    ),
)


##
# Environment configuration
##


@configclass
class TennisBouncingEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10
        self.scene.robot = ACEMATEROBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ball = BALL_CONFIGURATION.replace(prim_path="{ENV_REGEX_NS}/Ball")
        
        # root_state = self.scene.ball._data.default_root_state.clone()

        ball = RigidObject(cfg=BALL_CONFIGURATION)
        # root_state = ball.data.default_root_state.clone()
        # root_state[:, 7:10] = torch.tensor([[3.0, 0.0, 0.0]], device=sim.device) # X轴初速度 5m/s
        # ball.write_root_state_to_sim(root_state)
        # ball.reset()
        
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        
        # override rewards
        # self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link5"]
        # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link5"]
        # self.rewards.end_effector_heading_x_axis_tracking.params["asset_cfg"].body_names = ["link5"]
        # self.rewards.end_effector_heading_x_axis_velocity.params["asset_cfg"].body_names = ["link5"]
        self.rewards.end_ball_position_tracking.params["asset_cfg"].body_names = ["link5"]
        self.rewards.hit_ball_reward.params["asset_cfg"].body_names = ["link5"] 
        # override actions
        self.actions.arm_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=["joint1","joint2","joint3","joint4","joint5"], scale=5.0, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "link5"
        # self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

@configclass
class TennisBouncingEnvCfg_PLAY(TennisBouncingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False