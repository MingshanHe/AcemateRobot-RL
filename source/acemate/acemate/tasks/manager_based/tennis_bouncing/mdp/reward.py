# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
    
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def heading_x_axis_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    # extract the local z axis
    x_axis_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(curr_quat_w.shape[0], 1)
    des_x_w = quat_apply(des_quat_w, x_axis_local)   # desired Z axis in the world frame
    curr_x_w = quat_apply(curr_quat_w, x_axis_local) # current Z axis in the world frame
    # calculate the angle between the two z axes
    # cos_theta = a · b / (|a|*|b|)
    dot_product = torch.sum(des_x_w * curr_x_w, dim=1)
    # (clamp to [-1, 1])
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    return 1.0 - dot_product

def x_axis_striking_velocity_reward(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg,
    target_speed: float = 2.0,  # 假设目标击球速度是 2m/s
    dist_threshold: float = 0.1 # 只有距离小于 10cm 才开始奖励速度
) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    # obtain the desired and current orientations
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    x_axis_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(curr_quat_w.shape[0], 1)
    curr_x_w = quat_apply(curr_quat_w, x_axis_local)

    # obtain the  current orientations velocity
    #    body_vel_w: [num_envs, num_bodies, 6] (前3是线速度，后3是角速度)
    curr_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], :3]
    projected_speed = torch.sum(curr_vel_w * curr_x_w, dim=1)

    vel_error = torch.abs(projected_speed - target_speed)
    
    r_vel = torch.exp(-vel_error / 0.5) 

    is_close = (distance < dist_threshold).float()
    
    return is_close * r_vel






def calculate_table_tennis_rewards(ball_data, table_params):
    """
    ball_data: 包含当前帧和前一帧球的状态 (pos, vel)
    table_params: 球桌的边界坐标 [x_min, x_max, y_min, y_max]
    """
    reward_1 = 0.0
    reward_2 = 0.0
    
    # --- Reward 1: State Transition + Landing Prediction ---
    # 判定判定状态转换：球的速度方向是否从“靠近机器人”变为“远离机器人”
    # 假设机器人位于 x 负半区，球向 x 正方向飞代表击球成功
    is_hit = ball_data['prev_vel_x'] <= 0 and ball_data['curr_vel_x'] > 0
    
    if is_hit:
        # 基础分 (State Transition)
        base_bonus = 1.0
        
        # 预测落点分 (Landing Bonus)
        # 使用简化的抛体运动或模型预测球最终落地的位置 (pred_x, pred_y)
        pred_land_pos = predict_landing_point(ball_data['curr_pos'], ball_data['curr_vel'])
        
        # 计算预测落点与对方球桌中心或目标的距离
        target_pos = table_params['opponent_center']
        dist = torch.norm(pred_land_pos - target_pos)
        
        # 使用指数函数将距离映射到 [0, 1]
        landing_bonus = torch.exp(-dist) 
        
        reward_1 = base_bonus + landing_bonus # 范围 [1, 2]

    # --- Reward 2: Actual Landing Bonus ---
    # 判定球是否真正落在对方半场（通常由仿真引擎的碰撞事件触发）
    if ball_data['ball_has_bounced_on_table']:
        actual_pos = ball_data['bounce_pos']
        
        # 检查坐标是否在对方半场边界内
        if (table_params['x_min'] < actual_pos[0] < table_params['x_max'] and
            table_params['y_min'] < actual_pos[1] < table_params['y_max']):
            
            # 可以是固定值 1.0，也可以根据离边缘的距离给分
            reward_2 = 1.0 # 范围 [0, 1]
            
    return reward_1, reward_2




def distance_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_end_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    curr_ball_pos_w = env.scene["ball"].data.root_pos_w
    return torch.norm(curr_end_pos_w - curr_ball_pos_w, dim=1)

def hit_ball_flag(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_ball_vel_w = env.scene["ball"].data.root_lin_vel_w
    ball_vel_x = curr_ball_vel_w[:, 0]
    direction_mask = (ball_vel_x < 0).float()
    return direction_mask


def hit_ball_reward_velocity_tracking(env, target_vel_x=-2.0):
    curr_ball_vel_w = env.scene["ball"].data.root_lin_vel_w
    ball_vel_x = curr_ball_vel_w[:, 0]
    
    # 计算当前速度与目标的差距
    # 使用平方项可以惩罚较大的偏差
    error = ball_vel_x - target_vel_x
    
    # 转换为奖励：偏差越小，奖励越高（最高为0，或者加个偏移量）
    reward = error 
    return reward