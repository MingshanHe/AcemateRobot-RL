"""
Isaac Lab Hello World: Bouncing Tennis Ball
"""

import argparse
import csv  # 用于保存文件
import numpy as np
from isaaclab.app import AppLauncher

# 1. 启动仿真 App (必须在导入其他 isaaclab 模块之前)
parser = argparse.ArgumentParser(description="Bouncing Ball Demo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入仿真相关的模块 (必须在 App 启动后)
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

def main():
    # --- 配置仿真环境 ---
    sim_cfg = SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    
    # 设置主光照和地面
    sim.set_camera_view([2.0, 0.0, 1.0], [0.0, 0.0, 0.5]) # 设置相机位置
    
    # 创建地面 (Grid Floor)
    # 地面也需要物理属性，否则球可能弹不起来
    cfg_ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, 
            restitution=0.8  # 地面弹性
        )
    )
    cfg_ground.func("/World/Ground", cfg_ground)

    # --- 配置网球 ---
    # 网球标准: 半径 ~0.033m, 质量 ~0.057kg
    ball_cfg = RigidObjectCfg(
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
    sim.set_camera_view(
        eye=[0.0, 5.0, 2.0],   # 相机站在 Y=5 的位置
        target=[2.0, 0.0, 0.0] # 看着 X=2 的地方
    )
    # 在仿真中创建这个对象
    ball = RigidObject(cfg=ball_cfg)

    # --- 运行仿真循环 ---
    sim.reset() # 初始化物理引擎
    print("[INFO]: Simulation started. Look at the bouncing ball!")

    root_state = ball.data.default_root_state.clone()
    root_state[:, 7:10] = torch.tensor([[3.0, 0.0, 0.0]], device=sim.device) # X轴初速度 5m/s
    ball.write_root_state_to_sim(root_state)
    ball.reset()


    step_count = 0
    max_steps = 500  # 设置记录多少步（例如5秒钟的数据）
    data_log = []
    
    while simulation_app.is_running() and step_count < max_steps:
        # 1. 物理步进
        sim.step()
        ball.update(0.01)

        # 2. 提取球的位置 (World Frame)
        # ball.data.root_pos_w 的形状是 (num_envs, 3)
        pos = ball.data.root_pos_w[0].cpu().numpy() 
        
        # 3. 将当前步的时间和坐标存入列表
        data_log.append({
            "step": step_count,
            "time": step_count * 0.01,
            "x": pos[0],
            "y": pos[1],
            "z": pos[2]
        })

        step_count += 1

    # --- 4. 模拟结束后保存为 CSV ---
    csv_file = "ball_trajectory.csv"
    keys = data_log[0].keys()  # 获取列名: step, time, x, y, z

    with open(csv_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data_log)

    print(f"[INFO]: 数据已成功保存至 {csv_file}")
    simulation_app.close()

if __name__ == "__main__":
    main()