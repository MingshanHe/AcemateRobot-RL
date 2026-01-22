# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 1. 读取数据
# try:
#     df = pd.read_csv("/root/IsaacLab/scripts/acemate/test/ball_trajectory.csv")
# except FileNotFoundError:
#     print("错误：找不到 ball_trajectory.csv 文件，请先运行仿真脚本。")
#     exit()

# # 创建一个包含两个子图的画布：左边是 3D 轨迹，右边是高度随时间的变化
# fig = plt.figure(figsize=(14, 6))

# # --- 图 1: 3D 轨迹图 ---
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot(df['x'], df['y'], df['z'], label='Ball Path', color='blue', lw=2)
# ax1.set_xlabel('X (meters)')
# ax1.set_ylabel('Y (meters)')
# ax1.set_zlabel('Z (meters)')
# ax1.set_title('3D Ball Trajectory')
# ax1.grid(True)
# ax1.legend()

# # --- 图 2: Z 轴（高度）随时间的变化 ---
# # 这张图最能直观反映“弹跳（Bouncing）”的物理特性
# ax2 = fig.add_subplot(122)
# ax2.plot(df['time'], df['z'], color='red', label='Height (Z)')
# ax2.set_xlabel('Time (seconds)')
# ax2.set_ylabel('Height (meters)')
# ax2.set_title('Height vs Time (Bouncing)')
# ax2.axhline(0, color='black', linestyle='--') # 地面线
# ax2.grid(True, linestyle=':', alpha=0.7)
# ax2.legend()

# plt.tight_layout()
# plt.show()

import pandas as pd
from scipy.signal import find_peaks

# 读取数据
df = pd.read_csv("/root/IsaacLab/scripts/acemate/test/ball_trajectory.csv")

# 寻找 z 轴的所有波峰
# height 可以设置一个阈值，比如只找 z > 0.1 的峰值，防止记录地面微小震动
peaks, _ = find_peaks(df['z'], height=0.05)

print(f"检测到 {len(peaks)} 次弹跳最高点：")
for i, peak_idx in enumerate(peaks):
    z_peak = df.loc[peak_idx, 'z']
    x_peak = df.loc[peak_idx, 'x']
    t_peak = df.loc[peak_idx, 'time']
    print(f"第 {i+1} 次最高点: 时间={t_peak:.2f}s, z={z_peak:.4f}m, x={x_peak:.4f}m")