# -*- coding: utf-8 -*-
"""
详细调试：输出路径与障碍物的详细交互
"""
import numpy as np
from config_enhanced import EnhancedConfig
from particle_v2 import create_random_solution
from utils import spherical_to_cartesian
from cost_function import calculate_cost

config = EnhancedConfig(
    angle_range=np.pi/3,
    r_min_ratio=0.10,
    r_max_ratio=0.45
)
model = config.model
H = model['H']

print("=" * 70)
print("详细调试：路径-障碍物交互")
print("=" * 70)

# 创建一个随机解
np.random.seed(42)
spherical_sol = create_random_solution(config.n_var, config.var_min, config.var_max)
solution = spherical_to_cartesian(spherical_sol, model)

# 获取路径坐标
x = solution['x']
y = solution['y']
z = solution['z']

N = len(x)

# 计算绝对高度
x_all = x
y_all = y
z_abs = np.zeros(N)

for i in range(N):
    y_idx = int(np.clip(round(y[i]), 0, model['map_size_y'] - 1))
    x_idx = int(np.clip(round(x[i]), 0, model['map_size_x'] - 1))
    z_abs[i] = z[i] + H[y_idx, x_idx]

print(f"\n路径信息:")
print(f"  总点数: {N}")
print(f"  相对高度范围: [{z.min():.1f}, {z.max():.1f}]")
print(f"  绝对高度范围: [{z_abs.min():.1f}, {z_abs.max():.1f}]")

# 检查每个障碍物
threats = model['threats']
collision_count = 0
danger_count = 0

for i, threat in enumerate(threats):
    x_t, y_t, z_h, r_t, z_b = threat
    
    # 获取障碍物位置的地形高度
    y_idx = int(np.clip(round(y_t), 0, model['map_size_y'] - 1))
    x_idx = int(np.clip(round(x_t), 0, model['map_size_x'] - 1))
    terrain_h = H[y_idx, x_idx]
    
    threat_z_min = z_b + terrain_h
    threat_z_max = z_b + z_h + terrain_h
    
    print(f"\n障碍物{i+1}: ({x_t:.0f},{y_t:.0f}) R={r_t:.0f}")
    print(f"  绝对高度范围: [{threat_z_min:.1f}, {threat_z_max:.1f}]")
    
    # 检查与路径的交互
    has_collision = False
    has_danger = False
    
    for j in range(N - 1):
        segment_z_min = min(z_abs[j], z_abs[j+1])
        segment_z_max = max(z_abs[j], z_abs[j+1])
        
        # 检查垂直重叠
        vertical_margin = 2
        has_vertical_overlap = not (
            segment_z_max + vertical_margin < threat_z_min or 
            segment_z_min - vertical_margin > threat_z_max
        )
        
        if has_vertical_overlap:
            # 计算水平距离
            from utils import dist_point_to_segment
            dist = dist_point_to_segment(
                [x_t, y_t],
                [x_all[j], y_all[j]],
                [x_all[j+1], y_all[j+1]]
            )
            
            drone_size = 1
            danger_dist = 8
            
            if dist < (r_t + drone_size):
                has_collision = True
                print(f"  [碰撞] 段{j+1}: 水平距离={dist:.1f}, 高度重叠=[{segment_z_min:.1f},{segment_z_max:.1f}]")
                break
            elif dist < (r_t + drone_size + danger_dist):
                has_danger = True
    
    if has_collision:
        collision_count += 1
        print(f"  结论: 碰撞")
    elif has_danger:
        danger_count += 1
        print(f"  结论: 危险区")
    else:
        print(f"  结论: 安全")

print("\n" + "=" * 70)
print("统计:")
print(f"  碰撞障碍物数: {collision_count}/12")
print(f"  危险区障碍物数: {danger_count}/12")
print(f"  安全障碍物数: {12 - collision_count - danger_count}/12")

# 计算完整代价
cost = calculate_cost(solution, model, config.var_min)
print(f"\n完整代价:")
print(f"  J1={cost[0]:.3f}, J2={cost[1]:.3f}, J3={cost[2]:.3f}, J4={cost[3]:.3f}")

# 检查段长度
print(f"\n路径段长度检查:")
for j in range(N-1):
    dx = x[j+1] - x[j]
    dy = y[j+1] - y[j]
    dz = z_abs[j+1] - z_abs[j]
    seg_len = np.sqrt(dx**2 + dy**2 + dz**2)
    if seg_len <= config.var_min['r']:
        print(f"  段{j+1}: {seg_len:.2f} <= {config.var_min['r']:.2f} [太短！导致J1=inf]")
    else:
        print(f"  段{j+1}: {seg_len:.2f} [OK]")

if np.isinf(cost[0]):
    print("\n[关键问题] J1=inf，不是J2的问题！")
    print("原因：某些路径段太短（<= var_min['r']）")
    print("解决方案：")
    print("  1. 降低r_min_ratio（从0.10降到0.05）")
    print("  2. 或者修改J1计算逻辑，不要因为短段就返回inf")
elif np.isinf(cost[1]):
    print("\n分析：J2=inf，路径碰撞障碍物")
    print("可能原因：")
    print("  1. 障碍物太多太大（12个，半径40-70）")
    print("  2. 起终点高度范围140-180，很多障碍物在此范围内")
    print("  3. 智能初始化沿直线，随机初始化探索不足")
else:
    print("\n分析：J2正常，路径可行")
