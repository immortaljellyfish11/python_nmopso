# -*- coding: utf-8 -*-
"""
直接在代价函数内部添加调试输出
"""
import numpy as np
from config_enhanced import EnhancedConfig
from particle_v2 import create_random_solution
from utils import spherical_to_cartesian, dist_point_to_segment

config = EnhancedConfig(
    angle_range=np.pi/3,
    r_min_ratio=0.10,
    r_max_ratio=0.45
)
model = config.model
H = model['H']

# 创建一个随机解
np.random.seed(42)
spherical_sol = create_random_solution(config.n_var, config.var_min, config.var_max)
solution = spherical_to_cartesian(spherical_sol, model)

# 手动计算J2并添加详细输出
x = solution['x']
y = solution['y']
z = solution['z']

# 添加起终点
xs, ys, zs = model['start']
xf, yf, zf = model['end']
x_all = np.concatenate([[xs], x, [xf]])
y_all = np.concatenate([[ys], y, [yf]])
z_all = np.concatenate([[zs], z, [zf]])
N = len(x_all)

# 计算绝对高度
z_abs = np.zeros(N)
for i in range(N):
    y_idx = int(np.clip(round(y_all[i]), 0, model['map_size_y'] - 1))
    x_idx = int(np.clip(round(x_all[i]), 0, model['map_size_x'] - 1))
    z_abs[i] = z_all[i] + H[y_idx, x_idx]

# J2计算
threats = model['threats']
threat_num = len(threats)
drone_size = 1
danger_dist = 8
vertical_margin = 2
J_inf = 1e10

J2 = 0
n2 = 0
collision_segments = []

print("=" * 70)
print("逐段逐障碍物检查:")
print("=" * 70)

for i in range(threat_num):
    if len(threats[i]) == 4:
        threat_x, threat_y, threat_z, threat_radius = threats[i]
        z_base = 0
    else:
        threat_x, threat_y, threat_z_height, threat_radius, z_base = threats[i]
        threat_z = threat_z_height
    
    # 获取障碍物位置的地形高度
    threat_y_idx = int(np.clip(round(threat_y), 0, model['map_size_y'] - 1))
    threat_x_idx = int(np.clip(round(threat_x), 0, model['map_size_x'] - 1))
    terrain_height_at_threat = H[threat_y_idx, threat_x_idx]
    
    # 障碍物的绝对高度范围
    threat_z_min = z_base + terrain_height_at_threat
    threat_z_max = z_base + threat_z + terrain_height_at_threat
    
    for j in range(N - 1):
        segment_z_min = min(z_abs[j], z_abs[j+1])
        segment_z_max = max(z_abs[j], z_abs[j+1])
        
        # 检查垂直重叠
        has_vertical_overlap = not (
            segment_z_max + vertical_margin < threat_z_min or 
            segment_z_min - vertical_margin > threat_z_max
        )
        
        if not has_vertical_overlap:
            threat_cost = 0
        else:
            # 水平距离
            dist = dist_point_to_segment(
                [threat_x, threat_y],
                [x_all[j], y_all[j]],
                [x_all[j+1], y_all[j+1]]
            )
            
            if dist > (threat_radius + drone_size + danger_dist):
                threat_cost = 0
            elif dist < (threat_radius + drone_size):
                threat_cost = J_inf
                print(f"[碰撞] 障碍{i+1} vs 段{j+1}:")
                print(f"  障碍: ({threat_x:.0f},{threat_y:.0f}) R={threat_radius:.0f}, Z=[{threat_z_min:.1f},{threat_z_max:.1f}]")
                print(f"  段Z: [{segment_z_min:.1f},{segment_z_max:.1f}]")
                print(f"  水平距离: {dist:.2f} < {threat_radius + drone_size:.0f}")
                collision_segments.append((i+1, j+1))
            else:
                threat_cost = 1 - (dist - drone_size - threat_radius) / danger_dist
        
        n2 += 1
        J2 += threat_cost

if n2 > 0:
    J2 = J2 / n2

print("\n" + "=" * 70)
print(f"J2结果: {J2:.6f}")
print(f"碰撞段数: {len(collision_segments)}")
if collision_segments:
    print(f"碰撞详情: {collision_segments}")
else:
    print("无碰撞")
