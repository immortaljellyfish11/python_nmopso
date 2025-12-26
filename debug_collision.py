# -*- coding: utf-8 -*-
"""
调试碰撞检测 - 验证障碍物检测是否正确
"""
import numpy as np
from config_enhanced import EnhancedConfig
from particle_v2 import Particle
from cost_function import calculate_cost

# 创建配置
config = EnhancedConfig()
model = config.model

print("=" * 60)
print("障碍物信息:")
print("=" * 60)
for i, threat in enumerate(model['threats']):
    if len(threat) == 5:
        x, y, z_h, r, z_b = threat
        print(f"障碍物 {i+1}: 位置({x:.0f}, {y:.0f}), 底部高度={z_b:.0f}, 高度={z_h:.0f}, 半径={r:.0f}")

print("\n" + "=" * 60)
print("测试路径穿过障碍物的情况:")
print("=" * 60)

# 测试1: 创建一条直线路径穿过第一个障碍物
threat1 = model['threats'][0]
threat_x, threat_y = threat1[0], threat1[1]

# 创建粒子，路径直接穿过障碍物中心
particle = Particle()
particle.n_var = config.n_var

# 直线路径：从起点到障碍物中心到终点
start = model['start']
end = model['end']

# 设置航路点直接穿过障碍物
# 中间点设置为障碍物的xy位置，低高度
particle.position = {
    'r': np.full(config.n_var, 50.0),  # 固定步长
    'phi': np.full(config.n_var, 0.0),  # 不转向
    'psi': np.full(config.n_var, 0.0)   # 水平飞行
}

# 计算代价
particle.cost = calculate_cost(particle.position, model, config.var_min)

print(f"\n测试路径1 (水平飞行穿过障碍物):")
print(f"  J2 (障碍物代价) = {particle.cost[1]:.6f}")
if particle.cost[1] > 1e10:
    print("  [SUCCESS] 正确检测到碰撞!")
else:
    print("  [FAIL] 未检测到碰撞，存在bug!")

# 测试2: 创建一条高飞的路径（应该安全）
particle2 = Particle()
particle2.n_var = config.n_var
particle2.position = {
    'r': np.full(config.n_var, 50.0),
    'phi': np.full(config.n_var, 0.0),
    'psi': np.full(config.n_var, np.pi/6)  # 向上30度飞行
}

particle2.cost = calculate_cost(particle2.position, model, config.var_min)

print(f"\n测试路径2 (高空飞行避开障碍物):")
print(f"  J2 (障碍物代价) = {particle2.cost[1]:.6f}")
if particle2.cost[1] < 0.1:
    print("  [SUCCESS] 正确识别安全路径!")
else:
    print("  [WARNING] 安全路径被误判为危险!")

# 测试3: 打印实际路径坐标
print("\n" + "=" * 60)
print("详细路径坐标检查:")
print("=" * 60)

from utils import spherical_to_cartesian, transformation_matrix

def get_path_coords(position, model):
    """获取路径的笛卡尔坐标"""
    N = len(position['r'])
    start = model['start']
    end = model['end']
    
    path = [start]
    T = transformation_matrix(start, end)
    
    for i in range(N):
        x_sph, y_sph, z_sph = spherical_to_cartesian(
            position['r'][i],
            position['phi'][i],
            position['psi'][i]
        )
        point = start + T @ np.array([x_sph, y_sph, z_sph])
        path.append(point)
    
    path.append(end)
    return np.array(path)

path1 = get_path_coords(particle.position, model)
print(f"\n路径1 (水平飞行):")
print(f"  起点: ({path1[0][0]:.1f}, {path1[0][1]:.1f}, {path1[0][2]:.1f})")
print(f"  第1点: ({path1[1][0]:.1f}, {path1[1][1]:.1f}, {path1[1][2]:.1f})")
print(f"  第2点: ({path1[2][0]:.1f}, {path1[2][1]:.1f}, {path1[2][2]:.1f})")
print(f"  终点: ({path1[-1][0]:.1f}, {path1[-1][1]:.1f}, {path1[-1][2]:.1f})")

# 检查是否穿过障碍物
H = model['H']
threat1 = model['threats'][0]
threat_x, threat_y, threat_z_h, threat_r, threat_z_b = threat1

print(f"\n障碍物1信息:")
print(f"  位置: ({threat_x:.0f}, {threat_y:.0f})")
print(f"  底部高度: {threat_z_b:.0f}")
print(f"  顶部高度: {threat_z_b + threat_z_h:.0f}")
print(f"  半径: {threat_r:.0f}")

# 获取障碍物位置的地形高度
threat_y_idx = int(np.clip(round(threat_y), 0, model['map_size_y'] - 1))
threat_x_idx = int(np.clip(round(threat_x), 0, model['map_size_x'] - 1))
terrain_h = H[threat_y_idx, threat_x_idx]
print(f"  地形高度: {terrain_h:.1f}")
print(f"  绝对底部: {threat_z_b + terrain_h:.1f}")
print(f"  绝对顶部: {threat_z_b + threat_z_h + terrain_h:.1f}")

# 检查路径段是否穿过
for i in range(len(path1) - 1):
    p1 = path1[i]
    p2 = path1[i+1]
    
    # 计算水平距离
    dist = np.sqrt((p1[0] - threat_x)**2 + (p1[1] - threat_y)**2)
    
    # 获取路径点的地形高度
    y1_idx = int(np.clip(round(p1[1]), 0, model['map_size_y'] - 1))
    x1_idx = int(np.clip(round(p1[0]), 0, model['map_size_x'] - 1))
    terrain_h1 = H[y1_idx, x1_idx]
    
    z_abs1 = p1[2] + terrain_h1
    
    print(f"\n  路径段 {i+1}:")
    print(f"    起点: ({p1[0]:.1f}, {p1[1]:.1f}, z_rel={p1[2]:.1f}, z_abs={z_abs1:.1f})")
    print(f"    到障碍物中心的水平距离: {dist:.1f} (半径={threat_r:.0f})")
    
    if dist < threat_r + 10:
        if z_abs1 >= threat_z_b + terrain_h and z_abs1 <= threat_z_b + threat_z_h + terrain_h:
            print(f"    [COLLISION] 在障碍物高度范围内且水平距离小于半径!")
        else:
            print(f"    [SAFE] 水平距离近但高度不重叠")

print("\n" + "=" * 60)
