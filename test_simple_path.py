# -*- coding: utf-8 -*-
"""
简单测试：检查为什么所有解都是inf
"""
import numpy as np
from config_enhanced import EnhancedConfig
from particle_v2 import create_smart_solution
from utils import spherical_to_cartesian
from cost_function import calculate_cost

config = EnhancedConfig()
model = config.model

print("=" * 70)
print("环境信息:")
print("=" * 70)
print(f"起点: {model['start']}")
print(f"终点: {model['end']}")
print(f"地图大小: {model['map_size_x']} x {model['map_size_y']}")
print(f"高度范围: {model['zmin']} - {model['zmax']}")
print(f"\n障碍物数量: {len(model['threats'])}")

H = model['H']

for i, threat in enumerate(model['threats']):
    x, y, z_h, r, z_b = threat
    
    # 获取地形高度
    y_idx = int(np.clip(round(y), 0, model['map_size_y'] - 1))
    x_idx = int(np.clip(round(x), 0, model['map_size_x'] - 1))
    terrain_h = H[y_idx, x_idx]
    
    abs_bottom = z_b + terrain_h
    abs_top = z_b + z_h + terrain_h
    
    print(f"  障碍{i+1}: ({x:.0f},{y:.0f}) R={r:.0f}, "
          f"地形={terrain_h:.1f}, 高度{abs_bottom:.1f}-{abs_top:.1f}")

# 创建智能路径
print("\n" + "=" * 70)
print("测试智能初始化路径:")
print("=" * 70)

for test_id in range(5):
    spherical_solution = create_smart_solution(config.n_var, config.var_min, config.var_max, model)
    solution = spherical_to_cartesian(spherical_solution, model)
    cost = calculate_cost(solution, model, config.var_min)
    
    print(f"\n路径 {test_id+1}:")
    print(f"  代价: J1={cost[0]:.3f}, J2={cost[1]:.3f}, J3={cost[2]:.3f}, J4={cost[3]:.3f}")
    
    if np.isinf(cost[1]):
        print(f"  [碰撞] J2=inf")
    else:
        print(f"  [可行] J2正常")

# 测试从上方飞过所有障碍物
print("\n" + "=" * 70)
print("测试高空直线路径（应该安全）:")
print("=" * 70)

# 找出最高障碍物
max_threat_height = 0
for threat in model['threats']:
    x, y, z_h, r, z_b = threat
    y_idx = int(np.clip(round(y), 0, model['map_size_y'] - 1))
    x_idx = int(np.clip(round(x), 0, model['map_size_x'] - 1))
    terrain_h = H[y_idx, x_idx]
    abs_top = z_b + z_h + terrain_h
    max_threat_height = max(max_threat_height, abs_top)

print(f"最高障碍物顶部: {max_threat_height:.1f}")

# 创建高空直线路径
high_altitude_solution = {
    'r': np.full(config.n_var, 50.0),    # 中等步长
    'phi': np.full(config.n_var, 0.0),   # 直线
    'psi': np.full(config.n_var, np.pi/3)  # 向上60度
}

cost_high = calculate_cost(high_altitude_solution, model, config.var_min)
print(f"\n高空路径代价:")
print(f"  J1={cost_high[0]:.3f}, J2={cost_high[1]:.3f}, J3={cost_high[2]:.3f}, J4={cost_high[3]:.3f}")

if np.isinf(cost_high[1]):
    print("  [FAIL] 高空路径仍然碰撞！碰撞检测有bug!")
else:
    print("  [SUCCESS] 高空路径安全")

# 测试低空直线路径
print("\n" + "=" * 70)
print("测试低空直线路径（应该碰撞）:")
print("=" * 70)

low_altitude_solution = {
    'r': np.full(config.n_var, 50.0),
    'phi': np.full(config.n_var, 0.0),
    'psi': np.full(config.n_var, -np.pi/6)  # 向下30度
}

cost_low = calculate_cost(low_altitude_solution, model, config.var_min)
print(f"\n低空路径代价:")
print(f"  J1={cost_low[0]:.3f}, J2={cost_low[1]:.3f}, J3={cost_low[2]:.3f}, J4={cost_low[3]:.3f}")

if np.isinf(cost_low[1]):
    print("  [预期] 低空路径碰撞")
else:
    print("  [WARNING] 低空路径居然安全？")

print("\n" + "=" * 70)
