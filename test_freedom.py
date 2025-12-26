# -*- coding: utf-8 -*-
"""
快速测试增大自由度后的效果
"""
import numpy as np
from config_enhanced import EnhancedConfig
from nmopso_v2 import NMOPSO_V2

print("=" * 70)
print("测试增大自由度后的NMOPSO")
print("=" * 70)

# 配置（增大自由度）
config = EnhancedConfig(
    angle_range=np.pi/3,      # 60度转角
    r_min_ratio=0.10,         # 更灵活
    r_max_ratio=0.45          # 更大步长
)

config.max_iter = 100  # 快速测试，100次迭代
config.n_pop = 50
config.length_weight = 0.6

print(f"\n配置:")
print(f"  角度范围: {np.degrees(config.angle_range):.1f}°")
print(f"  r范围: [{config.var_min['r']:.1f}, {config.var_max['r']:.1f}]")
print(f"  种群大小: {config.n_pop}")
print(f"  迭代次数: {config.max_iter}")

# 运行（降低智能初始化比例）
optimizer = NMOPSO_V2(config, use_smart_init=True, smart_ratio=0.5)

print("\n开始优化...")
repository, best = optimizer.run()

print("\n" + "=" * 70)
print("结果:")
print("=" * 70)
print(f"存档大小: {len(repository)}")
print(f"最优解代价:")
print(f"  J1 (路径长度): {best.cost[0]:.3f}")
print(f"  J2 (障碍物): {best.cost[1]:.3f}")
print(f"  J3 (高度): {best.cost[2]:.3f}")
print(f"  J4 (平滑度): {best.cost[3]:.3f}")

if np.isinf(best.cost[1]):
    print("\n[失败] 仍然无法找到可行解")
    print("建议:")
    print("  1. 进一步增大角度范围到π/2")
    print("  2. 增加迭代次数到300")
    print("  3. 完全使用随机初始化（smart_ratio=0）")
else:
    print("\n[成功] 找到可行解！")
    print("增大自由度有效！")
