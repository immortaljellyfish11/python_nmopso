# -*- coding: utf-8 -*-
from config_enhanced import EnhancedConfig
from comparison_algorithms import StandardPSO, QPSO, DifferentialEvolution
from nmopso import NMOPSO
from utils import spherical_to_cartesian
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建配置
config_1 = EnhancedConfig(
    angle_range=np.pi/2,
    r_min_ratio=0.18,
    r_max_ratio=0.33
)
config = EnhancedConfig()
# 运行所有算法
print("="*70)
print("运行所有算法对比")
print("="*70)

algorithms = [
    ("NMOPSO", NMOPSO(config_1)),
    ("Standard PSO", StandardPSO(config.model, config)),
    ("QPSO", QPSO(config.model, config)),
    ("DE", DifferentialEvolution(config.model, config))
]

results = {}
nmopso_optimizer = None
for name, optimizer in algorithms:
    print(f"\n运行 {name}...")
    if name == "NMOPSO":
        best, repository = optimizer.run()
        nmopso_optimizer = optimizer
    else:
        best = optimizer.run()
    
    results[name] = {
        'best': best,
        'optimizer': optimizer
    }
    print(f"{name} 完成 - J1={best.cost[0]:.3f}, J2={best.cost[1]:.3f}, J3={best.cost[2]:.3f}, J4={best.cost[3]:.3f}")

# 先单独绘制NMOPSO的详细可视化结果
print("\n生成NMOPSO详细可视化...")
nmopso_optimizer.plot_results()

# 绘制对比图：左边1列合并的3D图，右边3列每个算法的XY平面和代价历史
print("\n生成对比可视化...")
fig = plt.figure(figsize=(20, 16))

# 左侧：合并的3D图（所有算法）
ax_3d = fig.add_subplot(4, 3, (1, 10), projection='3d')
colors = ['blue', 'green', 'orange', 'purple']
model = config.model

for (name, data), color in zip(results.items(), colors):
    best = data['best']
    best_cart = spherical_to_cartesian(best.position, model)
    x = np.concatenate([[model['start'][0]], best_cart['x'], [model['end'][0]]])
    y = np.concatenate([[model['start'][1]], best_cart['y'], [model['end'][1]]])
    z = np.concatenate([[model['start'][2]], best_cart['z'], [model['end'][2]]])
    
    ax_3d.plot(x, y, z, color=color, linewidth=2, label=name, alpha=0.8)
    ax_3d.plot(x, y, z, 'o', color=color, markersize=3)

# 绘制起点和终点
ax_3d.scatter(*model['start'], color='green', s=150, marker='o', label='起点', edgecolors='black', linewidths=2)
ax_3d.scatter(*model['end'], color='red', s=150, marker='*', label='终点', edgecolors='black', linewidths=2)

# 绘制障碍物
for threat in model['threats']:
    if len(threat) == 4:
        tx, ty, tz, tr = threat
        z_base = 0
    else:
        tx, ty, tz_height, tr, z_base = threat
        tz = tz_height
    
    u = np.linspace(0, 2*np.pi, 15)
    v = np.linspace(z_base, z_base+tz, 2)
    x_cyl = tr * np.outer(np.cos(u), np.ones(len(v))) + tx
    y_cyl = tr * np.outer(np.sin(u), np.ones(len(v))) + ty
    z_cyl = np.outer(np.ones(len(u)), v)
    ax_3d.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.2, color='pink')

ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.set_title('所有算法3D路径对比', fontsize=14, fontweight='bold')
ax_3d.legend(fontsize=10, loc='upper left')
ax_3d.grid(True)

# 右侧：每个算法的XY平面和代价历史（2列×4行）
# 右侧：每个算法的XY平面和代价历史（2列×4行）
for idx, (name, data) in enumerate(results.items()):
    best = data['best']
    optimizer = data['optimizer']
    
    # 获取笛卡尔坐标
    best_cart = spherical_to_cartesian(best.position, model)
    x = np.concatenate([[model['start'][0]], best_cart['x'], [model['end'][0]]])
    y = np.concatenate([[model['start'][1]], best_cart['y'], [model['end'][1]]])
    
    # XY平面图（第2列）
    ax2 = fig.add_subplot(4, 3, idx*3 + 2)
    ax2.plot(x, y, 'b-', linewidth=2, label='路径')
    ax2.plot(x, y, 'ro', markersize=4)
    ax2.scatter(model['start'][0], model['start'][1], color='green', s=100, marker='o', label='起点')
    ax2.scatter(model['end'][0], model['end'][1], color='red', s=100, marker='*', label='终点')
    
    for threat in model['threats']:
        if len(threat) == 4:
            tx, ty, tz, tr = threat
        else:
            tx, ty, tz_height, tr, z_base = threat
        circle = plt.Circle((tx, ty), tr, color='pink', alpha=0.3)
        ax2.add_patch(circle)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{name} - XY平面')
    ax2.legend(fontsize=8)
    ax2.grid(True)
    ax2.axis('equal')
    
    # 代价历史（第3列）
    ax3 = fig.add_subplot(4, 3, idx*3 + 3)
    history = np.array(optimizer.best_cost_history)
    iterations = np.arange(len(history))
    
    labels = ['J1 (长度)', 'J2 (障碍)', 'J3 (高度)', 'J4 (平滑)']
    for i in range(history.shape[1]):
        ax3.plot(iterations, history[:, i], label=labels[i])
    
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('代价值')
    ax3.set_title(f'{name} - 代价演化')
    ax3.legend(fontsize=8)
    ax3.grid(True)

plt.tight_layout()
plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
print("图像已保存为 comparison_results.png")
plt.show()