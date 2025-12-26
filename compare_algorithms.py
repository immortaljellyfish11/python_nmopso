"""
算法对比测试脚本 / Algorithm Comparison Test Script

对比NMOPSO与其他算法的性能
Compare NMOPSO performance with other algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append('E:\\ZJU\\25Sum_Fall\\algorithm\\WORK2\\python_nmopso')

from config_enhanced import SimpleConfig, EnhancedConfig
from nmopso import NMOPSO
from comparison_algorithms import StandardPSO, QPSO, DifferentialEvolution
from utils import spherical_to_cartesian

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_comparison(use_enhanced=False, angle_range=np.pi/4):
    """
    运行算法对比 / Run algorithm comparison
    
    参数 / Parameters:
        use_enhanced: 是否使用增强配置 / Whether to use enhanced configuration
        angle_range: 角度范围 / Angle range
    """
    print("\n" + "="*80)
    print("PSO算法家族性能对比测试 / PSO Algorithm Family Performance Comparison")
    print("="*80)
    
    # 创建配置
    if use_enhanced:
        config = EnhancedConfig(angle_range=angle_range)
        print(f"使用增强配置 / Using enhanced configuration")
    else:
        config = SimpleConfig()
        print(f"使用简化配置 / Using simple configuration")
    
    print(f"角度范围 / Angle range: {angle_range*180/np.pi:.1f}°")
    print(f"障碍物数量 / Number of obstacles: {len(config.model['threats'])}")
    
    # 减少迭代次数以加快测试
    max_iter = 300
    n_pop = 50
    
    results = {}
    
    # 1. NMOPSO
    print("\n" + "-"*80)
    start_time = time.time()
    nmopso = NMOPSO(config)
    config.max_iter = max_iter
    config.n_pop = n_pop
    global_best, repository = nmopso.run()
    nmopso_time = time.time() - start_time
    
    results['NMOPSO'] = {
        'solution': global_best,
        'cost': global_best.cost,
        'history': np.array(nmopso.best_cost_history),
        'time': nmopso_time,
        'repository_size': len(repository)
    }
    
    # 2. 标准PSO
    print("\n" + "-"*80)
    start_time = time.time()
    std_pso = StandardPSO(config.model, config)
    std_best = std_pso.run(max_iter=max_iter, n_pop=n_pop)
    std_pso_time = time.time() - start_time
    
    results['Standard PSO'] = {
        'solution': std_best,
        'cost': std_best.cost,
        'history': np.array(std_pso.best_cost_history),
        'time': std_pso_time
    }
    
    # 3. QPSO
    print("\n" + "-"*80)
    start_time = time.time()
    qpso = QPSO(config.model, config)
    qpso_best = qpso.run(max_iter=max_iter, n_pop=n_pop)
    qpso_time = time.time() - start_time
    
    results['QPSO'] = {
        'solution': qpso_best,
        'cost': qpso_best.cost,
        'history': np.array(qpso.best_cost_history),
        'time': qpso_time
    }
    
    # 4. DE
    print("\n" + "-"*80)
    start_time = time.time()
    de = DifferentialEvolution(config.model, config)
    de_best = de.run(max_iter=max_iter, n_pop=n_pop)
    de_time = time.time() - start_time
    
    results['DE'] = {
        'solution': de_best,
        'cost': de_best['cost'],
        'history': np.array(de.best_cost_history),
        'time': de_time
    }
    
    # 打印对比结果
    print("\n" + "="*80)
    print("性能对比总结 / Performance Comparison Summary")
    print("="*80)
    
    print("\n各算法最终代价值 / Final Cost Values:")
    print("-"*80)
    print(f"{'算法/Algorithm':<20} {'J1(长度)':<12} {'J2(障碍)':<12} {'J3(高度)':<12} {'J4(平滑)':<12} {'时间/s':<10}")
    print("-"*80)
    
    for alg_name, result in results.items():
        cost = result['cost']
        print(f"{alg_name:<20} {cost[0]:<12.4f} {cost[1]:<12.4f} {cost[2]:<12.4f} {cost[3]:<12.4f} {result['time']:<10.2f}")
    
    # 绘制对比图
    plot_comparison(results, config.model)
    
    return results


def plot_comparison(results, model):
    """
    绘制对比图 / Plot comparison charts
    """
    fig = plt.figure(figsize=(18, 10))
    
    # 1. 代价演化对比 (4个子图，每个目标一个)
    for obj_idx in range(4):
        ax = fig.add_subplot(3, 4, obj_idx + 1)
        obj_names = ['J1 (路径长度/Length)', 'J2 (障碍/Obstacle)', 
                    'J3 (高度/Altitude)', 'J4 (平滑度/Smoothness)']
        
        for alg_name, result in results.items():
            history = result['history']
            ax.plot(history[:, obj_idx], label=alg_name, linewidth=2)
        
        ax.set_xlabel('迭代次数 / Iterations')
        ax.set_ylabel('代价值 / Cost')
        ax.set_title(obj_names[obj_idx])
        ax.legend()
        ax.grid(True)
    
    # 2. 路径对比 (3D)
    ax3d = fig.add_subplot(3, 2, 3, projection='3d')
    
    colors = ['blue', 'red', 'green', 'orange']
    for (alg_name, result), color in zip(results.items(), colors):
        if 'DE' in alg_name:
            pos = result['solution']['position']
        else:
            pos = result['solution'].position
        
        cart_pos = spherical_to_cartesian(pos, model)
        x = np.concatenate([[model['start'][0]], cart_pos['x'], [model['end'][0]]])
        y = np.concatenate([[model['start'][1]], cart_pos['y'], [model['end'][1]]])
        z = np.concatenate([[model['start'][2]], cart_pos['z'], [model['end'][2]]])
        
        ax3d.plot(x, y, z, color=color, label=alg_name, linewidth=2, alpha=0.7)
    
    # 绘制障碍物
    for threat in model['threats']:
        if len(threat) == 4:
            tx, ty, tz, tr = threat
            z_base = 0
        else:
            tx, ty, tz_height, tr, z_base = threat
            tz = tz_height
        
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(z_base, z_base + tz, 2)
        x_cyl = tr * np.outer(np.cos(u), np.ones(len(v))) + tx
        y_cyl = tr * np.outer(np.sin(u), np.ones(len(v))) + ty
        z_cyl = np.outer(np.ones(len(u)), v)
        ax3d.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.2, color='red')
    
    ax3d.scatter(*model['start'], color='green', s=100, marker='o')
    ax3d.scatter(*model['end'], color='red', s=100, marker='*')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D路径对比 / 3D Path Comparison')
    ax3d.legend()
    
    # 3. 性能雷达图
    ax_radar = fig.add_subplot(3, 2, 4, projection='polar')
    
    categories = ['J1\n长度', 'J2\n障碍', 'J3\n高度', 'J4\n平滑']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for (alg_name, result), color in zip(results.items(), colors):
        values = result['cost'].tolist()
        values += values[:1]
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=alg_name, color=color)
        ax_radar.fill(angles, values, alpha=0.15, color=color)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_title('性能雷达图 / Performance Radar Chart')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax_radar.grid(True)
    
    # 4. 最终代价柱状图
    ax_bar = fig.add_subplot(3, 2, 5)
    
    alg_names = list(results.keys())
    x_pos = np.arange(len(alg_names))
    
    # 计算加权总代价
    weights = np.array([0.3, 0.4, 0.2, 0.1])
    total_costs = [np.dot(weights, results[alg]['cost']) for alg in alg_names]
    
    bars = ax_bar.bar(x_pos, total_costs, color=colors[:len(alg_names)])
    ax_bar.set_xlabel('算法 / Algorithm')
    ax_bar.set_ylabel('加权总代价 / Weighted Total Cost')
    ax_bar.set_title('算法性能对比 / Algorithm Performance Comparison')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(alg_names, rotation=15)
    ax_bar.grid(True, axis='y')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    # 5. 运行时间对比
    ax_time = fig.add_subplot(3, 2, 6)
    
    run_times = [results[alg]['time'] for alg in alg_names]
    bars = ax_time.bar(x_pos, run_times, color=colors[:len(alg_names)])
    ax_time.set_xlabel('算法 / Algorithm')
    ax_time.set_ylabel('运行时间 / Runtime (秒/s)')
    ax_time.set_title('运行时间对比 / Runtime Comparison')
    ax_time.set_xticks(x_pos)
    ax_time.set_xticklabels(alg_names, rotation=15)
    ax_time.grid(True, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax_time.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比图已保存为 algorithm_comparison.png")
    plt.show()


def test_angle_range_impact():
    """
    测试不同角度范围的影响 / Test impact of different angle ranges
    """
    print("\n" + "="*80)
    print("测试角度范围对算法性能的影响")
    print("Testing Impact of Angle Range on Algorithm Performance")
    print("="*80)
    
    angle_ranges = [np.pi/6, np.pi/4, np.pi/3, np.pi/2]  # 30°, 45°, 60°, 90°
    results_by_angle = {}
    
    for angle in angle_ranges:
        print(f"\n测试角度范围: {angle*180/np.pi:.0f}°")
        config = SimpleConfig()
        config.angle_range = angle
        config.max_iter = 50
        config.n_pop = 30
        
        nmopso = NMOPSO(config)
        global_best, _ = nmopso.run()
        
        results_by_angle[f"{angle*180/np.pi:.0f}°"] = {
            'cost': global_best.cost,
            'angle': angle
        }
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    obj_names = ['J1 (路径长度)', 'J2 (障碍)', 'J3 (高度)', 'J4 (平滑度)']
    
    for obj_idx, ax in enumerate(axes.flat):
        angles_deg = [r['angle']*180/np.pi for r in results_by_angle.values()]
        costs = [r['cost'][obj_idx] for r in results_by_angle.values()]
        
        ax.plot(angles_deg, costs, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('角度范围 / Angle Range (度/degrees)')
        ax.set_ylabel('代价值 / Cost Value')
        ax.set_title(obj_names[obj_idx])
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('angle_range_impact.png', dpi=150, bbox_inches='tight')
    print("\n角度范围影响图已保存为 angle_range_impact.png")
    plt.show()


if __name__ == "__main__":
    print("请选择测试模式 / Please select test mode:")
    print("1. 基础对比 (简单配置) / Basic comparison (simple config)")
    print("2. 增强对比 (复杂配置) / Enhanced comparison (complex config)")
    print("3. 角度范围影响测试 / Angle range impact test")
    print("4. 全部测试 / All tests")
    
    choice = input("请输入选择 (1-4) / Enter choice (1-4): ").strip()
    
    if choice == '1':
        results = run_comparison(use_enhanced=False, angle_range=np.pi/4)
    elif choice == '2':
        results = run_comparison(use_enhanced=True, angle_range=np.pi/4)
    elif choice == '3':
        test_angle_range_impact()
    elif choice == '4':
        print("\n运行所有测试...")
        run_comparison(use_enhanced=False, angle_range=np.pi/4)
        run_comparison(use_enhanced=True, angle_range=np.pi/4)
        test_angle_range_impact()
    else:
        print("默认运行基础对比 / Running basic comparison by default")
        results = run_comparison(use_enhanced=False, angle_range=np.pi/4)
