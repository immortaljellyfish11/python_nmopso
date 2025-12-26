"""
NMOPSO主算法 / NMOPSO Main Algorithm

导航变量多目标粒子群优化算法 (NMOPSO)
Navigation Variable-based Multi-objective Particle Swarm Optimization

基于论文: Navigation Variable-based Multi-objective Particle Swarm Optimization 
for UAV Path Planning with Kinematic Constraints

算法说明 / Algorithm Description:
==================

1. PSO基础 / PSO Basics:
   粒子群优化(PSO)是一种模拟鸟群觅食行为的仿生算法
   PSO simulates the foraging behavior of bird flocks
   
   - 每个粒子代表一个候选解 / Each particle represents a candidate solution
   - 粒子通过速度和位置更新在解空间中搜索 / Particles search by updating velocity and position
   - 粒子学习个体最优(pbest)和全局最优(gbest) / Particles learn from personal best and global best

2. 多目标优化 / Multi-objective Optimization:
   与单目标不同，多目标需要找到一组Pareto最优解
   Unlike single-objective, multi-objective finds a set of Pareto optimal solutions
   
   - Pareto支配: 解A在所有目标上不差于B，且至少一个目标更优
   - Pareto dominance: A is no worse than B in all objectives and better in at least one
   - Pareto前沿: 所有非支配解的集合
   - Pareto front: Set of all non-dominated solutions

3. NMOPSO创新 / NMOPSO Innovation:
   使用导航变量(r, phi, psi)而非直接坐标(x, y, z)
   Uses navigation variables (r, phi, psi) instead of direct coordinates
   
   - 自然满足运动学约束 / Naturally satisfies kinematic constraints
   - 生成平滑可飞行的路径 / Generates smooth flyable paths
   - 减少搜索空间维度 / Reduces search space dimensions

作者: GitHub Copilot
日期: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

from config import Config
from particle import Particle, create_random_solution, mutate, determine_domination
from utils import spherical_to_cartesian, dominates
from cost_function import calculate_cost
from grid import create_grid, find_grid_index, select_leader, delete_one_rep_member
# ============ 中文字体配置 / Chinese Font Configuration ============
import matplotlib
import platform

def setup_chinese_font():
    """
    配置matplotlib支持中文显示
    Configure matplotlib to support Chinese characters
    """
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常见中文字体
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':  # macOS
        fonts = ['PingFang SC', 'Heiti SC', 'STHeiti']
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'AR PL UMing CN']
    
    # 尝试设置字体
    for font in fonts:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font]
            matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            # print(f"成功设置中文字体: {font}")
            return True
        except:
            continue
    
    # 如果都失败，禁用中文警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    # print("警告: 未找到中文字体，中文可能无法正常显示")
    return False

# 在导入后立即配置字体
setup_chinese_font()
# ================================================================

class NMOPSO:
    """
    NMOPSO算法类 / NMOPSO Algorithm Class
    """
    
    def __init__(self, config=None):
        """
        初始化NMOPSO算法 / Initialize NMOPSO Algorithm
        """
        self.config = config if config is not None else Config()
        self.model = self.config.model
        
        # 设置变量边界 / Setup variable bounds
        self.setup_bounds()
        
        # 初始化粒子群 / Initialize swarm
        self.particles = []
        self.repository = []  # 存档(非支配解集) / Repository (non-dominated set)
        self.global_best = None
        
        # 记录历史 / Record history
        self.best_cost_history = []
    
    def setup_bounds(self):
        """
        设置变量边界 / Setup Variable Bounds
        支持可配置的角度范围
        """
        model = self.model
        n = model['n']
        
        # 位置边界 / Position bounds
        self.var_min = {}
        self.var_max = {}
        
        self.var_min['x'] = model['xmin']
        self.var_max['x'] = model['xmax']
        self.var_min['y'] = model['ymin']
        self.var_max['y'] = model['ymax']
        self.var_min['z'] = model['zmin']
        self.var_max['z'] = model['zmax']
        
        # 导航变量边界 / Navigation variable bounds
        # r: 每段的距离 / Distance of each segment
        self.var_max['r'] = 3 * np.linalg.norm(model['start'] - model['end']) / n
        self.var_min['r'] = self.var_max['r'] / 9
        
        # 角度范围（从config获取）/ Angle range (from config)
        angle_range = getattr(self.config, 'angle_range', np.pi / 4)
        self.var_min['psi'] = -angle_range
        self.var_max['psi'] = angle_range
        self.var_min['phi'] = -angle_range
        self.var_max['phi'] = angle_range
        
        # 速度边界 / Velocity bounds
        alpha = 0.5
        self.vel_min = {}
        self.vel_max = {}
        
        self.vel_max['r'] = alpha * (self.var_max['r'] - self.var_min['r'])
        self.vel_min['r'] = -self.vel_max['r']
        self.vel_max['psi'] = alpha * (self.var_max['psi'] - self.var_min['psi'])
        self.vel_min['psi'] = -self.vel_max['psi']
        self.vel_max['phi'] = alpha * (self.var_max['phi'] - self.var_min['phi'])
        self.vel_min['phi'] = -self.vel_max['phi']
        
        self.var_size = n
    
    def initialize_swarm(self):
        """
        初始化粒子群 / Initialize Particle Swarm
        确保至少有一个可行解 / Ensure at least one feasible solution
        """
        print("初始化粒子群... / Initializing swarm...")
        
        is_init = False
        attempt = 0
        
        while not is_init and attempt < 20:
            attempt += 1
            self.particles = []
            
            for i in range(self.config.n_pop):
                particle = Particle()
                
                # 随机初始化位置 / Random initialize position
                particle.position = create_random_solution(
                    self.var_size, self.var_min, self.var_max
                )
                
                # 初始化速度为零 / Initialize velocity to zero
                particle.velocity = {
                    'r': np.zeros(self.var_size),
                    'phi': np.zeros(self.var_size),
                    'psi': np.zeros(self.var_size)
                }
                
                # 评估 / Evaluation
                cart_position = spherical_to_cartesian(particle.position, self.model)
                particle.cost = calculate_cost(cart_position, self.model, self.var_min)
                
                # 初始化个体最优 / Initialize personal best
                particle.best['position'] = copy.deepcopy(particle.position)
                particle.best['cost'] = copy.deepcopy(particle.cost)
                
                self.particles.append(particle)
                
                # 检查是否有可行解 / Check for feasible solution
                if not np.any(particle.cost == np.inf):
                    is_init = True
        
        if not is_init:
            print("警告: 初始化未找到可行解，继续执行... / Warning: No feasible solution found, continuing...")
            is_init = True
        
        print(f"粒子群初始化完成 (尝试 {attempt} 次) / Swarm initialized (attempt {attempt})")
    
    def run(self):
        """
        运行NMOPSO算法 / Run NMOPSO Algorithm
        """
        # 初始化 / Initialize
        self.initialize_swarm()
        
        # 确定支配关系 / Determine domination
        self.particles = determine_domination(self.particles)
        
        # 初始化存档 / Initialize repository
        self.repository = [p for p in self.particles if not p.is_dominated]
        
        # 创建网格 / Create grid
        if len(self.repository) > 0:
            grid = create_grid(self.repository, self.config.n_grid, self.config.alpha)
            for i in range(len(self.repository)):
                self.repository[i] = find_grid_index(self.repository[i], grid)
            
            # 选择全局最优 / Select global best
            self.global_best = select_leader(self.repository, self.config.beta)
        else:
            print("警告: 初始存档为空 / Warning: Initial repository is empty")
            self.global_best = self.particles[0]
        
        # 主循环 / Main loop
        print(f"\n开始优化 (最大迭代次数: {self.config.max_iter}) / Starting optimization (max iterations: {self.config.max_iter})")
        print("=" * 80)
        
        w = self.config.w
        
        for it in range(self.config.max_iter):
            # 记录最优代价 / Record best cost
            self.best_cost_history.append(copy.deepcopy(self.global_best.cost))
            
            # 更新每个粒子 / Update each particle
            for i in range(len(self.particles)):
                particle = self.particles[i]
                
                # 选择领导者 / Select leader
                if len(self.repository) > 0:
                    self.global_best = select_leader(self.repository, self.config.beta)
                
                # 更新速度和位置 (针对每个导航变量) / Update velocity and position for each navigation variable
                for var in ['r', 'psi', 'phi']:
                    # PSO速度更新公式 / PSO velocity update formula
                    # v = w*v + c1*rand*(pbest - x) + c2*rand*(gbest - x)
                    particle.velocity[var] = (
                        w * particle.velocity[var] +
                        self.config.c1 * np.random.rand(self.var_size) * 
                        (particle.best['position'][var] - particle.position[var]) +
                        self.config.c2 * np.random.rand(self.var_size) * 
                        (self.global_best.position[var] - particle.position[var])
                    )
                    
                    # 限制速度 / Limit velocity
                    particle.velocity[var] = np.clip(
                        particle.velocity[var], 
                        self.vel_min[var], 
                        self.vel_max[var]
                    )
                    
                    # 更新位置 / Update position
                    particle.position[var] = particle.position[var] + particle.velocity[var]
                    
                    # 速度镜像 (边界处理) / Velocity mirroring (boundary handling)
                    out_of_range = (particle.position[var] < self.var_min[var]) | \
                                   (particle.position[var] > self.var_max[var])
                    particle.velocity[var][out_of_range] = -particle.velocity[var][out_of_range]
                    
                    # 限制位置 / Limit position
                    particle.position[var] = np.clip(
                        particle.position[var], 
                        self.var_min[var], 
                        self.var_max[var]
                    )
                
                # 评估 / Evaluation
                cart_position = spherical_to_cartesian(particle.position, self.model)
                particle.cost = calculate_cost(cart_position, self.model, self.var_min)
                
                # 变异操作 / Mutation operation
                pm = (1 - it / (self.config.max_iter - 1)) ** (1 / self.config.mu)
                if np.random.rand() < pm and len(self.repository) > 0:
                    new_position = mutate(
                        particle, self.repository, self.config.delta,
                        self.var_max, self.var_min
                    )
                    new_cart = spherical_to_cartesian(new_position, self.model)
                    new_cost = calculate_cost(new_cart, self.model, self.var_min)
                    
                    # 接受变异 / Accept mutation
                    if dominates(new_cost, particle.cost):
                        particle.position = new_position
                        particle.cost = new_cost
                    elif not dominates(particle.cost, new_cost) and np.random.rand() < 0.5:
                        particle.position = new_position
                        particle.cost = new_cost
                
                # 更新个体最优 / Update personal best
                if dominates(particle.cost, particle.best['cost']):
                    particle.best['position'] = copy.deepcopy(particle.position)
                    particle.best['cost'] = copy.deepcopy(particle.cost)
                elif not dominates(particle.best['cost'], particle.cost) and np.random.rand() < 0.5:
                    particle.best['position'] = copy.deepcopy(particle.position)
                    particle.best['cost'] = copy.deepcopy(particle.cost)
            
            # 更新存档 / Update repository
            self.particles = determine_domination(self.particles)
            non_dominated = [p for p in self.particles if not p.is_dominated]
            self.repository.extend(non_dominated)
            
            self.repository = determine_domination(self.repository)
            self.repository = [p for p in self.repository if not p.is_dominated]
            
            # 更新网格 / Update grid
            if len(self.repository) > 0:
                grid = create_grid(self.repository, self.config.n_grid, self.config.alpha)
                for i in range(len(self.repository)):
                    self.repository[i] = find_grid_index(self.repository[i], grid)
                
                # 限制存档大小 / Limit repository size
                while len(self.repository) > self.config.n_rep:
                    self.repository = delete_one_rep_member(self.repository, self.config.gamma)
            
            # 更新惯性权重 / Update inertia weight
            w = w * self.config.w_damp
            
            # 显示进度 / Show progress
            if (it + 1) % 50 == 0 or it == 0:
                print(f"迭代 {it+1:3d}/{self.config.max_iter} | "
                      f"存档大小: {len(self.repository):3d} | "
                      f"最优代价: [{', '.join([f'{c:.4f}' for c in self.global_best.cost])}]")
        
        # 选择最终最优解 / Select final best solution
        if len(self.repository) > 0:
            self.global_best = select_leader(self.repository, self.config.beta)
        
        print("=" * 80)
        print("优化完成! / Optimization completed!\n")
        
        return self.global_best, self.repository
    
    def plot_results(self, show_all_solutions=False):
        """
        绘制结果 / Plot Results
        """
        if self.global_best is None:
            print("请先运行算法 / Please run algorithm first")
            return
        
        # 获取最优路径 / Get best path
        best_cart = spherical_to_cartesian(self.global_best.position, self.model)
        
        # 创建图形 / Create figure
        fig = plt.figure(figsize=(15, 5))
        
        # 3D路径图 / 3D path plot
        ax1 = fig.add_subplot(131, projection='3d')
        self.plot_3d_path(ax1, best_cart, show_all_solutions)
        
        # XY平面图 / XY plane plot
        ax2 = fig.add_subplot(132)
        self.plot_2d_path(ax2, best_cart, 'xy')
        
        # 代价历史 / Cost history
        ax3 = fig.add_subplot(133)
        self.plot_cost_history(ax3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_3d_path(self, ax, solution, show_all=False):
        """
        绘制3D路径 / Plot 3D Path
        增强版：支持带高度信息的3D障碍物
        """
        x = np.concatenate([[self.model['start'][0]], solution['x'], [self.model['end'][0]]])
        y = np.concatenate([[self.model['start'][1]], solution['y'], [self.model['end'][1]]])
        z = np.concatenate([[self.model['start'][2]], solution['z'], [self.model['end'][2]]])
        
        # 绘制路径 / Plot path
        ax.plot(x, y, z, 'b-', linewidth=2, label='最优路径 / Best Path')
        ax.plot(x, y, z, 'ro', markersize=5)
        
        # 绘制起点和终点 / Plot start and end
        ax.scatter(*self.model['start'], color='green', s=100, marker='o', label='起点 / Start')
        ax.scatter(*self.model['end'], color='red', s=100, marker='*', label='终点 / End')
        
        # 绘制增强的3D障碍物 / Plot enhanced 3D obstacles
        for threat in self.model['threats']:
            if len(threat) == 4:
                tx, ty, tz, tr = threat
                z_base = 0
            else:
                tx, ty, tz_height, tr, z_base = threat
                tz = tz_height
            
            # 绘制圆柱体 / Draw cylinder
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(z_base, z_base + tz, 2)
            x_cyl = tr * np.outer(np.cos(u), np.ones(len(v))) + tx
            y_cyl = tr * np.outer(np.sin(u), np.ones(len(v))) + ty
            z_cyl = np.outer(np.ones(len(u)), v)
            
            # 使用不同颜色表示不同高度的障碍物
            # Use different colors for obstacles at different altitudes
            if z_base > 50:
                color = 'orange'  # 高空障碍
            elif tz > 200:
                color = 'red'     # 高障碍
            else:
                color = 'pink'    # 低障碍
            
            ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.4, color=color)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (高度 / Altitude)')
        ax.set_title('3D UAV路径 / 3D UAV Path')
        ax.legend()
        ax.grid(True)
    
    def plot_2d_path(self, ax, solution, plane='xy'):
        """
        绘制2D路径 / Plot 2D Path
        增强版：显示障碍物的高度信息
        """
        x = np.concatenate([[self.model['start'][0]], solution['x'], [self.model['end'][0]]])
        y = np.concatenate([[self.model['start'][1]], solution['y'], [self.model['end'][1]]])
        
        # 绘制路径 / Plot path
        ax.plot(x, y, 'b-', linewidth=2, label='最优路径 / Best Path')
        ax.plot(x, y, 'ro', markersize=5)
        
        # 绘制起点和终点 / Plot start and end
        ax.scatter(self.model['start'][0], self.model['start'][1], 
                  color='green', s=100, marker='o', label='起点 / Start')
        ax.scatter(self.model['end'][0], self.model['end'][1], 
                  color='red', s=100, marker='*', label='终点 / End')
        
        # 绘制障碍物（用颜色深浅表示高度）/ Plot obstacles with color indicating altitude
        for threat in self.model['threats']:
            if len(threat) == 4:
                tx, ty, tz, tr = threat
                z_base = 0
            else:
                tx, ty, tz_height, tr, z_base = threat
                tz = tz_height
            
            # 根据障碍物高度选择颜色深度
            # Choose color intensity based on obstacle altitude
            if z_base > 50:
                color = 'orange'
                alpha = 0.5
            elif tz > 200:
                color = 'red'
                alpha = 0.4
            else:
                color = 'lightcoral'
                alpha = 0.3
            
            circle = plt.Circle((tx, ty), tr, color=color, alpha=alpha, 
                              label=f'障碍 h={z_base}-{z_base+tz}' if z_base > 0 else None)
            ax.add_patch(circle)
            
            # 添加高度标注 / Add altitude annotation
            ax.text(tx, ty, f'{int(z_base)}-{int(z_base+tz)}', 
                   ha='center', va='center', fontsize=7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('XY平面路径 (颜色示高度) / XY Plane Path (Color indicates altitude)')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
    
    def plot_cost_history(self, ax):
        """
        绘制代价历史 / Plot Cost History
        """
        history = np.array(self.best_cost_history)
        iterations = np.arange(len(history))
        
        labels = ['J1 (路径长度/Length)', 'J2 (障碍/Obstacle)', 
                 'J3 (高度/Altitude)', 'J4 (平滑度/Smoothness)']
        
        for i in range(history.shape[1]):
            ax.plot(iterations, history[:, i], label=labels[i])
        
        ax.set_xlabel('迭代次数 / Iterations')
        ax.set_ylabel('代价值 / Cost Value')
        ax.set_title('代价演化 / Cost Evolution')
        ax.legend()
        ax.grid(True)


def main():
    """
    主函数 / Main Function
    """
    print("=" * 80)
    print("NMOPSO - 导航变量多目标粒子群优化算法")
    print("Navigation Variable-based Multi-objective Particle Swarm Optimization")
    print("=" * 80)
    print()
    
    # 创建并运行算法 / Create and run algorithm
    nmopso = NMOPSO()
    global_best, repository = nmopso.run()
    
    # 显示结果 / Display results
    print("\n最优解信息 / Best Solution Info:")
    print("-" * 80)
    print(f"代价值 / Cost values: {global_best.cost}")
    print(f"  - J1 (路径长度/Path Length): {global_best.cost[0]:.4f}")
    print(f"  - J2 (障碍代价/Obstacle Cost): {global_best.cost[1]:.4f}")
    print(f"  - J3 (高度代价/Altitude Cost): {global_best.cost[2]:.4f}")
    print(f"  - J4 (平滑度/Smoothness): {global_best.cost[3]:.4f}")
    print(f"存档大小 / Repository size: {len(repository)}")
    print("-" * 80)
    
    # 绘制结果 / Plot results
    nmopso.plot_results()


if __name__ == "__main__":
    main()
