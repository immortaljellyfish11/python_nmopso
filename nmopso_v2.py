"""
NMOPSO V2 - 智能初始化版本 / NMOPSO V2 - Smart Initialization Version

改进点 / Improvements:
1. 使用智能初始化策略，减少inf结果
2. 支持随机障碍物配置
3. 更稳定的收敛性能

基于原nmopso.py，主要改动在initialize_swarm()方法
Based on original nmopso.py, main change in initialize_swarm() method
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

from config import Config
from particle import Particle, mutate, determine_domination
from particle_v2 import create_smart_solution  # 🔑 使用智能初始化
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
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':  # macOS
        fonts = ['PingFang SC', 'Heiti SC', 'STHeiti']
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'AR PL UMing CN']
    
    for font in fonts:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font]
            matplotlib.rcParams['axes.unicode_minus'] = False
            return True
        except:
            continue
    
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    return False

setup_chinese_font()
# ================================================================


class NMOPSO_V2:
    """
    NMOPSO V2算法类 / NMOPSO V2 Algorithm Class
    使用智能初始化策略
    Uses smart initialization strategy
    """
    
    def __init__(self, config=None, use_smart_init=True, smart_ratio=0.7):
        """
        初始化NMOPSO V2算法 / Initialize NMOPSO V2 Algorithm
        
        参数 / Parameters:
            config: 配置对象 / Configuration object
            use_smart_init: 是否使用智能初始化 / Whether to use smart initialization
            smart_ratio: 智能初始化的粒子比例 [0,1] / Ratio of smart initialized particles
        """
        self.config = config if config is not None else Config()
        self.model = self.config.model if hasattr(self.config, 'model') else self.config.create_model()
        
        self.use_smart_init = use_smart_init
        self.smart_ratio = smart_ratio
        
        # 设置变量边界 / Setup variable bounds
        self.setup_bounds()
        
        # 初始化粒子群 / Initialize swarm
        self.particles = []
        self.repository = []
        self.global_best = None
        
        # 记录历史 / Record history
        self.best_cost_history = []
        self.inf_count_per_iter = []  # 记录每次迭代的inf数量 / Track inf count per iteration
    
    def setup_bounds(self):
        """设置变量边界 / Setup Variable Bounds"""
        model = self.model
        # 从config或model获取n_var / Get n_var from config or model
        if hasattr(self.config, 'n_var'):
            n = self.config.n_var
        else:
            n = model.get('n', 20)  # 默认值20 / Default value 20
        
        self.var_min = self.config.var_min
        self.var_max = self.config.var_max
        
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
        初始化粒子群 (智能版本) / Initialize Particle Swarm (Smart Version)
        
        改进策略 / Improvement Strategy:
        - 部分粒子使用智能初始化（沿起点-终点方向）
        - 部分粒子使用随机初始化（保持多样性）
        - 确保至少有一个可行解
        """
        print(f"\n初始化粒子群 ({self.config.n_pop} 个粒子)...")
        print(f"Initializing swarm ({self.config.n_pop} particles)...")
        
        if self.use_smart_init:
            n_smart = int(self.config.n_pop * self.smart_ratio)
            n_random = self.config.n_pop - n_smart
            print(f"  智能初始化 / Smart init: {n_smart} 个")
            print(f"  随机初始化 / Random init: {n_random} 个")
        else:
            n_smart = 0
            n_random = self.config.n_pop
            print(f"  全部随机初始化 / All random init: {n_random} 个")
        
        self.particles = []
        feasible_count = 0
        inf_count = 0
        
        for i in range(self.config.n_pop):
            particle = Particle()
            
            # 决定使用智能还是随机初始化 / Decide smart or random initialization
            if i < n_smart:
                # 智能初始化 / Smart initialization
                particle.position = create_smart_solution(
                    self.var_size, self.var_min, self.var_max, self.model,
                    direction_bias=0.7, perturbation=0.3
                )
            else:
                # 随机初始化 / Random initialization
                from particle import create_random_solution
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
            # 转换为笛卡尔坐标后计算代价 / Convert to Cartesian coordinates before cost calculation
            cart_position = spherical_to_cartesian(particle.position, self.model)
            particle.cost = calculate_cost(cart_position, self.model, self.var_min)
            
            # 统计可行解 / Count feasible solutions
            if not np.any(np.isinf(particle.cost)):
                feasible_count += 1
            else:
                inf_count += 1
            
            # 初始化个体最优 / Initialize personal best
            particle.best['position'] = copy.deepcopy(particle.position)
            particle.best['cost'] = copy.deepcopy(particle.cost)
            
            self.particles.append(particle)
        
        print(f"\n初始化统计 / Initialization Statistics:")
        print(f"  可行解 / Feasible: {feasible_count}/{self.config.n_pop} ({100*feasible_count/self.config.n_pop:.1f}%)")
        print(f"  inf解 / inf solutions: {inf_count}/{self.config.n_pop} ({100*inf_count/self.config.n_pop:.1f}%)")
        
        if feasible_count == 0:
            print("\n[WARNING] 没有找到可行解，算法可能难以收敛")
            print("          No feasible solution found, algorithm may struggle to converge")
        else:
            print("\n[SUCCESS] 粒子群初始化成功")
            print("          Swarm initialized successfully")
    
    def run(self):
        """运行NMOPSO V2算法 / Run NMOPSO V2 Algorithm"""
        # 初始化 / Initialize
        self.initialize_swarm()
        
        # 确定支配关系 / Determine domination
        self.particles = determine_domination(self.particles)
        
        # 初始化存档 / Initialize repository
        self.repository = [p for p in self.particles if not p.is_dominated]
        
        print(f"\n初始存档大小 / Initial repository size: {len(self.repository)}")
        
        # 创建网格 / Create grid
        if len(self.repository) > 0:
            grid = create_grid(self.repository, self.config.n_grid, self.config.alpha)
            for i in range(len(self.repository)):
                self.repository[i] = find_grid_index(self.repository[i], grid)
            
            # 选择全局最优 / Select global best
            length_weight = getattr(self.config, 'length_weight', 0.6)
            self.global_best = select_leader(self.repository, self.config.beta, length_weight)
        else:
            print("[WARNING] 初始存档为空，使用第一个粒子作为全局最优")
            print("          Initial repository empty, using first particle as global best")
            self.global_best = self.particles[0]
        
        # 主循环 / Main loop
        print(f"\n" + "=" * 80)
        print(f"开始优化 / Starting Optimization")
        print(f"最大迭代次数 / Max iterations: {self.config.max_iter}")
        print(f"种群大小 / Population size: {self.config.n_pop}")
        print(f"存档容量 / Repository capacity: {self.config.n_rep}")
        print("=" * 80 + "\n")
        
        w = self.config.w
        
        for it in range(self.config.max_iter):
            # 记录最优代价 / Record best cost
            self.best_cost_history.append(copy.deepcopy(self.global_best.cost))
            
            # 统计当前迭代的inf数量 / Count inf in current iteration
            current_inf = sum(1 for p in self.particles if np.any(np.isinf(p.cost)))
            self.inf_count_per_iter.append(current_inf)
            
            # 进度显示 / Progress display
            if (it + 1) % 10 == 0 or it == 0:
                print(f"迭代 {it+1:3d}/{self.config.max_iter}: "
                      f"存档={len(self.repository):3d}, "
                      f"inf={current_inf:3d}/{self.config.n_pop}, "
                      f"J=[{self.global_best.cost[0]:.3f}, {self.global_best.cost[1]:.3f}, "
                      f"{self.global_best.cost[2]:.3f}, {self.global_best.cost[3]:.3f}]")
            
            # 更新每个粒子 / Update each particle
            for i in range(len(self.particles)):
                particle = self.particles[i]
                
                # 选择领导者 / Select leader
                if len(self.repository) > 0:
                    length_weight = getattr(self.config, 'length_weight', 0.6)
                    self.global_best = select_leader(self.repository, self.config.beta, length_weight)
                
                # 更新速度和位置 / Update velocity and position
                for var in ['r', 'phi', 'psi']:
                    # PSO速度更新公式 / PSO velocity update formula
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
                    
                    # 速度镜像 / Velocity mirroring
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
                # 转换为笛卡尔坐标 / Convert to Cartesian coordinates
                cart_position = spherical_to_cartesian(particle.position, self.model)
                particle.cost = calculate_cost(cart_position, self.model, self.var_min)
                
                # 变异操作 / Mutation operation
                pm = (1 - it / (self.config.max_iter - 1)) ** (1 / self.config.mu)
                if np.random.rand() < pm and len(self.repository) > 0:
                    new_position = mutate(
                        particle, self.repository, self.config.delta,
                        self.var_max, self.var_min
                    )
                    # 转换为笛卡尔坐标 / Convert to Cartesian coordinates
                    new_cart = spherical_to_cartesian(new_position, self.model)
                    new_cost = calculate_cost(new_cart, self.model, self.var_min)
                    
                    # 接受变异 / Accept mutation
                    if dominates(new_cost, particle.cost):
                        particle.position = copy.deepcopy(new_position)
                        particle.cost = copy.deepcopy(new_cost)
                
                # 更新个体最优 / Update personal best
                if dominates(particle.cost, particle.best['cost']):
                    particle.best['position'] = copy.deepcopy(particle.position)
                    particle.best['cost'] = copy.deepcopy(particle.cost)
            
            # 确定支配关系 / Determine domination
            self.particles = determine_domination(self.particles)
            
            # 更新存档 / Update repository
            non_dominated = [p for p in self.particles if not p.is_dominated]
            self.repository.extend(non_dominated)
            
            self.repository = determine_domination(self.repository)
            self.repository = [p for p in self.repository if not p.is_dominated]
            
            # 更新网格 / Update grid
            if len(self.repository) > 0:
                grid = create_grid(self.repository, self.config.n_grid, self.config.alpha)
                for i in range(len(self.repository)):
                    self.repository[i] = find_grid_index(self.repository[i], grid)
                
                # 存档大小控制 / Repository size control
                if len(self.repository) > self.config.n_rep:
                    extra = len(self.repository) - self.config.n_rep
                    for _ in range(extra):
                        self.repository = delete_one_rep_member(self.repository)
        
        print("\n" + "=" * 80)
        print("[DONE] 优化完成 / Optimization Completed")
        print("=" * 80)
        print(f"最终存档大小 / Final repository size: {len(self.repository)}")
        print(f"最终全局最优代价 / Final global best cost: "
              f"[{self.global_best.cost[0]:.3f}, {self.global_best.cost[1]:.3f}, "
              f"{self.global_best.cost[2]:.3f}, {self.global_best.cost[3]:.3f}]")
        
        return self.repository, self.global_best
    
    def plot_results(self, show_all_pareto=False):
        """
        绘制结果 / Plot Results
        (继承原nmopso.py的绘图逻辑，这里简化)
        """
        from nmopso import NMOPSO
        # 创建临时NMOPSO对象来使用其plot_results方法
        temp = NMOPSO(self.config)
        temp.model = self.model
        temp.repository = self.repository
        temp.global_best = self.global_best
        temp.best_cost_history = self.best_cost_history
        temp.plot_results(show_all_pareto)


if __name__ == "__main__":
    print("NMOPSO V2 - 智能初始化版本")
    print("NMOPSO V2 - Smart Initialization Version")
    print("=" * 60)
    
    # 使用增强配置 / Use enhanced configuration
    from config_enhanced import EnhancedConfig
    config = EnhancedConfig()
    
    # 运行NMOPSO V2 / Run NMOPSO V2
    optimizer = NMOPSO_V2(config, use_smart_init=True, smart_ratio=0.7)
    repository, best = optimizer.run()
    
    # 绘制结果 / Plot results
    print("\n正在生成结果图...")
    print("Generating result plots...")
    optimizer.plot_results(show_all_pareto=True)
