"""
对比算法模块 / Comparison Algorithms Module

包含用于性能对比的经典算法实现：
1. 标准PSO (Standard PSO)
2. 量子粒子群优化 (QPSO - Quantum PSO)
3. 差分进化 (DE - Differential Evolution)

这些算法用于证明NMOPSO的优势
These algorithms are used to demonstrate the advantages of NMOPSO
"""

import numpy as np
import copy
from particle import Particle, create_random_solution
from utils import spherical_to_cartesian
from cost_function import calculate_cost


class StandardPSO:
    """
    标准PSO算法 / Standard PSO Algorithm
    
    与NMOPSO的区别：
    Differences from NMOPSO:
    1. 单目标优化（加权求和）/ Single objective (weighted sum)
    2. 没有存档机制 / No repository mechanism
    3. 简单的全局最优选择 / Simple global best selection
    
    教学要点 / Teaching Points:
    - PSO是群体智能算法，模拟鸟群觅食
    - 速度更新公式平衡探索与开发
    - 惯性权重递减提高收敛性
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.particles = []
        self.global_best = None
        self.best_cost_history = []
        
        # 目标权重 / Objective weights
        self.weights = np.array([0.3, 0.4, 0.2, 0.1])  # J1, J2, J3, J4的权重
    
    def single_objective_cost(self, multi_cost):
        """
        将多目标转换为单目标 / Convert multi-objective to single objective
        使用加权求和法 / Using weighted sum method
        """
        return np.dot(self.weights, multi_cost)
    
    def run(self, max_iter=300, n_pop=70):
        """
        运行标准PSO / Run Standard PSO
        """
        print("\n" + "="*80)
        print("运行标准PSO算法 / Running Standard PSO")
        print("="*80)
        
        # 初始化
        var_min = {}
        var_max = {}
        n = self.model['n']
        
        var_max['r'] = 3 * np.linalg.norm(self.model['start'] - self.model['end']) / n
        var_min['r'] = var_max['r'] / 9
        angle_range = np.pi / 4
        var_min['psi'] = -angle_range
        var_max['psi'] = angle_range
        var_min['phi'] = -angle_range
        var_max['phi'] = angle_range
        
        vel_min = {}
        vel_max = {}
        alpha = 0.5
        vel_max['r'] = alpha * (var_max['r'] - var_min['r'])
        vel_min['r'] = -vel_max['r']
        vel_max['psi'] = alpha * (var_max['psi'] - var_min['psi'])
        vel_min['psi'] = -vel_max['psi']
        vel_max['phi'] = alpha * (var_max['phi'] - var_min['phi'])
        vel_min['phi'] = -vel_max['phi']
        
        # 初始化粒子群
        for i in range(n_pop):
            p = Particle()
            p.position = create_random_solution(n, var_min, var_max)
            p.velocity = {'r': np.zeros(n), 'phi': np.zeros(n), 'psi': np.zeros(n)}
            
            cart_pos = spherical_to_cartesian(p.position, self.model)
            p.cost = calculate_cost(cart_pos, self.model, var_min)
            p.best['position'] = copy.deepcopy(p.position)
            p.best['cost'] = copy.deepcopy(p.cost)
            
            self.particles.append(p)
        
        # 找到初始全局最优
        best_idx = 0
        best_single_cost = self.single_objective_cost(self.particles[0].cost)
        for i in range(1, n_pop):
            single_cost = self.single_objective_cost(self.particles[i].cost)
            if single_cost < best_single_cost:
                best_single_cost = single_cost
                best_idx = i
        self.global_best = copy.deepcopy(self.particles[best_idx])
        
        # PSO主循环
        w = 1.0
        w_damp = 0.98
        c1, c2 = 1.5, 1.5
        
        for it in range(max_iter):
            self.best_cost_history.append(copy.deepcopy(self.global_best.cost))
            
            for i in range(n_pop):
                p = self.particles[i]
                
                # 更新速度和位置
                for var in ['r', 'psi', 'phi']:
                    p.velocity[var] = (
                        w * p.velocity[var] +
                        c1 * np.random.rand(n) * (p.best['position'][var] - p.position[var]) +
                        c2 * np.random.rand(n) * (self.global_best.position[var] - p.position[var])
                    )
                    p.velocity[var] = np.clip(p.velocity[var], vel_min[var], vel_max[var])
                    p.position[var] = p.position[var] + p.velocity[var]
                    
                    out_of_range = (p.position[var] < var_min[var]) | (p.position[var] > var_max[var])
                    p.velocity[var][out_of_range] = -p.velocity[var][out_of_range]
                    p.position[var] = np.clip(p.position[var], var_min[var], var_max[var])
                
                # 评估
                cart_pos = spherical_to_cartesian(p.position, self.model)
                p.cost = calculate_cost(cart_pos, self.model, var_min)
                
                # 更新个体最优
                if self.single_objective_cost(p.cost) < self.single_objective_cost(p.best['cost']):
                    p.best['position'] = copy.deepcopy(p.position)
                    p.best['cost'] = copy.deepcopy(p.cost)
                
                # 更新全局最优
                if self.single_objective_cost(p.cost) < self.single_objective_cost(self.global_best.cost):
                    self.global_best = copy.deepcopy(p)
            
            w = w * w_damp
            
            if (it + 1) % 50 == 0:
                print(f"迭代 {it+1}/{max_iter} | 最优代价: {self.global_best.cost} | "
                      f"加权和: {self.single_objective_cost(self.global_best.cost):.4f}")
        
        print("标准PSO完成 / Standard PSO completed")
        return self.global_best


class QPSO:
    """
    量子粒子群优化 (QPSO) / Quantum Particle Swarm Optimization
    
    特点 / Features:
    1. 基于量子力学原理 / Based on quantum mechanics
    2. 粒子以概率分布存在 / Particles exist in probability distribution
    3. 没有速度概念，直接更新位置 / No velocity, directly update position
    4. 全局搜索能力更强 / Stronger global search capability
    
    教学要点 / Teaching Points:
    - QPSO消除了速度的概念
    - 使用平均最优位置(mbest)引导搜索
    - 收缩扩张系数α控制探索与开发
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.particles = []
        self.global_best = None
        self.best_cost_history = []
        self.weights = np.array([0.3, 0.4, 0.2, 0.1])
    
    def single_objective_cost(self, multi_cost):
        return np.dot(self.weights, multi_cost)
    
    def run(self, max_iter=300, n_pop=70):
        """
        运行QPSO / Run QPSO
        """
        print("\n" + "="*80)
        print("运行量子PSO算法 / Running QPSO")
        print("="*80)
        
        # 初始化（与标准PSO相同）
        var_min = {}
        var_max = {}
        n = self.model['n']
        
        var_max['r'] = 3 * np.linalg.norm(self.model['start'] - self.model['end']) / n
        var_min['r'] = var_max['r'] / 9
        angle_range = np.pi / 4
        var_min['psi'] = -angle_range
        var_max['psi'] = angle_range
        var_min['phi'] = -angle_range
        var_max['phi'] = angle_range
        
        # 初始化粒子群
        for i in range(n_pop):
            p = Particle()
            p.position = create_random_solution(n, var_min, var_max)
            
            cart_pos = spherical_to_cartesian(p.position, self.model)
            p.cost = calculate_cost(cart_pos, self.model, var_min)
            p.best['position'] = copy.deepcopy(p.position)
            p.best['cost'] = copy.deepcopy(p.cost)
            
            self.particles.append(p)
        
        # 找到初始全局最优
        best_idx = np.argmin([self.single_objective_cost(p.cost) for p in self.particles])
        self.global_best = copy.deepcopy(self.particles[best_idx])
        
        # QPSO主循环
        for it in range(max_iter):
            self.best_cost_history.append(copy.deepcopy(self.global_best.cost))
            
            # 计算平均最优位置 (mbest) / Calculate mean best position
            mbest = {}
            for var in ['r', 'psi', 'phi']:
                mbest[var] = np.mean([p.best['position'][var] for p in self.particles], axis=0)
            
            # 收缩扩张系数 / Contraction-expansion coefficient
            alpha = 1.0 - 0.5 * it / max_iter  # 从1线性递减到0.5
            
            for i in range(n_pop):
                p = self.particles[i]
                
                # QPSO位置更新 / QPSO position update
                for var in ['r', 'psi', 'phi']:
                    phi = np.random.rand(n)
                    u = np.random.rand(n)
                    
                    # 局部吸引子 / Local attractor
                    p_local = phi * p.best['position'][var] + \
                             (1 - phi) * self.global_best.position[var]
                    
                    # 量子位置更新 / Quantum position update
                    if np.random.rand() < 0.5:
                        p.position[var] = p_local + alpha * np.abs(mbest[var] - p.position[var]) * np.log(1.0 / u)
                    else:
                        p.position[var] = p_local - alpha * np.abs(mbest[var] - p.position[var]) * np.log(1.0 / u)
                    
                    # 边界处理
                    p.position[var] = np.clip(p.position[var], var_min[var], var_max[var])
                
                # 评估
                cart_pos = spherical_to_cartesian(p.position, self.model)
                p.cost = calculate_cost(cart_pos, self.model, var_min)
                
                # 更新个体最优
                if self.single_objective_cost(p.cost) < self.single_objective_cost(p.best['cost']):
                    p.best['position'] = copy.deepcopy(p.position)
                    p.best['cost'] = copy.deepcopy(p.cost)
                
                # 更新全局最优
                if self.single_objective_cost(p.cost) < self.single_objective_cost(self.global_best.cost):
                    self.global_best = copy.deepcopy(p)
            
            if (it + 1) % 50 == 0:
                print(f"迭代 {it+1}/{max_iter} | 最优代价: {self.global_best.cost} | "
                      f"加权和: {self.single_objective_cost(self.global_best.cost):.4f}")
        
        print("QPSO完成 / QPSO completed")
        return self.global_best


class DifferentialEvolution:
    """
    差分进化算法 (DE) / Differential Evolution
    
    特点 / Features:
    1. 基于种群的进化算法 / Population-based evolutionary algorithm
    2. 通过差分变异产生新个体 / Generate new individuals through differential mutation
    3. 简单但强大的全局优化能力 / Simple but powerful global optimization
    
    教学要点 / Teaching Points:
    - DE属于进化算法家族，不同于PSO的群体智能
    - 变异、交叉、选择三个基本操作
    - F控制变异强度，CR控制交叉概率
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.population = []
        self.best_solution = None
        self.best_cost_history = []
        self.weights = np.array([0.3, 0.4, 0.2, 0.1])
    
    def single_objective_cost(self, multi_cost):
        return np.dot(self.weights, multi_cost)
    
    def run(self, max_iter=300, n_pop=70, F=0.8, CR=0.9):
        """
        运行差分进化 / Run Differential Evolution
        
        参数 / Parameters:
            F: 缩放因子 / Scaling factor (mutation intensity)
            CR: 交叉概率 / Crossover probability
        """
        print("\n" + "="*80)
        print("运行差分进化算法 / Running Differential Evolution")
        print("="*80)
        
        # 初始化
        var_min = {}
        var_max = {}
        n = self.model['n']
        
        var_max['r'] = 3 * np.linalg.norm(self.model['start'] - self.model['end']) / n
        var_min['r'] = var_max['r'] / 9
        angle_range = np.pi / 4
        var_min['psi'] = -angle_range
        var_max['psi'] = angle_range
        var_min['phi'] = -angle_range
        var_max['phi'] = angle_range
        
        # 初始化种群
        for i in range(n_pop):
            individual = create_random_solution(n, var_min, var_max)
            cart_pos = spherical_to_cartesian(individual, self.model)
            cost = calculate_cost(cart_pos, self.model, var_min)
            self.population.append({'position': individual, 'cost': cost})
        
        # 找到最优解
        best_idx = np.argmin([self.single_objective_cost(ind['cost']) for ind in self.population])
        self.best_solution = copy.deepcopy(self.population[best_idx])
        
        # DE主循环
        for it in range(max_iter):
            self.best_cost_history.append(copy.deepcopy(self.best_solution['cost']))
            
            new_population = []
            
            for i in range(n_pop):
                # 1. 变异 (Mutation): V = X_r1 + F * (X_r2 - X_r3)
                # 随机选择三个不同的个体
                indices = list(range(n_pop))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                
                mutant = {}
                for var in ['r', 'psi', 'phi']:
                    mutant[var] = (self.population[r1]['position'][var] + 
                                   F * (self.population[r2]['position'][var] - 
                                       self.population[r3]['position'][var]))
                    mutant[var] = np.clip(mutant[var], var_min[var], var_max[var])
                
                # 2. 交叉 (Crossover)
                trial = {}
                for var in ['r', 'psi', 'phi']:
                    cross_mask = np.random.rand(n) < CR
                    trial[var] = np.where(cross_mask, mutant[var], self.population[i]['position'][var])
                
                # 3. 选择 (Selection)
                cart_pos = spherical_to_cartesian(trial, self.model)
                trial_cost = calculate_cost(cart_pos, self.model, var_min)
                
                if self.single_objective_cost(trial_cost) < self.single_objective_cost(self.population[i]['cost']):
                    new_population.append({'position': trial, 'cost': trial_cost})
                    
                    # 更新全局最优
                    if self.single_objective_cost(trial_cost) < self.single_objective_cost(self.best_solution['cost']):
                        self.best_solution = {'position': copy.deepcopy(trial), 'cost': copy.deepcopy(trial_cost)}
                else:
                    new_population.append(copy.deepcopy(self.population[i]))
            
            self.population = new_population
            
            if (it + 1) % 50 == 0:
                print(f"迭代 {it+1}/{max_iter} | 最优代价: {self.best_solution['cost']} | "
                      f"加权和: {self.single_objective_cost(self.best_solution['cost']):.4f}")
        
        print("DE完成 / DE completed")
        # 将字典转换为Particle对象以保持一致性
        best_particle = Particle()
        best_particle.position = self.best_solution['position']
        best_particle.cost = self.best_solution['cost']
        best_particle.best['position'] = copy.deepcopy(self.best_solution['position'])
        best_particle.best['cost'] = copy.deepcopy(self.best_solution['cost'])

        return best_particle