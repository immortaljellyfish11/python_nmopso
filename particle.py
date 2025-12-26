"""
粒子类和相关操作 / Particle Class and Related Operations
定义粒子结构和粒子群操作
Defines particle structure and swarm operations
"""
import numpy as np
from utils import spherical_to_cartesian, dominates


class Particle:
    """
    粒子类 / Particle Class
    表示PSO算法中的一个粒子（一个候选解）
    Represents a particle (candidate solution) in PSO
    """
    def __init__(self):
        # 位置 (导航变量空间) / Position (in navigation variable space)
        self.position = {'r': None, 'phi': None, 'psi': None}
        
        # 速度 / Velocity
        self.velocity = {'r': None, 'phi': None, 'psi': None}
        
        # 代价值 / Cost values
        self.cost = None
        
        # 个体最优 / Personal best
        self.best = {
            'position': {'r': None, 'phi': None, 'psi': None},
            'cost': None
        }
        
        # Pareto支配标记 / Pareto dominance flag
        self.is_dominated = False
        
        # 网格索引 / Grid indices
        self.grid_index = None
        self.grid_sub_index = None


def create_random_solution(var_size, var_min, var_max):
    """
    创建随机解 / Create Random Solution
    初始化粒子的随机位置
    Initialize random position for particle
    
    参数 / Parameters:
        var_size: 变量维度 / Variable dimensions
        var_min, var_max: 变量边界 / Variable bounds
    """
    solution = {}
    solution['r'] = np.random.uniform(var_min['r'], var_max['r'], var_size)
    solution['phi'] = np.random.uniform(var_min['phi'], var_max['phi'], var_size)
    solution['psi'] = np.random.uniform(var_min['psi'], var_max['psi'], var_size)
    return solution


def mutate(particle, rep, delta, var_max, var_min):
    """
    变异操作 / Mutation Operation
    通过添加噪声来探索新的解空间
    Explore new solution space by adding noise
    
    变异有助于维持种群多样性，避免早熟收敛
    Mutation helps maintain population diversity and avoid premature convergence
    """
    n_var = len(particle.position['r'])
    pbest = particle.best
    
    # 变异强度系数 / Mutation intensity coefficient
    beta = np.tanh(delta * len(rep))
    
    # 添加高斯噪声 / Add Gaussian noise
    x_new = {}
    x_new['r'] = particle.position['r'] + np.random.randn(n_var) * pbest['position']['r'] * beta
    x_new['phi'] = particle.position['phi'] + np.random.randn(n_var) * pbest['position']['phi'] * beta
    x_new['psi'] = particle.position['psi'] + np.random.randn(n_var) * pbest['position']['psi'] * beta
    
    # 边界处理 / Boundary handling
    x_new['r'] = np.clip(x_new['r'], var_min['r'], var_max['r'])
    x_new['phi'] = np.clip(x_new['phi'], var_min['phi'], var_max['phi'])
    x_new['psi'] = np.clip(x_new['psi'], var_min['psi'], var_max['psi'])
    
    return x_new


def determine_domination(particles):
    """
    确定支配关系 / Determine Domination Relationships
    在种群中标记被支配的粒子
    Mark dominated particles in the population
    
    这是多目标优化的核心：找出Pareto前沿
    This is the core of multi-objective optimization: finding the Pareto front
    """
    n_pop = len(particles)
    
    if n_pop == 0:
        return particles
    
    # 初始化所有粒子为非支配 / Initialize all particles as non-dominated
    for particle in particles:
        particle.is_dominated = False
    
    if n_pop == 1:
        if np.any(particles[0].cost == np.inf):
            particles[0].is_dominated = True
        return particles
    
    # 比较所有粒子对 / Compare all particle pairs
    for i in range(n_pop - 1):
        if np.any(particles[i].cost == np.inf):
            particles[i].is_dominated = True
        
        for j in range(i + 1, n_pop):
            if dominates(particles[i].cost, particles[j].cost):
                particles[j].is_dominated = True
            
            if dominates(particles[j].cost, particles[i].cost):
                particles[i].is_dominated = True
    
    # 检查最后一个粒子 / Check the last particle
    if np.any(particles[-1].cost == np.inf):
        particles[-1].is_dominated = True
    
    for j in range(n_pop - 1):
        if dominates(particles[-1].cost, particles[j].cost):
            particles[j].is_dominated = True
        
        if dominates(particles[j].cost, particles[-1].cost):
            particles[-1].is_dominated = True
            break
    
    return particles
