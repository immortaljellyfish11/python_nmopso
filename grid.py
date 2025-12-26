"""
网格管理模块 / Grid Management Module
用于多目标优化中的解空间划分和选择
Used for solution space partitioning and selection in multi-objective optimization
"""
import numpy as np
from utils import roulette_wheel_selection


def create_grid(population, n_grid, alpha):
    """
    创建网格 / Create Grid
    将目标空间划分为网格，用于维持解的多样性
    Partition objective space into grids to maintain solution diversity
    
    网格机制是MOPSO的关键：在目标空间中均匀分布解
    Grid mechanism is key to MOPSO: distribute solutions uniformly in objective space
    
    参数 / Parameters:
        population: 粒子群 / Population
        n_grid: 网格数量 / Number of grids
        alpha: 膨胀率 / Inflation rate
    """
    # 提取所有代价值 / Extract all cost values
    costs = np.array([p.cost for p in population])
    
    # 计算每个目标的最小最大值 / Calculate min and max for each objective
    c_min = np.min(costs, axis=0)
    c_max = np.max(costs, axis=0)
    
    # 扩展边界 / Expand boundaries
    dc = c_max - c_min
    c_min = c_min - alpha * dc
    c_max = c_max + alpha * dc
    
    n_obj = len(c_min)
    
    # 创建网格结构 / Create grid structure
    grid = []
    for j in range(n_obj):
        # 在每个目标维度上创建网格边界 / Create grid boundaries for each objective
        cj = np.linspace(c_min[j], c_max[j], n_grid + 1)
        grid.append({
            'LB': np.concatenate([[-np.inf], cj]),
            'UB': np.concatenate([cj, [np.inf]])
        })
    
    return grid


def find_grid_index(particle, grid):
    """
    查找粒子的网格索引 / Find Grid Index for Particle
    确定粒子在网格中的位置
    Determine the position of particle in the grid
    """
    n_obj = len(particle.cost)
    n_grid = len(grid[0]['LB']) - 2
    
    particle.grid_sub_index = []
    
    # 找到每个目标维度的网格索引 / Find grid index for each objective dimension
    for j in range(n_obj):
        idx = np.where(particle.cost[j] < grid[j]['UB'])[0]
        if len(idx) == 0:
            particle.grid_sub_index.append(n_grid - 1)
        else:
            particle.grid_sub_index.append(idx[0])
    
    # 计算线性网格索引 / Calculate linear grid index
    particle.grid_index = particle.grid_sub_index[0]
    for j in range(1, n_obj):
        particle.grid_index = (particle.grid_index - 1) * n_grid + particle.grid_sub_index[j]
    
    return particle


def select_leader(repository, beta, length_weight=0.6):
    """
    选择领导者（改进版）/ Select Leader (Improved)
    从存档中选择一个粒子作为全局最优引导其他粒子
    Select a particle from repository as global best to guide other particles
    
    改进策略 / Improved Strategy:
    1. 基于网格密度的选择（维持多样性）
    2. 基于路径长度和平滑度的加权评分（解决路径过长问题）
    
    使用基于网格密度的选择策略：
    1. 网格中粒子越少，被选中概率越高（鼓励探索稀疏区域）
    2. 这有助于维持Pareto前沿的多样性
    
    参数 / Parameters:
        repository: 存档 / Repository
        beta: 网格选择压力 / Grid selection pressure
        length_weight: 路径长度权重 [0,1]，越大越重视短路径 / Path length weight
    
    Selection strategy based on grid density:
    1. Lower particle density -> higher selection probability (encourage exploration)
    2. Helps maintain diversity of Pareto front
    """
    if len(repository) == 0:
        return None
    
    if len(repository) == 1:
        return repository[0]
    
    # 提取代价值 / Extract costs
    costs = np.array([p.cost for p in repository])
    
    # 过滤inf值 / Filter inf values
    valid_mask = ~np.any(np.isinf(costs), axis=1)
    
    if not np.any(valid_mask):
        # 如果都是inf，使用原始方法 / If all inf, use original method
        return _select_leader_original(repository, beta)
    
    # 获取有效解 / Get valid solutions
    valid_costs = costs[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # 提取J1（路径长度）和J4（平滑度）
    # Extract J1 (path length) and J4 (smoothness)
    J1 = valid_costs[:, 0]
    J4 = valid_costs[:, 3]
    
    # 归一化 / Normalize
    J1_range = J1.max() - J1.min()
    J4_range = J4.max() - J4.min()
    
    if J1_range > 1e-6:
        J1_norm = (J1 - J1.min()) / J1_range
    else:
        J1_norm = np.zeros_like(J1)
    
    if J4_range > 1e-6:
        J4_norm = (J4 - J4.min()) / J4_range
    else:
        J4_norm = np.zeros_like(J4)
    
    # 计算综合评分（越小越好→越大越好）
    # Calculate composite score (smaller is better → larger is better)
    smoothness_weight = 1 - length_weight
    quality_scores = length_weight * (1 - J1_norm) + smoothness_weight * (1 - J4_norm)
    
    # 获取所有粒子的网格索引 / Get grid indices of all particles
    grid_indices = np.array([p.grid_index for p in repository])
    
    # 找到占据的网格 / Find occupied cells
    occupied_cells = np.unique(grid_indices)
    
    # 计算每个网格中的粒子数 / Count particles in each cell
    n_particles = np.array([np.sum(grid_indices == oc) for oc in occupied_cells])
    
    # 网格密度概率（粒子少的网格概率高）/ Grid density probability
    grid_prob = np.exp(-beta * n_particles)
    grid_prob = grid_prob / np.sum(grid_prob)
    
    # 选择网格 / Select cell
    selected_cell_idx = roulette_wheel_selection(grid_prob)
    selected_cell = occupied_cells[selected_cell_idx]
    
    # 从选中的网格中的有效粒子选择 / Select from valid particles in the cell
    cell_members = np.where(grid_indices == selected_cell)[0]
    
    # 找到该网格中的有效粒子 / Find valid particles in this cell
    cell_valid_mask = np.isin(cell_members, valid_indices)
    
    if not np.any(cell_valid_mask):
        # 如果该网格中没有有效粒子，随机选择
        # If no valid particles in this cell, select randomly
        return repository[np.random.choice(cell_members)]
    
    cell_valid_members = cell_members[cell_valid_mask]
    
    # 获取这些粒子在valid_indices中的位置
    # Get positions of these particles in valid_indices
    member_score_indices = [np.where(valid_indices == m)[0][0] for m in cell_valid_members]
    member_scores = quality_scores[member_score_indices]
    
    # 基于质量评分选择（评分高的概率大）
    # Select based on quality score (higher score, higher probability)
    score_prob = np.power(member_scores, 2)  # 平方增强差异 / Square to enhance difference
    
    # 检查sum避免NaN / Check sum to avoid NaN
    score_sum = score_prob.sum()
    if score_sum > 1e-10:
        score_prob = score_prob / score_sum
    else:
        # 如果所有评分相同，均匀分布 / If all scores are same, uniform distribution
        score_prob = np.ones_like(score_prob) / len(score_prob)
    
    selected_idx = np.random.choice(len(cell_valid_members), p=score_prob)
    selected_member = cell_valid_members[selected_idx]
    
    return repository[selected_member]


def _select_leader_original(repository, beta):
    """
    原始的领导者选择方法（仅基于网格密度）
    Original leader selection method (grid density only)
    """
    # 获取所有粒子的网格索引 / Get grid indices of all particles
    grid_indices = np.array([p.grid_index for p in repository])
    
    # 找到占据的网格 / Find occupied cells
    occupied_cells = np.unique(grid_indices)
    
    # 计算每个网格中的粒子数 / Count particles in each cell
    n_particles = np.array([np.sum(grid_indices == oc) for oc in occupied_cells])
    
    # 计算选择概率（粒子少的网格概率高）/ Calculate selection probability
    P = np.exp(-beta * n_particles)
    P = P / np.sum(P)
    
    # 选择网格 / Select cell
    selected_cell_idx = roulette_wheel_selection(P)
    selected_cell = occupied_cells[selected_cell_idx]
    
    # 从选中的网格中随机选择一个粒子 / Randomly select a particle from the cell
    selected_members = np.where(grid_indices == selected_cell)[0]
    selected_member = np.random.choice(selected_members)
    
    return repository[selected_member]


def delete_one_rep_member(repository, gamma):
    """
    删除一个存档成员 / Delete One Repository Member
    当存档满时，删除拥挤区域的粒子
    When repository is full, delete particles from crowded regions
    
    删除策略与选择策略相反：优先删除密集区域的粒子
    Deletion strategy is opposite to selection: prioritize deleting from dense regions
    """
    # 获取所有粒子的网格索引 / Get grid indices
    grid_indices = np.array([p.grid_index for p in repository])
    
    # 找到占据的网格 / Find occupied cells
    occupied_cells = np.unique(grid_indices)
    
    # 计算每个网格中的粒子数 / Count particles in each cell
    n_particles = np.array([np.sum(grid_indices == oc) for oc in occupied_cells])
    
    # 计算删除概率（粒子多的网格概率高）/ Calculate deletion probability
    P = np.exp(gamma * n_particles)
    P = P / np.sum(P)
    
    # 选择要删除的网格 / Select cell to delete from
    selected_cell_idx = roulette_wheel_selection(P)
    selected_cell = occupied_cells[selected_cell_idx]
    
    # 从选中的网格中随机选择一个粒子删除 / Randomly select a particle to delete
    selected_members = np.where(grid_indices == selected_cell)[0]
    selected_member = np.random.choice(selected_members)
    
    # 删除粒子 / Delete particle
    repository.pop(selected_member)
    
    return repository
