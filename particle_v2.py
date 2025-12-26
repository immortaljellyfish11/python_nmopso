"""
粒子类V2 - 智能初始化 / Particle Class V2 - Smart Initialization
在原particle.py基础上添加智能初始化策略
Adds smart initialization strategy on top of particle.py

改进点 / Improvements:
- 沿起点-终点方向初始化，减少初始碰撞
- 添加适度扰动保持多样性
- 避免过于激进的初始角度
"""
import numpy as np
from particle import Particle, mutate, determine_domination
from utils import spherical_to_cartesian


def create_smart_solution(var_size, var_min, var_max, model, 
                          direction_bias=0.7, perturbation=0.3):
    """
    创建智能解 / Create Smart Solution
    沿起点-终点方向初始化，同时保持多样性
    Initialize along start-end direction while maintaining diversity
    
    参数 / Parameters:
        var_size: 变量维度 (导航变量数量) / Variable dimensions (number of navigation variables)
        var_min, var_max: 变量边界 / Variable bounds
        model: 环境模型 (需要start和end位置) / Environment model (needs start and end positions)
        direction_bias: 方向偏置系数 [0,1]，越大越接近直线 / Direction bias [0,1], larger means closer to straight line
        perturbation: 扰动系数 [0,1]，越大多样性越高 / Perturbation [0,1], larger means more diversity
    
    返回 / Returns:
        solution: 导航变量字典 {'r', 'phi', 'psi'} / Navigation variable dict
    
    工作原理 / Working Principle:
        1. 计算起点到终点的方向向量
        2. 在该方向上生成航点，添加适度扰动
        3. 将航点转换为导航变量 (r, phi, psi)
        4. 这样可以避免初始就撞到障碍物导致inf
    """
    # 获取起点和终点 / Get start and end points
    xs, ys, zs = model['start']
    xf, yf, zf = model['end']
    
    # 计算方向向量 / Calculate direction vector
    direction = np.array([xf - xs, yf - ys, zf - zs])
    total_distance = np.linalg.norm(direction)
    direction_normalized = direction / total_distance if total_distance > 0 else np.array([1, 0, 0])
    
    # 初始化数组 / Initialize arrays
    r = np.zeros(var_size)
    phi = np.zeros(var_size)
    psi = np.zeros(var_size)
    
    # 为每个导航变量生成智能初始值 / Generate smart initial values for each navigation variable
    for i in range(var_size):
        # 计算沿方向的进度 / Calculate progress along direction
        progress = (i + 1) / (var_size + 1)  # 从0到1 / From 0 to 1
        
        # ========================================
        # r: 半径 (步长) / r: radius (step size)
        # ========================================
        # 基于总距离和变量数量估算合理步长
        # Estimate reasonable step size based on total distance and number of variables
        expected_step = total_distance / (var_size + 1)
        
        # 添加随机扰动，但保持在合理范围内
        # Add random perturbation but keep in reasonable range
        r_base = expected_step * (1 + perturbation * (np.random.rand() - 0.5))
        r[i] = np.clip(r_base, var_min['r'], min(var_max['r'], expected_step * 2))
        
        # ========================================
        # phi: 俯仰角 / phi: pitch angle
        # ========================================
        # 计算理想俯仰角 (基于高度变化)
        # Calculate ideal pitch angle (based on altitude change)
        dz = zf - zs
        dxy = np.sqrt((xf - xs)**2 + (yf - ys)**2)
        
        if dxy > 0:
            ideal_phi = np.arctan2(dz, dxy)  # 理想俯仰角 / Ideal pitch angle
            # 限制到安全范围 / Limit to safe range
            ideal_phi = np.clip(ideal_phi, -np.pi/6, np.pi/6)  # ±30度 / ±30 degrees
        else:
            ideal_phi = 0
        
        # 添加扰动，但偏向理想值
        # Add perturbation but bias towards ideal value
        phi_random = np.random.uniform(var_min['phi'], var_max['phi'])
        phi[i] = direction_bias * ideal_phi + (1 - direction_bias) * phi_random
        phi[i] = np.clip(phi[i], var_min['phi'], var_max['phi'])
        
        # ========================================
        # psi: 偏航角 / psi: yaw angle
        # ========================================
        # 计算理想偏航角 (基于水平方向)
        # Calculate ideal yaw angle (based on horizontal direction)
        dx = xf - xs
        dy = yf - ys
        
        if dx != 0 or dy != 0:
            ideal_psi = np.arctan2(dy, dx)
            if ideal_psi < 0:
                ideal_psi += 2 * np.pi  # 转换到[0, 2π] / Convert to [0, 2π]
        else:
            ideal_psi = 0
        
        # 添加扰动，但偏向理想值
        # Add perturbation but bias towards ideal value
        psi_random = np.random.uniform(var_min['psi'], var_max['psi'])
        psi[i] = direction_bias * ideal_psi + (1 - direction_bias) * psi_random
        psi[i] = np.clip(psi[i], var_min['psi'], var_max['psi'])
    
    solution = {
        'r': r,
        'phi': phi,
        'psi': psi
    }
    
    return solution


def create_random_solution(var_size, var_min, var_max):
    """
    创建随机解 / Create Random Solution
    (保留原始随机初始化方法，用于对比)
    (Keep original random initialization for comparison)
    """
    solution = {}
    solution['r'] = np.random.uniform(var_min['r'], var_max['r'], var_size)
    solution['phi'] = np.random.uniform(var_min['phi'], var_max['phi'], var_size)
    solution['psi'] = np.random.uniform(var_min['psi'], var_max['psi'], var_size)
    return solution


def test_smart_initialization():
    """
    测试智能初始化 / Test Smart Initialization
    """
    from config_enhanced import EnhancedConfig
    from cost_function import calculate_cost
    
    print("=" * 60)
    print("测试智能初始化 vs 随机初始化")
    print("Test Smart Initialization vs Random Initialization")
    print("=" * 60)
    
    # 创建配置 / Create configuration
    config = EnhancedConfig()
    model = config.create_model()
    
    var_size = config.n_var
    var_min = config.var_min
    var_max = config.var_max
    
    # 测试多次，统计inf的比例 / Test multiple times, count inf ratio
    n_tests = 50
    
    print(f"\n生成 {n_tests} 个初始解...")
    print(f"Generating {n_tests} initial solutions...")
    
    # 测试随机初始化 / Test random initialization
    print("\n1️⃣ 随机初始化 / Random Initialization:")
    random_inf_count = 0
    random_costs = []
    
    for i in range(n_tests):
        sol = create_random_solution(var_size, var_min, var_max)
        cost = calculate_cost(sol, model, var_min)
        random_costs.append(cost)
        if np.any(np.isinf(cost)):
            random_inf_count += 1
    
    print(f"   inf比例 / inf ratio: {random_inf_count}/{n_tests} ({100*random_inf_count/n_tests:.1f}%)")
    
    # 计算非inf的平均代价 / Calculate average cost for non-inf solutions
    valid_costs = [c for c in random_costs if not np.any(np.isinf(c))]
    if valid_costs:
        avg_cost = np.mean(valid_costs, axis=0)
        print(f"   平均代价 / Average cost (J1-J4): [{avg_cost[0]:.3f}, {avg_cost[1]:.3f}, {avg_cost[2]:.3f}, {avg_cost[3]:.3f}]")
    
    # 测试智能初始化 / Test smart initialization
    print("\n2️⃣ 智能初始化 / Smart Initialization:")
    smart_inf_count = 0
    smart_costs = []
    
    for i in range(n_tests):
        sol = create_smart_solution(var_size, var_min, var_max, model)
        cost = calculate_cost(sol, model, var_min)
        smart_costs.append(cost)
        if np.any(np.isinf(cost)):
            smart_inf_count += 1
    
    print(f"   inf比例 / inf ratio: {smart_inf_count}/{n_tests} ({100*smart_inf_count/n_tests:.1f}%)")
    
    # 计算非inf的平均代价 / Calculate average cost for non-inf solutions
    valid_costs = [c for c in smart_costs if not np.any(np.isinf(c))]
    if valid_costs:
        avg_cost = np.mean(valid_costs, axis=0)
        print(f"   平均代价 / Average cost (J1-J4): [{avg_cost[0]:.3f}, {avg_cost[1]:.3f}, {avg_cost[2]:.3f}, {avg_cost[3]:.3f}]")
    
    # 改进效果 / Improvement effect
    print("\n" + "=" * 60)
    print("📊 改进效果 / Improvement Effect:")
    print("=" * 60)
    
    inf_reduction = random_inf_count - smart_inf_count
    if random_inf_count > 0:
        reduction_pct = 100 * inf_reduction / random_inf_count
        print(f"✅ inf数量减少: {inf_reduction} 个 ({reduction_pct:.1f}%)")
        print(f"   inf count reduced: {inf_reduction} ({reduction_pct:.1f}%)")
    else:
        print("ℹ️  随机初始化也没有产生inf")
        print("   Random initialization didn't produce inf either")
    
    # 可视化一个例子 / Visualize one example
    print("\n" + "=" * 60)
    print("🔍 可视化对比 / Visualization Comparison")
    print("=" * 60)
    
    # 生成一个随机解 / Generate a random solution
    random_sol = create_random_solution(var_size, var_min, var_max)
    random_cost = calculate_cost(random_sol, model, var_min)
    
    # 生成一个智能解 / Generate a smart solution
    smart_sol = create_smart_solution(var_size, var_min, var_max, model)
    smart_cost = calculate_cost(smart_sol, model, var_min)
    
    print(f"\n随机解代价 / Random solution cost: [{random_cost[0]:.3f}, {random_cost[1]:.3f}, {random_cost[2]:.3f}, {random_cost[3]:.3f}]")
    print(f"智能解代价 / Smart solution cost:  [{smart_cost[0]:.3f}, {smart_cost[1]:.3f}, {smart_cost[2]:.3f}, {smart_cost[3]:.3f}]")
    
    # 转换为路径 / Convert to path
    from utils import spherical_to_cartesian
    random_path = spherical_to_cartesian(random_sol, model)
    smart_path = spherical_to_cartesian(smart_sol, model)
    
    print(f"\n随机解路径长度 / Random path length: {np.linalg.norm(np.diff(random_path, axis=0), axis=1).sum():.1f}")
    print(f"智能解路径长度 / Smart path length:  {np.linalg.norm(np.diff(smart_path, axis=0), axis=1).sum():.1f}")
    
    # 计算直线距离 / Calculate straight line distance
    start = np.array(model['start'])
    end = np.array(model['end'])
    straight_dist = np.linalg.norm(end - start)
    print(f"直线距离 / Straight distance:     {straight_dist:.1f}")
    
    print("\n[DONE] 测试完成")
    print("       Test completed")


if __name__ == "__main__":
    test_smart_initialization()
