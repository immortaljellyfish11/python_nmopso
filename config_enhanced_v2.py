"""
增强配置V2 - 随机障碍物生成 / Enhanced Configuration V2 - Random Obstacle Generation
在EnhancedConfig基础上添加随机障碍物生成功能
Adds random obstacle generation capability on top of EnhancedConfig

新增功能 / New Features:
- 随机生成障碍物位置和大小
- 可设置种子保证可复现性
- 自动避免障碍物遮挡起点/终点
"""
import numpy as np
from config_enhanced import EnhancedConfig


def generate_random_obstacles(n_obstacles=10,        # 从15降低到8，更稀疏
                               map_size_x=2050, map_size_y=1950,
                               radius_min=80,        # 从15增加到30
                               radius_max=200,        # 从40增加到60
                               height_min=80,        # 从30增加到80
                               height_max=450,       # 从120增加到250
                               start_pos=(50, 50), end_pos=(1900, 1900),
                               safe_distance=150,    # 从80增加到150，更安全
                               seed=None):
    """
    生成随机障碍物 / Generate Random Obstacles
    
    参数 / Parameters:
        n_obstacles: 障碍物数量 / Number of obstacles
        map_size_x, map_size_y: 地图尺寸 / Map size
        radius_min, radius_max: 半径范围 / Radius range
        height_min, height_max: 高度范围 / Height range
        start_pos, end_pos: 起点和终点位置 / Start and end positions
        safe_distance: 起点/终点周围的安全距离 / Safe distance around start/end
        seed: 随机种子 (用于可复现性) / Random seed (for reproducibility)
    
    返回 / Returns:
        obstacles: 障碍物列表，每个为 [x, y, z_height, radius, z_base]
    """
    if seed is not None:
        np.random.seed(seed)
    
    obstacles = []
    xs, ys = start_pos
    xf, yf = end_pos
    
    attempts = 0
    max_attempts = n_obstacles * 50  # 避免无限循环 / Avoid infinite loop
    
    while len(obstacles) < n_obstacles and attempts < max_attempts:
        attempts += 1
        
        # 生成随机位置 / Generate random position
        # 避开地图边缘 / Avoid map edges
        margin = radius_max + 20
        x = np.random.uniform(margin, map_size_x - margin)
        y = np.random.uniform(margin, map_size_y - margin)
        
        # 生成随机半径和高度 / Generate random radius and height
        radius = np.random.uniform(radius_min, radius_max)
        
        # 随机选择高度类型 / Randomly select height type
        height_type = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.4, 0.3])
        
        if height_type == 'low':
            height = np.random.uniform(height_min, (height_max - height_min) / 3 + height_min)
            z_base = 0
        elif height_type == 'medium':
            height = np.random.uniform((height_max - height_min) / 3 + height_min, 
                                     2 * (height_max - height_min) / 3 + height_min)
            z_base = np.random.uniform(0, 30)
        else:  # high
            height = np.random.uniform(2 * (height_max - height_min) / 3 + height_min, height_max)
            z_base = np.random.uniform(0, 50)
        
        # 检查是否太靠近起点或终点 / Check if too close to start or end
        dist_to_start = np.sqrt((x - xs)**2 + (y - ys)**2)
        dist_to_end = np.sqrt((x - xf)**2 + (y - yf)**2)
        
        if dist_to_start < safe_distance or dist_to_end < safe_distance:
            continue
        
        # 检查是否与已有障碍物重叠 / Check if overlaps with existing obstacles
        overlap = False
        for obs in obstacles:
            obs_x, obs_y, _, obs_radius, _ = obs
            dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if dist < (radius + obs_radius + 10):  # 障碍物之间至少间隔10 / At least 10 units between obstacles
                overlap = True
                break
        
        if not overlap:
            obstacles.append([x, y, height, radius, z_base])
    
    if len(obstacles) < n_obstacles:
        print(f"[WARNING] 只生成了 {len(obstacles)} 个障碍物，目标是 {n_obstacles} 个")
        print(f"          Only generated {len(obstacles)} obstacles, target was {n_obstacles}")
    
    return obstacles


class RandomConfig(EnhancedConfig):
    """
    随机配置类 / Random Configuration Class
    生成随机障碍物的配置
    Configuration with random obstacles
    """
    
    def __init__(self, n_obstacles=15, seed=None):
        """
        初始化随机配置 / Initialize Random Configuration
        
        参数 / Parameters:
            n_obstacles: 障碍物数量 / Number of obstacles
            seed: 随机种子 (None表示真随机) / Random seed (None for true random)
        """
        # 不调用父类init，因为我们要重新生成obstacles
        # Don't call parent init since we're regenerating obstacles
        self.n_obstacles = n_obstacles
        self.seed = seed
        
        # PSO参数（从EnhancedConfig复制）/ PSO Parameters (copied from EnhancedConfig)
        self.max_iter = 500
        self.n_pop = 100
        self.n_rep = 50
        self.w = 1.0
        self.w_damp = 0.98
        self.c1 = 1.5
        self.c2 = 1.5
        self.n_grid = 5
        self.alpha = 0.1
        self.beta = 2
        self.gamma = 2
        self.mu = 0.5
        self.delta = 20
        self.angle_range = np.pi / 4
        
        # 基本设置 / Basic settings
        self.n_var = 20
        self.var_min = {'r': 1, 'phi': 0, 'psi': 0}
        self.var_max = {'r': 100, 'phi': np.pi, 'psi': 2 * np.pi}
        
        # 地图参数 / Map parameters
        self.map_size_x = 2050
        self.map_size_y = 1950
        self.start_position = (50, 50, 20)
        self.end_position = (1900, 1900, 10)
        
        self.zmax = 150
        self.zmin = 0
        
        # 生成随机障碍物 / Generate random obstacles
        self.threats = generate_random_obstacles(
            n_obstacles=n_obstacles,
            map_size_x=self.map_size_x,
            map_size_y=self.map_size_y,
            radius_min=15,
            radius_max=40,
            height_min=30,
            height_max=120,
            start_pos=(self.start_position[0], self.start_position[1]),
            end_pos=(self.end_position[0], self.end_position[1]),
            safe_distance=80,
            seed=seed
        )
        
        print(f"\n[SUCCESS] 成功生成 {len(self.threats)} 个随机障碍物")
        print(f"          Successfully generated {len(self.threats)} random obstacles")
        if seed is not None:
            print(f"   使用种子 / Using seed: {seed}")
        
        # 打印障碍物统计 / Print obstacle statistics
        heights = [obs[2] for obs in self.threats]
        radii = [obs[3] for obs in self.threats]
        z_bases = [obs[4] for obs in self.threats]
        
        print(f"   高度范围 / Height range: {min(heights):.1f} - {max(heights):.1f}")
        print(f"   半径范围 / Radius range: {min(radii):.1f} - {max(radii):.1f}")
        print(f"   底部高度范围 / Base height range: {min(z_bases):.1f} - {max(z_bases):.1f}")
        
        # 创建完整的model字典 / Create complete model dictionary
        self.model = self._build_model()
    
    def _build_model(self):
        """构建完整的model字典 / Build complete model dictionary"""
        model = {}
        
        # 地图大小 / Map size
        model['map_size_x'] = self.map_size_x
        model['map_size_y'] = self.map_size_y
        
        # 创建简单地形 / Create simple terrain
        x = np.linspace(0, self.map_size_x, self.map_size_x)
        y = np.linspace(0, self.map_size_y, self.map_size_y)
        X, Y = np.meshgrid(x, y)
        
        # 地形高度场 / Terrain height field
        H = 20 * np.sin(X / 150) * np.cos(Y / 200) + \
            15 * np.sin(X / 100 + Y / 100)
        H = np.maximum(H, 0)
        model['H'] = H
        
        # 障碍物（已经生成）/ Obstacles (already generated)
        model['threats'] = self.threats
        
        # 起点和终点 / Start and end points
        model['start'] = np.array(self.start_position)
        model['end'] = np.array(self.end_position)
        
        # 其他参数 / Other parameters
        model['n'] = self.n_var
        model['xmin'] = 0
        model['xmax'] = self.map_size_x
        model['ymin'] = 0
        model['ymax'] = self.map_size_y
        model['zmin'] = self.zmin
        model['zmax'] = self.zmax
        
        return model


def test_random_generation():
    """
    测试随机障碍物生成 / Test Random Obstacle Generation
    """
    print("=" * 60)
    print("测试1: 固定种子生成 / Test 1: Fixed Seed Generation")
    print("=" * 60)
    
    config1 = RandomConfig(n_obstacles=10, seed=42)
    print(f"\n前3个障碍物 / First 3 obstacles:")
    for i, obs in enumerate(config1.threats[:3]):
        x, y, h, r, zb = obs
        print(f"  障碍物{i+1} / Obstacle {i+1}: 位置=({x:.1f}, {y:.1f}), 高度={h:.1f}, 半径={r:.1f}, 底部={zb:.1f}")
    
    print("\n" + "=" * 60)
    print("测试2: 相同种子应该生成相同结果 / Test 2: Same Seed Should Give Same Result")
    print("=" * 60)
    
    config2 = RandomConfig(n_obstacles=10, seed=42)
    
    # 验证是否相同 / Verify if same
    same = True
    for obs1, obs2 in zip(config1.threats, config2.threats):
        if not np.allclose(obs1, obs2):
            same = False
            break
    
    if same:
        print("[PASS] 相同种子生成相同结果")
        print("       Same seed generates identical results")
    else:
        print("[FAIL] 结果不同")
        print("       Results differ")
    
    print("\n" + "=" * 60)
    print("测试3: 真随机生成 / Test 3: True Random Generation")
    print("=" * 60)
    
    config3 = RandomConfig(n_obstacles=15, seed=None)
    
    print("\n" + "=" * 60)
    print("测试4: 验证安全距离约束 / Test 4: Verify Safe Distance Constraint")
    print("=" * 60)
    
    start = np.array([50, 50])
    end = np.array([800, 800])
    safe_distance = 80
    
    all_safe = True
    for i, obs in enumerate(config3.threats):
        x, y = obs[0], obs[1]
        pos = np.array([x, y])
        
        dist_start = np.linalg.norm(pos - start)
        dist_end = np.linalg.norm(pos - end)
        
        if dist_start < safe_distance or dist_end < safe_distance:
            print(f"[FAIL] 障碍物{i+1}违反安全距离: 到起点={dist_start:.1f}, 到终点={dist_end:.1f}")
            all_safe = False
    
    if all_safe:
        print("[PASS] 所有障碍物都满足安全距离约束")
        print("       All obstacles satisfy safe distance constraint")
    
    print("\n" + "=" * 60)
    print("所有测试完成 / All Tests Completed")
    print("=" * 60)


if __name__ == "__main__":
    test_random_generation()
