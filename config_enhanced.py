"""
增强配置文件 / Enhanced Configuration File
包含更复杂的3D地图和障碍物
Includes more complex 3D map and obstacles
"""
import numpy as np

class EnhancedConfig:
    """
    增强算法配置类 / Enhanced Algorithm Configuration Class
    改进：
    1. 更多障碍物，高度信息更明显
    2. 3D地形支持
    3. 可调节的初始角度范围
    """
    def __init__(self, angle_range=np.pi/4, r_min_ratio=0.15, r_max_ratio=0.35):
        """
        初始化配置 / Initialize Configuration
        
        参数 / Parameters:
            angle_range: 角度范围 / Angle range
            r_min_ratio: r最小值比例，防止点聚集 / r_min ratio to prevent clustering
            r_max_ratio: r最大值比例，限制最大步长 / r_max ratio to limit max step
        """
        # PSO参数 / PSO Parameters
        self.max_iter = 600
        self.n_pop = 100
        self.n_rep = 100
        
        # PSO系数 / PSO Coefficients
        self.w = 1.0
        self.w_damp = 0.98
        self.c1 = 1.5
        self.c2 = 1.5
        
        # 网格参数 / Grid Parameters
        self.n_grid = 5
        self.alpha = 0.1
        
        # 选择压力 / Selection Pressure
        self.beta = 2
        self.gamma = 2
        
        # 领导者选择权重（解决路径过长问题）
        # Leader selection weight (solve long path problem)
        self.length_weight = 0.6  # 0.7表示70%重视路径长度，30%重视平滑度
        
        # 变异参数 / Mutation Parameters
        self.mu = 0.5
        self.delta = 20
        
        # 角度范围（可调节）/ Angle range (adjustable)
        self.angle_range = angle_range
        
        # 导航变量数量 / Number of navigation variables
        self.n_var = 15
        
        # r范围比例参数 / r range ratio parameters
        self.r_min_ratio = r_min_ratio
        self.r_max_ratio = r_max_ratio
        
        # 变量边界（r范围将在创建模型后自适应计算）
        # Variable bounds (r range will be adaptively calculated after model creation)
        self.var_min = {'r': 10, 'phi': -angle_range, 'psi': -angle_range}
        self.var_max = {'r': 100, 'phi': angle_range, 'psi': angle_range}
        
        # 地图模型 / Map Model
        self.model = self.create_enhanced_model()
        
        # 自适应计算r范围（解决点聚集问题）
        # Adaptively calculate r range (solve clustering problem)
        self._recalculate_r_bounds()
    
    def create_enhanced_model(self):
        """
        创建增强的3D地图模型 / Create Enhanced 3D Map Model
        
        改进说明 / Improvements:
        1. 增加了障碍物数量（从6个到12个）
        2. 障碍物具有不同的高度
        3. 添加了简单的起伏地形
        4. 障碍物分布更复杂
        """
        model = {}
        
        # 地图大小 / Map size
        model['map_size_x'] = 850
        model['map_size_y'] = 850
        
        # 创建起伏地形 / Create undulating terrain
        # 使用正弦函数模拟山丘
        x = np.linspace(0, model['map_size_x'], model['map_size_x'])
        y = np.linspace(0, model['map_size_y'], model['map_size_y'])
        X, Y = np.meshgrid(x, y)
        
        # 地形高度场：波浪状地形
        # Terrain height field: wavy terrain
        H = 20 * np.sin(X / 150) * np.cos(Y / 200) + \
            15 * np.sin(X / 100 + Y / 100)
        H = np.maximum(H, 0)  # 确保非负 / Ensure non-negative
        
        model['H'] = H
        
        # 增强的障碍物配置 / Enhanced obstacle configuration
        # 每行: [x, y, z_height, radius, z_base]
        # z_height: 障碍物从地面到顶部的高度
        # z_base: 障碍物底部的高度
        
        model['threats'] = np.array([
            # 降低障碍物数量从12个到6个，增加可行性
            # Reduce obstacles from 12 to 6 for better feasibility
            
            # 低空障碍物 / Low altitude obstacles
            [150, 150, 150, 50, 0],    # 矮建筑
            [300, 200, 180, 60, 0],    # 中等建筑
            
            # 中空障碍物 / Medium altitude obstacles
            [450, 300, 250, 40, 50],   # 高塔
            [700, 500, 220, 45, 40],   # 中等塔
            
            # 高空障碍物 / High altitude obstacles
            [500, 650, 300, 70, 100],  # 高空禁飞区
            [250, 650, 190, 50, 0]     # 混合障碍物
        ])
        
        # 起点和终点 / Start and end points
        # 保持原始高度，算法需要学会绕开障碍物
        # 终点从(800,800)改为(750,850)，避开障碍物9(700,700)
        model['start'] = np.array([50, 50, 140])  # 原始高度
        model['end'] = np.array([750, 850, 180])  # 调整位置避开障碍物密集区
        
        # 路径节点数 / Number of path nodes
        model['n'] = 15  # 增加节点数以应对复杂环境
        
        # 搜索空间边界 / Search space bounds
        model['xmin'] = 1
        model['xmax'] = model['map_size_x']
        model['ymin'] = 1
        model['ymax'] = model['map_size_y']
        model['zmin'] = 80   # 最低飞行高度
        model['zmax'] = 250  # 最高飞行高度
        
        return model

    def _recalculate_r_bounds(self):
        """
        自适应计算r的边界，解决点聚集问题
        Adaptively calculate r bounds to solve clustering problem
        
        核心思想 / Core Idea:
        - 基于起点-终点直线距离计算合理的r范围
        - r_min设置较大值，强制点之间保持最小间距
        - r_max限制最大步长，避免过度跳跃
        """
        start = self.model['start']
        end = self.model['end']
        
        # 计算直线距离 / Calculate straight-line distance
        straight_dist = np.linalg.norm(end - start)
        
        # 每个导航变量应该覆盖的平均距离
        # Average distance each navigation variable should cover
        avg_segment_dist = straight_dist / (self.n_var + 1)
        
        # 设置r的范围（关键改进）
        # Set r range (key improvement)
        self.var_min['r'] = avg_segment_dist * self.r_min_ratio
        self.var_max['r'] = avg_segment_dist * self.r_max_ratio
        
        print(f"\n[自适应r范围] 基于直线距离: {straight_dist:.2f}")
        print(f"  平均段距离: {avg_segment_dist:.2f}")
        print(f"  r_min = {self.var_min['r']:.2f} (防止点聚集)")
        print(f"  r_max = {self.var_max['r']:.2f} (限制最大步长)")
        print(f"  比例范围: [{self.r_min_ratio:.2f}, {self.r_max_ratio:.2f}]")

        # 在 config_enhanced.py 中添加
    
    def generate_random_obstacles(num_obstacles=12, map_size_x=850, map_size_y=850,
                                 seed=None):
        """
        生成随机障碍物 / Generate Random Obstacles
        
        参数 / Parameters:
            num_obstacles: 障碍物数量 / Number of obstacles
            map_size_x, map_size_y: 地图尺寸 / Map size
            seed: 随机种子 / Random seed
        
        返回 / Returns:
            障碍物数组 / Obstacle array
        """
        if seed is not None:
            np.random.seed(seed)
        
        threats = []
        
        # 障碍物类型分布 / Obstacle type distribution
        num_low = num_obstacles // 3          # 低空障碍
        num_mid = num_obstacles // 3          # 中空障碍
        num_high = num_obstacles - num_low - num_mid  # 高空障碍
        
        # 生成低空障碍物 / Generate low altitude obstacles
        for _ in range(num_low):
            x = np.random.randint(100, map_size_x - 100)
            y = np.random.randint(100, map_size_y - 100)
            z_height = np.random.randint(100, 200)
            radius = np.random.randint(40, 70)
            z_base = 0
            threats.append([x, y, z_height, radius, z_base])
        
        # 生成中空障碍物 / Generate medium altitude obstacles
        for _ in range(num_mid):
            x = np.random.randint(100, map_size_x - 100)
            y = np.random.randint(100, map_size_y - 100)
            z_height = np.random.randint(150, 280)
            radius = np.random.randint(35, 60)
            z_base = np.random.randint(30, 80)
            threats.append([x, y, z_height, radius, z_base])
        
        # 生成高空障碍物 / Generate high altitude obstacles
        for _ in range(num_high):
            x = np.random.randint(100, map_size_x - 100)
            y = np.random.randint(100, map_size_y - 100)
            z_height = np.random.randint(200, 350)
            radius = np.random.randint(50, 80)
            z_base = np.random.randint(60, 150)
            threats.append([x, y, z_height, radius, z_base])
        
        return np.array(threats)
    
    

class SimpleConfig:
    """
    简化配置（用于测试和学习）/ Simplified Config (for testing and learning)
    """
    def __init__(self):
        self.max_iter = 100
        self.n_pop = 50
        self.n_rep = 30
        
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
        
        self.model = self.create_simple_model()
    
    def create_simple_model(self):
        """
        创建简单模型（更容易收敛）/ Create simple model (easier to converge)
        """
        model = {}
        
        model['map_size_x'] = 500
        model['map_size_y'] = 500
        model['H'] = np.zeros((500, 500))  # 平坦地形
        
        # 只有3个简单障碍物
        model['threats'] = np.array([
            [150, 150, 200, 60, 0],
            [300, 300, 200, 60, 0],
            [250, 380, 200, 50, 0]
        ])
        
        model['start'] = np.array([50, 50, 150])
        model['end'] = np.array([450, 450, 150])
        model['n'] = 8
        
        model['xmin'] = 1
        model['xmax'] = 500
        model['ymin'] = 1
        model['ymax'] = 500
        model['zmin'] = 100
        model['zmax'] = 200
        
        return model
