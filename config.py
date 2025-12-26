"""
配置文件 / Configuration File
定义NMOPSO算法的参数和地图模型
Defines parameters and map model for NMOPSO algorithm
"""
import numpy as np

class Config:
    """
    算法配置类 / Algorithm Configuration Class
    """
    def __init__(self):
        # PSO参数 / PSO Parameters
        self.max_iter = 500          # 最大迭代次数 / Maximum iterations
        self.n_pop = 100             # 种群大小 / Population size
        self.n_rep = 50              # 存档大小 / Repository size
        
        # PSO系数 / PSO Coefficients
        self.w = 1.0                 # 惯性权重 / Inertia weight
        self.w_damp = 0.98           # 惯性权重衰减 / Inertia weight damping
        self.c1 = 1.5                # 个体学习系数 / Personal learning coefficient
        self.c2 = 1.5                # 全局学习系数 / Global learning coefficient
        
        # 网格参数 / Grid Parameters
        self.n_grid = 5              # 每维网格数 / Number of grids per dimension
        self.alpha = 0.1             # 膨胀率 / Inflation rate
        
        # 选择压力 / Selection Pressure
        self.beta = 2                # 领导者选择压力 / Leader selection pressure
        self.gamma = 2               # 删除选择压力 / Deletion selection pressure
        
        # 变异参数 / Mutation Parameters
        self.mu = 0.5                # 变异率 / Mutation rate
        self.delta = 20              # 变异强度参数 / Mutation intensity parameter
        
        # 地图模型 / Map Model
        self.model = self.create_model()
    
    def create_model(self):
        """
        创建地图模型 / Create map model
        包括地形、障碍物、起点和终点
        Includes terrain, obstacles, start and end points
        """
        model = {}
        
        # 地图大小 / Map size (可以不使用地形图，简化为平面)
        model['map_size_x'] = 850
        model['map_size_y'] = 850
        
        # 创建简单地形 (高度场) / Create simple terrain (height field)
        # 这里用简单的平面，实际可以加载真实地形
        model['H'] = np.ones((model['map_size_y'], model['map_size_x'])) * 0  # 平坦地形
        
        # 障碍物 (圆柱形威胁区) / Obstacles (cylindrical threats)
        # 每行: [x, y, z, radius]
        model['threats'] = np.array([
            [200, 230, 250, 70],
            [600, 250, 250, 70],
            [450, 550, 250, 70],
            [700, 600, 250, 50],
            [200, 500, 250, 60],
            [500, 800, 250, 60]
        ])
        
        # 起点和终点 / Start and end points
        model['start'] = np.array([50, 50, 150])
        model['end'] = np.array([800, 800, 180])
        
        # 路径节点数 (不包括起点) / Number of path nodes (excluding start)
        model['n'] = 10
        
        # 搜索空间边界 / Search space bounds
        model['xmin'] = 1
        model['xmax'] = model['map_size_x']
        model['ymin'] = 1
        model['ymax'] = model['map_size_y']
        model['zmin'] = 100  # 相对于地面的高度 / Height relative to ground
        model['zmax'] = 200
        
        return model
