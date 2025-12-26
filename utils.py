"""
辅助函数模块 / Utility Functions Module
包含各种辅助计算函数
Contains various utility calculation functions
"""
import numpy as np

def transformation_matrix(r, phi, psi):
    """
    变换矩阵 / Transformation Matrix
    将球坐标变换到笛卡尔坐标
    Transform from spherical to Cartesian coordinates
    
    参数 / Parameters:
        r: 距离 / Distance
        phi: 方位角 (绕z轴) / Azimuth angle (around z-axis)
        psi: 俯仰角 (绕y轴) / Elevation angle (around y-axis)
    
    返回 / Returns:
        4x4变换矩阵 / 4x4 transformation matrix
    """
    # 绕z轴旋转 / Rotation around z-axis
    Rot_z = np.array([
        [np.cos(phi), -np.sin(phi), 0, 0],
        [np.sin(phi),  np.cos(phi), 0, 0],
        [0,            0,           1, 0],
        [0,            0,           0, 1]
    ])
    
    # 绕y轴旋转 / Rotation around y-axis
    Rot_y = np.array([
        [np.cos(-psi),  0, np.sin(-psi), 0],
        [0,             1, 0,            0],
        [-np.sin(-psi), 0, np.cos(-psi), 0],
        [0,             0, 0,            1]
    ])
    
    # 平移 / Translation
    Trans_x = np.array([
        [1, 0, 0, r],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    return Rot_z @ Rot_y @ Trans_x


def spherical_to_cartesian(solution, model):
    """
    球坐标到笛卡尔坐标转换 / Spherical to Cartesian Conversion
    将导航变量(r, phi, psi)转换为笛卡尔坐标(x, y, z)
    Convert navigation variables (r, phi, psi) to Cartesian coordinates (x, y, z)
    
    这是NMOPSO算法的核心创新：使用导航变量而非直接的坐标
    This is the core innovation of NMOPSO: using navigation variables instead of direct coordinates
    """
    r = solution['r']
    phi = solution['phi']
    psi = solution['psi']
    
    # 起点矩阵 / Start position matrix
    xs, ys, zs = model['start']
    start = np.array([
        [1, 0, 0, xs],
        [0, 1, 0, ys],
        [0, 0, 1, zs],
        [0, 0, 0, 1]
    ])
    
    # 计算起点到终点的方向 / Calculate direction from start to end
    dir_vector = model['end'] - model['start']
    phi_start = np.arctan2(dir_vector[1], dir_vector[0])
    psi_start = np.arctan2(dir_vector[2], np.linalg.norm(dir_vector[:2]))
    
    dir_matrix = transformation_matrix(0, phi_start, psi_start)
    start_position = start @ dir_matrix
    
    # 计算各路径点位置 / Calculate position of each path node
    n = model['n']
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    
    # 第一个节点 / First node
    T = transformation_matrix(r[0], phi[0], psi[0])
    pos = start_position @ T
    
    x[0] = np.clip(pos[0, 3], model['xmin'], model['xmax'])
    y[0] = np.clip(pos[1, 3], model['ymin'], model['ymax'])
    z[0] = np.clip(pos[2, 3], model['zmin'], model['zmax'])
    
    # 后续节点 / Subsequent nodes
    for i in range(1, n):
        T = T @ transformation_matrix(r[i], phi[i], psi[i])
        pos = start_position @ T
        
        x[i] = np.clip(pos[0, 3], model['xmin'], model['xmax'])
        y[i] = np.clip(pos[1, 3], model['ymin'], model['ymax'])
        z[i] = np.clip(pos[2, 3], model['zmin'], model['zmax'])
    
    return {'x': x, 'y': y, 'z': z}


def dist_point_to_segment(x, a, b):
    """
    计算点到线段的距离 / Calculate Distance from Point to Segment
    用于检测路径是否与障碍物碰撞
    Used to detect if path collides with obstacles
    
    参数 / Parameters:
        x: 点坐标 / Point coordinates
        a, b: 线段两端点 / Segment endpoints
    """
    a = np.array(a)
    b = np.array(b)
    x = np.array(x)
    
    d_ab = np.linalg.norm(a - b)
    d_ax = np.linalg.norm(a - x)
    d_bx = np.linalg.norm(b - x)
    
    if d_ab == 0:
        return d_ax
    
    # 检查x是否在线段ab之间 / Check if x is between segment ab
    if np.dot(a - b, x - b) * np.dot(b - a, x - a) >= 0:
        # 点到直线距离公式 / Point to line distance formula
        cross_prod = np.cross(b - a, x - a)
        if np.isscalar(cross_prod):
            dist = abs(cross_prod) / d_ab
        else:
            dist = np.linalg.norm(cross_prod) / d_ab
        return dist
    else:
        return min(d_ax, d_bx)


def dominates(x, y):
    """
    Pareto支配关系判断 / Pareto Dominance Check
    在多目标优化中，判断解x是否支配解y
    In multi-objective optimization, check if solution x dominates solution y
    
    支配定义：x在所有目标上不差于y，且至少在一个目标上优于y
    Dominance: x is no worse than y in all objectives and better in at least one
    """
    if isinstance(x, dict) and 'cost' in x:
        x = x['cost']
    if isinstance(y, dict) and 'cost' in y:
        y = y['cost']
    
    x = np.array(x)
    y = np.array(y)
    
    # x支配y的条件 / Conditions for x to dominate y
    return np.all(x <= y) and np.any(x < y) and np.all(x < np.inf)


def roulette_wheel_selection(P):
    """
    轮盘赌选择 / Roulette Wheel Selection
    基于概率分布选择索引
    Select index based on probability distribution
    
    这是经典的选择算子，概率越大被选中的可能性越高
    Classic selection operator: higher probability means higher chance of selection
    """
    r = np.random.rand()
    C = np.cumsum(P)  # 累积和 / Cumulative sum
    idx = np.where(r <= C)[0]
    if len(idx) > 0:
        return idx[0]
    return len(P) - 1
