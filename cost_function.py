"""
代价函数模块 / Cost Function Module (已修复3D障碍物判断)
"""
import numpy as np
from utils import dist_point_to_segment


def calculate_cost(solution, model, var_min):
    """
    计算路径代价 / Calculate Path Cost
    
    修复说明 / Fix Notes:
    - 修复了3D障碍物判断逻辑，允许无人机从障碍物下方/上方通过
    - Fixed 3D obstacle detection logic, allowing UAV to fly under/over obstacles
    
    这是一个多目标优化问题，包含4个目标：
    This is a multi-objective optimization problem with 4 objectives:
    
    1. J1: 路径长度代价 / Path length cost
       - 最小化路径长度，提高飞行效率
       - Minimize path length for flight efficiency
    
    2. J2: 威胁/障碍物代价 / Threat/obstacle cost
       - 避免碰撞，保持安全距离
       - Avoid collision and maintain safe distance
    
    3. J3: 高度代价 / Altitude cost
       - 保持在合理高度范围，节省能源
       - Maintain reasonable altitude to save energy
    
    4. J4: 平滑度代价 / Smoothness cost
       - 减少急转弯，考虑运动学约束
       - Reduce sharp turns considering kinematic constraints
    
    参数 / Parameters:
        solution: 解向量 (x, y, z坐标) / Solution vector (x, y, z coordinates)
        model: 地图模型 / Map model
        var_min: 变量下界 / Variable lower bounds
    
    返回 / Returns:
        包含4个目标值的数组 / Array with 4 objective values
    """
    J_inf = np.inf
    n = model['n']
    H = model['H']
    
    # 输入解 / Input solution
    x = solution['x']
    y = solution['y']
    z = solution['z']
    
    # 起点和终点 / Start and end points
    xs, ys, zs = model['start']
    xf, yf, zf = model['end']
    
    # 完整路径 / Full path
    x_all = np.concatenate([[xs], x, [xf]])
    y_all = np.concatenate([[ys], y, [yf]])
    z_all = np.concatenate([[zs], z, [zf]])
    
    N = len(x_all)
    
    # 计算绝对高度 (相对海平面) / Calculate absolute altitude (relative to sea level)
    z_abs = np.zeros(N)
    for i in range(N):
        # 获取地形高度 / Get terrain height
        y_idx = int(np.clip(round(y_all[i]), 0, model['map_size_y'] - 1))
        x_idx = int(np.clip(round(x_all[i]), 0, model['map_size_x'] - 1))
        z_abs[i] = z_all[i] + H[y_idx, x_idx]
    
    # ============================================
    # J1 - 路径长度代价 / Path Length Cost
    # ============================================
    traj_length = 0
    for i in range(N - 1):
        diff = np.array([x_all[i+1] - x_all[i], 
                        y_all[i+1] - y_all[i], 
                        z_abs[i+1] - z_abs[i]])
        segment_length = np.linalg.norm(diff)
        
        # 检查是否有重合点 / Check for coincident points
        if segment_length <= var_min['r']:
            traj_length = 0
            break
        traj_length += segment_length
    
    # 直线距离 / Straight line distance
    direct_dist = np.linalg.norm(np.array([xf, yf, zf]) - np.array([xs, ys, zs]))
    
    if traj_length == 0:
        J1 = J_inf
    else:
        J1 = abs(1 - direct_dist / traj_length)
    
    # ============================================
    # J2 - 威胁/障碍物代价 (已修复3D判断) / Threat/Obstacle Cost (Fixed 3D Detection)
    # ============================================
    threats = model['threats']
    threat_num = len(threats)
    
    drone_size = 1
    danger_dist = 8 * drone_size  # 危险距离 / Danger distance
    vertical_margin = 2  # 垂直安全余量 / Vertical safety margin
    
    J2 = 0
    n2 = 0
    
    for i in range(threat_num):
        # 兼容旧格式和新格式 / Compatible with old and new formats
        if len(threats[i]) == 4:
            threat_x, threat_y, threat_z, threat_radius = threats[i]
            z_base = 0
        else:
            threat_x, threat_y, threat_z_height, threat_radius, z_base = threats[i]
            threat_z = threat_z_height
        
        # 获取障碍物位置的地形高度
        threat_y_idx = int(np.clip(round(threat_y), 0, model['map_size_y'] - 1))
        threat_x_idx = int(np.clip(round(threat_x), 0, model['map_size_x'] - 1))
        terrain_height_at_threat = H[threat_y_idx, threat_x_idx]
    
        # 障碍物的绝对高度范围（相对海平面）
        threat_z_min = z_base + terrain_height_at_threat
        threat_z_max = z_base + threat_z + terrain_height_at_threat
        
        for j in range(N - 1):
            # 🔑 关键修复：精确的3D碰撞检测
            # Key fix: Accurate 3D collision detection
            segment_z_min = min(z_abs[j], z_abs[j+1])
            segment_z_max = max(z_abs[j], z_abs[j+1])
            
            # 检查高度是否重叠（加上安全余量）
            # Check if altitudes overlap (with safety margin)
            has_vertical_overlap = not (
                segment_z_max + vertical_margin < threat_z_min or 
                segment_z_min - vertical_margin > threat_z_max
            )
            
            if not has_vertical_overlap:
                # 无人机在障碍物上方或下方安全通过
                # UAV safely passes above or below the obstacle
                threat_cost = 0
            else:
                # 有高度重叠，检查水平距离
                # Has vertical overlap, check horizontal distance
                dist = dist_point_to_segment(
                    [threat_x, threat_y],
                    [x_all[j], y_all[j]],
                    [x_all[j+1], y_all[j+1]]
                )
                
                # 判断威胁等级 / Determine threat level
                if dist > (threat_radius + drone_size + danger_dist):
                    # 安全区域 / Safe zone
                    threat_cost = 0
                elif dist < (threat_radius + drone_size):
                    # 碰撞 / Collision
                    threat_cost = J_inf
                else:
                    # 危险区域 / Danger zone
                    threat_cost = 1 - (dist - drone_size - threat_radius) / danger_dist
            
            n2 += 1
            J2 += threat_cost
    
    if n2 > 0:
        J2 = J2 / n2
    
    # ============================================
    # J3 - 高度代价（优化：惩罚飞太低） / Altitude Cost (Optimized: Penalize Flying Too Low)
    # ============================================
    zmax = model['zmax']
    zmin = model['zmin']
    z_center = (zmax + zmin) / 2  # 理想高度 / Ideal altitude
    
    J3 = 0
    for i in range(n):
        if z[i] < 5:  # 飞太低，危险 / Flying too low, dangerous
            J3_node = J_inf
        elif z[i] < 20:  # 飞得很低，给予惩罚 / Flying low, penalize
            J3_node = 2.0 * (20 - z[i]) / 20
        else:
            # 偏离理想高度的程度 / Deviation from ideal altitude
            J3_node = abs(z[i] - z_center) / ((zmax - zmin) / 2)
        J3 += J3_node
    
    if n > 0:
        J3 = J3 / n
    
    # ============================================
    # J4 - 平滑度代价 / Smoothness Cost
    # ============================================
    # 通过航向角变化来衡量路径平滑度
    # Measure path smoothness by heading angle changes
    J4 = 0
    n4 = 0
    
    for i in range(N - 2):
        # 找到非零段 / Find non-zero segments
        segment1 = None
        for j in range(i, -1, -1):
            seg = np.array([x_all[j+1] - x_all[j],
                           y_all[j+1] - y_all[j],
                           z_abs[j+1] - z_abs[j]])
            if np.linalg.norm(seg) > 1e-6:
                segment1 = seg
                break
        
        segment2 = None
        for j in range(i, N - 2):
            seg = np.array([x_all[j+2] - x_all[j+1],
                           y_all[j+2] - y_all[j+1],
                           z_abs[j+2] - z_abs[j+1]])
            if np.linalg.norm(seg) > 1e-6:
                segment2 = seg
                break
        
        if segment1 is not None and segment2 is not None:
            # 计算航向角 / Calculate heading angle
            cross_prod = np.cross(segment1, segment2)
            dot_prod = np.dot(segment1, segment2)
            heading_angle = np.arctan2(np.linalg.norm(cross_prod), dot_prod)
            
            # 归一化到[0, 1] / Normalize to [0, 1]
            heading_angle = abs(heading_angle) / np.pi
            n4 += 1
            J4 += heading_angle
    
    if n4 > 0:
        J4 = J4 / n4
    
    # 返回所有目标值 / Return all objective values
    return np.array([J1, J2, J3, J4])
