# NMOPSO - 导航变量多目标粒子群优化算法

Navigation Variable-based Multi-objective Particle Swarm Optimization for UAV Path Planning

## 项目简介

本项目实现了NMOPSO算法用于无人机三维路径规划，包含以下特性：

- **3D障碍物检测**：精确的垂直重叠检测，允许UAV从障碍物上方/下方通过
- **智能初始化**：沿起点-终点方向初始化，显著降低inf比例
- **自适应r范围**：根据直线距离动态调整步长，防止点聚集
- **智能领导者选择**：平衡路径长度和平滑度，避免路径过长
- **随机障碍物生成**：支持种子设置的可复现随机环境

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行主程序

```python
python main.py
```

## 项目结构

```
python_nmopso/
├── main.py                      # 主运行脚本
├── README.md                    # 项目说明（本文件）
├── USER_GUIDE.md                # 详细使用指南
├── requirements.txt             # 依赖包列表
│
├── 核心算法文件 (源代码)
│   ├── nmopso.py                # NMOPSO算法v1（原版）
│   ├── nmopso_v2.py             # NMOPSO算法v2（改进版）
│   ├── particle.py              # 粒子类v1（原版）
│   ├── particle_v2.py           # 粒子类v2（智能初始化）
│   ├── config.py                # 配置类v1（基础）
│   ├── config_enhanced.py       # 配置类v2（增强+自适应）
│   ├── config_enhanced_v2.py    # 配置类v3（随机障碍物）
│   ├── cost_function.py         # 代价函数（已修复3D检测）
│   ├── cost_function_precopy.py # 代价函数原始备份
│   ├── grid.py                  # 网格管理（智能领导者选择）
│   └── utils.py                 # 工具函数
│
├── 对比算法
│   ├── comparison_algorithms.py  # PSO/QPSO/DE算法实现
│   └── compare_algorithms.py     # 算法对比脚本
│
└── 测试文件
    └── test_all.py              # 综合测试脚本
```

## 主要功能

### 1. 运行改进版NMOPSO

```python
from config_enhanced import EnhancedConfig
from nmopso_v2 import NMOPSO_V2

config = EnhancedConfig()
optimizer = NMOPSO_V2(config, use_smart_init=True, smart_ratio=0.7)
repository, best = optimizer.run()
optimizer.plot_results()
```

### 2. 运行原版NMOPSO

```python
from config import Config
from nmopso import NMOPSO

config = Config()
optimizer = NMOPSO(config)
repository, best = optimizer.run()
optimizer.plot_results()
```

### 3. 算法对比

```python
python compare_algorithms.py
```

比较NMOPSO、PSO、QPSO、DE算法的性能。

### 4. 使用随机障碍物

```python
from config_enhanced_v2 import RandomConfig
from nmopso_v2 import NMOPSO_V2

config = RandomConfig(n_obstacles=12, seed=42)
optimizer = NMOPSO_V2(config)
repository, best = optimizer.run()
```

## 核心改进

### 问题1: 3D障碍物检测
- **问题**：原版将所有障碍物视为无限高度圆柱体
- **解决**：添加精确的垂直重叠检测，允许上下飞行
- **文件**：`cost_function.py` (lines 118-145)

### 问题2: 点聚集问题
- **问题**：转角限制变大时，点容易聚集在一起
- **解决**：自适应计算r范围，强制最小间距
- **文件**：`config_enhanced.py` 
- **参数**：`r_min_ratio`, `r_max_ratio`

### 问题3: 路径过长问题
- **问题**：随机选择导致选中长但平滑的路径
- **解决**：基于路径长度和平滑度的加权选择
- **文件**：`grid.py` `select_leader()`
- **参数**：`length_weight`

## 参数调节

| 问题 | 参数 | 调节方向 | 建议值 |
|------|------|---------|--------|
| 点聚集严重 | `r_min_ratio` | 增大 | 0.15 → 0.20 |
| 路径僵硬 | `r_min_ratio` | 减小 | 0.15 → 0.10 |
| 路径太长 | `length_weight` | 增大 | 0.6 → 0.7 |
| 路径不平滑 | `length_weight` | 减小 | 0.6 → 0.4 |

## 性能对比

基于增强配置（12个障碍物，500次迭代）：
在config_enhance中有
``` python
#路径节点数 / Number of path nodes
model['n'] = 15  # 增加节点数以应对复杂环境
```

- **NMOPSO v2**: 最优路径长度，最佳3D避障
- **PSO**: 收敛较慢，路径质量一般
- **QPSO**: 探索性好，但计算量大
- **DE**: 稳定但缺乏多样性

详细对比见 `compare_algorithms.py` 输出。

## 引用

如果使用本代码，请引用原论文：
```
Navigation Variable-based Multi-objective Particle Swarm Optimization 
for UAV Path Planning with Kinematic Constraints
```

## 许可证

本项目仅用于学习和研究目的。

## 联系方式

如有问题，请查看 `USER_GUIDE.md` 获取详细说明。
