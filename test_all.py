"""
快速验证测试 / Quick Validation Test
测试所有主要功能是否正常工作
"""
import sys
sys.path.append('E:\\ZJU\\25Sum_Fall\\algorithm\\WORK2\\python_nmopso')

import numpy as np
from config import Config
from config_enhanced import SimpleConfig, EnhancedConfig
from nmopso import NMOPSO

def test_basic_nmopso():
    """测试基本NMOPSO功能"""
    print("\n" + "="*60)
    print("测试1: 基本NMOPSO算法")
    print("="*60)
    
    config = Config()
    config.max_iter = 30
    config.n_pop = 20
    
    try:
        nmopso = NMOPSO(config)
        global_best, repository = nmopso.run()
        
        print("\n✓ 测试通过!")
        print(f"  - 最优代价: {global_best.cost}")
        print(f"  - 存档大小: {len(repository)}")
        return True
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_config():
    """测试增强配置"""
    print("\n" + "="*60)
    print("测试2: 增强配置（更多障碍物）")
    print("="*60)
    
    config = EnhancedConfig(angle_range=np.pi/3)
    config.max_iter = 30
    config.n_pop = 20
    
    print(f"  - 障碍物数量: {len(config.model['threats'])}")
    print(f"  - 角度范围: {config.angle_range*180/np.pi:.1f}°")
    
    try:
        nmopso = NMOPSO(config)
        global_best, repository = nmopso.run()
        
        print("\n✓ 测试通过!")
        print(f"  - 最优代价: {global_best.cost}")
        return True
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        return False

def test_large_angle():
    """测试大角度范围"""
    print("\n" + "="*60)
    print("测试3: 大角度范围 (π/2)")
    print("="*60)
    
    config = SimpleConfig()
    config.angle_range = np.pi / 2
    config.max_iter = 50
    config.n_pop = 30
    
    try:
        nmopso = NMOPSO(config)
        global_best, repository = nmopso.run()
        
        print("\n✓ 测试通过!")
        print(f"  - 最优代价: {global_best.cost}")
        print("  注意: 大角度范围可能需要更多迭代才能收敛")
        return True
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        return False

def test_comparison_algorithms():
    """测试对比算法"""
    print("\n" + "="*60)
    print("测试4: 对比算法")
    print("="*60)
    
    from comparison_algorithms import StandardPSO, QPSO, DifferentialEvolution
    
    config = SimpleConfig()
    results = {}
    
    # 测试标准PSO
    try:
        print("\n  测试标准PSO...")
        std_pso = StandardPSO(config.model, config)
        std_best = std_pso.run(max_iter=30, n_pop=20)
        results['Standard PSO'] = std_best.cost
        print("  ✓ 标准PSO测试通过")
    except Exception as e:
        print(f"  ✗ 标准PSO测试失败: {e}")
        return False
    
    # 测试QPSO
    try:
        print("\n  测试QPSO...")
        qpso = QPSO(config.model, config)
        qpso_best = qpso.run(max_iter=30, n_pop=20)
        results['QPSO'] = qpso_best.cost
        print("  ✓ QPSO测试通过")
    except Exception as e:
        print(f"  ✗ QPSO测试失败: {e}")
        return False
    
    # 测试DE
    try:
        print("\n  测试DE...")
        de = DifferentialEvolution(config.model, config)
        de_best = de.run(max_iter=30, n_pop=20)
        results['DE'] = de_best['cost']
        print("  ✓ DE测试通过")
    except Exception as e:
        print(f"  ✗ DE测试失败: {e}")
        return False
    
    print("\n✓ 所有对比算法测试通过!")
    print("\n各算法最终代价:")
    for alg, cost in results.items():
        print(f"  {alg}: {cost}")
    return True

def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("NMOPSO Python实现 - 功能验证测试")
    print("="*60)
    
    results = []
    
    # 运行测试
    results.append(("基本NMOPSO", test_basic_nmopso()))
    results.append(("增强配置", test_enhanced_config()))
    results.append(("大角度范围", test_large_angle()))
    results.append(("对比算法", test_comparison_algorithms()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! Python实现完全正常工作!")
        print("\n后续步骤:")
        print("  1. 运行 compare_algorithms.py 查看详细对比")
        print("  2. 阅读 ALGORITHM_ANALYSIS.md 了解算法细节和优化建议")
        print("  3. 根据需求调整config_enhanced.py中的参数")
    else:
        print("\n!!!  部分测试失败，请检查错误信息")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
