# 从其他模块导入必要的类和配置
from config import CONFIG
from simulator import EnergySystemSimulator
from analyzer import Analyzer

if __name__ == '__main__':
    # 从配置文件中读取是否运行参数扫描
    run_scan = CONFIG["ENABLE_OPTIMIZATION_SCAN"]
    
    try:
        # 1. 初始化仿真器 (加载气象数据)
        simulator = EnergySystemSimulator(CONFIG)
        
        # 2. 根据配置决定执行何种任务
        if run_scan:
            simulator.run_optimization_scan()
        else:
            print("--- Running Single Detailed Simulation  ---")
            # 2a. 运行单次仿真
            results = simulator.run_single_simulation(CONFIG)
            
            # 2b. 初始化分析器
            analyzer = Analyzer(CONFIG)
            
            # 2c. 生成完整报告
            analyzer.generate_full_report(results)
            
    except FileNotFoundError as e:
        print(f"\n[致命错误] {e}")
        print("请确保已下载CMFD数据，并将其解压至脚本所在目录下的 'wind', 'srad', 'temp' 文件夹中。")
    except Exception as e:
        print(f"\n[发生未知错误] {e}")
