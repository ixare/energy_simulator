import numpy as np
import copy
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS 
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# 从项目中导入必要的类和配置
from config import CONFIG
from simulator import EnergySystemSimulator
from analyzer import Analyzer

# ==============================================================================
# SECTION 1: 定义遗传算法的优化问题
# ==============================================================================
class EnergySystemOptimizationProblem(Problem):

    def __init__(self, base_config, simulator):
        self.base_config = base_config
        self.simulator = simulator

        # 初始化一个列表来存储所有评估过的解决方案的历史记录
        self.history = []

        self.variables = {
            "PV__capacity_kw": [700, 900],
            "AWES__count": [6, 10],
            "BATTERY__capacity_kwh": [4000, 8000],
            "HYDROGEN__tank__capacity_kg": [3000, 10000]
        }
        
        var_names = list(self.variables.keys())
        n_var = len(var_names)
        xl = np.array([self.variables[k][0] for k in var_names])
        xu = np.array([self.variables[k][1] for k in var_names])
        n_obj = 2

        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        all_results_f = []

        for single_solution_x in x:
            temp_config = copy.deepcopy(self.base_config)
            
            current_params = {}
            for i, key in enumerate(self.variables.keys()):
                val = int(np.round(single_solution_x[i])) if "count" in key else single_solution_x[i]
                current_params[key] = val
                
                d = temp_config
                parts = key.split("__")
                for p in parts[:-1]:
                    d = d.setdefault(p, {})
                d[parts[-1]] = val
            
            print(f"--- Evaluating: {current_params} ---")

            temp_config['LIFETIME_YEARS'] = 10
            results_df = self.simulator.run_single_simulation(temp_config)
            
            analyzer = Analyzer(temp_config)
            perf_metrics = analyzer._analyze_performance(results_df, is_print=False)
            eco_metrics = analyzer._calculate_economics(results_df, "baseline")
            
            lcoe = eco_metrics['lcoe']
            reliability = perf_metrics['reliability']

            f1 = lcoe if np.isfinite(lcoe) else 1e10
            f2 = 100 - reliability if np.isfinite(reliability) else 1e10
            
            print(f"--- Result: LCOE={f1:.2f}, Reliability={100-f2:.3f}% ---")
            
            all_results_f.append([f1, f2])

            # 将当前解的参数和结果存入历史记录
            history_entry = current_params.copy()
            history_entry['LCOE_yuan_kWh'] = f1
            history_entry['Reliability_%'] = 100 - f2
            self.history.append(history_entry)

        out["F"] = np.array(all_results_f)


# ==============================================================================
# SECTION 2: 运行遗传算法
# ==============================================================================
if __name__ == '__main__':
    print("Initializing simulator and loading meteorological data...")
    simulator = EnergySystemSimulator(CONFIG)
    print("Initialization complete.")

    problem = EnergySystemOptimizationProblem(base_config=CONFIG, simulator=simulator)

    algorithm = NSGA2(
        pop_size=80, # 种群数
        n_offsprings=10,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=10),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 40) # 迭代次数

    print("\nStarting NSGA-II optimization...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True,
        verbose=True
    )
    print("Optimization finished.")

    # --- 结果分析与可视化 ---
    
    # 1. 帕累托最优解集
    print("\n" + "="*50)
    print("--- Optimization Results: Pareto Optimal Set ---")
    print("="*50)
    
    pareto_solutions = []
    if res.X is not None and res.F is not None:
        for x, f in zip(res.X, res.F):
            solution = {}
            for i, key in enumerate(problem.variables.keys()):
                val = int(np.round(x[i])) if "count" in key else x[i]
                solution[key] = val
            solution['LCOE_yuan_kWh'] = f[0]
            solution['Reliability_%'] = 100 - f[1]
            pareto_solutions.append(solution)
        
    results_df = pd.DataFrame(pareto_solutions)
    
    if not results_df.empty:
        results_df.rename(columns={
            "PV__capacity_kw": "PV_kW",
            "AWES__count": "Wind_Count",
            "BATTERY__capacity_kwh": "Battery_kWh",
            "HYDROGEN__tank__capacity_kg": "H2_kg"
        }, inplace=True)

        pd.options.display.float_format = '{:.2f}'.format
        print(results_df)
    else:
        print("No feasible solution found on the Pareto front.")

    # 2. 从历史记录中筛选并输出前20个最佳解 (按LCOE排序)
    print("\n" + "="*50)
    print("--- Top 20 Best Solutions Found (sorted by LCOE) ---")
    print("="*50)
    
    if problem.history:
        # 将历史记录转换为DataFrame
        history_df = pd.DataFrame(problem.history)
        
        # 筛选出有效的解（LCOE < 1e10）
        valid_history_df = history_df[history_df['LCOE_yuan_kWh'] < 1e9].copy()
        
        # 按 LCOE 升序排序
        sorted_history_df = valid_history_df.sort_values(by='LCOE_yuan_kWh', ascending=True)
        
        # 重命名列以保持一致性
        sorted_history_df.rename(columns={
            "PV__capacity_kw": "PV_kW",
            "AWES__count": "Wind_Count",
            "BATTERY__capacity_kwh": "Battery_kWh",
            "HYDROGEN__tank__capacity_kg": "H2_kg"
        }, inplace=True)
        
        # 输出前20行
        print(sorted_history_df.head(20))
    else:
        print("No history was recorded.")

    # 3. 绘图
    if not results_df.empty:
        try:
            from plot_pareto_front import plot_pareto_front
            plot_pareto_front(
                scan_results_df=results_df,
                cost_col='LCOE_yuan_kWh',
                benefit_col='Reliability_%',
                color_by_col='PV_kW'
            )
        except ImportError:
            print("\nCould not import 'plot_pareto_front'. Please run plotting manually.")
        except Exception as e:
            print(f"\nAn error occurred during plotting: {e}")