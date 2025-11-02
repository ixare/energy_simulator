# -*- coding: utf-8 -*-
"""
能量平衡堆叠面积图绘制模块
功能: 接收仿真结果DataFrame，绘制指定时间段内的能量供需平衡图。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

try:
    from models import PVSystem, AWESystem
    from config import font, title_font, CONFIG as main_config
except ImportError:
    # 如果作为独立脚本运行，定义一些默认值以避免错误
    print("警告: 未能从项目中导入'models'和'config'。将使用内置的默认配置进行绘图。")
    from matplotlib.font_manager import FontProperties
    font = FontProperties(family='sans-serif', size=12)
    title_font = FontProperties(family='sans-serif', size=16, weight='bold')
    main_config = None


def plot_energy_balance_stacked_area(
    results_df: pd.DataFrame, 
    config: Dict[str, Any], 
    start_date: str, 
    end_date: str, 
    title: str = "能量平衡分析"
):

    print(f"\n--- 正在为 '{title}' 生成能量平衡图 ---")
    
    # 1. 筛选指定时间段的数据
    df = results_df.loc[start_date:end_date].copy()
    if df.empty:
        print(f"警告: 在指定日期范围 {start_date} 到 {end_date} 内未找到数据。")
        return

    # 2. 数据准备
    if 'pv_kwh' not in df.columns or 'wind_kwh' not in df.columns:
        print("错误: 结果数据中缺少 'pv_kwh' 或 'wind_kwh' 列。")
        print("将统一显示为'可再生能源发电'。请确保仿真器已正确记录分项发电量。")

        df['pv_kwh'] = df.get('generation_kwh', pd.Series(0, index=df.index))
        df['wind_kwh'] = pd.Series(0, index=df.index)
    else:
        print("成功加载精确的分项发电量数据 (pv_kwh, wind_kwh)。")

    # 3. 定义供应侧和需求侧的能量流
    # 供应侧
    supply_sources = {
        '光伏发电': df['pv_kwh'],
        '风力发电': df['wind_kwh'],
        '电池放电': df['battery_discharge_kwh'],
        '燃料电池发电': df.get('fc_to_load_kwh', pd.Series(0, index=df.index))
    }
    
    # 需求侧（处理为负值以便在图下方显示）
    demand_sinks = {
        '用户负载': -df['load_kwh'],
        '寄生损耗': -df['parasitic_load_kwh'],
        '电池充电': -df['battery_charge_kwh'],
        '电解槽制氢': -df.get('power_consumed_by_h2_prod_kwh', pd.Series(0, index=df.index))
    }
    
    supply_df = pd.DataFrame(supply_sources)
    demand_df = pd.DataFrame(demand_sinks)

    # 4. 绘图
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 定义颜色
    supply_colors = ['gold', 'deepskyblue', 'lightgreen', 'plum']
    demand_colors = ['coral', 'grey', 'limegreen', 'orchid']

    ax.stackplot(df.index, supply_df.T, labels=supply_df.columns, colors=supply_colors, alpha=0.8,
                 edgecolor='black', linewidth=0.5)
    
    ax.stackplot(df.index, demand_df.T, labels=demand_df.columns, colors=demand_colors, alpha=0.8,
                 edgecolor='black', linewidth=0.5)

    # 绘制总发电量和总负载的参考线
    ax.plot(df.index, df['generation_kwh'], color='black', linestyle='--', linewidth=2, label='总发电量')
    total_demand = df['load_kwh'] + df['parasitic_load_kwh']
    ax.plot(df.index, total_demand, color='red', linestyle='-', linewidth=2, label='总系统需求')

    # 格式化图表
    ax.set_title(title, fontproperties=title_font, pad=20)
    ax.set_xlabel('日期和时间', fontproperties=font)
    ax.set_ylabel('功率 (kW)', fontproperties=font)
    ax.axhline(0, color='black', linewidth=1.5)
    ax.tick_params(axis='x', rotation=20)
    
    # 优化图例
    handles, labels = ax.get_legend_handles_labels()
    # 重新排序图例以匹配堆叠顺序
    order = list(range(len(supply_df.columns))) + \
            list(range(len(supply_df.columns), len(supply_df.columns) + len(demand_df.columns))) + \
            [len(handles)-2, len(handles)-1]
    
    import matplotlib.patches as mpatches
    legend_elements = [mpatches.Patch(color=c, label=l) for c, l in zip(supply_colors, supply_df.columns)]
    legend_elements += [mpatches.Patch(color=c, label=l) for c, l in zip(demand_colors, demand_df.columns)]
    legend_elements += [plt.Line2D([0], [0], color='black', linestyle='--', lw=2, label='总发电量'),
                        plt.Line2D([0], [0], color='red', linestyle='-', lw=2, label='总系统需求')]
    
    ax.legend(handles=legend_elements, prop=font, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()


# ==============================================================================
# SECTION: 独立运行入口
# ==============================================================================
if __name__ == '__main__':
    print("--- 正在以独立模式运行能量平衡绘图模块 ---")
    
    # 创建一个模拟的、符合格式的DataFrame用于测试
    start_time = '2023-07-10 00:00:00'
    end_time = '2023-07-16 23:00:00'
    index = pd.date_range(start=start_time, end=end_time, freq='H')
    n = len(index)
    
    # 模拟数据
    data = {
        'generation_kwh': 500 * (1 - np.cos(np.linspace(0, 4 * np.pi, n))) + np.random.rand(n) * 100 + 150,
        'load_kwh': 200 + 100 * np.sin(np.linspace(0, 14 * np.pi, n)) + np.random.rand(n) * 30,
        'parasitic_load_kwh': np.full(n, 5.0),
        'battery_discharge_kwh': pd.Series(np.random.rand(n) * 100).rolling(6).mean().fillna(0),
        'battery_charge_kwh': pd.Series(np.random.rand(n) * 150).rolling(6).mean().fillna(0),
    }
    mock_results_df = pd.DataFrame(data, index=index)
    
    # 清理模拟数据，确保逻辑正确
    is_surplus = mock_results_df['generation_kwh'] > (mock_results_df['load_kwh'] + mock_results_df['parasitic_load_kwh'])
    mock_results_df.loc[is_surplus, 'battery_discharge_kwh'] = 0
    mock_results_df.loc[~is_surplus, 'battery_charge_kwh'] = 0

    # 创建一个模拟的config字典
    mock_config = {
        "PV": {}, "WIND": {}, "LOCATION": {},
    }
    
    # 调用绘图函数
    plot_energy_balance_stacked_area(
        results_df=mock_results_df,
        config=mock_config,
        start_date='2023-07-11',
        end_date='2023-07-13',
        title="夏季典型周能量平衡分析 (模拟数据)"
    )