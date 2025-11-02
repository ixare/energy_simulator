# -*- coding: utf-8 -*-
"""
储能持续时间曲线绘制模块
功能: 在仿真结果中找到最长的连续能量亏空期，并绘制此期间储能系统的状态变化。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from typing import Dict, Any, Tuple
try:
    from config import font, title_font
except ImportError:
    print("警告: 未能从项目中导入'config'。将使用内置的默认配置进行绘图。")
    from matplotlib.font_manager import FontProperties
    font = FontProperties(family='sans-serif', size=12)
    title_font = FontProperties(family='sans-serif', size=16, weight='bold')


def find_longest_deficit_period(results_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, int]:

    df = results_df.copy()
    
    # 1. 计算净能量 (发电 - 总需求)
    df['net_energy'] = df['generation_kwh'] - (df['load_kwh'] + df['parasitic_load_kwh'])
    
    # 2. 标记亏空时段 (净能量 < 0)
    df['is_deficit'] = df['net_energy'] < 0
    
    # 3. 识别连续亏空期的开始和结束
    # `shift(1)` 用于找到状态变化的点
    # `cumsum()` 为每个连续块分配一个唯一的ID
    df['deficit_block'] = (df['is_deficit'] != df['is_deficit'].shift(1)).cumsum()
    
    # 4. 计算每个连续亏空块的长度
    deficit_periods = df[df['is_deficit']].groupby('deficit_block').size()
    
    if deficit_periods.empty:
        return None, None, 0
        
    # 5. 找到最长的亏空块
    longest_block_id = deficit_periods.idxmax()
    longest_period_df = df[df['deficit_block'] == longest_block_id]
    
    start_time = longest_period_df.index.min()
    end_time = longest_period_df.index.max()
    duration_hours = len(longest_period_df)
    
    return start_time, end_time, duration_hours


def plot_storage_duration_curve(
    results_df: pd.DataFrame, 
    config: Dict[str, Any],
    title: str = "最长能量亏空期内储能系统续航能力分析"
):

    print(f"\n--- 正在为 '{title}' 生成储能持续时间曲线 ---")

    # 1. 找到最长的能量亏空期
    start_time, end_time, duration = find_longest_deficit_period(results_df)
    
    if duration == 0:
        print("未在仿真结果中找到任何能量亏空期，无需绘图。")
        return

    print(f"找到最长连续亏空期: 从 {start_time} 到 {end_time}，持续 {duration} 小时。")

    # 2. 筛选该时期的数据
    period_df = results_df.loc[start_time:end_time].copy()
    
    # 3. 数据准备
    # 将储能状态转换为百分比
    period_df['battery_soc_percent'] = (period_df['battery_soc_kwh'] / config["BATTERY"]["capacity_kwh"]) * 100
    if config["HYDROGEN"]["enabled"]:
        period_df['h2_storage_percent'] = (period_df['h2_storage_kg'] / config["HYDROGEN"]["tank"]["capacity_kg"]) * 100
    
    # 创建以小时为单位的X轴
    period_df['duration_hours'] = np.arange(len(period_df))

    # 4. 绘图
    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(16, 9))

    # 绘制电池SOC曲线
    ax1.plot(period_df['duration_hours'], period_df['battery_soc_percent'], color='cyan', linewidth=2.5, label='电池SOC')
    ax1.set_xlabel('能量亏空持续时间 (小时)', fontproperties=font, fontsize=14)
    ax1.set_ylabel('电池荷电状态 (SOC %)', fontproperties=font, fontsize=14, color='cyan')
    ax1.tick_params(axis='y', labelcolor='cyan', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 找到电池SOC最低点
    min_soc_val = period_df['battery_soc_percent'].min()
    min_soc_hour = period_df['battery_soc_percent'].idxmin()
    min_soc_duration = period_df.loc[min_soc_hour, 'duration_hours']
    ax1.axhline(min_soc_val, color='red', linestyle=':', linewidth=2, label=f'最低SOC: {min_soc_val:.1f}%')
    ax1.text(period_df['duration_hours'].max(), min_soc_val, f' {min_soc_val:.1f}%', color='red', va='center', ha='left', fontproperties=font)


    # 如果启用了氢能，则绘制氢储量曲线
    if config["HYDROGEN"]["enabled"]:
        ax2 = ax1.twinx()
        ax2.plot(period_df['duration_hours'], period_df['h2_storage_percent'], color='lime', linestyle='--', linewidth=2.5, label='氢储量')
        ax2.set_ylabel('氢储罐储量 (%)', fontproperties=font, fontsize=14, color='lime')
        ax2.tick_params(axis='y', labelcolor='lime', labelsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(False) # 避免网格重叠

    # 5. 合并图例和格式化
    lines1, labels1 = ax1.get_legend_handles_labels()
    if config["HYDROGEN"]["enabled"]:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, prop=font, loc='best')
    else:
        ax1.legend(prop=font, loc='best')
        
    full_title = f"{title}\n(最长亏空期: {start_time.strftime('%Y-%m-%d %H:%M')} 起, 持续{duration}小时)"
    fig.suptitle(full_title, fontproperties=title_font)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# ==============================================================================
# SECTION: 独立运行入口
# ==============================================================================
if __name__ == '__main__':
    print("--- 正在以独立模式运行储能持续时间曲线绘图模块 ---")
    
    # 创建一个模拟的、符合格式的DataFrame用于测试
    index = pd.date_range(start='2023-01-01', periods=24 * 10, freq='H')
    n = len(index)
    
    # 模拟数据
    # 创造一个从第50小时到第150小时的漫长亏空期
    gen = np.full(n, 150.0)
    gen[50:150] = 20.0 # 亏空期发电量很低
    load = np.full(n, 100.0)
    
    data = {
        'generation_kwh': gen,
        'load_kwh': load,
        'parasitic_load_kwh': np.full(n, 5.0),
        'battery_soc_kwh': np.linspace(18000, 3000, n), # 模拟电池电量持续下降
        'h2_storage_kg': np.linspace(3000, 2000, n), # 模拟氢储量持续下降
    }
    mock_results_df = pd.DataFrame(data, index=index)
    
    # 创建一个模拟的config字典
    mock_config = {
        "BATTERY": {"capacity_kwh": 20000}, 
        "HYDROGEN": {"enabled": True, "tank": {"capacity_kg": 3500}},
    }
    
    # 调用绘图函数
    plot_storage_duration_curve(
        results_df=mock_results_df,
        config=mock_config
    )