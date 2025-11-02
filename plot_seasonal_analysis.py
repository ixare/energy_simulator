# -*- coding: utf-8 -*-
"""
月度/季节性特征分析绘图模块
功能: 绘制关键性能指标（如发电、负载、储能SOC）按月份分布的小提琴图。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

try:
    from config import font, title_font
except ImportError:
    print("警告: 未能从项目中导入'config'。将使用内置的默认配置进行绘图。")
    from matplotlib.font_manager import FontProperties
    font = FontProperties(family='sans-serif', size=12)
    title_font = FontProperties(family='sans-serif', size=16, weight='bold')


def plot_monthly_violin(
    results_df: pd.DataFrame, 
    config: Dict[str, Any],
    title: str = "关键指标月度分布特征分析"
):

    print(f"\n--- 正在为 '{title}' 生成月度特征分析图 ---")

    # 1. 数据预处理
    df = results_df.copy()
    df['month'] = df.index.month
    if 'battery_soc_kwh' in df.columns:
        df['battery_soc_percent'] = (df['battery_soc_kwh'] / config["BATTERY"]["capacity_kwh"]) * 100
    else:
        df['battery_soc_percent'] = np.nan
    if 'curtailment_kwh' in df.columns:
        df_curtailment = df[df['curtailment_kwh'] > 0.1].copy()
    else:
        df_curtailment = pd.DataFrame(columns=df.columns)

    # 2. 绘图
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 创建一个2x2的子图布局，并明确设置总标题
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # plt.figtext(x, y, text, ...) 
    plt.figtext(0.5, 0.98, title, ha='center', va='top', fontproperties=title_font)

    # --- 子图1: 小时发电量 ---
    ax1 = axes[0, 0]
    sns.violinplot(x='month', y='generation_kwh', data=df, ax=ax1, inner='quartile', cut=0)
    ax1.set_title('小时发电量 (kW) 月度分布', fontproperties=font)
    ax1.set_xlabel(None)
    ax1.set_ylabel('发电功率 (kW)', fontproperties=font)

    # --- 子图2: 小时负载 ---
    ax2 = axes[0, 1]
    sns.violinplot(x='month', y='load_kwh', data=df, ax=ax2, inner='quartile', cut=0)
    ax2.set_title('小时负载 (kW) 月度分布', fontproperties=font)
    ax2.set_xlabel(None)
    ax2.set_ylabel('负载功率 (kW)', fontproperties=font)

    # --- 子图3: 电池SOC ---
    ax3 = axes[1, 0]
    if not df['battery_soc_percent'].isnull().all():
        sns.violinplot(x='month', y='battery_soc_percent', data=df, ax=ax3, inner='quartile', cut=0)
        ax3.set_ylim(0, 100)
    ax3.set_title('电池SOC (%) 月度分布', fontproperties=font)
    ax3.set_xlabel('月份', fontproperties=font)
    ax3.set_ylabel('荷电状态 (SOC %)', fontproperties=font)

    # --- 子图4: 弃电量 ---
    ax4 = axes[1, 1]
    if not df_curtailment.empty:
        sns.violinplot(x='month', y='curtailment_kwh', data=df_curtailment, ax=ax4, inner='quartile', cut=0)
        ax4.set_title('小时弃电量 (kW) 月度分布 (仅统计发生弃电时段)', fontproperties=font)
    else:
        ax4.text(0.5, 0.5, '全年无弃电', ha='center', va='center', transform=ax4.transAxes, fontproperties=font, fontsize=16)
        ax4.set_title('小时弃电量 (kW) 月度分布', fontproperties=font)
    
    ax4.set_xlabel('月份', fontproperties=font)
    ax4.set_ylabel('弃电功率 (kW)', fontproperties=font)
    
    # hspace 是子图之间的水平间距，wspace 是垂直间距
    # top, bottom, left, right 控制图表边缘的留白
    plt.subplots_adjust(
        left=0.07, 
        right=0.97, 
        top=0.92,
        bottom=0.08, 
        hspace=0.3,
        wspace=0.2
    )
    
    plt.show()
# ==============================================================================
# SECTION: 独立运行入口
# ==============================================================================
if __name__ == '__main__':
    print("--- 正在以独立模式运行季节性特征分析绘图模块 ---")
    
    # 1. 创建时间索引
    time_index = pd.date_range(start='2023-01-01', end='2023-12-31 23:00:00', freq='H')
    n_points = len(time_index)
    
    # 2. 计算各个分量
    
    # 发电量模拟：夏季高，冬季低，有日夜循环
    seasonal_gen_factor = 300 - 250 * np.cos(2 * np.pi * (time_index.dayofyear - 180) / 365.25)
    daily_gen_factor = 200 * (1 - np.cos(2 * np.pi * time_index.hour / 24))
    simulated_generation = (seasonal_gen_factor + daily_gen_factor + np.random.rand(n_points) * 50).clip(0)

    # 负载模拟：冬季高，夏季次高，有日夜循环
    seasonal_load_factor = 150 + 100 * np.cos(2 * np.pi * (time_index.dayofyear - 30) / 365.25)
    daily_load_factor = 50 * np.sin(2 * np.pi * time_index.hour / 24)
    simulated_load = (seasonal_load_factor + daily_load_factor + np.random.rand(n_points) * 20).clip(0)

    # 电池SOC模拟：夏季普遍高，冬季普遍低
    base_soc_percent = (70 - 20 * np.cos(2 * np.pi * (time_index.dayofyear - 180) / 365.25) + np.random.randn(n_points) * 5).clip(15, 95)
    
    # 弃电模拟：当发电远大于负载时，有一定概率发生
    surplus = (simulated_generation - simulated_load).clip(0)
    simulated_curtailment = surplus * np.random.choice([0, 1], n_points, p=[0.8, 0.2])

    # 3. 构建DataFrame
    mock_results_df = pd.DataFrame(
        {
            'generation_kwh': simulated_generation,
            'load_kwh': simulated_load,
            # 将SOC百分比转换回绝对值kWh
            'battery_soc_kwh': (base_soc_percent / 100) * 20000,
            'curtailment_kwh': simulated_curtailment,
        },
        index=time_index
    )
    
    # 4. 创建模拟的config字典
    mock_config = {
        "BATTERY": {"capacity_kwh": 20000},
    }
    
    # 5. 调用绘图函数
    plot_monthly_violin(
        results_df=mock_results_df,
        config=mock_config
    )