# -*- coding: utf-8 -*-
"""
日度发电量趋势图绘制模块
功能: 接收仿真结果DataFrame，绘制光伏和风力按日聚合的总发电量曲线。
"""

import pandas as pd
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

def plot_daily_generation(
    results_df: pd.DataFrame, 
    config: Dict[str, Any],
    title: str = "年度光伏与风力日发电量趋势"
):

    print(f"\n--- 正在生成 '{title}' 图表 ---")

    if 'pv_kwh' not in results_df.columns or 'wind_kwh' not in results_df.columns:
        print("错误: 结果数据中缺少 'pv_kwh' 或 'wind_kwh' 列，无法绘制日发电量图。")
        return
        
    daily_generation_df = results_df[['pv_kwh', 'wind_kwh']].resample('D').sum()

    # 2. 绘图
    sns.set_theme(style="darkgrid", palette="colorblind")
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(
        daily_generation_df.index, 
        daily_generation_df['pv_kwh'], 
        label='光伏日发电量 (kWh)', 
        color='gold', 
        linewidth=2.5, 
        zorder=3
    )

    ax.plot(
        daily_generation_df.index, 
        daily_generation_df['wind_kwh'], 
        label='风力日发电量 (kWh)', 
        color='deepskyblue', 
        linewidth=1.5,  
        linestyle='-',  
        zorder=2       

    )
    
    ax.fill_between(
        daily_generation_df.index, daily_generation_df['pv_kwh'], daily_generation_df['wind_kwh'],
        where=daily_generation_df['pv_kwh'] >= daily_generation_df['wind_kwh'],
        facecolor='gold', alpha=0.2, interpolate=True
    )
    ax.fill_between(
        daily_generation_df.index, daily_generation_df['pv_kwh'], daily_generation_df['wind_kwh'],
        where=daily_generation_df['pv_kwh'] < daily_generation_df['wind_kwh'],
        facecolor='deepskyblue', alpha=0.2, interpolate=True
    )

    # 3. 格式化图表
    ax.set_title(title, fontproperties=title_font, pad=20)
    ax.set_xlabel('日期', fontproperties=font)
    ax.set_ylabel('日总发电量 (kWh)', fontproperties=font)
    ax.legend(prop=font, loc='best')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()