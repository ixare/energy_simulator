# -*- coding: utf-8 -*-
"""
发电量日周期-年季节性热力图绘制模块
功能: 接收仿真结果，绘制展示全年各个月份24小时平均发电功率的热力图。
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import calendar

try:
    from config import font, title_font
except ImportError:
    print("警告: 未能从项目中导入'config'。将使用内置的默认配置进行绘图。")
    from matplotlib.font_manager import FontProperties
    font = FontProperties(family='sans-serif', size=12)
    title_font = FontProperties(family='sans-serif', size=16, weight='bold')


def plot_diurnal_heatmap(
    results_df: pd.DataFrame, 
    config: Dict[str, Any],
    title: str = "风力发电季节性-日周期特征热力图"
):
    print(f"\n--- 正在生成 '{title}' 图表 ---")

    # 1. 数据准备
    df = results_df.copy()
    if 'wind_kwh' not in df.columns:
        print("错误: 结果数据中缺少 'wind_kwh' 列，无法绘制热力图。")
        return
        
    # 添加 'month' 和 'hour' 列用于分组
    df['month'] = df.index.month
    df['hour'] = df.index.hour

    # 2. 数据聚合
    # 按月份和小时分组，计算平均发电功率 (因为是小时数据, kwh即为kw)
    # 使用 pivot_table 或 groupby().unstack() 转换为2D矩阵
    heatmap_data = df.pivot_table(
        values='wind_kwh', 
        index='month', 
        columns='hour', 
        aggfunc='mean'
    )

    # 3. 绘图
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(22, 10))

    # 使用 seaborn 绘制热力图
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap='inferno',
        linewidths=.5,
        annot=False,
        cbar_kws={'label': '平均发电功率 (kW)'}
    )

    # 4. 格式化图表
    ax.set_title(title, fontproperties=title_font, pad=20)
    ax.set_xlabel('一天中的小时', fontproperties=font)
    ax.set_ylabel('月份', fontproperties=font)
    
    # 将 Y 轴的数字标签替换为月份名称
    month_names = [calendar.month_abbr[i] for i in heatmap_data.index]
    ax.set_yticklabels(month_names, rotation=0, fontproperties=font)
    
    # 设置颜色条标签的字体
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('平均发电功率 (kW)', fontproperties=font)

    plt.tight_layout()
    plt.show()

