# -*- coding: utf-8 -*-
"""
帕累托前沿图绘制模块
功能: 接收参数扫描结果DataFrame，识别并绘制帕累托最优前沿。
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


def identify_pareto_front(df: pd.DataFrame, cost_col: str, benefit_col: str) -> pd.Series:

    is_pareto = pd.Series(True, index=df.index)
    for i, row in df.iterrows():
        # 检查是否存在任何其他点优于当前点
        # "优于"定义为：成本更低或相等，且效益更高或相等，并且不能两者都相等
        dominated = (
            (df[cost_col] <= row[cost_col]) & 
            (df[benefit_col] >= row[benefit_col]) &
            ((df[cost_col] < row[cost_col]) | (df[benefit_col] > row[benefit_col]))
        ).any()
        
        if dominated:
            is_pareto[i] = False
            
    return is_pareto


def plot_pareto_front(
    scan_results_df: pd.DataFrame,
    cost_col: str = 'LCOE_yuan_kWh',
    benefit_col: str = 'Reliability_%',
    color_by_col: str = 'PV_kW',
    title: str = "参数扫描优化结果 - 帕累托前沿分析"
):

    if scan_results_df.empty:
        print("警告: 传入的参数扫描结果为空，无法绘图。")
        return
        
    print(f"\n--- 正在为 '{title}' 生成帕累托前沿图 ---")

    # 1. 识别帕累托前沿
    df = scan_results_df.copy()
    df['is_pareto'] = identify_pareto_front(df, cost_col, benefit_col)
    
    pareto_front = df[df['is_pareto']].sort_values(by=cost_col)

    # 2. 绘图
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 9))

    # 使用 seaborn 绘制散点图，并根据第三维度着色
    scatter = sns.scatterplot(
        data=df,
        x=cost_col,
        y=benefit_col,
        hue=color_by_col,
        palette='viridis',
        s=80,
        alpha=0.7,
        ax=ax,
        legend='full'
    )
    
    # 突出显示帕累托前沿点
    ax.scatter(
        pareto_front[cost_col],
        pareto_front[benefit_col],
        color='red',
        s=120,
        edgecolor='black',
        linewidth=1.5,
        marker='*',
        label='帕累托最优解'
    )
    
    # 连接帕累托前沿点
    ax.plot(
        pareto_front[cost_col],
        pareto_front[benefit_col],
        color='red',
        linestyle='--',
        linewidth=2,
        label='帕累托前沿'
    )

    # 3. 格式化图表
    ax.set_title(title, fontproperties=title_font, pad=20)
    ax.set_xlabel("平准化度电成本 (LCOE, 元/kWh)", fontproperties=font)
    ax.set_ylabel("供电可靠性 (%)", fontproperties=font)
    
    # 优化坐标轴范围
    ax.set_xlim(left=max(0, df[cost_col].min() * 0.95), right=df[cost_col].max() * 1.05)
    ax.set_ylim(bottom=max(0, df[benefit_col].min() * 0.98), top=min(100.1, df[benefit_col].max() * 1.02))


    handles, labels = ax.get_legend_handles_labels()
    
    ax.get_legend().remove()

    pareto_labels = ['帕累托最优解', '帕累托前沿']
    pareto_handles = [h for h, l in zip(handles, labels) if l in pareto_labels]
    
    color_legend_title = color_by_col
    color_handles = [h for h, l in zip(handles, labels) if l not in pareto_labels and l != '']
    color_labels = [l for l in labels if l not in pareto_labels and l != '']

    
    # 第一个图例：颜色图例
    legend1 = fig.legend(
        handles=color_handles,
        labels=color_labels,
        prop=font,
        loc='upper left',
        bbox_to_anchor=(0.86, 0.85),
        title=color_legend_title,
        title_fontproperties=font
    )
    
    # 第二个图例：帕累托指标图例
    legend2 = fig.legend(
        handles=pareto_handles,
        labels=pareto_labels,
        prop=font,
        loc='upper left',
        bbox_to_anchor=(0.86, 0.55),
        title="关键指标",
        title_fontproperties=font
    )
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()



# ==============================================================================
# SECTION: 独立运行入口
# ==============================================================================
if __name__ == '__main__':
    print("--- 正在以独立模式运行帕累托前沿绘图模块 ---")
    
    # 创建一个模拟的参数扫描结果DataFrame
    data = {
        'PV_kW': [800, 800, 800, 1000, 1000, 1000, 1200, 1200, 1200, 1000],
        'Battery_kWh': [7000, 8000, 9000, 8000, 9000, 10000, 9000, 10000, 11000, 8500],
        'LCOE_yuan_kWh': [2.5, 2.7, 2.9, 2.4, 2.6, 2.8, 2.5, 2.7, 2.9, 2.45],
        'Reliability_%': [99.80, 99.90, 99.92, 99.91, 99.95, 99.97, 99.96, 99.98, 99.99, 99.93],
    }
    mock_scan_results_df = pd.DataFrame(data)
    
    # 添加一个被支配的点用于测试
    dominated_point = pd.DataFrame([{
        'PV_kW': 1000, 'Battery_kWh': 9500, 'LCOE_yuan_kWh': 2.65, 'Reliability_%': 99.94
    }])
    mock_scan_results_df = pd.concat([mock_scan_results_df, dominated_point], ignore_index=True)
    
    # 调用绘图函数
    plot_pareto_front(
        scan_results_df=mock_scan_results_df,
        color_by_col='Battery_kWh'
    )