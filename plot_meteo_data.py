# -*- coding: utf-8 -*-
"""
气象数据年度可视化脚本
功能: 读取并绘制指定年份的逐时太阳辐射(srad)和风速(wind)数据曲线。
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.font_manager import FontProperties

# ==============================================================================
# SECTION 1: 用户配置
# ==============================================================================
# --- 数据路径配置 ---
# 将此路径修改为你的 data 文件夹所在的根目录
# 例如，如果你的结构是 E:/my_project/data/，那么这里就填 "E:/my_project"
DATA_ROOT_PATH = Path.cwd() # 使用当前工作目录，如果脚本在项目根目录，则无需修改
DATA_FOLDER_NAME = "data"     # data 文件夹的名称

# --- 仿真参数配置 ---
SIMULATION_YEAR = 2024        # 数据年份
LOCATION = {
    "lon": 80.1,
    "lat": 32.3
}

# --- 绘图配置 ---
PLOT_STYLE = "darkgrid"        # 图表风格 (e.g., "darkgrid", "whitegrid", "ticks")
SRAD_COLOR = "gold"
WIND_COLOR = "deepskyblue"
SRAD_TREND_COLOR = "orangered"
WIND_TREND_COLOR = "royalblue"
TREND_WINDOW_HOURS = 24 * 7    # 趋势线计算窗口 (7天)

# --- 中文显示配置 ---
try:
    # 优先使用Windows下的黑体
    font = FontProperties(fname=r"c:\windows\fonts\simhei.ttf", size=12)
    title_font = FontProperties(fname=r"c:\windows\fonts\simhei.ttf", size=16)
except IOError:
    print("未找到SimHei字体，尝试使用系统默认中文字体。如果显示乱码，请手动指定字体路径。")
    font = FontProperties(family='SimHei', size=12)
    title_font = FontProperties(family='SimHei', size=16)

# ==============================================================================
# SECTION 2: 数据加载函数
# ==============================================================================
def load_and_prepare_data(root_path: Path, data_folder: str, year: int, lon: float, lat: float) -> pd.DataFrame:

    print("--- 正在加载气象数据 ---")
    all_series = {}
    
    for var in ['srad', 'wind']:
        print(f"  正在处理变量: {var}...")
        # 拼接数据文件夹的完整路径
        folder_path = root_path / data_folder / var
        
        # 定义文件名匹配模式
        pattern = f"{var}_CMFD_V0200_B-01_03hr_010deg_{year}??.nc"
        
        # 查找所有匹配的文件
        files = sorted(list(folder_path.glob(pattern)))
        
        if not files:
            raise FileNotFoundError(
                f"在路径 '{folder_path}' 下未找到匹配 '{pattern}' 的数据文件。\n"
                f"请检查:\n"
                f"1. DATA_ROOT_PATH 和 DATA_FOLDER_NAME 配置是否正确。\n"
                f"2. SIMULATION_YEAR ({year}) 是否与数据文件名年份一致。"
            )

        with xr.open_mfdataset(files, combine='by_coords', parallel=True) as ds:
            series_3hr = ds[var].sel(lon=lon, lat=lat, method='nearest')
            series_3hr.load()
            resampler = series_3hr.resample(time='1H')
            if var == 'srad':
                series_1hr = resampler.asfreq().interpolate_na(dim='time', method='spline', order=3)
            else:
                series_1hr = resampler.asfreq().interpolate_na(dim='time', method='linear')
            
            all_series[var] = series_1hr.to_series().clip(lower=0)
            print(f"  变量 '{var}' 加载并处理完成。")
            
    return pd.DataFrame(all_series)

# ==============================================================================
# SECTION 3: 绘图函数
# ==============================================================================
def plot_annual_meteo_curves(df: pd.DataFrame, year: int):

    print("\n--- 开始生成可视化图表 ---")
    sns.set_theme(style=PLOT_STYLE, palette="colorblind")
    
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # --- 计算滚动平均趋势 ---
    df['srad_trend'] = df['srad'].rolling(window=TREND_WINDOW_HOURS, center=True).mean()
    df['wind_trend'] = df['wind'].rolling(window=TREND_WINDOW_HOURS, center=True).mean()
    
    # --- 绘制左Y轴：太阳辐射 ---
    ax1.plot(df.index, df['srad'], label='太阳辐射 (逐时)', color=SRAD_COLOR, alpha=0.4, linewidth=1)
    ax1.plot(df.index, df['srad_trend'], label=f'太阳辐射 ({TREND_WINDOW_HOURS//24}日趋势)', color=SRAD_TREND_COLOR, linewidth=2.5)
    ax1.set_xlabel('日期', fontproperties=font, fontsize=14)
    ax1.set_ylabel('太阳总辐射 (W/m²)', fontproperties=font, fontsize=14, color=SRAD_COLOR)
    ax1.tick_params(axis='y', labelcolor=SRAD_COLOR, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_ylim(bottom=0)

    # --- 绘制右Y轴：风速 ---
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['wind'], label='风速 (逐时)', color=WIND_COLOR, alpha=0.4, linewidth=1)
    ax2.plot(df.index, df['wind_trend'], label=f'风速 ({TREND_WINDOW_HOURS//24}日趋势)', color=WIND_TREND_COLOR, linewidth=2.5, linestyle='--')
    ax2.set_ylabel('10米高风速 (m/s)', fontproperties=font, fontsize=14, color=WIND_COLOR)
    ax2.tick_params(axis='y', labelcolor=WIND_COLOR, labelsize=12)
    ax2.set_ylim(bottom=0)

    # --- 合并图例 ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, prop=font, loc='upper center', ncol=4)
    
    # --- 设置标题和布局 ---
    fig.suptitle(f'{year}年 经纬度({LOCATION["lon"]}, {LOCATION["lat"]}) 气象数据逐时变化', fontproperties=title_font)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- 显示图表 ---
    print("--- 图表生成完毕，即将显示 ---")
    plt.show()

# ==============================================================================
# SECTION 4: 主程序入口
# ==============================================================================
if __name__ == '__main__':
    try:
        # 1. 加载并准备数据
        meteo_data = load_and_prepare_data(
            root_path=DATA_ROOT_PATH,
            data_folder=DATA_FOLDER_NAME,
            year=SIMULATION_YEAR,
            lon=LOCATION['lon'],
            lat=LOCATION['lat']
        )
        
        # 2. 绘制图表
        plot_annual_meteo_curves(meteo_data, SIMULATION_YEAR)
        
    except FileNotFoundError as e:
        print(f"\n[致命错误] 数据文件未找到:")
        print(e)
    except Exception as e:
        print(f"\n[发生未知错误]: {e}")