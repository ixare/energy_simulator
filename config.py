from pathlib import Path
from typing import Dict, Any
from matplotlib.font_manager import FontProperties

try:
    font = FontProperties(fname=r"c:\windows\fonts\simhei.ttf", size=12)
    title_font = FontProperties(fname=r"c:\windows\fonts\simhei.ttf", size=16)
except IOError:
    font = FontProperties(family='SimHei', size=12)
    title_font = FontProperties(family='SimHei', size=16)

# ==============================================================================
# SECTION 1: 全局配置
# ==============================================================================
CONFIG: Dict[str, Any] = {
    # --- 1. 基础与仿真控制 ---
    "BASE_PATH": Path.cwd(),                      # 项目根目录路径，Path.cwd()表示当前工作目录。
    "DATA_FOLDER": "data",                        # 存放气象数据的文件夹名称。
    "SIMULATION_YEARS": [2024],                   # 用于仿真的气象数据年份列表。目前模型仅使用列表中的第一个年份。
    "LIFETIME_YEARS": 10,                          # 系统的设计寿命（年），用于经济性分析（如LCOE计算）和设备衰减模拟。
    "ENABLE_OPTIMIZATION_SCAN": True,            # 是否启用参数扫描模式。True: 运行多组合优化；False: 运行单次详细仿真。

    # --- 2. 地理位置信息 ---
    "LOCATION": {
        "lon": 80.1,                              # 站点经度 (degrees East)。
        "lat": 32.3,                              # 站点纬度 (degrees North)。
        "altitude_m": 5250,                       # 站点海拔高度 (米)，用于修正风机输出功率。
    },

    # --- 3. 用户用电负荷模型 ---
    "LOAD": {
        # 基础负荷定义，按季节和昼夜划分。
        "base_load": {
            "winter_night_kw": 70,                # 冬季（10月-3月）夜间（18:00-06:00）基础负荷 (kW)。
            "winter_day_kw": 25,                  # 冬季白天的基础负荷 (kW)。
            "summer_night_kw": 35,                # 夏季（4月-9月）夜间的基础负荷 (kW)。
            "summer_day_kw": 20,                  # 夏季白天的基础负荷 (kW)。
        },
        "random_fluctuation_percent": 0.1,        # 在基础负荷上增加的随机波动百分比 (例如, 0.1 表示 ±10% 的波动)。
        # 科研任务事件模拟。
        "science_events": {
            "enabled": True,                      # 是否启用科研事件模拟。
            "events_per_year": 10,                # 每年随机发生的科研事件次数。
            "duration_hours_range": (24, 72),     # 每个事件的持续时间范围（小时）。
            "extra_power_kw_range": (50, 100),    # 每个事件期间额外增加的功率范围 (kW)。
        }
    },

    # --- 4. 光伏发电系统 ---
    "PV": {
        "capacity_kw": 800,                      # 光伏系统总装机容量 (kW)。
        "efficiency": 0.40,                       # 光伏组件在标准测试条件下的光电转换效率 (未来技术设定)。
        "reference": "NREL 2040 PV Roadmap",      # 技术参考或说明。
        "temp_coefficient": -0.003,               # 功率温度系数 (%/°C)，表示电池温度每升高1摄氏度，功率下降的百分比。
        "noct": 45,                               # 正常工作电池温度 (°C)，用于计算实际工作时的电池板温度。
        "annual_degradation_rate": 0.005,         # 年性能衰减率 (例如, 0.005 表示每年衰减0.5%)。
    },

    # --- 5. 系留式风力发电系统 ---
    "AWES": {
        "count": 8,                               # AWES 单元的数量
        "nominal_power_kw_per_unit": 60,          # 单个单元的额定/名义功率 (kW)，主要用于经济和环境评估。
        # 额定输出功率，这是发电机的电气系统上限
        "rated_power_kw_per_unit": 75,
        # 达到额定功率时的风速 (在运行高度)
        "rated_wind_speed_ms": 15.0,
        "operational_altitude_m": 500,            # 平均运行高度 (米)。

        # --- 2045年先进空气动力学参数 ---
        "wing_area_m2": 20,                       # 单个单元的翼面积 (平方米)。
        "lift_to_drag_ratio": 25.0,               # [2045预测] 翼型的升阻比。得益于主动翼型控制和先进复合材料，远高于当前水平(10-15)。
        "lift_coeff": 1.4,                        # 翼型的平均升力系数，用于计算有效升阻比。
        
        # --- 2045年先进材料与结构参数 (缆绳) ---
        "tether_diameter_mm": 8.0,                # 缆绳直径(毫米)。基于碳纳米管等未来材料，实现高强度与低直径。
        "tether_drag_coeff": 0.9,                 # 缆绳的阻力系数。可通过缆绳外形优化（如增加整流纹理）来降低。

        # --- 系统运行与效率参数 ---
        "elevation_angle_deg": 30.0,              # 系统的平均飞行仰角（度）。用于计算余弦损失和缆绳长度。
        "pumping_cycle_efficiency": 0.88,         # 泵动循环效率。表示 (发电能量 - 回收能量) / 发电能量 的比率。
        "total_efficiency": 0.5,                 # [2045预测] 包含发电机、传动、电力电子的总系统效率，高于当前的0.4。
        "annual_degradation_rate": 0.010,         # [2045预测] 年性能衰减率。新材料和设计使其更耐久。

        # --- 基础运行边界 ---
        "min_operational_wind_speed_ms": 3.0,     # 最小可运行风速 (m/s)。
        "max_operational_wind_speed_ms": 25.0,    # 最大可运行风速 (m/s)。

        "wind_shear_alpha": {
        "day": 0.14,      # 白天 (不稳定大气)，风混合较好，alpha较小
        "night": 0.25     # 夜晚 (稳定大气)，风层化明显，alpha较大
    }
    },

    # --- 6. 锂电池储能系统 ---
    "BATTERY": {
        "capacity_kwh": 6000,                    # 电池储能系统的总额定容量 (kWh)。
        "initial_soc_percent": 0.6,               # 仿真开始时电池的初始荷电状态 (State of Charge, SOC) 百分比。
        "max_soc_percent": 0.95,                  # 允许的最高SOC，以保护电池不过充。
        "min_soc_percent": 0.15,                  # 允许的最低SOC，以保护电池不过放。
        "c_rate": 0.5,                            # 电池的额定充放电倍率 (C-rate)，决定了最大充放电功率 (功率 = C率 * 容量)。
        # 动态效率曲线，描述了不同充放电倍率下的效率。
        "efficiency_curve": {
            "c_points": [0, 0.2, 0.5, 0.8, 1.0],  # C率的关键点。
            "eff_points": [0.88, 0.93, 0.94, 0.92, 0.89], # 对应C率下的充放电效率。
        },
        "capacity_degradation_per_kwh_throughput": 1e-7, # 每kWh能量吞吐量（充+放）导致的容量衰减系数。
    },

    # --- 7. 氢储能系统 ---
    "HYDROGEN": {
        "enabled": True,                          # 是否启用氢储能系统。
        # 电解槽，用于制氢。
        "electrolyzer": {
            "capacity_kw": 350,                   # 电解槽的额定输入功率 (kW)。
            "efficiency": 0.9,                    # 电解槽的效率（基于低热值LHV）。
            "min_load_percent": 0.1,              # 最小运行负载百分比 (例如, 0.1 表示必须在10%额定功率以上运行)。
            "aux_power_percent_of_input": 0.10,   # 例如，额外消耗电解槽输入功率10%的电能
            "annual_degradation_rate": 0.015,     # 电解槽年效率衰减率 (1.5%)
        },
        
        # 储氢罐。
        "tank": {
            "capacity_kg": 10000,                  # 储氢罐的总容量 (kg)。
            "initial_storage_percent": 0.5,       # 仿真开始时的初始储氢量百分比。
        },
        # 燃料电池，用于发电。
        "fuel_cell": {
            "capacity_kw": 120,                   # 燃料电池的额定输出功率 (kW)。
            "efficiency": 0.60,                   # 燃料电池的发电效率。
            "min_load_percent": 0.1,              # 最小运行负载百分比。
            "annual_degradation_rate": 0.015,     # 燃料电池年效率衰减率 (1.5%)
        },
        # 氢能系统高级控制逻辑参数。
        "control_logic": {
            "recharge_start_percent": 0.7,        # 当电池SOC低于此值时，燃料电池启动为电池充电。
            "recharge_target_percent": 0.80,      # 燃料电池为电池充电的目标SOC。
            "direct_supply_soc_percent": 0.30,    # 当电池SOC低于此值且有缺口时，燃料电池直接为负载供电。
        },
    },

    # --- 8. 辅助设备与系统损耗 ---
    "AUXILIARY_LOAD": {
        "base_parasitic_kw": 2.0,                 # 系统的基础寄生损耗 (kW)，如监控、热管理等。
        "inverter_loss_factor": 0.03,             # 逆变器损耗系数，损耗功率 = 此系数 * 逆变器处理的总功率。
    },

    # --- 9. 经济性评估参数 ---
    "ECONOMICS": {
        "discount_rate": 0.05,                    # 贴现率，用于计算LCOE。
        "cost_scenarios": ["baseline", "optimistic", "pessimistic"], # 定义三种成本情景。
        # 基准情景下的单位成本。
        "baseline": {
            "capex_pv_per_kw": 2500,              # 光伏系统的单位资本支出 (元/kW)。
            "capex_awes_per_kw": 11000,           # 风电系统的单位资本支出 (元/kW)。
            "capex_battery_per_kwh": 1000,        # 电池系统的单位资本支出 (元/kWh)。
            "capex_hydrogen_sys_per_kw": 12000,   # 氢能系统的单位资本支出 (元/kW，按电解槽和燃料电池中较大功率计算)。
            "opex_percent_of_capex": 0.02,        # 年运维成本占总资本支出的百分比。
        },
        # 乐观情景下的单位成本。
        "optimistic": {
            "capex_pv_per_kw": 1800,
            "capex_awes_per_kw": 8000,
            "capex_battery_per_kwh": 700,
            "capex_hydrogen_sys_per_kw": 9000,
            "opex_percent_of_capex": 0.015,
        },
        # 悲观情景下的单位成本。
        "pessimistic": {
            "capex_pv_per_kw": 3500,
            "capex_awes_per_kw": 14000,
            "capex_battery_per_kwh": 1400,
            "capex_hydrogen_sys_per_kw": 15000,
            "opex_percent_of_capex": 0.025,
        },
    },

    # --- 10. 环境影响评估参数 ---
    "ENVIRONMENTAL": {
        "land_use_pv_m2_per_kw": 15,              # 每kW光伏安装所需的土地面积 (平方米)。
        "land_use_awes_m2_per_unit": 200,      # 每台风机所需的土地面积 (平方米)。
        "water_consumption_L_per_kg_h2": 10,      # 每生产1公斤氢气所需的水量 (升)。
        # 设备生命周期的隐含碳足迹（制造、运输等过程的碳排放）。
        "carbon_footprint": {
            "pv_per_kw": 500,                     # 每kW光伏的碳足迹 (kgCO2e)。
            "awes_per_kw": 800,                    # 每kW风电的碳足迹 (kgCO2e)。
            "battery_per_kwh": 150,               # 每kWh电池的碳足迹 (kgCO2e)。
            "hydrogen_sys_per_kw": 800,           # 每kW氢能系统的碳足迹 (kgCO2e)。
        },
    },

    # --- 11. 参数扫描配置 ---
"OPTIMIZATION_SCAN_PARAMS": {

    "PV__capacity_kw": [250, 300, 350, 400, 450, 500, 550, 600, 650, 700],

    "BATTERY__capacity_kwh": [2500, 3000, 3500, 4000, 4500, 5000, 5500],

    "HYDROGEN__tank__capacity_kg": [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
    
    "AWES__count": [3, 4, 5, 6, 7, 8, 9],
}
}
