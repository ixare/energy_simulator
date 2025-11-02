import pandas as pd
import numpy as np
from typing import Dict, Any

# ==============================================================================
# SECTION 2: 能源系统模型类
# ==============================================================================
class LoadProfile:
    def __init__(self, config: Dict[str, Any]): self.config = config
    def generate(self, time_index: pd.DatetimeIndex) -> pd.Series:
        base_config = self.config["base_load"]
        load = pd.Series(index=time_index, dtype=float)
        for ts in time_index:
            month, hour = ts.month, ts.hour; is_winter = (month >= 10) or (month <= 3); is_night = (hour >= 18) or (hour < 6)
            if is_winter and is_night: base = base_config["winter_night_kw"]
            elif is_winter: base = base_config["winter_day_kw"]
            elif is_night: base = base_config["summer_night_kw"]
            else: base = base_config["summer_day_kw"]
            load[ts] = base * (1 + np.random.uniform(-self.config["random_fluctuation_percent"], self.config["random_fluctuation_percent"]))
        if self.config["science_events"]["enabled"]:
            total_hours = len(time_index)
            for _ in range(self.config["science_events"]["events_per_year"]):
                start_hour = np.random.randint(0, total_hours - 72); duration = np.random.randint(*self.config["science_events"]["duration_hours_range"]); extra_power = np.random.uniform(*self.config["science_events"]["extra_power_kw_range"])
                load.iloc[start_hour : start_hour + duration] += extra_power
        return load

class PVSystem:
    def __init__(self, config: Dict[str, Any]): self.config = config
    def get_power(self, srad: pd.Series, temp: pd.Series, degradation_factor: float) -> pd.Series:
        capacity = self.config["capacity_kw"] * degradation_factor
        base_power = capacity * (srad / 1000); cell_temp = temp + (self.config["noct"] - 20) / 800 * srad
        temp_correction = 1 + self.config["temp_coefficient"] * (cell_temp - 25); pv_power = (base_power * temp_correction).clip(lower=0)
        return pv_power

class AWESystem:
    def __init__(self, config: Dict[str, Any], location: Dict[str, Any]):
        """
        初始化系留式风能系统 (AWES) 模型。
        该模型基于一个简化的物理公式，其功率输出与升阻比的平方成正比。
        """
        self.config = config
        self.location = location
        self.count = config["count"]
        self.operational_altitude_m = config["operational_altitude_m"]
        self.wing_area_m2 = config["wing_area_m2"]
        self.ld_ratio = config["lift_to_drag_ratio"]
        self.total_efficiency = config["total_efficiency"]
        self.min_wind = config["min_operational_wind_speed_ms"]
        self.max_wind = config["max_operational_wind_speed_ms"]
        self.alpha = config["wind_shear_alpha"]
        self.rated_power_per_unit = config["rated_power_kw_per_unit"]
    def _calculate_air_density(self, temp_series: pd.Series) -> pd.Series:
        pressure_hpa = 1013.25 * (1 - 2.25577e-5 * self.location["altitude_m"])**5.25588
        temp_kelvin = temp_series + 273.15
        return (pressure_hpa * 100) / (287.058 * temp_kelvin)
    
    def _correct_wind_speed(self, wind_speed: pd.Series) -> pd.Series:
        """使用风切变公式修正风速到运行高度。"""
        H1 = 10
        H2 = self.operational_altitude_m
        
        # 创建一个与wind_speed索引相同的Series，用于存放alpha值
        alpha_series = pd.Series(index=wind_speed.index, dtype=float)
        
        # 根据小时判断是白天还是夜晚
        day_hours = (wind_speed.index.hour >= 7) & (wind_speed.index.hour < 19)
        night_hours = ~day_hours
        
        alpha_series[day_hours] = self.config["wind_shear_alpha"]["day"]
        alpha_series[night_hours] = self.config["wind_shear_alpha"]["night"]

        if H2 > H1:
            return wind_speed * (H2 / H1) ** alpha_series
        else:
            return wind_speed
    def get_power(self, wind_speed: pd.Series, temp: pd.Series, degradation_factor: float) -> pd.Series:
        """
        计算 AWES 的输出功率，采用2045年实用化模型。
        该模型包含了缆绳阻力、仰角损失和泵动循环效率的修正。
        """
        # 1. 基础计算
        air_density = self._calculate_air_density(temp)
        corrected_wind_speed = self._correct_wind_speed(wind_speed)

        # 2. 计算系统有效升阻比 (E_eff)，包含缆绳阻力影响
        elevation_angle_rad = np.deg2rad(self.config.get("elevation_angle_deg", 30.0))
        tether_length = self.operational_altitude_m / np.sin(elevation_angle_rad)
        tether_diameter = self.config.get("tether_diameter_mm", 8.0) / 1000
        tether_drag_coeff = self.config.get("tether_drag_coeff", 0.9)
        lift_coeff = self.config.get("lift_coeff", 1.4)

        # E_eff 公式
        inverse_E_eff = (1 / self.ld_ratio) + \
                        (tether_drag_coeff * tether_diameter * tether_length) / \
                        (3 * lift_coeff * self.wing_area_m2)
        E_eff = 1 / inverse_E_eff

        # 3. 计算考虑了仰角损失的理论毛功率 (单位: W)
        # 公式: P_gross = (2/27) * ρ * A * (Vw * cos(β))^3 * E_eff^2
        effective_wind_speed = corrected_wind_speed * np.cos(elevation_angle_rad)
        
        power_per_unit_gross_watts = (2/27) * air_density * self.wing_area_m2 * (effective_wind_speed**3) * (E_eff**2)
        pumping_efficiency = self.config.get("pumping_cycle_efficiency", 0.88)
        # 转换为 kW 并应用所有效率和设备数量
        total_net_power_kw_theoretical = (power_per_unit_gross_watts / 1000) * \
                                         self.total_efficiency * \
                                         pumping_efficiency * \
                                         self.count * \
                                         degradation_factor

        # 步骤 5: 应用运行风速限制
        # 首先，将超出运行范围的风速对应的功率直接置为0
        total_net_power_kw_theoretical[(corrected_wind_speed < self.min_wind) | (corrected_wind_speed > self.max_wind)] = 0

        # 步骤 6: 应用额定功率限制
        # 然后，对所有仍在工作范围内的功率点，应用额定功率的上限
        total_rated_power_kw = self.rated_power_per_unit * self.count * degradation_factor
        total_net_power_kw = total_net_power_kw_theoretical.clip(lower=0, upper=total_rated_power_kw)

        # 返回最终结果
        return total_net_power_kw

class BatteryStorage:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.capacity_kwh = config["capacity_kwh"]
        # 保存初始能量值
        self.initial_soc_kwh = self.capacity_kwh * config["initial_soc_percent"]
        self.soc_kwh = self.initial_soc_kwh
        
        self.max_soc_kwh = self.capacity_kwh * config["max_soc_percent"]
        self.min_soc_kwh = self.capacity_kwh * config["min_soc_percent"]
        self.max_charge_power_kw = self.capacity_kwh * config["c_rate"]
        self.max_discharge_power_kw = self.capacity_kwh * config["c_rate"]

    def get_dynamic_efficiency(self, power_kw: float) -> float:
        c_rate = abs(power_kw) / self.capacity_kwh if self.capacity_kwh > 0 else 0
        curve = self.config["efficiency_curve"]
        return np.interp(c_rate, curve["c_points"], curve["eff_points"])

    def degrade(self, throughput_kwh: float):
        degradation = throughput_kwh * self.config["capacity_degradation_per_kwh_throughput"]
        self.capacity_kwh *= (1 - degradation)

        self.max_soc_kwh = self.capacity_kwh * self.config["max_soc_percent"]
        self.min_soc_kwh = self.capacity_kwh * self.config["min_soc_percent"]


class HydrogenStorage:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_enabled = config["enabled"]
        elec_conf = config["electrolyzer"]
        tank_conf = config["tank"]
        fc_conf = config["fuel_cell"]
        
        # 保存初始配置，用于衰减计算
        self.initial_elec_efficiency = elec_conf["efficiency"]
        self.initial_fc_efficiency = fc_conf["efficiency"]
        self.elec_degradation_rate = elec_conf.get("annual_degradation_rate", 0)
        self.fc_degradation_rate = fc_conf.get("annual_degradation_rate", 0)

        # 初始化当前效率
        self.current_elec_efficiency = self.initial_elec_efficiency
        self.current_fc_efficiency = self.initial_fc_efficiency
        
        self.elec_capacity_kw = elec_conf["capacity_kw"]
        # 使用当前效率计算转换系数
        self.h2_per_kwh = (1 / 33.33) * self.current_elec_efficiency
        
        self.tank_capacity_kg = tank_conf["capacity_kg"]
        self.initial_storage_kg = self.tank_capacity_kg * tank_conf["initial_storage_percent"]
        self.storage_kg = self.initial_storage_kg
        
        self.fc_capacity_kw = fc_conf["capacity_kw"]
        # 使用当前效率计算转换系数
        self.kwh_per_h2 = 33.33 * self.current_fc_efficiency

    def degrade(self):
        """
        每年调用一次，用于更新电解槽和燃料电池的效率。
        """
        # 更新当前效率
        self.current_elec_efficiency *= (1 - self.elec_degradation_rate)
        self.current_fc_efficiency *= (1 - self.fc_degradation_rate)
        
        # 重新计算依赖于效率的转换系数
        self.h2_per_kwh = (1 / 33.33) * self.current_elec_efficiency
        self.kwh_per_h2 = 33.33 * self.current_fc_efficiency
        print(f"  [衰减更新] 氢系统: 电解槽效率 -> {self.current_elec_efficiency:.4f}, 燃料电池效率 -> {self.current_fc_efficiency:.4f}")
