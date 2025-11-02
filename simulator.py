import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import itertools
import copy


from models import LoadProfile, PVSystem, AWESystem, BatteryStorage, HydrogenStorage
from analyzer import Analyzer
from plot_pareto_front import plot_pareto_front

class EnergySystemSimulator:
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config; self.met_data = self._load_meteorological_data()
        print("\n--- 能源系统仿真器初始化完毕 ---")

    def _load_meteorological_data(self) -> pd.DataFrame:
        print("--- 正在加载基础气象数据 ---"); all_series = {}
        year_to_load = self.base_config["SIMULATION_YEARS"][0]
        base_path = Path(self.base_config["BASE_PATH"])
        data_folder_name = self.base_config["DATA_FOLDER"]
        
        for var in ['wind', 'srad', 'temp']:
            folder_path = base_path / data_folder_name / var; pattern = f"{var}_CMFD_V0200_B-01_03hr_010deg_{year_to_load}??.nc"
            files = sorted(list(folder_path.glob(pattern)))
            if not files: raise FileNotFoundError(f"在路径'{folder_path}'下未找到匹配'{pattern}'的数据文件")
            ds = xr.open_mfdataset(files, combine='by_coords')
            series_3hr = ds[var].sel(lon=self.base_config["LOCATION"]["lon"], lat=self.base_config["LOCATION"]["lat"], method='nearest')
            series_3hr.load()
            resampler = series_3hr.resample(time='1H')
            if var == 'srad':
                series_1hr = resampler.asfreq().interpolate_na(dim='time', method='spline', order=3)
            else:
                series_1hr = resampler.asfreq().interpolate_na(dim='time', method='linear')

            all_series[var] = series_1hr.to_series().clip(lower=0)
        return pd.DataFrame(all_series)

    def run_single_simulation(self, config_to_run: Dict[str, Any], force_single_year: bool = False) -> pd.DataFrame:
        load_model = LoadProfile(config_to_run["LOAD"]); pv_model = PVSystem(config_to_run["PV"])
        wind_model = AWESystem(config_to_run["AWES"], config_to_run["LOCATION"])
        battery = BatteryStorage(config_to_run["BATTERY"]); hydrogen = HydrogenStorage(config_to_run["HYDROGEN"])
        all_years_history = []
        simulation_years = 1 if force_single_year else config_to_run['LIFETIME_YEARS']
        print(f"\n--- 开始进行 {simulation_years} 年全生命周期仿真 ---")
        for year_index in range(simulation_years):
            if config_to_run['LIFETIME_YEARS'] > 1: print(f"  正在仿真第 {year_index + 1}/{config_to_run['LIFETIME_YEARS']} 年...")
            pv_deg_factor = (1 - config_to_run["PV"]["annual_degradation_rate"]) ** year_index
            wind_deg_factor = (1 - config_to_run["AWES"]["annual_degradation_rate"]) ** year_index
            load_series = load_model.generate(self.met_data.index)
            pv_power = pv_model.get_power(self.met_data['srad'], self.met_data['temp'], pv_deg_factor)
            wind_power = wind_model.get_power(self.met_data['wind'], self.met_data['temp'], wind_deg_factor)
            
            year_history_df = self._simulation_loop(load_series, pv_power, wind_power, battery, hydrogen, config_to_run)


            year_history_df.index = self.met_data.index + pd.DateOffset(years=year_index)

            throughput_kwh = year_history_df['battery_charge_kwh'].sum() + year_history_df['battery_discharge_kwh'].sum()
            battery.degrade(throughput_kwh)
            hydrogen.degrade()
            all_years_history.append(year_history_df)
        full_lifetime_results_df = pd.concat(all_years_history); print("--- 全生命周期仿真完成 ---")
        full_lifetime_results_df.sort_index(inplace=True)  
        return full_lifetime_results_df

    def _simulation_loop(self, load, pv_power, wind_power, battery, hydrogen, config) -> pd.DataFrame:
            history = []
            aux_config = config["AUXILIARY_LOAD"]
            hydro_config = config["HYDROGEN"]
            
            recharge_start_kwh = battery.capacity_kwh * hydro_config["control_logic"]["recharge_start_percent"]
            recharge_target_kwh = battery.capacity_kwh * hydro_config["control_logic"]["recharge_target_percent"]
            direct_supply_soc_kwh = battery.capacity_kwh * hydro_config["control_logic"]["direct_supply_soc_percent"]
            elec_min_power_kw = hydro_config["electrolyzer"]["capacity_kw"] * hydro_config["electrolyzer"]["min_load_percent"]
            fc_min_power_kw = hydro_config["fuel_cell"]["capacity_kw"] * hydro_config["fuel_cell"]["min_load_percent"]

            time_step_hours = 1.0

            for h in range(len(load)):
                hr = {
                    'load_kwh': load.iloc[h], 'pv_kwh': pv_power.iloc[h], 'wind_kwh': wind_power.iloc[h],
                    'generation_kwh': pv_power.iloc[h] + wind_power.iloc[h], 'parasitic_load_kwh': 0,
                    'battery_charge_kwh': 0, 'battery_discharge_kwh': 0, 'h2_produced_kg': 0,
                    'h2_consumed_kg': 0, 'shortage_kwh': 0, 'curtailment_kwh': 0,
                    'fc_to_battery_kwh': 0, 'fc_to_load_kwh': 0,'storage_loss_kwh': 0,
                }

                # 寄生损耗分为基础部分和逆变器部分，逆变器损耗在确定功率流后计算
                base_parasitic_kwh = aux_config["base_parasitic_kw"] * time_step_hours
                effective_load_kwh = hr['load_kwh'] + base_parasitic_kwh
                net_energy_kwh = hr['generation_kwh'] - effective_load_kwh

                inverter_loss_kwh = 0

                if net_energy_kwh > 0:  # 能量盈余
                    # 逆变器损耗应用于所有发电量
                    inverter_loss_kwh = hr['generation_kwh'] * aux_config["inverter_loss_factor"]
                    remaining_energy_kwh = net_energy_kwh - inverter_loss_kwh
                    
                    # 电池充电逻辑 (不变)
                    charge_power_limit_kw = battery.max_charge_power_kw
                    energy_to_charge_kwh = min(remaining_energy_kwh, charge_power_limit_kw * time_step_hours)
                    charge_power_kw = energy_to_charge_kwh / time_step_hours
                    charge_efficiency = battery.get_dynamic_efficiency(charge_power_kw)
                    energy_stored_in_battery_kwh = energy_to_charge_kwh * charge_efficiency
                    actual_energy_stored = min(energy_stored_in_battery_kwh, battery.max_soc_kwh - battery.soc_kwh)
                    power_consumed_by_charging_kwh = actual_energy_stored / charge_efficiency if charge_efficiency > 0 else float('inf')
                    hr['storage_loss_kwh'] += (power_consumed_by_charging_kwh - actual_energy_stored) # 记录电池充电损耗
                    battery.soc_kwh += actual_energy_stored
                    hr['battery_charge_kwh'] = actual_energy_stored
                    remaining_energy_kwh -= power_consumed_by_charging_kwh

                    # --- 制氢逻辑，增加辅助设备损耗 ---
                    if hydrogen.is_enabled and remaining_energy_kwh > 0:
                        elec_conf = hydro_config["electrolyzer"]
                        aux_loss_factor = elec_conf["aux_power_percent_of_input"]
                        
                        # 总可用功率 = 净剩余功率 / (1 + 辅助损耗因子)
                        total_power_for_h2_kw = remaining_energy_kwh / time_step_hours
                        power_to_electrolyzer_kw = total_power_for_h2_kw / (1 + aux_loss_factor)

                        if power_to_electrolyzer_kw >= elec_min_power_kw:
                            power_to_electrolyzer_kw = min(power_to_electrolyzer_kw, elec_conf["capacity_kw"])
                            
                            h2_produced_kg = (power_to_electrolyzer_kw * time_step_hours) * hydrogen.h2_per_kwh
                            actual_h2_produced_kg = min(h2_produced_kg, hydrogen.tank_capacity_kg - hydrogen.storage_kg)
                            hydrogen.storage_kg += actual_h2_produced_kg
                            hr['h2_produced_kg'] = actual_h2_produced_kg

                            # 计算实际消耗的总电能（电解槽+辅助设备）
                            actual_power_to_elec_kwh = (actual_h2_produced_kg / hydrogen.h2_per_kwh) if hydrogen.h2_per_kwh > 0 else 0
                            total_power_consumed_by_h2_prod_kwh = actual_power_to_elec_kwh * (1 + aux_loss_factor)

                            # 记录制氢损耗 (输入电能 - 氢气等效热值)
                            h2_energy_equivalent_kwh = actual_h2_produced_kg * 33.33
                            hr['storage_loss_kwh'] += (total_power_consumed_by_h2_prod_kwh - h2_energy_equivalent_kwh)
                            remaining_energy_kwh -= total_power_consumed_by_h2_prod_kwh

                    hr['curtailment_kwh'] = max(0, remaining_energy_kwh)

                else:  # 能量亏空
                    energy_needed_kwh = -net_energy_kwh
                    
                    # 计算逆变器损耗，需要从储能获取更多能量来弥补
                    loss_factor = aux_config["inverter_loss_factor"]
                    gross_energy_needed_from_storage = energy_needed_kwh / (1 - loss_factor)
                    inverter_loss_kwh = gross_energy_needed_from_storage - energy_needed_kwh
                    
                    # 从电池放电
                    discharge_power_limit_kw = battery.max_discharge_power_kw
                    energy_available_from_battery_kwh = min(battery.soc_kwh - battery.min_soc_kwh, discharge_power_limit_kw * time_step_hours)
                    discharge_power_kw = min(gross_energy_needed_from_storage / time_step_hours, discharge_power_limit_kw)
                    discharge_efficiency = battery.get_dynamic_efficiency(discharge_power_kw)
                    
                    # 电池能提供的有效能量（考虑了自身效率）
                    effective_energy_from_battery_kwh = energy_available_from_battery_kwh * discharge_efficiency
                    
                    # 实际从储能系统获取的能量
                    energy_supplied_from_storage = min(gross_energy_needed_from_storage, effective_energy_from_battery_kwh)
                    
                    # 反算出从电池中抽取的能量
                    energy_drawn_from_battery_kwh = energy_supplied_from_storage / discharge_efficiency if discharge_efficiency > 0 else float('inf')
                    
                    hr['storage_loss_kwh'] += (energy_drawn_from_battery_kwh - energy_supplied_from_storage) # 记录电池放电损耗
                    battery.soc_kwh -= energy_drawn_from_battery_kwh
                    hr['battery_discharge_kwh'] = energy_drawn_from_battery_kwh

                    # 扣除逆变器损耗后，实际到达负载的能量
                    energy_delivered_to_load_from_batt = energy_supplied_from_storage * (1 - loss_factor)
                    
                    remaining_shortage_kwh = energy_needed_kwh - energy_delivered_to_load_from_batt
                    
                    # 燃料电池紧急供电
                    if hydrogen.is_enabled and remaining_shortage_kwh > 0 and battery.soc_kwh <= direct_supply_soc_kwh:
                        # 同样需要考虑逆变器损耗
                        gross_shortage_from_fc = remaining_shortage_kwh / (1 - loss_factor)
                        inverter_loss_fc_part = gross_shortage_from_fc - remaining_shortage_kwh
                        inverter_loss_kwh += inverter_loss_fc_part
                        
                        power_needed_from_fc_kw = gross_shortage_from_fc / time_step_hours
                        
                        if power_needed_from_fc_kw >= fc_min_power_kw:
                            max_e_from_fc_power = hydrogen.fc_capacity_kw * time_step_hours
                            max_e_from_h2_storage = hydrogen.storage_kg * hydrogen.kwh_per_h2
                            e_fc_to_gen = min(gross_shortage_from_fc, max_e_from_fc_power, max_e_from_h2_storage)
                            
                            h2_consumed_for_load = (e_fc_to_gen / hydrogen.kwh_per_h2) if hydrogen.kwh_per_h2 > 0 else 0
                            # 录燃料电池发电损耗 (消耗氢气热值 - 输出电能)
                            h2_energy_consumed_kwh = h2_consumed_for_load * 33.33
                            hr['storage_loss_kwh'] += (h2_energy_consumed_kwh - e_fc_to_gen)

                            hydrogen.storage_kg -= h2_consumed_for_load
                            hr['fc_to_load_kwh'] = e_fc_to_gen * (1 - loss_factor) # 记录净输出
                            hr['h2_consumed_kg'] += h2_consumed_for_load
                            remaining_shortage_kwh -= hr['fc_to_load_kwh']

                    hr['shortage_kwh'] = max(0, remaining_shortage_kwh)

                # 燃料电池为电池充电逻辑
                if hydrogen.is_enabled and battery.soc_kwh < recharge_start_kwh:
                    # 只有当燃料电池额定功率大于最小运行时，才考虑启动
                    if hydrogen.fc_capacity_kw >= fc_min_power_kw:
                        chg_eff_fc = battery.get_dynamic_efficiency(hydrogen.fc_capacity_kw)
                        e_needed_batt = recharge_target_kwh - battery.soc_kwh
                        
                        # 假设充电时以额定功率运行
                        max_p_fc = hydrogen.fc_capacity_kw
                        max_e_fc_storage = hydrogen.storage_kg * hydrogen.kwh_per_h2
                        
                        # 计算此小时内能产生的电量
                        e_to_gen_potential = max_p_fc * time_step_hours
                        
                        # 实际发电量受限于所需电量和储氢量
                        e_to_gen = min(e_needed_batt / chg_eff_fc if chg_eff_fc > 0 else float('inf'), e_to_gen_potential, max_e_fc_storage)
                        
                        # 仅当产生的能量有意义时才启动
                        if e_to_gen > 0:
                            h2_cons = (e_to_gen / hydrogen.kwh_per_h2) if hydrogen.kwh_per_h2 > 0 else 0

                            # 记录燃料电池为电池充电的损耗
                            h2_energy_consumed_kwh_for_batt = h2_cons * 33.33
                            e_into_batt = e_to_gen * chg_eff_fc
                            hr['storage_loss_kwh'] += (h2_energy_consumed_kwh_for_batt - e_into_batt)

                            hydrogen.storage_kg -= h2_cons
                            e_into_batt = e_to_gen * chg_eff_fc
                            battery.soc_kwh += e_into_batt
                            hr['h2_consumed_kg'] += h2_cons
                            hr['fc_to_battery_kwh'] = e_into_batt
                # 更新总寄生损耗
                hr['parasitic_load_kwh'] = base_parasitic_kwh + inverter_loss_kwh
                
                hr['battery_soc_kwh'] = battery.soc_kwh
                hr['h2_storage_kg'] = hydrogen.storage_kg
                history.append(hr)
                
            return pd.DataFrame(history, index=load.index)
    def run_optimization_scan(self):
        print("\n--- Running Optimization Parameter Scan ---"); scan_params = self.base_config["OPTIMIZATION_SCAN_PARAMS"]
        keys, values = zip(*scan_params.items()); scan_results_list = []
        for v_combination in itertools.product(*values):
            temp_config = copy.deepcopy(self.base_config); print("\n--- 扫描配置 ---"); config_str_list = []
            for key, val in zip(keys, v_combination):
                config_str_list.append(f"{key.split('__')[-1]}: {val}"); d = temp_config; parts = key.split("__")
                for p in parts[:-1]: d = d.setdefault(p, {})
                d[parts[-1]] = val
            print(" | ".join(config_str_list))

            analysis_lifetime = 10 # 或者从主CONFIG中获取
            temp_config['LIFETIME_YEARS'] = analysis_lifetime
            
            # 运行1年的仿真
            single_year_results_df = self.run_single_simulation(temp_config, force_single_year=True)
            
            analyzer = Analyzer(temp_config)
            perf = analyzer._analyze_performance(single_year_results_df, is_print=False)

            # 传入单年结果，但告知经济性计算函数使用完整的生命周期进行扩展计算
            eco = analyzer._calculate_economics(single_year_results_df, "baseline", force_lifetime_scaling=True)
            env = analyzer._calculate_environment(perf)
            current_result = {"PV_kW": temp_config["PV"]["capacity_kw"], "Battery_kWh": temp_config["BATTERY"]["capacity_kwh"],
                              "H2_kg": temp_config["HYDROGEN"]["tank"]["capacity_kg"], "Reliability_%": perf["reliability"],
                              "LCOE_yuan_kWh": eco["lcoe"], "Land_Use_m2": env["land_use_m2"]}
            scan_results_list.append(current_result)
        results_df = pd.DataFrame(scan_results_list); print("\n" + "="*80); print("--- 参数扫描结果对比 ---")
        print(results_df.round(2).to_string(index=False)); print("="*80)
        
        viable_solutions = results_df[results_df['Reliability_%'] >= 99.9]
        if not viable_solutions.empty:
            best_solution = viable_solutions.loc[viable_solutions['LCOE_yuan_kWh'].idxmin()]
            print("\n--- 推荐最优方案 (可靠性 > 99.9% & LCOE最低) ---")
            print(best_solution.to_string())
        else:
            print("\n--- 未找到满足可靠性 > 99.9% 的方案 ---")
        print("="*80)

        
        if not results_df.empty:
            plot_pareto_front(
                scan_results_df=results_df,
                cost_col='LCOE_yuan_kWh',
                benefit_col='Reliability_%',
                color_by_col='PV_kW'
            )
