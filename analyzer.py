import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from config import font, title_font
from models import BatteryStorage, HydrogenStorage
from plot_energy_balance import plot_energy_balance_stacked_area
from plot_storage_duration import plot_storage_duration_curve
from plot_seasonal_analysis import plot_monthly_violin
from plot_daily_generation import plot_daily_generation
from plot_diurnal_heatmap import plot_diurnal_heatmap

class Analyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate_full_report(self, results_df: pd.DataFrame):
        """主入口函数，调用所有分析和绘图方法。"""
        print("\n" + "="*50)
        print("  能源系统仿真综合分析报告")
        print("="*50)

        # 1. 性能分析
        performance_metrics = self._analyze_performance(results_df)
        
        # 2. 经济性分析
        self._analyze_economics(results_df)
        
        # 3. 环境影响评估
        self._analyze_environment(performance_metrics)
        
        print("="*50)
        
        # 4. 可视化
        self._plot_results_single_year(results_df)
        self._plot_seasonal_balance(results_df)
        # 绘制储能持续时间曲线
        plot_storage_duration_curve(results_df, self.config)
        
        plot_monthly_violin(results_df, self.config)

        plot_daily_generation(results_df, self.config)

        plot_diurnal_heatmap(results_df, self.config)
    def _analyze_performance(self, results_df: pd.DataFrame, is_print=True) -> Dict[str, float]:
            if is_print: print("\n--- 1. 关键性能指标 ---")
            lifetime = max(1, self.config.get('LIFETIME_YEARS', 1))
            
            # --- 基础数据汇总 ---
            total_user_load = results_df['load_kwh'].sum()
            total_pv_generation = results_df['pv_kwh'].sum()
            total_wind_generation = results_df['wind_kwh'].sum()
            total_generation = total_pv_generation + total_wind_generation
            total_shortage = results_df['shortage_kwh'].sum()
            total_curtailment = results_df['curtailment_kwh'].sum()
            total_parasitic_load = results_df['parasitic_load_kwh'].sum()
            # 明确统计储能转换损耗
            total_storage_loss = results_df['storage_loss_kwh'].sum()

            # --- 计算储能净贡献 ---
            temp_battery = BatteryStorage(self.config["BATTERY"])
            temp_hydrogen = HydrogenStorage(self.config["HYDROGEN"])
            initial_batt_e = temp_battery.initial_soc_kwh
            initial_h2_e = temp_hydrogen.initial_storage_kg * 33.33 # 使用热值计算
            final_batt_e = results_df['battery_soc_kwh'].iloc[-1]
            final_h2_e = results_df['h2_storage_kg'].iloc[-1] * 33.33 # 使用热值计算
            net_storage_contribution = (initial_batt_e + initial_h2_e) - (final_batt_e + final_h2_e)

            # --- 计算总需求与可靠性 ---
            total_system_demand = total_user_load + total_parasitic_load
            reliability = 100 * (1 - total_shortage / total_system_demand) if total_system_demand > 0 else 100
            
            # 能量平衡校验
            energy_sources = total_generation + net_storage_contribution
            energy_sinks = total_system_demand + total_curtailment + total_storage_loss
            balance_check = energy_sources - energy_sinks

            if is_print:
                print(f"  仿真周期: {results_df.index.min().date()} to {results_df.index.max().date()}")
                print(f"  供电可靠性: {reliability:.5f} %")
                print(f"\n  --- 能量平衡分析 (年均) ---")
                print(f"    [+] 总发电量:        {total_generation / lifetime:,.2f} kWh/year")
                print(f"        - 光伏发电:      {total_pv_generation / lifetime:,.2f} kWh/year")
                print(f"        - 风力发电:      {total_wind_generation / lifetime:,.2f} kWh/year")
                print(f"    [+] 储能净贡献:      {net_storage_contribution / lifetime:,.2f} kWh/year")
                print(f"    ---------------------------------------------")
                print(f"    [=] 总可用能源:      {energy_sources / lifetime:,.2f} kWh/year\n")

                print(f"    [-] 总用户负荷:      {total_user_load / lifetime:,.2f} kWh/year")
                print(f"    [-] 总寄生损耗:      {total_parasitic_load / lifetime:,.2f} kWh/year")
                print(f"    [-] 总系统需求:      {total_system_demand / lifetime:,.2f} kWh/year")
                print(f"    [-] 储能与转换损耗:  {total_storage_loss / lifetime:,.2f} kWh/year")
                print(f"    [-] 总弃电量:        {total_curtailment / lifetime:,.2f} kWh/year")
                print(f"    ---------------------------------------------")
                print(f"    [=] 总能源去向:      {energy_sinks / lifetime:,.2f} kWh/year\n")

                print(f"  校验: [总可用] - [总去向] = {balance_check / lifetime:,.4f} kWh/year (应接近于0)")
                print(f"  总电力缺口:          {total_shortage / lifetime:,.2f} kWh/year")
            
            return {"reliability": reliability, "total_h2_produced_kg": results_df['h2_produced_kg'].sum()}


    def _calculate_economics(self, results_df: pd.DataFrame, scenario: str, force_lifetime_scaling: bool = False) -> Dict[str, float]:
        eco_conf = self.config["ECONOMICS"]
        scenario_conf = eco_conf[scenario]
        lifetime = self.config["LIFETIME_YEARS"]
        discount_rate = eco_conf["discount_rate"]
        
        # --- 1. 计算所有资本支出的现值 ---
        
        # 初始资本支出
        capex_pv = self.config["PV"]["capacity_kw"] * scenario_conf["capex_pv_per_kw"]
        capex_awes = self.config["AWES"]["count"] * self.config["AWES"]["nominal_power_kw_per_unit"] * scenario_conf["capex_awes_per_kw"]
        capex_battery = self.config["BATTERY"]["capacity_kwh"] * scenario_conf["capex_battery_per_kwh"]
        h2_power = max(self.config["HYDROGEN"]["electrolyzer"]["capacity_kw"], self.config["HYDROGEN"]["fuel_cell"]["capacity_kw"])
        capex_hydrogen = h2_power * scenario_conf["capex_hydrogen_sys_per_kw"] if self.config["HYDROGEN"]["enabled"] else 0
        
        total_initial_capex = capex_pv + capex_awes + capex_battery + capex_hydrogen

        # 重置成本
        battery_lifetime_years = 15 # 假设电池寿命15年
        replacement_costs_pv = 0
        if lifetime > battery_lifetime_years:
            print(f"  [经济性提示] 项目寿命({lifetime}年) > 电池寿命({battery_lifetime_years}年)，正在计算重置成本...")
            num_replacements = (lifetime - 1) // battery_lifetime_years
            for i in range(1, num_replacements + 1):
                replacement_year = i * battery_lifetime_years
                replacement_cost = self.config["BATTERY"]["capacity_kwh"] * scenario_conf["capex_battery_per_kwh"]
                # 将未来成本折现回其现值
                replacement_costs_pv += replacement_cost / ((1 + discount_rate) ** replacement_year)
        
        # 总投资成本的现值 = 初始投资 + 所有重置投资的现值
        total_investment_pv = total_initial_capex + replacement_costs_pv

        # --- 2. 计算所有运维成本的现值 ---
        annual_opex = total_initial_capex * scenario_conf["opex_percent_of_capex"]
        opex_pv = 0
        for year in range(1, lifetime + 1):
            opex_pv += annual_opex / ((1 + discount_rate) ** year)

        # --- 3. 计算总生命周期成本的现值 ---
        total_lcc_pv = total_investment_pv + opex_pv

        # --- 4. 计算总生命周期发电量的现值 ---
        total_load_served_in_sim = results_df['load_kwh'].sum() - results_df['shortage_kwh'].sum()
        
        if force_lifetime_scaling:
            avg_annual_load_served = total_load_served_in_sim
        else:
            avg_annual_load_served = total_load_served_in_sim / lifetime if lifetime > 0 else 0
            
        total_energy_production_pv = 0
        for year in range(1, lifetime + 1):
            # 假设年发电量不变，将其折现
            total_energy_production_pv += avg_annual_load_served / ((1 + discount_rate) ** year)

        # --- 5. 计算 LCOE ---
        # LCOE = 总生命周期成本的现值 / 总生命周期发电量的现值
        lcoe = total_lcc_pv / total_energy_production_pv if total_energy_production_pv > 0 else float('inf')
        
        # 为了报告，我们仍然可以计算年化成本
        crf = (discount_rate * (1 + discount_rate)**lifetime) / ((1 + discount_rate)**lifetime - 1) if lifetime > 0 else 0
        total_annualized_cost = total_lcc_pv * crf
        
        return {"total_capex": total_initial_capex, "lcoe": lcoe, "annualized_cost": total_annualized_cost}


    def _analyze_economics(self, results_df: pd.DataFrame):
        print("\n--- 2. 经济性评估 ---")
        for scenario in self.config["ECONOMICS"]["cost_scenarios"]:
            economics = self._calculate_economics(results_df, scenario)
            print(f"  情景: {scenario:<12} | "
                  f"LCOE: {economics['lcoe']:.2f} 元/kWh | "
                  f"年化成本: {economics['annualized_cost']/10000:,.2f} 万元 | "
                  f"CAPEX: {economics['total_capex']/10000:,.0f} 万元")

    def _calculate_environment(self, perf_metrics: Dict[str, float]) -> Dict[str, float]:
        env_config = self.config["ENVIRONMENTAL"]
        pv_kw = self.config["PV"]["capacity_kw"]
        
        awes_count = self.config["AWES"]["count"]
        awes_kw = awes_count * self.config["AWES"]["nominal_power_kw_per_unit"]
        land_use_awes = awes_count * env_config["land_use_awes_m2_per_unit"]
        
        battery_kwh = self.config["BATTERY"]["capacity_kwh"]
        h2_power = max(self.config["HYDROGEN"]["electrolyzer"]["capacity_kw"], self.config["HYDROGEN"]["fuel_cell"]["capacity_kw"])
        
        land_use = pv_kw * env_config["land_use_pv_m2_per_kw"] + land_use_awes
        
        water = (perf_metrics.get("total_h2_produced_kg", 0) / self.config.get("LIFETIME_YEARS", 1)) * env_config["water_consumption_L_per_kg_h2"]
        
        carbon = (pv_kw * env_config["carbon_footprint"]["pv_per_kw"] + 
                  awes_kw * env_config["carbon_footprint"]["awes_per_kw"] + 
                  battery_kwh * env_config["carbon_footprint"]["battery_per_kwh"] + 
                  h2_power * env_config["carbon_footprint"]["hydrogen_sys_per_kw"])
                  
        return {"land_use_m2": land_use, "water_L_annual_actual": water, "carbon_footprint_kgCO2e": carbon}

    def _analyze_environment(self, perf_metrics: Dict[str, float]):
        print("\n--- 3. 环境影响评估 ---")
        env_impact = self._calculate_environment(perf_metrics)
        print(f"  预计总土地占用: {env_impact['land_use_m2']:,.0f} m^2 (约 {env_impact['land_use_m2']/666.7:.1f} 亩)")
        print(f"  年均水资源消耗 (制氢): {env_impact['water_L_annual_actual']:,.0f} L/year")
        print(f"  设备生命周期隐含碳足迹: {env_impact['carbon_footprint_kgCO2e']/1000:,.2f} 吨CO2e")

    def _plot_results_single_year(self, results_df: pd.DataFrame):

        print("\n--- 正在生成可视化图表 ---")
        # 风格切换点
        sns.set_theme(style="darkgrid", palette="colorblind") 

        first_year = self.config["SIMULATION_YEARS"][0]
        year_df = results_df[results_df.index.year == first_year].copy()
        
        # --- 图1: 全年逐时储能状态变化 ---
        year_df['battery_soc_percent'] = (year_df['battery_soc_kwh'] / self.config["BATTERY"]["capacity_kwh"]) * 100
        year_df['h2_storage_percent'] = (year_df['h2_storage_kg'] / self.config["HYDROGEN"]["tank"]["capacity_kg"]) * 100
        year_df['battery_soc_percent_smooth'] = year_df['battery_soc_percent'].rolling(window=24*7, center=True).mean()

        fig1, ax1 = plt.subplots(figsize=(20, 10))
        
        ax1.plot(year_df.index, year_df['battery_soc_percent'], label='电池 SOC (逐时)', color='deepskyblue', alpha=0.3, linewidth=1)
        ax1.plot(year_df.index, year_df['battery_soc_percent_smooth'], label='电池 SOC (7日趋势)', color='dodgerblue', linewidth=2.5)
        
        ax1.set_xlabel('日期', fontproperties=font, fontsize=14)
        ax1.set_ylabel('电池荷电状态 SOC (%)', fontproperties=font, fontsize=14, color='dodgerblue')
        ax1.tick_params(axis='y', labelcolor='dodgerblue', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.set_ylim(0, 100)
        
        ax2 = ax1.twinx()
        ax2.plot(year_df.index, year_df['h2_storage_percent'], label='氢储量 (%)', color='lightgreen', linestyle='--', linewidth=2.5)
        ax2.set_ylabel('氢储罐储量 (%)', fontproperties=font, fontsize=14, color='lightgreen')
        ax2.tick_params(axis='y', labelcolor='lightgreen', labelsize=12)
        ax2.set_ylim(0, 100)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, prop=font, loc='best')
        
        fig1.suptitle('全年储能系统状态变化 (电池 & 氢)', fontproperties=title_font)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # --- 图2: 冬季典型周能源供需与储能状态 ---
        start_date = f'{first_year}-01-15'
        end_date = f'{first_year}-01-21'
        winter_week_df = year_df.loc[start_date:end_date].copy()
        
        gen = winter_week_df['generation_kwh']
        load = winter_week_df['load_kwh']

        fig2, ax1 = plt.subplots(figsize=(20, 10))
        
        ax1.plot(winter_week_df.index, gen, label='总发电 (kW)', color='lime', linewidth=2)
        ax1.plot(winter_week_df.index, load, label='用电负荷 (kW)', color='coral', linestyle='--', linewidth=2)
        
        ax1.fill_between(winter_week_df.index, load, gen, where=gen >= load, facecolor='green', alpha=0.3, interpolate=True, label='能量盈余')
        ax1.fill_between(winter_week_df.index, load, gen, where=gen < load, facecolor='red', alpha=0.3, interpolate=True, label='能量亏空')
        
        ax1.set_xlabel('日期', fontproperties=font, fontsize=14)
        ax1.set_ylabel('功率 (kW)', fontproperties=font, fontsize=14)
        ax1.axhline(0, color='white', linewidth=0.7, linestyle='--')
        ax1.set_ylim(bottom=min(0, load.min() - 20), top=max(gen.max(), load.max()) + 20)
        ax1.tick_params(axis='both', labelsize=12)

        ax3 = ax1.twinx()
        ax3.plot(winter_week_df.index, winter_week_df['battery_soc_percent'], label='电池SOC (%)', color='deepskyblue', linewidth=2.5)
        ax3.set_ylabel('荷电状态 SOC (%)', fontproperties=font, fontsize=14, color='deepskyblue')
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='y', labelcolor='deepskyblue', labelsize=12)
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, prop=font, loc='upper left', ncol=2)

        plt.title('冬季典型周能源供需与储能状态', fontproperties=title_font)
        fig2.tight_layout()
        plt.show()

    def _plot_seasonal_balance(self, results_df: pd.DataFrame):
        """
        调用外部模块绘制季节性的能量平衡图。
        """
        first_year = self.config["SIMULATION_YEARS"][0]
        
        # 绘制冬季典型周
        plot_energy_balance_stacked_area(
            results_df=results_df,
            config=self.config,
            start_date=f'{first_year}-01-15',
            end_date=f'{first_year}-01-21',
            title=f'{first_year}年 冬季典型周能量平衡分析'
        )
        
        # 绘制夏季典型周
        plot_energy_balance_stacked_area(
            results_df=results_df,
            config=self.config,
            start_date=f'{first_year}-07-15',
            end_date=f'{first_year}-07-21',
            title=f'{first_year}年 夏季典型周能量平衡分析'
        )
