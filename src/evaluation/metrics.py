"""
Metrik Kinerja Sistem Kontrol HVAC Gedung Pintar.
Menghitung konsumsi energi, penghematan, kenyamanan termal, kestabilan, dan beban puncak.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


def compute_energy_saving(baseline_energy: float, ppo_energy: float) -> float:
    """
    Menghitung persentase penghematan energi PPO terhadap baseline:
    Energy Saving (%) = ((Baseline Energy - PPO Energy) / Baseline Energy) * 100
    """
    if baseline_energy <= 1e-6:
        return 0.0
    return float(((baseline_energy - ppo_energy) / baseline_energy) * 100.0)


def calculate_scenario_metrics(
    trajectory_df: pd.DataFrame, 
    controller_name: str,
    scenario_name: str,
    comfort_min: float = 22.0,
    comfort_max: float = 25.0,
) -> Dict[str, Any]:
    """
    Menghitung seluruh metrik standar dari trajectory data frame episode.
    """
    # 1. Total energy consumption (kWh)
    total_energy_kwh = float(trajectory_df["energy_step_kwh"].sum())

    # 2. Rata-rata suhu ruangan (C)
    avg_indoor_temp = float(trajectory_df["indoor_temp"].mean())
    std_indoor_temp = float(trajectory_df["indoor_temp"].std())

    # 3. Kenyamanan termal saat zona berpenghuni / jam kerja
    occupied_mask = trajectory_df["is_occupied"] == True
    if occupied_mask.sum() > 0:
        occ_records = trajectory_df[occupied_mask]
        # Persentase step yang melanggar batas kenyamanan
        violations = occ_records["is_comfort_violated"].sum()
        comfort_violation_pct = float((violations / len(occ_records)) * 100.0)
        
        # Degree-hours akumulasi deviasi
        # degree_deviation * hours_per_step (0.25 jam per step)
        degree_hours = float((occ_records["temp_deviation_c"] * 0.25).sum())
        avg_temp_occupied = float(occ_records["indoor_temp"].mean())
    else:
        comfort_violation_pct = 0.0
        degree_hours = 0.0
        avg_temp_occupied = avg_indoor_temp

    # 4. Total reward episode
    total_reward = float(trajectory_df["total_reward"].sum())
    avg_step_reward = float(trajectory_df["total_reward"].mean())

    # 5. Kestabilan kontrol HVAC (jumlah perubahan kontrol signifikan > 0.25 C atau delta action > 0.08)
    setpoint_diffs = np.abs(trajectory_df["setpoint"].diff().dropna())
    hvac_control_changes = int((setpoint_diffs > 0.25).sum())

    # 6. Peak energy demand (kW)
    peak_power_kw = float(trajectory_df["electric_power_kw"].max())
    avg_power_kw = float(trajectory_df["electric_power_kw"].mean())

    return {
        "controller": controller_name,
        "scenario": scenario_name,
        "total_energy_kwh": round(total_energy_kwh, 3),
        "avg_indoor_temp_c": round(avg_indoor_temp, 2),
        "std_indoor_temp_c": round(std_indoor_temp, 2),
        "avg_temp_occupied_c": round(avg_temp_occupied, 2),
        "comfort_violation_pct": round(comfort_violation_pct, 2),
        "degree_hours_c_h": round(degree_hours, 2),
        "total_reward": round(total_reward, 2),
        "avg_step_reward": round(avg_step_reward, 4),
        "hvac_control_changes": hvac_control_changes,
        "peak_power_kw": round(peak_power_kw, 3),
        "avg_power_kw": round(avg_power_kw, 3),
        "total_steps": len(trajectory_df),
    }
