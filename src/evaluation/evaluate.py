"""
Pipeline Evaluasi Komparatif PPO vs Baseline pada Berbagai Skenario Gedung Pintar.
Menghasilkan trajectory CSV, tabel metrik agregat, dan evaluasi multi-seed independen.
"""

import os
import sys

# Pastikan root direktori ada di sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import argparse
from typing import Dict, List, Any
import yaml
import numpy as np
import pandas as pd
from tabulate import tabulate
from stable_baselines3 import PPO

from src.environment.building_env import SmartBuildingEnv
from src.baselines.rule_based import (
    RuleBasedController,
    OccupancyAwareController,
    FixedThermostatController,
)
from src.evaluation.metrics import calculate_scenario_metrics, compute_energy_saving


def run_episode(env: SmartBuildingEnv, controller: Any) -> pd.DataFrame:
    """
    Menjalankan 1 episode penuh simulasi gedung dan merekam seluruh data trajectory per step.
    """
    obs, info = env.reset()
    records = []
    done = False

    while not done:
        action, _ = controller.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        # Rekam data komprehensif per langkah
        record = {
            "step": step_info["step"],
            "day": step_info["day"],
            "hour": step_info["hour"],
            "indoor_temp": step_info["indoor_temp"],
            "outdoor_temp": step_info["outdoor_temp"],
            "occupancy": step_info["occupancy"],
            "setpoint": step_info["setpoint"],
            "cooling_power_w": step_info["cooling_power_w"],
            "electric_power_kw": step_info["electric_power_kw"],
            "energy_step_kwh": step_info["energy_step_kwh"],
            "cumulative_energy_kwh": step_info["cumulative_energy_kwh"],
            "cop": step_info["cop"],
            "temp_deviation_c": step_info["temp_deviation_c"],
            "is_comfort_violated": step_info["is_comfort_violated"],
            "is_occupied": step_info["is_occupied"],
            "reward_energy": step_info["reward_energy"],
            "reward_comfort": step_info["reward_comfort"],
            "reward_stability": step_info["reward_stability"],
            "total_reward": step_info["total_reward"],
        }
        records.append(record)
        obs = next_obs

    return pd.DataFrame(records)


def evaluate_all(
    config_path: str = "config/experiment_config.yaml",
    model_path: str = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    metrics_dir = paths.get("outputs_metrics", "outputs/metrics")
    best_model_dir = paths.get("models_best", "models/best_model")
    os.makedirs(metrics_dir, exist_ok=True)

    if model_path is None:
        target_path = os.path.join(best_model_dir, "ppo_smart_building.zip")
        if not os.path.exists(target_path):
            target_path = os.path.join(best_model_dir, "best_model.zip")
        model_path = target_path

    print("=" * 75)
    print("EVALUASI MULTI-SKENARIO KONTROL HVAC GEDUNG PINTAR")
    print(f"Model PPO: {model_path}")
    print("=" * 75)

    # Muat Agen PPO
    ppo_agent = PPO.load(model_path)

    # Inisialisasi Baseline Controllers
    controllers = {
        "PPO Agent": ppo_agent,
        "Rule-Based (Scheduled)": RuleBasedController(occupied_setpoint=22.5, setback_setpoint=26.0),
        "Occupancy-Aware": OccupancyAwareController(occupied_setpoint=22.5, unoccupied_setpoint=26.0),
        "Fixed Thermostat (22.5 C)": FixedThermostatController(constant_setpoint=22.5),
    }

    # Skenario pengujian yang representatif
    scenarios = ["standard", "heatwave", "high_occupancy", "weekend"]
    eval_seeds = config.get("project", {}).get("eval_seeds", [42, 101, 2024])

    all_metrics = []
    trajectories_sample = {}

    for scn in scenarios:
        print(f"\n>>> Mengevaluasi Skenario: [{scn.upper()}] ...")
        for ctrl_name, ctrl in controllers.items():
            seed_metrics = []

            for seed_idx, seed in enumerate(eval_seeds):
                # Buat environment independen untuk evaluasi yang adil
                env = SmartBuildingEnv(config=config, scenario=scn, seed=seed)
                traj_df = run_episode(env, ctrl)

                # Simpan trajectory seed pertama untuk visualisasi
                if seed_idx == 0:
                    traj_filename = f"trajectory_{scn}_{ctrl_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
                    traj_path = os.path.join(metrics_dir, traj_filename)
                    traj_df.to_csv(traj_path, index=False)
                    trajectories_sample[(scn, ctrl_name)] = traj_df

                m = calculate_scenario_metrics(
                    trajectory_df=traj_df,
                    controller_name=ctrl_name,
                    scenario_name=scn,
                    comfort_min=config["comfort"]["comfort_temp_min_c"],
                    comfort_max=config["comfort"]["comfort_temp_max_c"],
                )
                m["seed"] = seed
                seed_metrics.append(m)

            # Rata-rata dari multi-seed
            avg_m = {
                "controller": ctrl_name,
                "scenario": scn,
                "total_energy_kwh": round(float(np.mean([x["total_energy_kwh"] for x in seed_metrics])), 2),
                "energy_kwh_std": round(float(np.std([x["total_energy_kwh"] for x in seed_metrics])), 2),
                "avg_indoor_temp_c": round(float(np.mean([x["avg_indoor_temp_c"] for x in seed_metrics])), 2),
                "comfort_violation_pct": round(float(np.mean([x["comfort_violation_pct"] for x in seed_metrics])), 2),
                "degree_hours_c_h": round(float(np.mean([x["degree_hours_c_h"] for x in seed_metrics])), 2),
                "total_reward": round(float(np.mean([x["total_reward"] for x in seed_metrics])), 2),
                "hvac_control_changes": int(np.mean([x["hvac_control_changes"] for x in seed_metrics])),
                "peak_power_kw": round(float(np.mean([x["peak_power_kw"] for x in seed_metrics])), 2),
            }
            all_metrics.append(avg_m)

    # Konversi hasil evaluasi ke DataFrame
    summary_df = pd.DataFrame(all_metrics)

    # Hitung Energy Saving (%) terhadap Rule-Based Baseline untuk setiap skenario
    savings_list = []
    for _, row in summary_df.iterrows():
        scn = row["scenario"]
        base_row = summary_df[
            (summary_df["scenario"] == scn) & (summary_df["controller"] == "Rule-Based (Scheduled)")
        ]
        if not base_row.empty:
            base_energy = base_row.iloc[0]["total_energy_kwh"]
            saving = compute_energy_saving(base_energy, row["total_energy_kwh"])
        else:
            saving = 0.0
        savings_list.append(round(saving, 2))

    summary_df["energy_saving_vs_rule_based_pct"] = savings_list

    # Simpan CSV hasil evaluasi akhir
    summary_csv_path = os.path.join(metrics_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # Simpan juga format JSON terstruktur
    json_path = os.path.join(metrics_dir, "scenario_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    # Tampilkan tabel perbandingan yang rapi di terminal
    display_cols = [
        "scenario",
        "controller",
        "total_energy_kwh",
        "energy_saving_vs_rule_based_pct",
        "avg_indoor_temp_c",
        "comfort_violation_pct",
        "degree_hours_c_h",
        "peak_power_kw",
        "hvac_control_changes",
    ]
    print("\n" + "=" * 90)
    print("RINGKASAN METRIK EVALUASI KOMPARATIF")
    print("=" * 90)
    print(tabulate(summary_df[display_cols], headers="keys", tablefmt="github", showindex=False))
    print(f"\n[OK] Evaluasi selesai. Hasil disimpan di: {summary_csv_path}")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi Agen PPO dan Baseline")
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    evaluate_all(config_path=args.config, model_path=args.model)
