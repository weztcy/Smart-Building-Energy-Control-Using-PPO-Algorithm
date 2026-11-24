"""
Generator 10 Visualisasi Wajib Hasil Eksperimen Smart Building Energy Control Using PPO.
Menghasilkan grafik beresolusi tinggi (300 DPI) di direktori outputs/figures/.
"""

import os
import sys

# Pastikan root direktori ada di sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Atur gaya visualisasi modern & profesional
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})

# Palet warna konsisten
COLOR_PPO = "#1f77b4"           # Biru
COLOR_RULE = "#ff7f0e"          # Oranye
COLOR_OCC = "#2ca02c"           # Hijau
COLOR_FIXED = "#d62728"         # Merah
COLOR_OUTDOOR = "#7f7f7f"       # Abu-abu
COLOR_COMFORT_ZONE = "#2ca02c"  # Hijau lembut


def plot_01_training_reward_curve(history_csv: str, save_path: str):
    """1. Training reward curve dengan moving average."""
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    ax.plot(df["episode"], df["episode_reward"], alpha=0.35, color="#1f77b4", label="Episode Reward (Aktual)")
    ax.plot(df["episode"], df["running_reward_20ep"], color="#0b3c61", linewidth=2.5, label="Moving Average (20 Episode)")

    ax.set_title("Kurva Pembelajaran Reward Pelatihan PPO", fontweight="bold", pad=12)
    ax.set_xlabel("Episode Pelatihan")
    ax.set_ylabel("Total Reward Episode")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Grafik 1] Disimpan: {save_path}")


def plot_02_indoor_temperature_profile(traj_ppo_path: str, traj_rule_path: str, save_path: str):
    """2. Indoor temperature profile over time untuk PPO dan baseline."""
    df_ppo = pd.read_csv(traj_ppo_path)
    df_rule = pd.read_csv(traj_rule_path)

    # Ambil 3 hari pertama (288 langkah @ 15 menit) agar detail pola harian terlihat jelas
    steps = min(288, len(df_ppo))
    time_hours = np.arange(steps) * 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)

    # Zona kenyamanan termal 22 C - 25 C
    ax.axhspan(22.0, 25.0, color="#a1dab4", alpha=0.35, label="Rentang Nyaman ASHRAE 55 (22-25°C)")
    ax.axhline(22.0, color="#41b6c4", linestyle="--", linewidth=1.0)
    ax.axhline(25.0, color="#41b6c4", linestyle="--", linewidth=1.0)

    ax.plot(time_hours, df_rule["indoor_temp"].iloc[:steps], color=COLOR_RULE, linewidth=2.0, label="Rule-Based (Scheduled)")
    ax.plot(time_hours, df_ppo["indoor_temp"].iloc[:steps], color=COLOR_PPO, linewidth=2.2, label="PPO Agent (Adaptif)")

    ax.set_title("Profil Suhu Ruangan (Indoor Temperature): PPO vs Rule-Based (3 Hari Pertama)", fontweight="bold", pad=12)
    ax.set_xlabel("Waktu Simulasi (Jam)")
    ax.set_ylabel("Suhu Ruangan (°C)")
    ax.set_ylim(20.5, 27.5)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Grafik 2] Disimpan: {save_path}")


def plot_03_outdoor_vs_indoor_temperature(traj_ppo_path: str, save_path: str):
    """3. Hubungan suhu lingkungan luar dan suhu dalam ruangan."""
    df_ppo = pd.read_csv(traj_ppo_path)
    steps = min(288, len(df_ppo))
    time_hours = np.arange(steps) * 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, dpi=300)

    # Subplot 1: Deret waktu suhu luar vs suhu dalam
    ax1.plot(time_hours, df_ppo["outdoor_temp"].iloc[:steps], color="#e41a1c", linestyle=":", linewidth=2.0, label="Suhu Lingkungan Luar (Tout)")
    ax1.plot(time_hours, df_ppo["indoor_temp"].iloc[:steps], color=COLOR_PPO, linewidth=2.2, label="Suhu Ruangan Terkendali (Tin - PPO)")
    ax1.axhspan(22.0, 25.0, color="#a1dab4", alpha=0.3, label="Zona Nyaman (22-25°C)")
    ax1.set_ylabel("Suhu (°C)")
    ax1.set_title("Dinamika Suhu Luar vs Suhu Ruangan Terkendali PPO", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Subplot 2: Kebutuhan daya pendinginan HVAC
    ax2.plot(time_hours, df_ppo["electric_power_kw"].iloc[:steps], color="#377eb8", linewidth=2.0, label="Daya Listrik HVAC (kW)")
    ax2.set_ylabel("Daya HVAC (kW)")
    ax2.set_xlabel("Waktu Simulasi (Jam)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Grafik 3] Disimpan: {save_path}")


def plot_04_energy_consumption_comparison(summary_csv: str, save_path: str):
    """4. Perbandingan konsumsi energi PPO dan baseline per skenario."""
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    palette = {
        "PPO Agent": COLOR_PPO,
        "Rule-Based (Scheduled)": COLOR_RULE,
        "Occupancy-Aware": COLOR_OCC,
        "Fixed Thermostat (22.5 C)": COLOR_FIXED,
    }

    sns.barplot(
        data=df,
        x="scenario",
        y="total_energy_kwh",
        hue="controller",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Perbandingan Total Konsumsi Energi HVAC (7 Hari)", fontweight="bold", pad=12)
    ax.set_xlabel("Skenario Simulasi")
    ax.set_ylabel("Total Konsumsi Energi (kWh)")
    ax.legend(title="Pengendali", loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Beri label nilai pada bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}", (p.get_x() + p.get_width() / 2., height / 2.),
                        ha="center", va="center", fontsize=8.5, color="white", fontweight="bold", rotation=90)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Grafik 4] Disimpan: {save_path}")


def plot_05_cumulative_energy_consumption(metrics_dir: str, save_path: str):
    """5. Akumulasi konsumsi energi sepanjang periode simulasi 7 hari."""
    files = {
        "PPO Agent": "trajectory_standard_PPO_Agent.csv",
        "Rule-Based (Scheduled)": "trajectory_standard_Rule-Based_Scheduled.csv",
        "Occupancy-Aware": "trajectory_standard_Occupancy-Aware.csv",
        "Fixed Thermostat (22.5 C)": "trajectory_standard_Fixed_Thermostat_22.5_C.csv",
    }
    colors = {
        "PPO Agent": COLOR_PPO,
        "Rule-Based (Scheduled)": COLOR_RULE,
        "Occupancy-Aware": COLOR_OCC,
        "Fixed Thermostat (22.5 C)": COLOR_FIXED,
    }

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    for name, fname in files.items():
        fpath = os.path.join(metrics_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            time_days = np.arange(len(df)) * 0.25 / 24.0
            ax.plot(time_days, df["cumulative_energy_kwh"], label=name, color=colors[name], linewidth=2.3)

    ax.set_title("Akumulasi Konsumsi Energi HVAC Selama 7 Hari (Skenario Standar)", fontweight="bold", pad=12)
    ax.set_xlabel("Hari Simulasi")
    ax.set_ylabel("Energi Kumulatif (kWh)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Grafik 5] Disimpan: {save_path}")


def plot_06_hvac_control_behavior(traj_ppo_path: str, traj_rule_path: str, save_path: str):
    """6. Perilaku kontrol HVAC (setpoint & laju pendinginan)."""
    df_ppo = pd.read_csv(traj_ppo_path)
    df_rule = pd.read_csv(traj_rule_path)
    steps = min(192, len(df_ppo))  # 48 jam (2 hari)
    time_hours = np.arange(steps) * 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, dpi=300)

    # Setpoint
    ax1.plot(time_hours, df_rule["setpoint"].iloc[:steps], color=COLOR_RULE, linestyle="--", linewidth=2.0, label="Setpoint Rule-Based")
    ax1.plot(time_hours, df_ppo["setpoint"].iloc[:steps], color=COLOR_PPO, linewidth=2.2, label="Setpoint PPO")
    ax1.set_ylabel("Setpoint Suhu (°C)")
    ax1.set_title("Perilaku Kontrol HVAC: Dinamika Setpoint dan Daya Listrik", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Electric Power
    ax2.plot(time_hours, df_rule["electric_power_kw"].iloc[:steps], color=COLOR_RULE, alpha=0.8, linewidth=1.8, label="Daya Listrik Rule-Based (kW)")
    ax2.plot(time_hours, df_ppo["electric_power_kw"].iloc[:steps], color=COLOR_PPO, linewidth=2.0, label="Daya Listrik PPO (kW)")
    ax2.set_ylabel("Daya HVAC (kW)")
    ax2.set_xlabel("Waktu Simulasi (Jam)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Grafik 6] Disimpan: {save_path}")


def plot_07_occupancy_profile(traj_ppo_path: str, save_path: str):
    """7. Profil okupansi penghuni selama simulasi 7 hari."""
    df = pd.read_csv(traj_ppo_path)
    time_days = np.arange(len(df)) * 0.25 / 24.0

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)

    ax.fill_between(time_days, df["occupancy"], color="#41b6c4", alpha=0.45, label="Jumlah Penghuni Aktual")
    ax.plot(time_days, df["occupancy"], color="#081d58", linewidth=1.5)

    ax.set_title("Profil Okupansi Gedung Perkantoran Selama 7 Hari (Senin - Minggu)", fontweight="bold", pad=12)
    ax.set_xlabel("Hari Simulasi (0 = Senin ... 6 = Minggu)")
    ax.set_ylabel("Jumlah Penghuni (Orang)")
    ax.set_ylim(-1, 30)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Tambahkan anotasi akhir pekan
    ax.axvspan(5.0, 7.0, color="#feb24c", alpha=0.2, label="Akhir Pekan (Sabtu - Minggu)")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Grafik 7] Disimpan: {save_path}")


def plot_08_comfort_analysis(summary_csv: str, save_path: str):
    """8. Analisis kenyamanan termal (persentase pelanggaran & degree-hours)."""
    df = pd.read_csv(summary_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    palette = {
        "PPO Agent": COLOR_PPO,
        "Rule-Based (Scheduled)": COLOR_RULE,
        "Occupancy-Aware": COLOR_OCC,
        "Fixed Thermostat (22.5 C)": COLOR_FIXED,
    }

    # Pelanggaran kenyamanan (%)
    sns.barplot(
        data=df,
        x="scenario",
        y="comfort_violation_pct",
        hue="controller",
        palette=palette,
        ax=ax1,
        edgecolor="black",
    )
    ax1.set_title("Persentase Pelanggaran Batas Nyaman (% Waktu Berpenghuni)", fontweight="bold")
    ax1.set_xlabel("Skenario")
    ax1.set_ylabel("Pelanggaran Kenyamanan (%)")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Degree-hours akumulasi
    sns.barplot(
        data=df,
        x="scenario",
        y="degree_hours_c_h",
        hue="controller",
        palette=palette,
        ax=ax2,
        edgecolor="black",
    )
    ax2.set_title("Akumulasi Deviasi Kenyamanan (Degree-Hours °C·h)", fontweight="bold")
    ax2.set_xlabel("Skenario")
    ax2.set_ylabel("Degree-Hours (°C·h)")
    ax2.get_legend().remove()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Grafik 8] Disimpan: {save_path}")


def plot_09_reward_component_analysis(traj_ppo_path: str, save_path: str):
    """9. Analisis kontribusi komponen reward (energy, comfort, stability)."""
    df = pd.read_csv(traj_ppo_path)
    steps = min(288, len(df))
    time_hours = np.arange(steps) * 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    ax.plot(time_hours, df["reward_energy"].iloc[:steps], color="#e41a1c", linewidth=1.8, label="Penalti Energi")
    ax.plot(time_hours, df["reward_comfort"].iloc[:steps], color="#377eb8", linewidth=2.0, label="Penalti Kenyamanan")
    ax.plot(time_hours, df["reward_stability"].iloc[:steps], color="#4daf4a", linewidth=1.5, linestyle=":", label="Penalti Stabilitas Kontrol")
    ax.plot(time_hours, df["total_reward"].iloc[:steps], color="#000000", linewidth=2.2, label="Total Reward")

    ax.set_title("Dekomposisi Komponen Fungsi Reward PPO Sepanjang Waktu", fontweight="bold", pad=12)
    ax.set_xlabel("Waktu Simulasi (Jam)")
    ax.set_ylabel("Nilai Komponen Reward")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Grafik 9] Disimpan: {save_path}")


def plot_10_scenario_comparison(summary_csv: str, save_path: str):
    """10. Perbandingan metrik utama (Trade-off Energi vs Kenyamanan)."""
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    markers = {
        "PPO Agent": "o",
        "Rule-Based (Scheduled)": "s",
        "Occupancy-Aware": "^",
        "Fixed Thermostat (22.5 C)": "D",
    }
    colors = {
        "standard": "#1f77b4",
        "heatwave": "#d62728",
        "high_occupancy": "#ff7f0e",
        "weekend": "#2ca02c",
    }

    for _, row in df.iterrows():
        ctrl = row["controller"]
        scn = row["scenario"]
        ax.scatter(
            row["total_energy_kwh"],
            row["comfort_violation_pct"],
            label=f"{ctrl} - {scn}" if scn == "standard" else None,
            marker=markers.get(ctrl, "o"),
            color=colors.get(scn, "#333333"),
            s=120,
            edgecolors="black",
            zorder=4,
        )
        ax.annotate(
            f"{ctrl[:3]}-{scn[:4]}",
            (row["total_energy_kwh"] + 1.5, row["comfort_violation_pct"] + 0.3),
            fontsize=8.5,
        )

    ax.set_title("Trade-off Kinerja: Konsumsi Energi vs Pelanggaran Kenyamanan", fontweight="bold", pad=12)
    ax.set_xlabel("Total Konsumsi Energi (kWh) -> Lebih Rendah Lebih Baik")
    ax.set_ylabel("Pelanggaran Kenyamanan (%) -> Lebih Rendah Lebih Baik")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Tambahkan panah ideal Pareto
    ax.annotate(
        "Kondisi Ideal\n(Hemat & Nyaman)",
        xy=(df["total_energy_kwh"].min() * 0.95, df["comfort_violation_pct"].min()),
        xytext=(df["total_energy_kwh"].min() * 1.05, df["comfort_violation_pct"].max() * 0.7),
        arrowprops=dict(facecolor="green", shrink=0.08, width=2, headwidth=8),
        fontweight="bold",
        color="green",
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Grafik 10] Disimpan: {save_path}")


def generate_all_figures(
    config_path: str = "config/experiment_config.yaml",
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    fig_dir = paths.get("outputs_figures", "outputs/figures")
    metrics_dir = paths.get("outputs_metrics", "outputs/metrics")
    logs_dir = paths.get("outputs_logs", "outputs/logs")
    os.makedirs(fig_dir, exist_ok=True)

    history_csv = os.path.join(logs_dir, "training_history.csv")
    summary_csv = os.path.join(metrics_dir, "evaluation_summary.csv")
    traj_ppo = os.path.join(metrics_dir, "trajectory_standard_PPO_Agent.csv")
    traj_rule = os.path.join(metrics_dir, "trajectory_standard_Rule-Based_Scheduled.csv")

    print("=" * 70)
    print("MEMBUAT 10 VISUALISASI HASIL EKSPERIMEN LENGKAP")
    print(f"Direktori Output: {fig_dir}")
    print("=" * 70)

    # 1. Training reward curve
    if os.path.exists(history_csv):
        plot_01_training_reward_curve(history_csv, os.path.join(fig_dir, "01_training_reward_curve.png"))

    # 2. Indoor temperature profile
    if os.path.exists(traj_ppo) and os.path.exists(traj_rule):
        plot_02_indoor_temperature_profile(traj_ppo, traj_rule, os.path.join(fig_dir, "02_indoor_temperature_profile.png"))

    # 3. Outdoor vs indoor temperature
    if os.path.exists(traj_ppo):
        plot_03_outdoor_vs_indoor_temperature(traj_ppo, os.path.join(fig_dir, "03_outdoor_vs_indoor_temperature.png"))

    # 4. Energy consumption comparison
    if os.path.exists(summary_csv):
        plot_04_energy_consumption_comparison(summary_csv, os.path.join(fig_dir, "04_energy_consumption_comparison.png"))

    # 5. Cumulative energy consumption
    plot_05_cumulative_energy_consumption(metrics_dir, os.path.join(fig_dir, "05_cumulative_energy_consumption.png"))

    # 6. HVAC control behavior
    if os.path.exists(traj_ppo) and os.path.exists(traj_rule):
        plot_06_hvac_control_behavior(traj_ppo, traj_rule, os.path.join(fig_dir, "06_hvac_control_behavior.png"))

    # 7. Occupancy profile
    if os.path.exists(traj_ppo):
        plot_07_occupancy_profile(traj_ppo, os.path.join(fig_dir, "07_occupancy_profile.png"))

    # 8. Comfort analysis
    if os.path.exists(summary_csv):
        plot_08_comfort_analysis(summary_csv, os.path.join(fig_dir, "08_comfort_analysis.png"))

    # 9. Reward component analysis
    if os.path.exists(traj_ppo):
        plot_09_reward_component_analysis(traj_ppo, os.path.join(fig_dir, "09_reward_component_analysis.png"))

    # 10. Scenario comparison
    if os.path.exists(summary_csv):
        plot_10_scenario_comparison(summary_csv, os.path.join(fig_dir, "10_scenario_comparison.png"))

    print("=" * 70)
    print("SELURUH 10 VISUALISASI BERHASIL DIBUAT DENGAN SUKSES!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_figures()
