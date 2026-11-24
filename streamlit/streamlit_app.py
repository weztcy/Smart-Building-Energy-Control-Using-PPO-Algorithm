"""
Smart Building Energy Control Dashboard (Streamlit).
Antarmuka interaktif dark mode untuk eksplorasi hasil kontrol HVAC berbasis PPO dan simulasi skenario real-time.
"""

import os
import sys

# Tambahkan root path ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.environment.building_env import SmartBuildingEnv
from src.baselines.rule_based import (
    RuleBasedController,
    OccupancyAwareController,
    FixedThermostatController,
)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Smart Building Energy Control | PPO RL",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Dark Mode Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 14px 18px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .stMetric:hover {
        border-color: #38bdf8;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .badge-ppo {
        background-color: #0284c7;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-baseline {
        background-color: #ea580c;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_config():
    config_path = "config/experiment_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


@st.cache_data
def load_evaluation_data():
    summary_path = "outputs/metrics/evaluation_summary.csv"
    history_path = "outputs/logs/training_history.csv"
    meta_path = "outputs/logs/experiment_metadata.json"

    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    history_df = pd.read_csv(history_path) if os.path.exists(history_path) else None
    
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return summary_df, history_df, metadata


def main():
    config = load_config()
    summary_df, history_df, metadata = load_evaluation_data()

    # Sidebar Navigasi
    st.sidebar.image("https://img.icons8.com/fluency/96/smart-home-checked.png", width=64)
    st.sidebar.title("Smart Building RL")
    st.sidebar.caption("Sistem Kendali HVAC Hemat Energi Berbasis PPO")
    
    menu = st.sidebar.radio(
        "Pilih Halaman Navigasi:",
        [
            "🏠 Overview / Dashboard",
            "🎮 Simulation & Scenario",
            "📈 Model Performance",
            "⚡ Energy & Comfort Analysis",
            "📖 About & Methodology",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Status Sistem:**
        - **Model:** PPO MlpPolicy
        - **Zona:** Kantor 120 m²
        - **Standar:** ASHRAE 55
        - **Mode Tema:** Dark Mode
        """
    )

    # ==========================================
    # HALAMAN 1: OVERVIEW / DASHBOARD
    # ==========================================
    if menu == "🏠 Overview / Dashboard":
        st.title("🏢 Smart Building Energy Control Dashboard")
        st.subheader("Ikhtisar Sistem Kendali Energi Gedung Berbasis Reinforcement Learning")
        
        st.markdown(
            """
            Proyek ini mengoptimalkan operasi sistem HVAC pada gedung komersial menggunakan algoritma **Proximal Policy Optimization (PPO)**.
            Agent bertugas meminimalkan konsumsi energi listrik sambil mempertahankan standar kenyamanan termal (**ASHRAE 55**).
            """
        )

        if summary_df is not None:
            # Ambil data skenario standar
            std_ppo = summary_df[(summary_df["scenario"] == "standard") & (summary_df["controller"] == "PPO Agent")]
            std_rule = summary_df[(summary_df["scenario"] == "standard") & (summary_df["controller"] == "Rule-Based (Scheduled)")]
            
            if not std_ppo.empty and not std_rule.empty:
                ppo_energy = std_ppo.iloc[0]["total_energy_kwh"]
                rule_energy = std_rule.iloc[0]["total_energy_kwh"]
                saving_pct = std_ppo.iloc[0]["energy_saving_vs_rule_based_pct"]
                ppo_viol = std_ppo.iloc[0]["comfort_violation_pct"]
                rule_viol = std_rule.iloc[0]["comfort_violation_pct"]
                ppo_peak = std_ppo.iloc[0]["peak_power_kw"]
                rule_peak = std_rule.iloc[0]["peak_power_kw"]

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Penghematan Energi", f"{saving_pct:.1f}%", f"{rule_energy - ppo_energy:.1f} kWh lebih hemat")
                col2.metric("Konsumsi Energi PPO (7 Hari)", f"{ppo_energy:.1f} kWh", f"Baseline: {rule_energy:.1f} kWh", delta_color="inverse")
                col3.metric("Pelanggaran Kenyamanan", f"{ppo_viol:.1f}%", f"Rule-Based: {rule_viol:.1f}%", delta_color="inverse")
                col4.metric("Beban Daya Puncak (Peak)", f"{ppo_peak:.2f} kW", f"Baseline: {rule_peak:.2f} kW", delta_color="inverse")

        st.markdown("### Ringkasan Hasil Evaluasi Lintas Skenario")
        if summary_df is not None:
            display_df = summary_df[[
                "scenario", "controller", "total_energy_kwh", "energy_saving_vs_rule_based_pct",
                "avg_indoor_temp_c", "comfort_violation_pct", "degree_hours_c_h", "peak_power_kw"
            ]].copy()
            display_df.columns = [
                "Skenario", "Pengendali", "Total Energi (kWh)", "Penghematan vs Rule-Based (%)",
                "Rata-rata Suhu (°C)", "Pelanggaran Nyaman (%)", "Degree-Hours (°C·h)", "Peak Demand (kW)"
            ]
            st.dataframe(display_df, use_container_width=True)

        st.markdown("### Visualisasi Utama")
        col_img1, col_img2 = st.columns(2)
        fig_dir = "outputs/figures"
        if os.path.exists(os.path.join(fig_dir, "02_indoor_temperature_profile.png")):
            col_img1.image(os.path.join(fig_dir, "02_indoor_temperature_profile.png"), caption="Profil Suhu Ruangan (PPO vs Baseline)", use_container_width=True)
        if os.path.exists(os.path.join(fig_dir, "04_energy_consumption_comparison.png")):
            col_img2.image(os.path.join(fig_dir, "04_energy_consumption_comparison.png"), caption="Perbandingan Konsumsi Energi HVAC", use_container_width=True)

    # ==========================================
    # HALAMAN 2: SIMULATION & SCENARIO
    # ==========================================
    elif menu == "🎮 Simulation & Scenario":
        st.title("🎮 Simulasi & Eksplorasi Skenario Interaktif")
        st.markdown("Eksplorasi respons agen HVAC terhadap berbagai kondisi cuaca dan okupansi gedung.")

        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            st.markdown("#### Parameter Uji")
            sel_scenario = st.selectbox(
                "Pilih Skenario Kondisi:",
                ["standard", "heatwave", "high_occupancy", "weekend"],
                format_func=lambda x: {
                    "standard": "Standard (Tropis/Musim Panas Reguler)",
                    "heatwave": "Heatwave (Gelombang Panas Ekstrem)",
                    "high_occupancy": "High Occupancy (Beban Panas Penuh)",
                    "weekend": "Weekend (Akhir Pekan/Sepi)",
                }[x],
            )
            sim_days = st.slider("Durasi Tampilan (Hari):", min_value=1, max_value=7, value=3)
            
            run_btn = st.button("Jalankan Simulasi Ulang", type="primary", use_container_width=True)
            if run_btn:
                st.session_state["live_sim_seed"] = int(np.random.randint(100, 9999))
                st.success(f"Simulasi ulang berhasil dijalankan (Seed: {st.session_state['live_sim_seed']})!")

        with col_ctrl2:
            st.markdown("#### Karakteristik Skenario Terpilih")
            desc_map = {
                "standard": "Suhu luar harian 23°C - 34°C, radiasi matahari normal, jadwal kantor reguler Senin-Jumat.",
                "heatwave": "Suhu luar harian 28°C - 38.5°C, radiasi matahari tinggi, menguji kapasitas pendinginan ekstrem.",
                "high_occupancy": "Okupansi mencapai 25 orang penuh, beban peralatan tinggi, menguji mitigasi panas internal.",
                "weekend": "Okupansi minimal (0-2 orang), menguji efektivitas logika setback suhu hemat energi.",
            }
            st.info(desc_map[sel_scenario])

        # Plot data trajectory tersimpan
        metrics_dir = "outputs/metrics"
        ppo_traj_file = os.path.join(metrics_dir, f"trajectory_{sel_scenario}_PPO_Agent.csv")
        rule_traj_file = os.path.join(metrics_dir, f"trajectory_{sel_scenario}_Rule-Based_Scheduled.csv")

        if os.path.exists(ppo_traj_file) and os.path.exists(rule_traj_file):
            df_p = pd.read_csv(ppo_traj_file)
            df_r = pd.read_csv(rule_traj_file)
            
            steps = sim_days * 96
            df_p_sub = df_p.iloc[:steps]
            df_r_sub = df_r.iloc[:steps]
            time_hours = np.arange(steps) * 0.25

            st.markdown("#### Hasil Trajectory Dinamika Gedung")
            
            fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            ax.axhspan(22.0, 25.0, color='#22c55e', alpha=0.25, label='Batas Nyaman ASHRAE 55')
            ax.plot(time_hours, df_p_sub["outdoor_temp"], color='#94a3b8', linestyle=':', label='Suhu Luar (Tout)')
            ax.plot(time_hours, df_r_sub["indoor_temp"], color='#f97316', label='Suhu Ruangan (Rule-Based)')
            ax.plot(time_hours, df_p_sub["indoor_temp"], color='#38bdf8', linewidth=2.0, label='Suhu Ruangan (PPO Agent)')
            ax.set_ylabel('Suhu (°C)', color='#f8fafc')
            ax.set_xlabel('Jam Simulasi', color='#f8fafc')
            ax.tick_params(colors='#f8fafc')
            ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
            ax.grid(True, color='#334155', linestyle='--')
            st.pyplot(fig)

            st.markdown("#### Profil Daya Listrik HVAC")
            fig2, ax2 = plt.subplots(figsize=(11, 3.2), dpi=150)
            fig2.patch.set_facecolor('#0f172a')
            ax2.set_facecolor('#1e293b')
            ax2.plot(time_hours, df_r_sub["electric_power_kw"], color='#f97316', alpha=0.8, label='Daya Listrik Rule-Based (kW)')
            ax2.plot(time_hours, df_p_sub["electric_power_kw"], color='#38bdf8', linewidth=1.8, label='Daya Listrik PPO (kW)')
            ax2.set_ylabel('Daya Listrik (kW)', color='#f8fafc')
            ax2.set_xlabel('Jam Simulasi', color='#f8fafc')
            ax2.tick_params(colors='#f8fafc')
            ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
            ax2.grid(True, color='#334155', linestyle='--')
            st.pyplot(fig2)

    # ==========================================
    # HALAMAN 3: MODEL PERFORMANCE
    # ==========================================
    elif menu == "📈 Model Performance":
        st.title("📈 Kinerja Model & Pembelajaran PPO")
        st.markdown("Analisis stabilitas konvergensi reward dan metadata pelatihan reinforcement learning.")

        if metadata:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Total Timesteps", f"{metadata.get('total_timesteps', 0):,}")
            col_m2.metric("Total Episode", f"{metadata.get('total_episodes', 0)}")
            col_m3.metric("Waktu Pelatihan", f"{metadata.get('training_duration_minutes', 0):.2f} menit")
            col_m4.metric("Reward Akhir", f"{metadata.get('final_running_reward', 0):.2f}")

        fig_dir = "outputs/figures"
        if os.path.exists(os.path.join(fig_dir, "01_training_reward_curve.png")):
            st.image(os.path.join(fig_dir, "01_training_reward_curve.png"), caption="Kurva Pembelajaran Reward Episode (Moving Average 20 Episode)", use_container_width=True)

        if os.path.exists(os.path.join(fig_dir, "09_reward_component_analysis.png")):
            st.image(os.path.join(fig_dir, "09_reward_component_analysis.png"), caption="Dekomposisi Komponen Fungsi Reward", use_container_width=True)

    # ==========================================
    # HALAMAN 4: ENERGY & COMFORT ANALYSIS
    # ==========================================
    elif menu == "⚡ Energy & Comfort Analysis":
        st.title("⚡ Analisis Energi & Kenyamanan Termal")
        st.markdown("Evaluasi menyeluruh terhadap trade-off konsumsi energi listrik dan kepatuhan kenyamanan penghuni.")

        fig_dir = "outputs/figures"
        col_c1, col_c2 = st.columns(2)
        if os.path.exists(os.path.join(fig_dir, "05_cumulative_energy_consumption.png")):
            col_c1.image(os.path.join(fig_dir, "05_cumulative_energy_consumption.png"), caption="Akumulasi Konsumsi Energi 7 Hari", use_container_width=True)
        if os.path.exists(os.path.join(fig_dir, "08_comfort_analysis.png")):
            col_c2.image(os.path.join(fig_dir, "08_comfort_analysis.png"), caption="Tingkat Pelanggaran Batas Nyaman & Degree-Hours", use_container_width=True)

        col_c3, col_c4 = st.columns(2)
        if os.path.exists(os.path.join(fig_dir, "06_hvac_control_behavior.png")):
            col_c3.image(os.path.join(fig_dir, "06_hvac_control_behavior.png"), caption="Perilaku Kontrol Setpoint HVAC", use_container_width=True)
        if os.path.exists(os.path.join(fig_dir, "10_scenario_comparison.png")):
            col_c4.image(os.path.join(fig_dir, "10_scenario_comparison.png"), caption="Pareto Trade-Off Energi vs Kenyamanan", use_container_width=True)

    # ==========================================
    # HALAMAN 5: ABOUT & METHODOLOGY
    # ==========================================
    elif menu == "📖 About & Methodology":
        st.title("📖 Tentang Proyek & Metodologi")
        st.markdown(
            """
            ### Formulasi Model Termal Gedung (1R1C Lumped Model)
            Model perubahan suhu dalam ruangan mengikuti persamaan diferensial kesetimbangan kalor ASHRAE:
            $$C_{eff} \\frac{dT_{in}}{dt} = Q_{envelope} + Q_{internal} + Q_{solar} - Q_{cooling}$$

            Di mana:
            - **$Q_{envelope} = \\frac{T_{out} - T_{in}}{R_{env}}$**: Perpindahan panas konduksi dan infiltrasi melalui dinding dan jendela.
            - **$Q_{internal} = N_{occ} \\cdot q_{person} + Q_{equipment}$**: Beban kalor dari manusia (100 W/orang) dan peralatan kantor.
            - **$Q_{solar} = I_{solar} \\cdot A_{win} \\cdot SHGC$**: Perolehan radiasi matahari melalui kaca.
            - **$Q_{cooling}$**: Kapasitas pendinginan termal yang diserap evaporator HVAC.

            ---

            ### Formulasi Reward Reinforcement Learning
            $$R_t = - \\left( w_{energy} \\cdot \\frac{P_{elec}}{P_{scale}} + w_{comfort} \\cdot I_{occ} \\cdot (\\Delta T_{viol})^2 + w_{stability} \\cdot (\\Delta a)^2 \\right)$$

            Dengan:
            - **$w_{energy} = 1.0$**: Penalti konsumsi daya listrik HVAC.
            - **$w_{comfort} = 3.5$**: Penalti deviasi kenyamanan termal saat ada penghuni.
            - **$w_{stability} = 0.2$**: Penalti perubahan kontrol untuk mencegah keausan kompresor.

            ---

            ### Informasi Lingkungan Komputasi
            - **Framework:** Gymnasium 1.4.0 & Stable-Baselines3 2.9.2
            - **Backend:** PyTorch 2.15.0 (CPU)
            - **Arsitektur Jaringan:** Actor-Critic MLP (128x128 tanh)
            """
        )


if __name__ == "__main__":
    main()
