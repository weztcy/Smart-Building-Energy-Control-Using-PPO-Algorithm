"""
Pengujian Environment Gymnasium dan Model Termal Gedung.
"""

import os
import yaml
import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env

from src.environment.building_env import SmartBuildingEnv
from src.environment.thermal_model import BuildingThermalModel, WeatherGenerator, OccupancyGenerator


@pytest.fixture
def config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_thermal_model_physics(config):
    """Menguji kestabilan dan arah perpindahan panas pada model termal."""
    model = BuildingThermalModel(config)
    
    # 1. Ketika suhu luar lebih panas dari suhu dalam, selubung harus mentransfer panas ke dalam
    q_env, q_int, q_sol, q_passive = model.compute_heat_gains(
        t_in=24.0, t_out=32.0, solar_rad=500.0, n_occupants=10
    )
    assert q_env > 0, "Konduksi harus positif jika luar lebih panas"
    assert q_int > 0, "Beban internal harus positif"
    assert q_sol > 0, "Radiasi matahari harus positif"
    assert q_passive > q_env

    # 2. HVAC pendinginan harus aktif bila Tin > Setpoint
    q_cool, cop, p_elec = model.step_hvac(t_in=26.0, t_out=30.0, setpoint_c=22.0)
    assert q_cool > 0.0, "HVAC harus mendinginkan saat suhu di atas setpoint"
    assert p_elec > 0.0, "Daya listrik HVAC harus lebih besar dari 0 saat aktif"
    assert 1.5 <= cop <= 5.0, f"COP di luar rentang realistis: {cop}"

    # 3. HVAC mati jika suhu ruangan sudah di bawah atau sama dengan setpoint
    q_cool_off, _, p_elec_off = model.step_hvac(t_in=21.0, t_out=30.0, setpoint_c=23.0)
    assert q_cool_off == 0.0
    assert p_elec_off == 0.0


def test_weather_and_occupancy_generators():
    """Menguji generator cuaca dan okupansi pada berbagai skenario."""
    weather = WeatherGenerator(scenario="standard", seed=42)
    t_out_day, solar_day = weather.get_weather(0, 13.0)
    t_out_night, solar_night = weather.get_weather(0, 2.0)

    # Siang harus lebih panas dan ada sinar matahari, malam tidak ada matahari
    assert t_out_day > t_out_night
    assert solar_day > 200.0
    assert solar_night == 0.0

    occ_gen = OccupancyGenerator(max_occupants=25, scenario="standard", seed=42)
    occ_workday_workhour = occ_gen.get_occupancy(0, 10.0)   # Senin 10:00
    occ_workday_night = occ_gen.get_occupancy(0, 23.0)      # Senin 23:00
    occ_weekend = occ_gen.get_occupancy(5, 10.0)            # Sabtu 10:00

    assert occ_workday_workhour > 10
    assert occ_workday_night == 0
    assert occ_weekend <= 2


def test_gymnasium_check_env(config):
    """Pemeriksaan ketat kepatuhan Gymnasium API menggunakan check_env."""
    env = SmartBuildingEnv(config=config, scenario="standard", seed=42)
    # Gymnasium check_env akan memeriksa spaces, bounds, step output dsb.
    check_env(env)


def test_environment_step_and_reset(config):
    """Menguji eksekusi reset dan episode beberapa step."""
    env = SmartBuildingEnv(config=config, scenario="standard", seed=42)
    obs, info = env.reset(seed=42)

    assert obs.shape == (8,)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert "indoor_temp" in info

    total_reward = 0.0
    for _ in range(96):  # Simulasi 1 hari (96 time steps @ 15 min)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        assert not np.isnan(reward), "Reward bernilai NaN!"
        assert not np.isnan(next_obs).any(), "Observation mengandung NaN!"
        assert 15.0 <= step_info["indoor_temp"] <= 35.0, f"Suhu tidak realistis: {step_info['indoor_temp']}"
        total_reward += reward

    assert not terminated
    assert step_info["cumulative_energy_kwh"] > 0.0


def test_scenarios_initialization(config):
    """Menguji bahwa seluruh skenario evaluasi dapat diinisialisasi tanpa error."""
    for scn in ["standard", "heatwave", "high_occupancy", "weekend", "mild"]:
        env = SmartBuildingEnv(config=config, scenario=scn, seed=123)
        obs, _ = env.reset()
        assert obs is not None
        next_obs, r, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        assert not term
