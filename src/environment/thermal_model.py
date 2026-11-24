"""
Model Termal Gedung dan Generator Kondisi Lingkungan (Weather & Occupancy).
Menggunakan pendekatan lumped capacitance 1R1C berbasis prinsip termodinamika ASHRAE.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np


class WeatherGenerator:
    """
    Generator profil cuaca (suhu luar ruangan dan radiasi matahari) realistis.
    Mendukung skenario standar tropis/musim panas, heatwave (ekstrem), dan sejuk (mild).
    """

    def __init__(
        self, 
        scenario: str = "standard", 
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        self.scenario = scenario
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(42)
        self._configure_scenario()

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng

    def _configure_scenario(self):
        if self.scenario == "heatwave":
            self.base_temp_mean = 32.5     # Suhu rata-rata harian (C)
            self.temp_amplitude = 6.0      # Variasi siang-malam (+- 6 C)
            self.peak_solar_w = 950.0      # Radiasi matahari puncak (W/m2)
            self.noise_std = 0.4
        elif self.scenario == "mild":
            self.base_temp_mean = 24.0
            self.temp_amplitude = 4.5
            self.peak_solar_w = 650.0
            self.noise_std = 0.3
        else:  # standard (tropis/musim panas)
            self.base_temp_mean = 28.5
            self.temp_amplitude = 5.5      # Rentang harian 23 C - 34 C
            self.peak_solar_w = 800.0
            self.noise_std = 0.35

    def get_weather(self, day_index: int, hour: float) -> Tuple[float, float]:
        """
        Mengembalikan (outdoor_temp_c, solar_radiation_w_m2) untuk waktu tertentu.
        """
        # Variasi antar hari (multi-day synoptic weather wave)
        synoptic_wave = 1.2 * np.sin(2.0 * np.pi * day_index / 3.5)
        
        # Fluktuasi diurnal: suhu puncak terjadi sekitar jam 14:00 - 15:00
        diurnal_phase = (hour - 14.5) / 24.0 * 2.0 * np.pi
        noise_t = float(self.rng.normal(0.0, self.noise_std))
        t_out = (
            self.base_temp_mean 
            + synoptic_wave 
            + self.temp_amplitude * np.cos(diurnal_phase) 
            + noise_t
        )

        # Radiasi matahari: hanya muncul saat matahari terbit hingga terbenam (06:00 - 18:00)
        if 6.0 <= hour <= 18.0:
            solar_phase = (hour - 6.0) / 12.0 * np.pi
            noise_sol = float(self.rng.normal(0.0, 15.0))
            solar_rad = self.peak_solar_w * np.sin(solar_phase) + noise_sol
            solar_rad = max(0.0, solar_rad)
        else:
            solar_rad = 0.0

        return float(t_out), float(solar_rad)


class OccupancyGenerator:
    """
    Generator profil okupansi kantor komersial.
    Membedakan hari kerja (Senin-Jumat) dan akhir pekan (Sabtu-Minggu).
    """

    def __init__(
        self, 
        max_occupants: int = 25, 
        scenario: str = "standard", 
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        self.max_occupants = max_occupants
        self.scenario = scenario
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(42)

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng

    def get_occupancy(self, day_index: int, hour: float) -> int:
        """
        Mengembalikan jumlah penghuni ruangan pada waktu tertentu.
        """
        is_weekend = (day_index % 7) >= 5  # Sabtu dan Minggu

        if self.scenario == "weekend":
            if 9.0 <= hour <= 14.0:
                return int(self.rng.choice([0, 1, 2]))
            return 0

        if is_weekend:
            if 10.0 <= hour <= 16.0:
                return int(self.rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
            return 0

        # Hari kerja (Senin - Jumat)
        if hour < 7.0 or hour >= 19.5:
            return 0
        elif 7.0 <= hour < 8.5:
            ratio = (hour - 7.0) / 1.5
            n = int(self.max_occupants * ratio * 0.7)
            offset = int(self.rng.integers(-1, 2))
            return max(0, min(self.max_occupants, n + offset))
        elif 8.5 <= hour < 12.0:
            peak = self.max_occupants if self.scenario == "high_occupancy" else int(self.max_occupants * 0.9)
            offset = int(self.rng.integers(-2, 3))
            return max(0, min(self.max_occupants, peak + offset))
        elif 12.0 <= hour < 13.0:
            lunch_ratio = 0.35 if self.scenario == "high_occupancy" else 0.25
            n = int(self.max_occupants * lunch_ratio)
            offset = int(self.rng.integers(-1, 2))
            return max(0, min(self.max_occupants, n + offset))
        elif 13.0 <= hour < 17.5:
            peak = self.max_occupants if self.scenario == "high_occupancy" else int(self.max_occupants * 0.85)
            offset = int(self.rng.integers(-2, 3))
            return max(0, min(self.max_occupants, peak + offset))
        elif 17.5 <= hour < 19.5:
            ratio = (19.5 - hour) / 2.0
            n = int(self.max_occupants * ratio * 0.6)
            offset = int(self.rng.integers(-1, 2))
            return max(0, min(self.max_occupants, n + offset))
        return 0


class BuildingThermalModel:
    """
    Model termodinamika zona gedung berbasis RC lumped capacitance (1R1C).
    Memodelkan perpindahan panas konduksi selubung, radiasi surya, beban internal,
    serta dinamika sistem HVAC pendingin.
    """

    def __init__(self, config: Dict):
        thermal_cfg = config.get("building_thermal", {})
        hvac_cfg = config.get("hvac", {})
        sim_cfg = config.get("simulation", {})

        self.c_eff = float(thermal_cfg.get("effective_heat_capacity_j_per_k", 1.2e7))
        self.r_env = float(thermal_cfg.get("thermal_resistance_k_per_w", 0.0045))
        self.shgc = float(thermal_cfg.get("solar_heat_gain_coeff", 0.45))
        self.window_area = float(thermal_cfg.get("window_area_m2", 20.0))
        self.heat_per_person = float(thermal_cfg.get("internal_heat_per_person_w", 100.0))
        self.equip_base_power = float(thermal_cfg.get("equipment_base_power_w", 300.0))
        self.equip_occ_power = float(thermal_cfg.get("equipment_occupied_power_w", 1200.0))

        # Parameter HVAC
        self.max_cooling_w = float(hvac_cfg.get("max_cooling_power_w", 12000.0))
        self.cop_nominal = float(hvac_cfg.get("cop_nominal", 3.5))
        self.cop_temp_coeff = float(hvac_cfg.get("cop_temp_coeff", 0.02))
        self.fan_power_w = float(hvac_cfg.get("fan_power_w", 250.0))
        self.cooling_efficiency = float(hvac_cfg.get("cooling_efficiency", 0.95))
        self.min_setpoint = float(hvac_cfg.get("min_setpoint_c", 20.0))
        self.max_setpoint = float(hvac_cfg.get("max_setpoint_c", 26.0))

        # Parameter waktu
        self.dt_minutes = float(sim_cfg.get("time_step_minutes", 15.0))
        self.dt_seconds = self.dt_minutes * 60.0

        # Kontroler termostat proporsional internal (gain W/K)
        self.kp_hvac = 2500.0

    def compute_heat_gains(
        self, 
        t_in: float, 
        t_out: float, 
        solar_rad: float, 
        n_occupants: int
    ) -> Tuple[float, float, float, float]:
        """
        Menghitung perolehan panas pasif (W).
        Mengembalikan: (q_envelope, q_internal, q_solar, q_total_passive)
        """
        q_envelope = (t_out - t_in) / self.r_env
        q_people = n_occupants * self.heat_per_person
        equip_factor = min(1.0, n_occupants / 10.0) if n_occupants > 0 else 0.1
        q_equip = self.equip_base_power + self.equip_occ_power * equip_factor
        q_internal = q_people + q_equip
        q_solar = solar_rad * self.window_area * self.shgc
        q_total_passive = q_envelope + q_internal + q_solar
        return q_envelope, q_internal, q_solar, q_total_passive

    def step_hvac(
        self, 
        t_in: float, 
        t_out: float, 
        setpoint_c: float
    ) -> Tuple[float, float, float]:
        """
        Menghitung kapasitas pendinginan termal (W), COP, dan daya listrik (W).
        Mengembalikan: (q_cooling_w, cop, p_electric_w)
        """
        temp_error = max(0.0, t_in - setpoint_c)
        if temp_error > 0.05:
            q_requested = self.kp_hvac * temp_error
            q_cooling = min(self.max_cooling_w, q_requested) * self.cooling_efficiency
        else:
            q_cooling = 0.0

        cop = max(1.5, self.cop_nominal * (1.0 - self.cop_temp_coeff * (t_out - 25.0)))

        if q_cooling > 0.0:
            p_compressor = q_cooling / cop
            p_electric = p_compressor + self.fan_power_w
        else:
            p_electric = 0.0

        return float(q_cooling), float(cop), float(p_electric)

    def next_indoor_temperature(
        self, 
        t_in: float, 
        q_total_passive: float, 
        q_cooling: float, 
        rng: Optional[np.random.Generator] = None,
        noise_std: float = 0.04
    ) -> float:
        """
        Memperbarui suhu ruangan untuk time step berikutnya berdasarkan hukum pertama termodinamika.
        C_eff * dT/dt = Q_passive - Q_cooling
        """
        net_heat_flow = q_total_passive - q_cooling
        dt_temperature = (net_heat_flow * self.dt_seconds) / self.c_eff
        
        noise = float(rng.normal(0.0, noise_std)) if (rng is not None and noise_std > 0) else 0.0
        new_t_in = t_in + dt_temperature + noise
        
        return float(np.clip(new_t_in, 10.0, 45.0))
