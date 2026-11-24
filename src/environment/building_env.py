"""
Gymnasium Environment untuk Simulasi Kontrol Energi Gedung Pintar Berbasis RL.
Memenuhi standar Gymnasium API dengan observation_space dan action_space kontinu.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.environment.thermal_model import (
    BuildingThermalModel,
    WeatherGenerator,
    OccupancyGenerator,
)


class SmartBuildingEnv(gym.Env):
    """
    Environment Gym untuk simulasi sistem termal gedung dan kontrol HVAC.
    
    Observation Space: Box(8,) [-1.0, 1.0]
        0: Indoor temperature (dinormalisasi ke [-1, 1])
        1: Outdoor temperature (dinormalisasi ke [-1, 1])
        2: Tingkat okupansi (dinormalisasi ke [-1, 1])
        3: Sinus waktu dalam sehari sin(2*pi*hour/24)
        4: Kosinus waktu dalam sehari cos(2*pi*hour/24)
        5: Indikator hari kerja (1.0 = hari kerja, -1.0 = akhir pekan)
        6: Setpoint sebelumnya (dinormalisasi ke [-1, 1])
        7: Indikator jam operasional kantor (1.0 = jam kerja sibuk, -1.0 = non-operasional)
        
    Action Space: Box(1,) [-1.0, 1.0]
        a in [-1.0, 1.0] merepresentasikan setpoint suhu pendinginan:
        T_set = 23.0 + a * 3.0 (rentang [20.0 C, 26.0 C])
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        config: Dict[str, Any],
        scenario: str = "standard",
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.scenario = scenario
        self._initial_seed = seed

        # Sub-konfigurasi
        sim_cfg = config.get("simulation", {})
        hvac_cfg = config.get("hvac", {})
        comfort_cfg = config.get("comfort", {})
        reward_cfg = config.get("reward", {})

        # Parameter waktu simulasi
        self.dt_minutes = int(sim_cfg.get("time_step_minutes", 15))
        self.steps_per_hour = 60 // self.dt_minutes
        self.steps_per_day = 24 * self.steps_per_hour
        self.episode_days = int(sim_cfg.get("episode_days", 7))
        self.max_steps = self.episode_days * self.steps_per_day

        # Parameter HVAC dan Comfort
        self.min_setpoint = float(hvac_cfg.get("min_setpoint_c", 20.0))
        self.max_setpoint = float(hvac_cfg.get("max_setpoint_c", 26.0))
        self.default_setpoint = float(hvac_cfg.get("default_setpoint_c", 23.0))

        self.comfort_temp_min = float(comfort_cfg.get("comfort_temp_min_c", 22.0))
        self.comfort_temp_max = float(comfort_cfg.get("comfort_temp_max_c", 25.0))
        self.unoccupied_temp_min = float(comfort_cfg.get("unoccupied_temp_min_c", 18.0))
        self.unoccupied_temp_max = float(comfort_cfg.get("unoccupied_temp_max_c", 28.0))
        self.work_hour_start = float(comfort_cfg.get("work_hour_start", 8))
        self.work_hour_end = float(comfort_cfg.get("work_hour_end", 18))
        self.max_occupants = int(comfort_cfg.get("max_occupants", 25))

        # Bobot Reward
        self.w_energy = float(reward_cfg.get("weight_energy", 1.0))
        self.w_comfort = float(reward_cfg.get("weight_comfort", 3.5))
        self.w_stability = float(reward_cfg.get("weight_stability", 0.2))
        self.energy_scale = float(reward_cfg.get("energy_scale_factor", 5.0))
        self.comfort_power = float(reward_cfg.get("comfort_power", 2.0))

        # Model Termal dan Generator
        self.thermal_model = BuildingThermalModel(config)
        self.weather_gen = WeatherGenerator(scenario=scenario)
        self.occupancy_gen = OccupancyGenerator(
            max_occupants=self.max_occupants, 
            scenario=scenario
        )

        # Spaces
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(8,), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )

        # State internal
        self.current_step = 0
        self.indoor_temp = 24.0
        self.prev_action = np.array([0.0], dtype=np.float32)
        self.prev_setpoint = self.default_setpoint

        # Statistik kumulatif episode
        self.episode_energy_kwh = 0.0
        self.episode_comfort_violation_steps = 0
        self.episode_degree_hours = 0.0
        self.episode_action_changes = 0
        self.episode_peak_power_kw = 0.0

        if seed is not None:
            self.reset(seed=seed)

    def _get_time_features(self, step: int) -> Tuple[int, float, bool, bool]:
        """
        Mengonversi step index menjadi (day_index, hour, is_workday, is_work_hour).
        """
        step_in_day = step % self.steps_per_day
        hour = (step_in_day * self.dt_minutes) / 60.0
        day_index = (step // self.steps_per_day) % 7
        is_workday = day_index < 5
        is_work_hour = is_workday and (self.work_hour_start <= hour < self.work_hour_end)
        return day_index, hour, is_workday, is_work_hour

    def _normalize_obs(
        self,
        t_in: float,
        t_out: float,
        n_occ: int,
        hour: float,
        is_workday: bool,
        prev_setpoint: float,
        is_work_hour: bool,
    ) -> np.ndarray:
        """
        Menghasilkan representasi observation ternormalisasi dalam rentang [-1.0, 1.0].
        """
        t_in_norm = np.clip((t_in - 23.0) / 7.0, -1.0, 1.0)
        t_out_norm = np.clip((t_out - 28.0) / 12.0, -1.0, 1.0)
        occ_norm = np.clip((n_occ / max(1, self.max_occupants)) * 2.0 - 1.0, -1.0, 1.0)
        hour_angle = 2.0 * np.pi * hour / 24.0
        sin_hour = np.sin(hour_angle)
        cos_hour = np.cos(hour_angle)
        workday_val = 1.0 if is_workday else -1.0
        work_hour_val = 1.0 if is_work_hour else -1.0
        setpoint_norm = np.clip((prev_setpoint - 23.0) / 3.0, -1.0, 1.0)

        obs = np.array(
            [
                t_in_norm,
                t_out_norm,
                occ_norm,
                sin_hour,
                cos_hour,
                workday_val,
                setpoint_norm,
                work_hour_val,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Sinkronisasi RNG generator anak dari Gymnasium generator
        self.weather_gen.set_rng(self.np_random)
        self.occupancy_gen.set_rng(self.np_random)

        self.current_step = 0
        # Inisialisasi suhu ruangan menggunakan self.np_random
        self.indoor_temp = float(self.np_random.uniform(23.5, 25.0))
        self.prev_action = np.array([0.0], dtype=np.float32)
        self.prev_setpoint = self.default_setpoint

        # Reset metrik akumulasi
        self.episode_energy_kwh = 0.0
        self.episode_comfort_violation_steps = 0
        self.episode_degree_hours = 0.0
        self.episode_action_changes = 0
        self.episode_peak_power_kw = 0.0

        day_index, hour, is_workday, is_work_hour = self._get_time_features(self.current_step)
        t_out, _ = self.weather_gen.get_weather(day_index, hour)
        n_occ = self.occupancy_gen.get_occupancy(day_index, hour)

        obs = self._normalize_obs(
            t_in=self.indoor_temp,
            t_out=t_out,
            n_occ=n_occ,
            hour=hour,
            is_workday=is_workday,
            prev_setpoint=self.prev_setpoint,
            is_work_hour=is_work_hour,
        )

        info = {
            "indoor_temp": self.indoor_temp,
            "outdoor_temp": t_out,
            "occupancy": n_occ,
            "step": self.current_step,
            "setpoint": self.prev_setpoint,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)
        action_val = float(action[0])
        mid_setpoint = (self.max_setpoint + self.min_setpoint) / 2.0
        span = (self.max_setpoint - self.min_setpoint) / 2.0
        target_setpoint = mid_setpoint + action_val * span

        action_delta = abs(action_val - float(self.prev_action[0]))
        if action_delta > 0.08:
            self.episode_action_changes += 1

        day_index, hour, is_workday, is_work_hour = self._get_time_features(self.current_step)
        t_out, solar_rad = self.weather_gen.get_weather(day_index, hour)
        n_occ = self.occupancy_gen.get_occupancy(day_index, hour)

        q_env, q_int, q_sol, q_passive = self.thermal_model.compute_heat_gains(
            t_in=self.indoor_temp,
            t_out=t_out,
            solar_rad=solar_rad,
            n_occupants=n_occ,
        )
        q_cooling, cop, p_electric = self.thermal_model.step_hvac(
            t_in=self.indoor_temp,
            t_out=t_out,
            setpoint_c=target_setpoint,
        )

        hours_per_step = self.dt_minutes / 60.0
        power_kw = p_electric / 1000.0
        energy_step_kwh = power_kw * hours_per_step
        self.episode_energy_kwh += energy_step_kwh
        self.episode_peak_power_kw = max(self.episode_peak_power_kw, power_kw)

        new_indoor_temp = self.thermal_model.next_indoor_temperature(
            t_in=self.indoor_temp,
            q_total_passive=q_passive,
            q_cooling=q_cooling,
            rng=self.np_random,
        )

        is_occupied_zone = n_occ > 0 or is_work_hour
        if is_occupied_zone:
            t_low = self.comfort_temp_min
            t_high = self.comfort_temp_max
        else:
            t_low = self.unoccupied_temp_min
            t_high = self.unoccupied_temp_max

        if new_indoor_temp < t_low:
            temp_deviation = t_low - new_indoor_temp
        elif new_indoor_temp > t_high:
            temp_deviation = new_indoor_temp - t_high
        else:
            temp_deviation = 0.0

        is_violated = temp_deviation > 0.05
        if is_violated and is_occupied_zone:
            self.episode_comfort_violation_steps += 1
            self.episode_degree_hours += temp_deviation * hours_per_step

        r_energy = - self.w_energy * (power_kw / self.energy_scale)
        occ_weight = 1.0 if is_occupied_zone else 0.2
        r_comfort = - self.w_comfort * occ_weight * (temp_deviation ** self.comfort_power)
        r_stability = - self.w_stability * (action_delta ** 2)

        total_reward = float(r_energy + r_comfort + r_stability)

        self.indoor_temp = new_indoor_temp
        self.prev_action = action.copy()
        self.prev_setpoint = target_setpoint
        self.current_step += 1

        terminated = bool(new_indoor_temp < 10.0 or new_indoor_temp > 45.0)
        truncated = bool(self.current_step >= self.max_steps)

        next_day, next_hour, next_workday, next_work_hour = self._get_time_features(self.current_step)
        next_tout, _ = self.weather_gen.get_weather(next_day, next_hour)
        next_nocc = self.occupancy_gen.get_occupancy(next_day, next_hour)

        next_obs = self._normalize_obs(
            t_in=self.indoor_temp,
            t_out=next_tout,
            n_occ=next_nocc,
            hour=next_hour,
            is_workday=next_workday,
            prev_setpoint=self.prev_setpoint,
            is_work_hour=next_work_hour,
        )

        info = {
            "step": self.current_step,
            "hour": hour,
            "day": day_index,
            "indoor_temp": self.indoor_temp,
            "outdoor_temp": t_out,
            "occupancy": n_occ,
            "setpoint": target_setpoint,
            "cooling_power_w": q_cooling,
            "electric_power_kw": power_kw,
            "energy_step_kwh": energy_step_kwh,
            "cop": cop,
            "temp_deviation_c": temp_deviation,
            "is_comfort_violated": is_violated,
            "is_occupied": is_occupied_zone,
            "reward_energy": r_energy,
            "reward_comfort": r_comfort,
            "reward_stability": r_stability,
            "total_reward": total_reward,
            "cumulative_energy_kwh": self.episode_energy_kwh,
            "cumulative_degree_hours": self.episode_degree_hours,
            "peak_power_kw": self.episode_peak_power_kw,
        }

        return next_obs, total_reward, terminated, truncated, info

    def render(self):
        pass
