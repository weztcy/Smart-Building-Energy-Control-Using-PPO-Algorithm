"""
Baseline Controllers untuk Smart Building HVAC Control.
Kompatibel dengan antarmuka Stable-Baselines3 (predict(obs, state=None, deterministic=True)).
"""

from typing import Tuple, Optional, Any
import numpy as np


class RuleBasedController:
    """
    Rule-Based Controller standar (Scheduled Programmable Thermostat).
    Strategi industri konvensional:
    - Jam operasional kantor (Senin-Jumat, 08:00 - 18:00): setpoint nyaman 22.5 C.
    - Di luar jam kantor dan akhir pekan: setback mode 26.0 C (hemat energi).
    """

    def __init__(
        self, 
        occupied_setpoint: float = 22.5, 
        setback_setpoint: float = 26.0,
        work_hour_start: float = 8.0,
        work_hour_end: float = 18.0,
    ):
        self.occupied_setpoint = occupied_setpoint
        self.setback_setpoint = setback_setpoint
        self.work_hour_start = work_hour_start
        self.work_hour_end = work_hour_end

    def _setpoint_to_action(self, setpoint: float) -> np.ndarray:
        # Konversi setpoint [20.0 C, 26.0 C] ke aksi kontinu [-1.0, 1.0]
        # T_set = 23.0 + a * 3.0 => a = (T_set - 23.0) / 3.0
        action_val = (setpoint - 23.0) / 3.0
        return np.array([np.clip(action_val, -1.0, 1.0)], dtype=np.float32)

    def predict(
        self, 
        observation: np.ndarray, 
        state: Optional[Any] = None, 
        episode_start: Optional[Any] = None, 
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Memprediksi aksi berdasarkan feature observation:
        obs[5]: is_workday (1.0 = hari kerja, -1.0 = akhir pekan)
        obs[7]: is_work_hour (1.0 = jam operasional, -1.0 = non-operasional)
        """
        # Fitur is_work_hour langsung menandakan jam kerja hari kerja
        is_work_hour = observation[7] > 0.0

        if is_work_hour:
            target_setpoint = self.occupied_setpoint
        else:
            target_setpoint = self.setback_setpoint

        action = self._setpoint_to_action(target_setpoint)
        return action, state


class OccupancyAwareController:
    """
    Occupancy-Aware Rule-Based Controller.
    Menggunakan sensor okupansi real-time:
    - Jika ruangan terisi (okupansi > 0): setpoint nyaman 22.5 C.
    - Jika ruangan kosong: setback 26.0 C.
    """

    def __init__(
        self, 
        occupied_setpoint: float = 22.5, 
        unoccupied_setpoint: float = 26.0
    ):
        self.occupied_setpoint = occupied_setpoint
        self.unoccupied_setpoint = unoccupied_setpoint

    def _setpoint_to_action(self, setpoint: float) -> np.ndarray:
        action_val = (setpoint - 23.0) / 3.0
        return np.array([np.clip(action_val, -1.0, 1.0)], dtype=np.float32)

    def predict(
        self, 
        observation: np.ndarray, 
        state: Optional[Any] = None, 
        episode_start: Optional[Any] = None, 
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Any]]:
        # obs[2]: occ_norm = (n_occ / max_occ) * 2 - 1
        # Ruangan terisi jika occ_norm > -1.0 + 1e-4
        is_occupied = observation[2] > -0.95
        if is_occupied:
            target_setpoint = self.occupied_setpoint
        else:
            target_setpoint = self.unoccupied_setpoint

        action = self._setpoint_to_action(target_setpoint)
        return action, state


class FixedThermostatController:
    """
    Fixed Thermostat Controller (Tradisional tanpa otomasi jadwal).
    Menjaga setpoint konstan 22.5 C secara terus menerus selama 24/7.
    """

    def __init__(self, constant_setpoint: float = 22.5):
        self.constant_setpoint = constant_setpoint
        action_val = (self.constant_setpoint - 23.0) / 3.0
        self.fixed_action = np.array([np.clip(action_val, -1.0, 1.0)], dtype=np.float32)

    def predict(
        self, 
        observation: np.ndarray, 
        state: Optional[Any] = None, 
        episode_start: Optional[Any] = None, 
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Any]]:
        return self.fixed_action.copy(), state
