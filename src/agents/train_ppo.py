"""
Pipeline Pelatihan Agen Reinforcement Learning (PPO) untuk Kontrol HVAC Gedung Pintar.
Menggunakan Stable-Baselines3 dengan callback logging performa dan penyimpanan model terbaik.
"""

import os
import sys

# Pastikan root direktori ada di sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import json
import random
import argparse
from typing import Dict, Any
import yaml
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.building_env import SmartBuildingEnv


class TrainingMetricsCallback(BaseCallback):
    """
    Callback khusus untuk merekam metrik reward, energi, dan kenyamanan selama training.
    """

    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_log = []
        self.running_rewards = []
        self.current_ep_reward = 0.0
        self.current_ep_length = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None:
            self.current_ep_reward += float(rewards[0])
            self.current_ep_length += 1

        if dones is not None and dones[0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            self.timesteps_log.append(self.num_timesteps)

            # Hitung moving average reward 20 episode terakhir
            window = min(len(self.episode_rewards), 20)
            avg_rew = float(np.mean(self.episode_rewards[-window:]))
            self.running_rewards.append(avg_rew)

            if len(self.episode_rewards) % 10 == 0:
                print(
                    f"[Step {self.num_timesteps:6d}] "
                    f"Episode {len(self.episode_rewards):4d} | "
                    f"Ep Reward: {self.current_ep_reward:8.2f} | "
                    f"Avg (20 ep): {avg_rew:8.2f}"
                )

            self.current_ep_reward = 0.0
            self.current_ep_length = 0

        return True


def set_all_seeds(seed: int):
    """Menetapkan seed ke seluruh generator angka acak untuk reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ppo(
    config_path: str = "config/experiment_config.yaml",
    total_timesteps: int = None,
    seed: int = None,
):
    """
    Fungsi utama pelatihan PPO.
    """
    # 1. Muat konfigurasi
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_seed = seed if seed is not None else int(config.get("project", {}).get("seed", 42))
    set_all_seeds(exp_seed)

    training_cfg = config.get("rl_training", {})
    timesteps = total_timesteps if total_timesteps is not None else int(training_cfg.get("total_timesteps", 100000))

    paths = config.get("paths", {})
    best_model_dir = paths.get("models_best", "models/best_model")
    checkpoints_dir = paths.get("models_checkpoints", "models/checkpoints")
    logs_dir = paths.get("outputs_logs", "outputs/logs")
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 70)
    print("MEMULAI PELATIHAN PROXIMAL POLICY OPTIMIZATION (PPO)")
    print(f"Total Timesteps  : {timesteps}")
    print(f"Random Seed      : {exp_seed}")
    print(f"Learning Rate    : {training_cfg.get('learning_rate', 0.0003)}")
    print(f"Batch Size       : {training_cfg.get('batch_size', 64)}")
    print(f"N Steps Horizon  : {training_cfg.get('n_steps', 2048)}")
    print(f"Discount (gamma) : {training_cfg.get('gamma', 0.99)}")
    print("=" * 70)

    # 2. Bangun Environment Pelatihan & Evaluasi
    train_env = DummyVecEnv([lambda: SmartBuildingEnv(config=config, scenario="standard", seed=exp_seed)])
    eval_env = DummyVecEnv([lambda: SmartBuildingEnv(config=config, scenario="standard", seed=exp_seed + 999)])

    # 3. Setup Callbacks
    metrics_cb = TrainingMetricsCallback(check_freq=500)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # 4. Inisialisasi Model PPO
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(training_cfg.get("learning_rate", 0.0003)),
        n_steps=int(training_cfg.get("n_steps", 2048)),
        batch_size=int(training_cfg.get("batch_size", 64)),
        n_epochs=int(training_cfg.get("n_epochs", 10)),
        gamma=float(training_cfg.get("gamma", 0.99)),
        gae_lambda=float(training_cfg.get("gae_lambda", 0.95)),
        clip_range=float(training_cfg.get("clip_range", 0.2)),
        ent_coef=float(training_cfg.get("ent_coef", 0.005)),
        vf_coef=float(training_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=exp_seed,
    )

    # 5. Jalankan Training
    start_time = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[metrics_cb, eval_cb],
        progress_bar=False,
    )
    training_duration = time.time() - start_time

    # 6. Simpan Model Akhir dan Checkpoint
    final_model_path = os.path.join(checkpoints_dir, "ppo_final.zip")
    model.save(final_model_path)
    
    # Salin juga ke best_model jika eval_callback belum menyimpan
    best_target = os.path.join(best_model_dir, "ppo_smart_building.zip")
    eval_best_path = os.path.join(best_model_dir, "best_model.zip")
    if os.path.exists(eval_best_path):
        import shutil
        shutil.copyfile(eval_best_path, best_target)
    else:
        model.save(best_target)

    # 7. Ekspor Riwayat Training ke CSV
    import pandas as pd
    history_df = pd.DataFrame({
        "episode": range(1, len(metrics_cb.episode_rewards) + 1),
        "timestep": metrics_cb.timesteps_log,
        "episode_reward": metrics_cb.episode_rewards,
        "running_reward_20ep": metrics_cb.running_rewards,
        "episode_length": metrics_cb.episode_lengths,
    })
    history_csv_path = os.path.join(logs_dir, "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)

    # 8. Catat Metadata Eksperimen
    metadata = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "total_timesteps": timesteps,
        "total_episodes": len(metrics_cb.episode_rewards),
        "training_duration_seconds": round(training_duration, 2),
        "training_duration_minutes": round(training_duration / 60.0, 2),
        "final_running_reward": float(metrics_cb.running_rewards[-1]) if metrics_cb.running_rewards else 0.0,
        "seed": exp_seed,
        "saved_model_paths": {
            "final": final_model_path,
            "best": best_target,
        },
    }
    meta_json_path = os.path.join(logs_dir, "experiment_metadata.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print("PELATIHAN PPO SELESAI")
    print(f"Durasi Training : {training_duration / 60.0:.2f} menit ({training_duration:.1f} detik)")
    print(f"Total Episode   : {len(metrics_cb.episode_rewards)}")
    print(f"Final Reward    : {metadata['final_running_reward']:.2f}")
    print(f"Model Tersimpan : {best_target}")
    print(f"History CSV     : {history_csv_path}")
    print("=" * 70)

    train_env.close()
    eval_env.close()
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih Agen PPO Smart Building HVAC")
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    train_ppo(config_path=args.config, total_timesteps=args.timesteps, seed=args.seed)
