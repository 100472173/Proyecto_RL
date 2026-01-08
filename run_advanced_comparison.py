"""
Script de comparación avanzada para Breakout con Reward Shaping y Curriculum Learning.
Ejecuta 4 variantes experimentales y genera un reporte comparativo justo.
"""
import os
import json
import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Tuple

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# Importamos las clases originales del usuario
from breakout_env import BreakoutEnv
from run_experiments import run_experiment, STANDARD_ENV, DQN_PARAMS
from metrics import generate_metrics_report

# =============================================================================
# 1. ENTORNO CON REWARD SHAPING AVANZADO
# =============================================================================

class ShapedBreakoutEnv(BreakoutEnv):
    """
    Extiende BreakoutEnv para añadir Reward Shaping avanzado.
    """
    def __init__(self, use_shaping: bool = True, **kwargs):

        kwargs.pop('reward_shaping', None)
        
        # Forzamos reward_shaping=False en el padre para manejarlo nosotros manualmente
        super().__init__(reward_shaping=False, **kwargs)
        
        self.use_shaping = use_shaping
        self.last_ball_y = 0.0
        self.last_potential = 0.0
        
        self.paddle_x = 0
        self.ball_x = 0
        self.ball_vy = 0
        self.last_ball_vy = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.last_ball_y = self.ball_y
        self.last_ball_vy = self.ball_vy
        
        dist = abs((self.paddle_x + self.paddle_width / 2) - self.ball_x)
        self.last_potential = -dist
        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = super().step(action)
        
        if not self.use_shaping:
            return obs, reward, terminated, truncated, info

        shaping_reward = 0.0

        current_dist = abs((self.paddle_x + self.paddle_width / 2) - self.ball_x)
        current_potential = -current_dist
        gamma = 0.99

        if self.ball_vy > 0:
            alignment_shaping = (gamma * current_potential - self.last_potential)
            shaping_reward += alignment_shaping * 0.1
        
        self.last_potential = current_potential

        if self.ball_vy < 0 and self.last_ball_vy > 0 and self.ball_y > self.paddle_y - 10:
             shaping_reward += 1.0  # Bonus por devolver la bola
             
             hit_offset = (self.ball_x - (self.paddle_x + self.paddle_width/2)) / (self.paddle_width/2)
             if abs(hit_offset) > 0.5:
                 shaping_reward += 0.5  # Incentivar tiros angulados

        self.last_ball_y = self.ball_y
        self.last_ball_vy = self.ball_vy

        # Sumar shaping al reward base
        total_reward = reward + shaping_reward
        
        return obs, total_reward, terminated, truncated, info

def make_shaped_env(use_shaping=False, **kwargs):
    def _init():
        env = ShapedBreakoutEnv(use_shaping=use_shaping, **kwargs)
        env = Monitor(env)
        return env
    return _init

def build_shaped_vec_env(env_kwargs, use_shaping=False):
    env = DummyVecEnv([make_shaped_env(use_shaping=use_shaping, **env_kwargs)])
    env = VecFrameStack(env, n_stack=4)
    return env

TOTAL_TIMESTEPS = 1_500_000

def get_experiment_configs():
    """Genera las 4 configuraciones requeridas."""

    # Curriculum Combinado
    combined_phases = [
        {
            "name": "phase1_easy",
            "timesteps": 200_000,
            "env_kwargs": {
                "ball_speed": 0.75, "paddle_width": 1.3, 
                "brick_rows": 4, "brick_cols": 8, "max_steps": 4_000
            }
        },
        {
            "name": "phase2_medium",
            "timesteps": 300_000,
            "env_kwargs": {
                "ball_speed": 0.85, "paddle_width": 1.2, 
                "brick_rows": 5, "brick_cols": 9, "max_steps": 5_000
            }
        },
        {
            "name": "phase3_almostStandard",
            "timesteps": 400_000,
            "env_kwargs": {
                "ball_speed": 0.95, "paddle_width": 1.1, 
                "brick_rows": 6, "brick_cols": 10, "max_steps": 6_000
            }
        },
        {
            "name": "phase4_standard",
            "timesteps": 600_000,
            "env_kwargs": STANDARD_ENV.copy()
        }
    ]

    baseline_phases = [{
        "name": "standard",
        "timesteps": TOTAL_TIMESTEPS,
        "env_kwargs": STANDARD_ENV.copy()
    }]

    experiments = {}

    # Solo ejecutar curriculum_with_shaping
    # Baseline
    # experiments["baseline_no_shaping"] = {
    #     "description": "Baseline clásico",
    #     "phases": baseline_phases,
    #     "use_shaping": False
    # }

    # # Baseline con  Reward Shaping
    # experiments["baseline_with_shaping"] = {
    #     "description": "Baseline con Reward Shaping",
    #     "phases": baseline_phases,
    #     "use_shaping": True
    # }

    # # Curriculum
    # experiments["curriculum_no_shaping"] = {
    #     "description": "Curriculum Combinado clásico",
    #     "phases": combined_phases,
    #     "use_shaping": False
    # }

    # Curriculum con Reward Shaping
    experiments["curriculum_with_shaping"] = {
        "description": "Curriculum Combinado + Reward Shaping",
        "phases": combined_phases,
        "use_shaping": True
    }

    return experiments


class FairEvalCallback(BaseCallback):

    def __init__(self, eval_env, eval_freq=10000):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_logs = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=5, deterministic=True
            )
            self.eval_logs.append({
                "step": self.num_timesteps,
                "mean_reward": mean_reward,
                "std_reward": std_reward
            })
            if self.verbose > 0:
                print(f"Evaluacion {self.num_timesteps} pasos: {mean_reward:.2f} +/- {std_reward:.2f}")
        return True

def run_comparison():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    configs = get_experiment_configs()
    results = {}

    fair_eval_env = build_shaped_vec_env(STANDARD_ENV, use_shaping=False)

    print(f"{'='*80}")
    print(f"INICIANDO COMPARATIVA DE 4 MODELOS")
    print(f"Métrica de Evaluación: Reward medio en entorno estándar")
    print(f"{'='*80}\n")

    for exp_name, config in configs.items():
        print(f"\n     Ejecutando: {exp_name.upper()}")
        print(f"     Desc: {config['description']}")
        
        # Crear estructura de directorios
        exp_dir = os.path.join(output_dir, exp_name)
        model_dir = os.path.join(exp_dir, "models")
        log_dir = os.path.join(exp_dir, "logs")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup del modelo
        model = None
        eval_callback = FairEvalCallback(fair_eval_env, eval_freq=10000)
        
        total_steps = 0
        use_shaping = config["use_shaping"]

        # Bucle de fases
        for phase in config["phases"]:
            phase_env_kwargs = phase["env_kwargs"]
            timesteps = phase["timesteps"]
            
            train_env = build_shaped_vec_env(phase_env_kwargs, use_shaping=use_shaping)
            
            if model is None:
                model = DQN(
                    env=train_env,
                    **DQN_PARAMS
                )
            else:
                model.set_env(train_env)
            
            # Entrenar
            model.learn(
                total_timesteps=timesteps, 
                reset_num_timesteps=False,
                callback=eval_callback,
                progress_bar=True
            )
            
            train_env.close()
            total_steps += timesteps

        # Guardar modelo
        model.save(os.path.join(model_dir, "dqn_final"))
        
        # Guardar métricas en formato compatible con analyze_results.py
        timesteps_list = [log["step"] for log in eval_callback.eval_logs]
        rewards_list = [log["mean_reward"] for log in eval_callback.eval_logs]
        rewards_std_list = [log["std_reward"] for log in eval_callback.eval_logs]
        
        training_metrics = {
            "timesteps": timesteps_list,
            "rewards": rewards_list,
            "rewards_std": rewards_std_list
        }
        
        metrics_path = os.path.join(log_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(training_metrics, f, indent=2)
        
        # Generar reporte de métricas
        report = generate_metrics_report(
            experiment_name=exp_name,
            rewards=np.array(rewards_list),
            timesteps=np.array(timesteps_list),
            thresholds=[15.0, 30.0, 45.0]
        )
        report["final_reward_mean"] = float(rewards_list[-1])
        report["final_reward_std"] = float(rewards_std_list[-1])
        
        report_path = os.path.join(exp_dir, "metrics_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Guardar configuración
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        results[exp_name] = eval_callback.eval_logs
        print(f"✓ {exp_name} finalizado. Resultado final: {eval_callback.eval_logs[-1]['mean_reward']:.2f}")
        print(f"  Métricas guardadas: {metrics_path}")
        print(f"  Reporte guardado: {report_path}")

    # Generar Reporte Final
    report_lines = []
    report_lines.append("\n" + "="*80)
    report_lines.append("REPORTE COMPARATIVO FINAL")
    report_lines.append("="*80)
    report_lines.append(f"{'Experimento':<30} | {'Reward Final':<12} | {'Max Reward':<12}")
    report_lines.append("-" * 60)
    
    for name, logs in results.items():
        rewards = [l['mean_reward'] for l in logs]
        final_r = rewards[-1]
        max_r = max(rewards)
        line = f"{name:<30} | {final_r:<12.2f} | {max_r:<12.2f}"
        report_lines.append(line)
    
    report_lines.append("="*80)
    
    # Imprimir en consola
    for line in report_lines:
        print(line)
    
    # Guardar en archivo
    report_file = os.path.join(output_dir, "comparison_report.txt")
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n✓ Reporte comparativo guardado: {report_file}")
    print(f"✓ Todos los resultados guardados en: {output_dir}/")
    print(f"\nPara generar gráficas ejecuta:")
    print(f"  python analyze_results.py --results-dir {output_dir} --output-dir figures_comparison")
    
    fair_eval_env.close()

if __name__ == "__main__":
    run_comparison()