"""
Script de comparación avanzada para Breakout con Reward Shaping y Curriculum Learning.
Ejecuta 4 variantes experimentales y genera un reporte comparativo justo.
"""
import os
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
        # CORRECCIÓN: Eliminamos 'reward_shaping' de kwargs si existe
        # para evitar el conflicto de "multiple values for keyword argument".
        kwargs.pop('reward_shaping', None)
        
        # Forzamos reward_shaping=False en el padre para manejarlo nosotros manualmente
        super().__init__(reward_shaping=False, **kwargs)
        
        self.use_shaping = use_shaping
        self.last_ball_y = 0.0
        self.last_potential = 0.0
        
        # Inicialización de seguridad para evitar errores en el primer reset
        self.paddle_x = 0
        self.ball_x = 0
        self.ball_vy = 0
        self.last_ball_vy = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.last_ball_y = self.ball_y
        self.last_ball_vy = self.ball_vy
        
        # Calcular potencial inicial: -Distancia
        dist = abs((self.paddle_x + self.paddle_width / 2) - self.ball_x)
        self.last_potential = -dist
        return obs, info

    def step(self, action):
        # 1. Ejecutar paso físico original
        obs, reward, terminated, truncated, info = super().step(action)
        
        if not self.use_shaping:
            return obs, reward, terminated, truncated, info

        # 2. Calcular componentes de Shaping
        shaping_reward = 0.0
        
        # A) Alignment (Potential-based Shaping)
        current_dist = abs((self.paddle_x + self.paddle_width / 2) - self.ball_x)
        current_potential = -current_dist
        gamma = 0.99
        
        # Solo aplicamos shaping de alineación si la pelota está bajando
        if self.ball_vy > 0:
            alignment_shaping = (gamma * current_potential - self.last_potential)
            shaping_reward += alignment_shaping * 0.1
        
        self.last_potential = current_potential

        # B) Return Bonus & Aiming
        # Detectamos si la velocidad vertical cambió de positiva (bajando) a negativa (subiendo)
        # y si está cerca de la altura de la pala.
        if self.ball_vy < 0 and self.last_ball_vy > 0 and self.ball_y > self.paddle_y - 10:
             shaping_reward += 1.0  # Bonus por devolver la bola
             
             # C) Aiming (Edge Bonus)
             hit_offset = (self.ball_x - (self.paddle_x + self.paddle_width/2)) / (self.paddle_width/2)
             if abs(hit_offset) > 0.5:
                 shaping_reward += 0.5  # Incentivar tiros angulados

        # Actualizar estado previo
        self.last_ball_y = self.ball_y
        self.last_ball_vy = self.ball_vy

        # Sumar shaping al reward base
        total_reward = reward + shaping_reward
        
        return obs, total_reward, terminated, truncated, info

# Wrapper para usar nuestra env class en el pipeline existente
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

# =============================================================================
# 2. DEFINICIÓN DE EXPERIMENTOS
# =============================================================================

TOTAL_TIMESTEPS = 500_000  # Reducido para ejemplo rápido, ajustar según necesidad

def get_experiment_configs():
    """Genera las 4 configuraciones requeridas."""
    
    # --- Configuración Base (Environment Parameters) ---
    # Curriculum Combinado: Fácil -> Medio -> Difícil
    combined_phases = [
        {
            "name": "phase1_easy",
            "timesteps": int(TOTAL_TIMESTEPS * 0.2),
            "env_kwargs": {
                "ball_speed": 0.6, "paddle_width": 1.5, 
                "brick_rows": 3, "brick_cols": 8, "max_steps": 5000
            }
        },
        {
            "name": "phase2_medium",
            "timesteps": int(TOTAL_TIMESTEPS * 0.3),
            "env_kwargs": {
                "ball_speed": 0.8, "paddle_width": 1.2, 
                "brick_rows": 4, "brick_cols": 10, "max_steps": 6000
            }
        },
        {
            "name": "phase3_standard",
            "timesteps": int(TOTAL_TIMESTEPS * 0.5),
            "env_kwargs": STANDARD_ENV.copy()
        }
    ]

    baseline_phases = [{
        "name": "standard",
        "timesteps": TOTAL_TIMESTEPS,
        "env_kwargs": STANDARD_ENV.copy()
    }]

    experiments = {}

    # 1. Baseline Sin Shaping
    experiments["baseline_no_shape"] = {
        "description": "Baseline clásico (Standard Env, No Shaping)",
        "phases": baseline_phases,
        "use_shaping": False
    }

    # 2. Baseline Con Shaping
    experiments["baseline_with_shape"] = {
        "description": "Baseline con Reward Shaping (Tracking+Return)",
        "phases": baseline_phases,
        "use_shaping": True
    }

    # 3. Curriculum Sin Shaping
    experiments["curriculum_no_shape"] = {
        "description": "Curriculum Combinado (Speed+Width+Bricks), No Shaping",
        "phases": combined_phases,
        "use_shaping": False
    }

    # 4. Curriculum Con Shaping
    experiments["curriculum_with_shape"] = {
        "description": "Curriculum Combinado + Reward Shaping",
        "phases": combined_phases,
        "use_shaping": True
    }

    return experiments

# =============================================================================
# 3. EJECUCIÓN Y EVALUACIÓN JUSTA
# =============================================================================

class FairEvalCallback(BaseCallback):
    """
    Callback que evalúa SIEMPRE en el entorno estándar SIN shaping.
    Esto asegura que la métrica de comparación sea justa para todos.
    """
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
                print(f"Fair Eval @ {self.num_timesteps}: {mean_reward:.2f} +/- {std_reward:.2f}")
        return True

def run_comparison():
    output_dir = "results_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    configs = get_experiment_configs()
    results = {}

    # Entorno de Evaluación Justa: SIEMPRE Estándar y SIN Shaping
    # Así comparamos manzanas con manzanas (rendimiento real en el juego)
    fair_eval_env = build_shaped_vec_env(STANDARD_ENV, use_shaping=False)

    print(f"{'='*80}")
    print(f"INICIANDO COMPARATIVA DE 4 MODELOS")
    print(f"Métrica de Evaluación: Reward Medio en Entorno Estándar (Sin Shaping)")
    print(f"{'='*80}\n")

    for exp_name, config in configs.items():
        print(f"\n---> Ejecutando: {exp_name.upper()}")
        print(f"     Desc: {config['description']}")
        
        # Setup del modelo
        model = None
        eval_callback = FairEvalCallback(fair_eval_env, eval_freq=10000)
        
        total_steps = 0
        use_shaping = config["use_shaping"]

        # Bucle de fases (para curriculum o baseline)
        for phase in config["phases"]:
            phase_env_kwargs = phase["env_kwargs"]
            timesteps = phase["timesteps"]
            
            # Construir entorno de entrenamiento (con o sin shaping según config)
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

        # Guardar resultados
        results[exp_name] = eval_callback.eval_logs
        model.save(os.path.join(output_dir, f"model_{exp_name}"))
        print(f"✓ {exp_name} finalizado. Resultado final (Fair): {eval_callback.eval_logs[-1]['mean_reward']:.2f}")

    # Generar Reporte Final
    print("\n" + "="*80)
    print("REPORTE COMPARATIVO FINAL")
    print("="*80)
    print(f"{'Experimento':<30} | {'Reward Final':<12} | {'Max Reward':<12}")
    print("-" * 60)
    
    for name, logs in results.items():
        rewards = [l['mean_reward'] for l in logs]
        final_r = rewards[-1]
        max_r = max(rewards)
        print(f"{name:<30} | {final_r:<12.2f} | {max_r:<12.2f}")
    
    fair_eval_env.close()

if __name__ == "__main__":
    run_comparison()