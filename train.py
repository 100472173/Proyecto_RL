"""
Entrenamiento de DQN en el entorno Breakout custom.
"""
import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from breakout_env import BreakoutEnv


def make_env(render_mode=None, **kwargs):
    """Crea el entorno Breakout custom."""
    def _init():
        env = BreakoutEnv(render_mode=render_mode, **kwargs)
        env = Monitor(env)
        return env
    return _init


def train_dqn(
    total_timesteps=500_000,
    save_freq=50_000,
    model_dir='models',
    log_dir='logs',
    # Parámetros del entorno (modificables para curriculum learning)
    ball_speed=1.0,
    paddle_width=1.0,
    brick_rows=6,
    brick_cols=10,
    max_steps=10000,
    reward_shaping=False
):
    """
    Entrena un agente DQN en Breakout custom.
    
    Args:
        total_timesteps: Pasos totales de entrenamiento
        save_freq: Frecuencia de guardado de checkpoints
        model_dir: Directorio para modelos
        log_dir: Directorio para logs
        
        # Parámetros del entorno:
        ball_speed: Multiplicador velocidad pelota (1.0 = normal)
        paddle_width: Multiplicador ancho pala (1.0 = normal)
        brick_rows: Filas de ladrillos
        brick_cols: Columnas de ladrillos
        max_steps: Máximo pasos por episodio
        reward_shaping: Si dar recompensa extra por golpear pelota
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Parámetros del entorno
    env_kwargs = {
        "ball_speed": ball_speed,
        "paddle_width": paddle_width,
        "brick_rows": brick_rows,
        "brick_cols": brick_cols,
        "max_steps": max_steps,
        "reward_shaping": reward_shaping
    }
    
    # Crear entorno con frame stacking
    env = DummyVecEnv([make_env(**env_kwargs)])
    env = VecFrameStack(env, n_stack=4)
    
    # Entorno de evaluación
    eval_env = DummyVecEnv([make_env(**env_kwargs)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix='dqn_breakout'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True
    )
    
    # Crear agente DQN
    model = DQN(
        policy='CnnPolicy',
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log=log_dir,
        verbose=1
    )
    
    print("=" * 60)
    print("ENTRENAMIENTO DQN - BREAKOUT CUSTOM")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Dispositivo: {model.device}")
    print("-" * 60)
    print("Parámetros del entorno:")
    print(f"  Ball speed: {ball_speed}x")
    print(f"  Paddle width: {paddle_width}x")
    print(f"  Bricks: {brick_rows} x {brick_cols}")
    print(f"  Max steps: {max_steps}")
    print(f"  Reward shaping: {reward_shaping}")
    print("=" * 60)
    print("\nVer métricas en TensorBoard:")
    print(f"  tensorboard --logdir {log_dir}")
    print("=" * 60 + "\n")
    
    # Entrenar
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False
    )
    
    # Guardar modelo final
    final_path = os.path.join(model_dir, 'dqn_breakout_final')
    model.save(final_path)
    print(f"\nModelo final guardado en {final_path}")
    
    # Evaluación
    print("\nEvaluando modelo...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    eval_env.close()
    
    return model


if __name__ == '__main__':
    # Entrenamiento normal (sin curriculum)
    model = train_dqn(
        total_timesteps=500_000,
        save_freq=50_000,
        # Parámetros normales del juego
        ball_speed=1.0,
        paddle_width=1.0,
        brick_rows=6,
        brick_cols=10,
        max_steps=10000,
        reward_shaping=False
    )
