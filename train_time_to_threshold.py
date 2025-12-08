"""
Entrenamiento DQN con métrica de "time to threshold": mide en qué timestep
el agente supera un umbral de recompensa media (en evaluación).
No toca el train.py original.
"""
import os
import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
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


class TimeToThresholdCallback(BaseCallback):
    """Evalúa periódicamente y detiene el entrenamiento al superar un umbral.

    Guarda el log en log_dir/time_to_threshold.txt con:
    - steps cuando se supera el umbral
    - tiempo de pared transcurrido
    - mean_reward alcanzada
    """

    def __init__(self, threshold_score, eval_env, eval_freq=10_000, n_eval_episodes=5, log_dir=None, verbose=1):
        super().__init__(verbose)
        self.threshold_score = threshold_score
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = log_dir
        self.last_eval = 0
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Evaluar cada eval_freq timesteps
        if (self.num_timesteps - self.last_eval) < self.eval_freq:
            return True

        self.last_eval = self.num_timesteps
        mean_reward, _ = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True
        )

        if self.verbose:
            print(f"[TimeToThreshold] steps={self.num_timesteps:,} mean_reward={mean_reward:.2f}")

        if mean_reward >= self.threshold_score:
            elapsed = time.time() - self.start_time if self.start_time else 0.0
            msg = (
                f"Umbral alcanzado: mean_reward={mean_reward:.2f} >= {self.threshold_score} | "
                f"steps={self.num_timesteps:,} | elapsed={elapsed:.1f}s"
            )
            print(msg)
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(os.path.join(self.log_dir, "time_to_threshold.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            return False  # detener entrenamiento

        return True


def train_time_to_threshold(
    total_timesteps=500_000,
    threshold_score=15.0,
    threshold_eval_episodes=5,
    save_freq=50_000,
    model_dir='models',
    log_dir='logs',
    # Parámetros del entorno
    ball_speed=1.0,
    paddle_width=1.0,
    brick_rows=6,
    brick_cols=10,
    max_steps=10000,
    reward_shaping=False,
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env_kwargs = {
        "ball_speed": ball_speed,
        "paddle_width": paddle_width,
        "brick_rows": brick_rows,
        "brick_cols": brick_cols,
        "max_steps": max_steps,
        "reward_shaping": reward_shaping,
    }

    # Entorno de entrenamiento y eval (igual que train.py)
    env = DummyVecEnv([make_env(**env_kwargs)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_env(**env_kwargs)])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Callbacks habituales
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

    # Callback de time-to-threshold
    ttt_callback = TimeToThresholdCallback(
        threshold_score=threshold_score,
        eval_env=eval_env,
        eval_freq=10_000,
        n_eval_episodes=threshold_eval_episodes,
        log_dir=log_dir,
        verbose=1,
    )

    callbacks = [checkpoint_callback, eval_callback, ttt_callback]

    # Agente DQN (mismos hiperparámetros que train.py)
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
    print("ENTRENAMIENTO DQN - TIME TO THRESHOLD")
    print("=" * 60)
    print(f"Total timesteps (máx): {total_timesteps:,}")
    print(f"Umbral de score medio: {threshold_score}")
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

    # Entrenar (se detiene si alcanza el umbral)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False
    )

    # Guardar modelo final (o el alcanzado si se detuvo antes)
    final_path = os.path.join(model_dir, 'dqn_breakout_time_to_threshold')
    model.save(final_path)
    print(f"\nModelo guardado en {final_path}")

    # Evaluación final
    print("\nEvaluando modelo...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    eval_env.close()

    return model


if __name__ == '__main__':
    train_time_to_threshold(
        total_timesteps=500_000,
        threshold_score=15.0,
        threshold_eval_episodes=5,
        save_freq=50_000,
        # Entorno estándar
        ball_speed=1.0,
        paddle_width=1.0,
        brick_rows=6,
        brick_cols=10,
        max_steps=10000,
        reward_shaping=False,
    )
