"""
Entrenamiento DQN con curriculum learning en Breakout custom.
Fases sin reward shaping, terminando en configuración estándar.
"""
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from breakout_env import BreakoutEnv


PHASES = [
    {
        "name": "phase1_easy",
        "timesteps": 300_000,
        "env_kwargs": {
            "ball_speed": 0.6,
            "paddle_width": 1.5,
            "brick_rows": 2,
            "brick_cols": 6,
            "max_steps": 500,
            "reward_shaping": False,
        },
    },
    {
        "name": "phase2_medium",
        "timesteps": 600_000,
        "env_kwargs": {
            "ball_speed": 0.8,
            "paddle_width": 1.2,
            "brick_rows": 4,
            "brick_cols": 8,
            "max_steps": 1_000,
            "reward_shaping": False,
        },
    },
    {
        "name": "phase3_standard",
        "timesteps": 600_000,
        "env_kwargs": {
            "ball_speed": 1.0,
            "paddle_width": 1.0,
            "brick_rows": 6,
            "brick_cols": 10,
            "max_steps": 7_000,
            "reward_shaping": False,
        },
    },
]


def make_env(render_mode=None, **kwargs):
    """Devuelve una función creadora de entornos (necesaria para SB3)."""
    def _init():
        env = BreakoutEnv(render_mode=render_mode, **kwargs)
        env = Monitor(env)
        return env
    return _init


def build_vec_env(env_kwargs, n_stack=4):
    """Crea el vec env con frame stacking."""
    env = DummyVecEnv([make_env(**env_kwargs)])
    env = VecFrameStack(env, n_stack=n_stack)
    return env


def train_curriculum(
    phases=PHASES,
    model_dir="models",
    log_dir="logs",
    save_freq=50_000,
    eval_freq=10_000,
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = None
    total_so_far = 0

    for i, phase in enumerate(phases, start=1):
        phase_name = phase["name"]
        phase_steps = phase["timesteps"]
        env_kwargs = phase["env_kwargs"]

        # Crear entornos de training y eval para la fase
        train_env = build_vec_env(env_kwargs)
        eval_env = build_vec_env(env_kwargs)

        # Callbacks
        ckpt_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=model_dir,
            name_prefix=f"dqn_{phase_name}"
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, f"best_{phase_name}"),
            log_path=log_dir,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )

        if model is None:
            # Primera fase: crear modelo nuevo
            model = DQN(
                policy="CnnPolicy",
                env=train_env,
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
                verbose=1,
            )
        else:
            # Reutilizar pesos y cambiar entorno
            model.set_env(train_env)

        print(f"\n=== Fase {i}: {phase_name} | pasos: {phase_steps} ===")
        model.learn(
            total_timesteps=phase_steps,
            reset_num_timesteps=False,
            callback=[ckpt_cb, eval_cb],
            progress_bar=False,
        )

        total_so_far += phase_steps
        model.save(os.path.join(model_dir, f"dqn_{phase_name}_last.zip"))
        print(f"Fase {phase_name} completada. Pasos acumulados: {total_so_far}")

    # Guardar modelo final
    final_path = os.path.join(model_dir, "dqn_curriculum_final.zip")
    model.save(final_path)
    print(f"\nEntrenamiento completo. Modelo final guardado en: {final_path}")

    # Evaluación final (igual que train.py)
    print("\nEvaluando modelo final...")
    # Usamos el último eval_env creado en la última fase
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    train_curriculum()
