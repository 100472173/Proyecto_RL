"""
Curriculum + time-to-threshold:
- Avanza de fase cuando la media de reward en eval >= 90% del umbral.
- En la última fase espera al 100% del umbral.
- Mide cuántos timesteps y tiempo de pared tarda en alcanzarse el umbral final.
- No toca train.py ni train_curriculum.py.
"""
import os
import time
from typing import List, Dict, Any

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from breakout_env import BreakoutEnv


# Fases por defecto (sin shaping), terminando en estándar
DEFAULT_PHASES: List[Dict[str, Any]] = [
    {
        "name": "phase1_easy",
        "env_kwargs": {
            "ball_speed": 0.7,
            "paddle_width": 1.8,
            "brick_rows": 3,
            "brick_cols": 8,
            "max_steps": 6_000,
            "reward_shaping": False,
        },
    },
    {
        "name": "phase2_medium",
        "env_kwargs": {
            "ball_speed": 0.9,
            "paddle_width": 1.4,
            "brick_rows": 5,
            "brick_cols": 10,
            "max_steps": 8_000,
            "reward_shaping": False,
        },
    },
    {
        "name": "phase3_standard",
        "env_kwargs": {
            "ball_speed": 1.0,
            "paddle_width": 1.0,
            "brick_rows": 6,
            "brick_cols": 10,
            "max_steps": 10_000,
            "reward_shaping": False,
        },
    },
]


def make_env(render_mode=None, **kwargs):
    def _init():
        env = BreakoutEnv(render_mode=render_mode, **kwargs)
        env = Monitor(env)
        return env
    return _init


def build_vec_env(env_kwargs, n_stack=4):
    env = DummyVecEnv([make_env(**env_kwargs)])
    env = VecFrameStack(env, n_stack=n_stack)
    return env


def train_curriculum_time_to_threshold(
    phases: List[Dict[str, Any]] = DEFAULT_PHASES,
    threshold_score: float = 15.0,
    threshold_eval_episodes: int = 5,
    eval_freq: int = 10_000,
    total_timesteps_cap: int = 1_000_000,
    model_dir: str = "models",
    log_dir: str = "logs",
    save_freq: int = 50_000,
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "time_to_threshold_curriculum.txt")

    model = None
    total_steps = 0
    start_time = time.time()
    reached = False

    for phase_idx, phase in enumerate(phases):
        env_kwargs = phase["env_kwargs"]
        phase_name = phase["name"]
        train_env = build_vec_env(env_kwargs)
        eval_env = build_vec_env(env_kwargs)

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=threshold_eval_episodes,
            deterministic=True,
        )

        # Objetivo de la fase: 90% del umbral salvo la última (100%)
        is_last = (phase_idx == len(phases) - 1)
        phase_target = threshold_score if is_last else threshold_score * 0.9

        # Crear/continuar modelo
        if model is None:
            model = DQN(
                policy='CnnPolicy',
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
            model.set_env(train_env)

        ckpt_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=model_dir,
            name_prefix=f"dqn_{phase_name}"
        )

        print(f"\n=== {phase_name} | target={phase_target} | eval cada {eval_freq} pasos ===")

        while total_steps < total_timesteps_cap:
            # Entrenar en bloques de eval_freq para poder evaluar y decidir transición
            model.learn(
                total_timesteps=eval_freq,
                reset_num_timesteps=False,
                callback=[ckpt_cb, eval_cb],
                progress_bar=False,
            )
            total_steps += eval_freq

            mean_reward, _ = evaluate_policy(
                model, eval_env, n_eval_episodes=threshold_eval_episodes, deterministic=True
            )
            elapsed = time.time() - start_time
            print(
                f"[Eval] steps={total_steps:,}  mean_reward={mean_reward:.2f}  "
                f"target={phase_target:.2f}  elapsed={elapsed:.1f}s"
            )

            if mean_reward >= phase_target:
                msg = (
                    f"Fase {phase_name} superada en {total_steps:,} steps | "
                    f"mean_reward={mean_reward:.2f} | elapsed={elapsed:.1f}s"
                )
                print(msg)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                # Guardar checkpoint de fase
                model.save(os.path.join(model_dir, f"dqn_{phase_name}_last.zip"))
                if is_last:
                    reached = True
                break

            if total_steps >= total_timesteps_cap:
                print("Se alcanzó el límite de pasos global sin llegar al target.")
                break

        # Limpia entornos de eval para no consumir recursos en la siguiente fase
        eval_env.close()

        if reached or total_steps >= total_timesteps_cap:
            break

    # Guardar modelo final
    final_path = os.path.join(model_dir, "dqn_curriculum_time_to_threshold.zip")
    model.save(final_path)
    print(f"\nEntrenamiento terminado. Modelo guardado en: {final_path}")

    # Log final
    elapsed_total = time.time() - start_time
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"FINAL | reached={reached} | steps={total_steps:,} | elapsed={elapsed_total:.1f}s\n"
        )

    return model


if __name__ == "__main__":
    train_curriculum_time_to_threshold(
        phases=DEFAULT_PHASES,
        threshold_score=15.0,  # objetivo experto
        threshold_eval_episodes=5,
        eval_freq=10_000,
        total_timesteps_cap=1_000_000,
        model_dir="models",
        log_dir="logs",
        save_freq=50_000,
    )
