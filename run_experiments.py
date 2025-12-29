"""
Script unificado para ejecutar experimentos de curriculum learning.

Permite ejecutar cualquier experimento definido en experiment_configs.py
y recolecta métricas automáticamente para comparación.
"""
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Optional

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from breakout_env import BreakoutEnv
from callbacks import MetricsCollectorCallback
from experiment_configs import (
    get_experiment,
    get_all_experiments,
    DQN_PARAMS,
    STANDARD_ENV,
    print_experiment_summary,
)
from metrics import generate_metrics_report


def make_env(render_mode=None, **kwargs):
    """Crea el entorno Breakout custom."""
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


def run_experiment(
    experiment_name: str,
    output_dir: str = "results",
    eval_freq: int = 10_000,
    n_eval_episodes: int = 10,
    save_checkpoints: bool = True,
    checkpoint_freq: int = 100_000,
    seed: Optional[int] = None,
    verbose: int = 1,
) -> Dict:
    """
    Ejecuta un experimento completo.
    
    Args:
        experiment_name: Nombre del experimento (de experiment_configs.py)
        output_dir: Directorio base para resultados
        eval_freq: Frecuencia de evaluación
        n_eval_episodes: Episodios por evaluación
        save_checkpoints: Si guardar checkpoints intermedios
        checkpoint_freq: Frecuencia de checkpoints
        seed: Semilla para reproducibilidad
        verbose: Nivel de verbosidad
        
    Returns:
        Diccionario con resultados y métricas
    """
    # Obtener configuración
    config = get_experiment(experiment_name)
    
    # Crear directorios
    exp_dir = os.path.join(output_dir, experiment_name)
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Guardar configuración
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENTO: {experiment_name.upper()}")
    print("=" * 70)
    print(f"Descripción: {config['description']}")
    print(f"Timesteps totales: {config['total_timesteps']:,}")
    print(f"Número de fases: {len(config['phases'])}")
    print(f"Output: {exp_dir}")
    print("\n⚠️  ESTRATEGIA DE EVALUACIÓN:")
    print(f"   - Entrenamiento: Usa el entorno de cada fase (varía)")
    print(f"   - Evaluación: SIEMPRE en entorno ESTÁNDAR (6×10 ladrillos, ball_speed=1.0)")
    print(f"   - Esto permite comparar todos los métodos de forma justa")
    print("=" * 70 + "\n")
    
    # Crear entorno de evaluación (siempre en configuración estándar)
    eval_env = build_vec_env(STANDARD_ENV)
    
    # Callback de métricas
    metrics_callback = MetricsCollectorCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        log_path=os.path.join(log_dir, "training_metrics.json"),
        verbose=verbose,
    )
    
    model = None
    total_timesteps_so_far = 0
    
    # Ejecutar cada fase
    for i, phase in enumerate(config['phases'], start=1):
        phase_name = phase["name"]
        phase_timesteps = phase["timesteps"]
        env_kwargs = phase["env_kwargs"]
        
        print(f"\n{'='*70}")
        print(f"FASE {i}/{len(config['phases'])}: {phase_name}")
        print(f"{'='*70}")
        print(f"⚙️  Entrenamiento:")
        print(f"   - Ball speed: {env_kwargs['ball_speed']}")
        print(f"   - Paddle width: {env_kwargs['paddle_width']}")
        print(f"   - Bricks: {env_kwargs['brick_rows']}×{env_kwargs['brick_cols']}")
        print(f"   - Timesteps: {phase_timesteps:,}")
        print(f"\n📊 Evaluación: SIEMPRE en entorno ESTÁNDAR")
        print(f"   - Ball speed: {STANDARD_ENV['ball_speed']}")
        print(f"   - Paddle width: {STANDARD_ENV['paddle_width']}")
        print(f"   - Bricks: {STANDARD_ENV['brick_rows']}×{STANDARD_ENV['brick_cols']}")
        print(f"{'='*70}\n")
        
        # Crear entorno de training
        train_env = build_vec_env(env_kwargs)
        
        # Registrar transición de fase
        if model is not None:
            metrics_callback.record_phase_transition(phase_name, {
                "env_kwargs": env_kwargs,
                "phase_index": i,
            })
        
        # Callbacks
        callbacks = [metrics_callback]
        
        if save_checkpoints:
            ckpt_cb = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=model_dir,
                name_prefix=f"dqn_{phase_name}",
            )
            callbacks.append(ckpt_cb)
        
        if model is None:
            # Primera fase: crear modelo nuevo
            dqn_kwargs = DQN_PARAMS.copy()
            dqn_kwargs["tensorboard_log"] = log_dir
            
            if seed is not None:
                dqn_kwargs["seed"] = seed
            
            model = DQN(
                env=train_env,
                **dqn_kwargs,
            )
        else:
            # Reutilizar pesos y cambiar entorno
            model.set_env(train_env)
        
        # Entrenar
        model.learn(
            total_timesteps=phase_timesteps,
            reset_num_timesteps=False,
            callback=callbacks,
            progress_bar=True,
        )
        
        total_timesteps_so_far += phase_timesteps
        
        # Guardar modelo de fase
        phase_model_path = os.path.join(model_dir, f"dqn_{phase_name}.zip")
        model.save(phase_model_path)
        print(f"\nFase {phase_name} completada. Modelo guardado: {phase_model_path}")
        print(f"Timesteps acumulados: {total_timesteps_so_far:,}")
    
    # Guardar modelo final
    final_model_path = os.path.join(model_dir, "dqn_final.zip")
    model.save(final_model_path)
    print(f"\n✓ Modelo final guardado: {final_model_path}")
    
    # Guardar métricas
    metrics_callback.save_metrics()
    
    # Evaluación final
    print("\nEvaluación final en configuración estándar...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Reward final: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Generar reporte de métricas
    rewards, timesteps = metrics_callback.get_arrays()
    report = generate_metrics_report(
        experiment_name=experiment_name,
        rewards=rewards,
        timesteps=timesteps,
        thresholds=[15.0, 30.0, 45.0],  # 25%, 50%, 75% de 60 ladrillos
    )
    report["final_reward_mean"] = float(mean_reward)
    report["final_reward_std"] = float(std_reward)
    
    # Guardar reporte
    report_path = os.path.join(exp_dir, "metrics_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Reporte de métricas guardado: {report_path}")
    
    # Limpiar
    train_env.close()
    eval_env.close()
    
    return {
        "experiment_name": experiment_name,
        "model_path": final_model_path,
        "metrics": metrics_callback.get_metrics(),
        "report": report,
        "final_reward": (mean_reward, std_reward),
    }


def run_all_experiments(
    output_dir: str = "results",
    experiments: list = None,
    **kwargs,
) -> Dict[str, Dict]:
    """
    Ejecuta todos los experimentos (o una selección).
    
    Args:
        output_dir: Directorio base para resultados
        experiments: Lista de nombres de experimentos (None = todos)
        **kwargs: Argumentos adicionales para run_experiment
        
    Returns:
        Diccionario con resultados de cada experimento
    """
    all_experiments = get_all_experiments()
    
    if experiments is None:
        experiments = list(all_experiments.keys())
    
    results = {}
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("EJECUTANDO TODOS LOS EXPERIMENTOS")
    print(f"Experimentos: {experiments}")
    print(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    for exp_name in experiments:
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENTO: {exp_name}")
        print(f"{'#'*70}")
        
        try:
            result = run_experiment(
                experiment_name=exp_name,
                output_dir=output_dir,
                **kwargs,
            )
            results[exp_name] = result
        except Exception as e:
            print(f"ERROR en {exp_name}: {e}")
            results[exp_name] = {"error": str(e)}
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("RESUMEN DE EXPERIMENTOS")
    print("=" * 70)
    print(f"Duración total: {duration}")
    
    for exp_name, result in results.items():
        if "error" in result:
            print(f"  {exp_name}: ERROR - {result['error']}")
        else:
            mean, std = result["final_reward"]
            print(f"  {exp_name}: {mean:.2f} ± {std:.2f}")
    
    print("=" * 70)
    
    # Guardar resumen general
    summary_path = os.path.join(output_dir, "experiments_summary.json")
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "experiments": {
            name: {
                "final_reward_mean": r.get("final_reward", (0, 0))[0] if "final_reward" in r else None,
                "final_reward_std": r.get("final_reward", (0, 0))[1] if "final_reward" in r else None,
                "error": r.get("error"),
            }
            for name, r in results.items()
        }
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Resumen guardado: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar experimentos de curriculum learning"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default=None,
        help="Nombre del experimento a ejecutar (None = todos)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Frecuencia de evaluación",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para reproducibilidad",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar experimentos disponibles",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print_experiment_summary()
        return
    
    if args.experiment:
        run_experiment(
            experiment_name=args.experiment,
            output_dir=args.output_dir,
            eval_freq=args.eval_freq,
            seed=args.seed,
        )
    else:
        run_all_experiments(
            output_dir=args.output_dir,
            eval_freq=args.eval_freq,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
