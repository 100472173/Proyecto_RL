"""
Teacher-Student Curriculum Learning basado en Learning Progress.

Implementación de Automatic Curriculum Learning usando un Multi-Armed Bandit
como "Teacher" que selecciona tareas basándose en el Learning Progress del estudiante.

Basado en:
- Graves et al., 2017 - "Automated Curriculum Learning for Neural Networks"
- Matiisen et al., 2017 - "Teacher-Student Curriculum Learning"

Arquitectura:
    - Teacher: Non-Stationary Multi-Armed Bandit (Boltzmann Exploration)
    - Student: DQN agent (stable-baselines3)
    - Arms/Tasks: Configuraciones de Breakout (Easy → Hard)
    - Feedback: Learning Progress = |mean(R_recent) - mean(R_past)|
"""

import os
import json
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from breakout_env import BreakoutEnv
from experiment_configs import DQN_PARAMS, STANDARD_ENV

import gymnasium as gym


# =============================================================================
# TASK REGISTRY: Definición de tareas de fácil a difícil
# =============================================================================

TASK_REGISTRY: Dict[int, Dict] = {
    0: {
        "name": "trivial",
        "description": "Muy fácil: bola lenta, paddle grande, pocos ladrillos",
        "ball_speed": 0.5,
        "paddle_width": 2.0,
        "brick_rows": 2,
        "brick_cols": 10,
        "max_steps": 5000,
        "reward_shaping": False,
    },
    1: {
        "name": "very_easy",
        "description": "Muy fácil+: bola lenta, paddle grande",
        "ball_speed": 0.5,
        "paddle_width": 1.8,
        "brick_rows": 3,
        "brick_cols": 10,
        "max_steps": 5000,
        "reward_shaping": False,
    },
    2: {
        "name": "easy",
        "description": "Fácil: bola media-lenta, paddle grande",
        "ball_speed": 0.65,
        "paddle_width": 1.5,
        "brick_rows": 4,
        "brick_cols": 10,
        "max_steps": 5500,
        "reward_shaping": False,
    },
    3: {
        "name": "easy_medium",
        "description": "Fácil-Medio: bola media, paddle medio-grande",
        "ball_speed": 0.75,
        "paddle_width": 1.3,
        "brick_rows": 4,
        "brick_cols": 10,
        "max_steps": 6000,
        "reward_shaping": False,
    },
    4: {
        "name": "medium",
        "description": "Medio: bola media, paddle normal+",
        "ball_speed": 0.85,
        "paddle_width": 1.15,
        "brick_rows": 5,
        "brick_cols": 10,
        "max_steps": 6500,
        "reward_shaping": False,
    },
    5: {
        "name": "medium_hard",
        "description": "Medio-Difícil: casi estándar",
        "ball_speed": 0.92,
        "paddle_width": 1.05,
        "brick_rows": 5,
        "brick_cols": 10,
        "max_steps": 7000,
        "reward_shaping": False,
    },
    6: {
        "name": "hard",
        "description": "Difícil: configuración casi estándar",
        "ball_speed": 0.97,
        "paddle_width": 1.0,
        "brick_rows": 6,
        "brick_cols": 10,
        "max_steps": 7000,
        "reward_shaping": False,
    },
    7: {
        "name": "standard",
        "description": "Estándar Atari: configuración normal",
        "ball_speed": 1.0,
        "paddle_width": 1.0,
        "brick_rows": 6,
        "brick_cols": 10,
        "max_steps": 7000,
        "reward_shaping": False,
    },
    8: {
        "name": "god_mode",
        "description": "God Mode: más difícil que Atari estándar",
        "ball_speed": 1.15,
        "paddle_width": 0.85,
        "brick_rows": 6,
        "brick_cols": 10,
        "max_steps": 7000,
        "reward_shaping": False,
    },
}


# =============================================================================
# BANDIT TEACHER: Multi-Armed Bandit con Learning Progress
# =============================================================================

class BanditTeacher:
    """
    Non-Stationary Multi-Armed Bandit que selecciona tareas basándose en
    Learning Progress usando exploración Boltzmann (Softmax).
    
    Learning Progress: LP_k = |mean(R_recent) - mean(R_past)|
    
    El valor absoluto es crucial porque:
    - Mejora en reward positivo = progreso
    - Reducción en reward negativo (perder más lento) = también progreso
    
    Política Boltzmann:
    P(k) = exp(LP_k / τ) / Σ exp(LP_j / τ)
    """
    
    def __init__(
        self,
        n_tasks: int,
        window_size: int = 20,
        recent_window: int = 5,
        temperature: float = 0.5,
        min_samples: int = 3,
        epsilon_exploration: float = 0.1,
    ):
        """
        Args:
            n_tasks: Número de tareas/brazos disponibles
            window_size: Tamaño total de la ventana de historial por tarea
            recent_window: Tamaño de la ventana "reciente" para calcular LP
            temperature: τ para Boltzmann exploration (menor = más greedy)
            min_samples: Mínimo de samples antes de calcular LP (evita ruido inicial)
            epsilon_exploration: Probabilidad de exploración aleatoria
        """
        self.n_tasks = n_tasks
        self.window_size = window_size
        self.recent_window = recent_window
        self.temperature = temperature
        self.min_samples = min_samples
        self.epsilon_exploration = epsilon_exploration
        
        # Historial de rewards por tarea
        self.reward_history: Dict[int, deque] = {
            k: deque(maxlen=window_size) for k in range(n_tasks)
        }
        
        # Tracking de métricas
        self.task_counts: Dict[int, int] = {k: 0 for k in range(n_tasks)}
        self.learning_progress: Dict[int, float] = {k: 0.0 for k in range(n_tasks)}
        self.task_selection_history: List[int] = []
        
        # Task actual
        self.current_task: int = 0
        
    def record_reward(self, task_id: int, reward: float) -> None:
        """Registra el reward de un episodio para una tarea."""
        self.reward_history[task_id].append(reward)
        self.task_counts[task_id] += 1
        
    def compute_learning_progress(self, task_id: int) -> float:
        """
        Calcula Learning Progress para una tarea.
        
        LP = |mean(R_recent) - mean(R_past)|
        
        Donde:
        - R_recent: últimos `recent_window` rewards
        - R_past: rewards anteriores en la ventana
        """
        history = list(self.reward_history[task_id])
        
        # Si no hay suficientes muestras, retornar LP alto para fomentar exploración
        if len(history) < self.min_samples:
            return 1.0  # Alta prioridad para tareas no exploradas
            
        if len(history) < self.recent_window + 1:
            # No hay suficiente historial para dividir en recent/past
            return 0.5
        
        # Dividir en recent y past
        recent = history[-self.recent_window:]
        past = history[:-self.recent_window]
        
        if len(past) == 0:
            return 0.5
            
        mean_recent = np.mean(recent)
        mean_past = np.mean(past)
        
        # Learning Progress = cambio absoluto
        lp = abs(mean_recent - mean_past)
        
        # Normalizar por la escala típica de rewards (opcional)
        # Esto evita que tareas con rewards muy altos dominen
        scale = max(abs(mean_recent), abs(mean_past), 1.0)
        lp_normalized = lp / scale
        
        return lp_normalized
    
    def compute_all_learning_progress(self) -> Dict[int, float]:
        """Calcula LP para todas las tareas."""
        for task_id in range(self.n_tasks):
            self.learning_progress[task_id] = self.compute_learning_progress(task_id)
        return self.learning_progress
    
    def sample_task(self) -> int:
        """
        Selecciona la siguiente tarea usando Boltzmann Exploration.
        
        P(k) = exp(LP_k / τ) / Σ exp(LP_j / τ)
        
        Returns:
            task_id: Índice de la tarea seleccionada
        """
        # Epsilon-greedy exploration para garantizar que todas las tareas se prueban
        if np.random.random() < self.epsilon_exploration:
            task = np.random.randint(0, self.n_tasks)
            self.current_task = task
            self.task_selection_history.append(task)
            return task
        
        # Calcular Learning Progress para todas las tareas
        lp_values = self.compute_all_learning_progress()
        
        # Boltzmann distribution (softmax)
        lp_array = np.array([lp_values[k] for k in range(self.n_tasks)])
        
        # Clip para estabilidad numérica
        lp_array = np.clip(lp_array, 0, 10)
        
        # Softmax con temperatura
        exp_values = np.exp(lp_array / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        # Muestrear
        task = np.random.choice(self.n_tasks, p=probabilities)
        
        self.current_task = task
        self.task_selection_history.append(task)
        
        return task
    
    def get_task_probabilities(self) -> np.ndarray:
        """Retorna las probabilidades actuales de selección."""
        lp_values = self.compute_all_learning_progress()
        lp_array = np.array([lp_values[k] for k in range(self.n_tasks)])
        lp_array = np.clip(lp_array, 0, 10)
        exp_values = np.exp(lp_array / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas del bandit para logging."""
        stats = {
            "current_task": self.current_task,
            "task_counts": dict(self.task_counts),
            "learning_progress": {k: round(v, 4) for k, v in self.learning_progress.items()},
            "probabilities": {k: round(p, 4) for k, p in enumerate(self.get_task_probabilities())},
            "recent_tasks": self.task_selection_history[-20:] if self.task_selection_history else [],
        }
        return stats


# =============================================================================
# CURRICULUM CALLBACK: Integración con SB3
# =============================================================================

class BanditCurriculumCallback(BaseCallback):
    """
    Callback de SB3 que:
    1. Detecta fin de episodio
    2. Reporta reward al BanditTeacher
    3. Reconfigura el entorno dinámicamente según la tarea seleccionada
    """
    
    def __init__(
        self,
        teacher: BanditTeacher,
        task_registry: Dict[int, Dict],
        log_freq: int = 10,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ):
        """
        Args:
            teacher: Instancia de BanditTeacher
            task_registry: Diccionario de configuraciones de tareas
            log_freq: Cada cuántos episodios loggear estadísticas
            eval_env: Entorno de evaluación (siempre en config estándar)
            eval_freq: Frecuencia de evaluación en timesteps
            n_eval_episodes: Episodios por evaluación
            verbose: Nivel de verbosidad
        """
        super().__init__(verbose)
        self.teacher = teacher
        self.task_registry = task_registry
        self.log_freq = log_freq
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Tracking
        self.episode_count = 0
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.last_eval_timestep = 0
        
        # Métricas para análisis
        self.metrics_history: List[Dict] = []
        self.eval_history: List[Dict] = []
        
    def _on_training_start(self) -> None:
        """Inicialización al comenzar el entrenamiento."""
        # Configurar la primera tarea
        initial_task = self.teacher.sample_task()
        self._reconfigure_env(initial_task)
        
        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("BANDIT CURRICULUM LEARNING - INICIADO")
            print("=" * 60)
            print(f"Número de tareas: {self.teacher.n_tasks}")
            print(f"Temperatura Boltzmann: {self.teacher.temperature}")
            print(f"Tarea inicial: {initial_task} ({self.task_registry[initial_task]['name']})")
            print("=" * 60 + "\n")
    
    def _on_step(self) -> bool:
        """Llamado en cada step del entrenamiento."""
        # Acumular reward del step
        # En VecEnv, los rewards están en self.locals["rewards"]
        rewards = self.locals.get("rewards", [0])
        self.current_episode_reward += rewards[0]
        
        # Detectar fin de episodio
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self._on_episode_end()
        
        # Evaluación periódica en entorno estándar
        if self.eval_env is not None:
            if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
                self._run_evaluation()
                self.last_eval_timestep = self.num_timesteps
        
        return True
    
    def _on_episode_end(self) -> None:
        """Procesamiento al finalizar un episodio."""
        self.episode_count += 1
        episode_reward = self.current_episode_reward
        
        # Registrar reward con el teacher
        current_task = self.teacher.current_task
        self.teacher.record_reward(current_task, episode_reward)
        self.episode_rewards.append(episode_reward)
        
        # Seleccionar siguiente tarea
        next_task = self.teacher.sample_task()
        
        # Reconfigurar entorno si cambió la tarea
        if next_task != current_task:
            self._reconfigure_env(next_task)
        
        # Logging periódico
        if self.episode_count % self.log_freq == 0:
            self._log_statistics(current_task, next_task, episode_reward)
        
        # Reset para siguiente episodio
        self.current_episode_reward = 0.0
    
    def _reconfigure_env(self, task_id: int) -> None:
        """
        Reconfigura el entorno de entrenamiento con la nueva tarea.
        
        Maneja la estructura DummyVecEnv -> VecTransposeImage -> BreakoutEnv
        """
        task_config = self.task_registry[task_id]
        
        # Navegar la estructura de wrappers para llegar al env base
        env = self.training_env
        
        # Si es VecTransposeImage, acceder al venv interno
        if hasattr(env, 'venv'):
            env = env.venv
        
        # Si es DummyVecEnv, acceder al env[0]
        if hasattr(env, 'envs'):
            base_env = env.envs[0]
            # Desenvolver cualquier wrapper adicional
            while hasattr(base_env, 'env'):
                base_env = base_env.env
        else:
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
        
        # Ahora base_env debería ser BreakoutEnv
        if hasattr(base_env, 'ball_speed_multiplier'):
            base_env.ball_speed_multiplier = task_config["ball_speed"]
            base_env.paddle_width_multiplier = task_config["paddle_width"]
            base_env.brick_rows = task_config["brick_rows"]
            base_env.brick_cols = task_config["brick_cols"]
            base_env.max_steps = task_config["max_steps"]
            base_env.reward_shaping = task_config.get("reward_shaping", False)
            
            if self.verbose > 1:
                print(f"  [Env Reconfigurado] Task {task_id}: {task_config['name']}")
        else:
            if self.verbose > 0:
                print(f"  [WARN] No se pudo reconfigurar el entorno (tipo: {type(base_env)})")
    
    def _run_evaluation(self) -> None:
        """Ejecuta evaluación en el entorno estándar."""
        if self.eval_env is None:
            return
            
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
        )
        
        eval_data = {
            "timesteps": self.num_timesteps,
            "episode": self.episode_count,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "current_task": self.teacher.current_task,
        }
        self.eval_history.append(eval_data)
        
        if self.verbose > 0:
            print(f"\n[EVAL @ {self.num_timesteps:,}] Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"  Task actual: {self.teacher.current_task} ({self.task_registry[self.teacher.current_task]['name']})")
    
    def _log_statistics(self, prev_task: int, next_task: int, episode_reward: float) -> None:
        """Loggea estadísticas del curriculum."""
        stats = self.teacher.get_statistics()
        probs = self.teacher.get_task_probabilities()
        
        # Guardar métricas
        metrics = {
            "timesteps": self.num_timesteps,
            "episode": self.episode_count,
            "episode_reward": episode_reward,
            "prev_task": prev_task,
            "next_task": next_task,
            **stats,
        }
        self.metrics_history.append(metrics)
        
        if self.verbose > 0:
            print(f"\n[Episode {self.episode_count}] Timesteps: {self.num_timesteps:,}")
            print(f"  Reward: {episode_reward:.2f} | Task: {prev_task} -> {next_task}")
            print(f"  Task counts: {stats['task_counts']}")
            print(f"  Learning Progress: ", end="")
            for k, lp in stats['learning_progress'].items():
                print(f"T{k}:{lp:.3f} ", end="")
            print()
            print(f"  Probabilities: ", end="")
            for k, p in enumerate(probs):
                print(f"T{k}:{p:.2%} ", end="")
            print()
    
    def _on_training_end(self) -> None:
        """Guardado de métricas al finalizar."""
        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("BANDIT CURRICULUM LEARNING - FINALIZADO")
            print("=" * 60)
            print(f"Episodios totales: {self.episode_count}")
            print(f"Timesteps totales: {self.num_timesteps:,}")
            print(f"\nDistribución final de tareas:")
            for k, count in self.teacher.task_counts.items():
                pct = count / max(sum(self.teacher.task_counts.values()), 1) * 100
                print(f"  Task {k} ({self.task_registry[k]['name']}): {count} ({pct:.1f}%)")
            print("=" * 60)


# =============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# =============================================================================

def train_with_bandit_curriculum(
    total_timesteps: int = 1_500_000,
    output_dir: str = "results/bandit_curriculum",
    temperature: float = 0.5,
    window_size: int = 20,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 5,
    seed: int = 42,
    verbose: int = 1,
) -> Dict:
    """
    Entrena un agente DQN usando Bandit Curriculum Learning.
    
    Args:
        total_timesteps: Total de timesteps de entrenamiento
        output_dir: Directorio para guardar resultados
        temperature: Temperatura para Boltzmann exploration
        window_size: Tamaño de ventana para calcular Learning Progress
        eval_freq: Frecuencia de evaluación (timesteps)
        n_eval_episodes: Episodios por evaluación
        seed: Semilla para reproducibilidad
        verbose: Nivel de verbosidad
        
    Returns:
        Dict con métricas y paths de modelos guardados
    """
    # Crear directorios
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("TEACHER-STUDENT CURRICULUM LEARNING - BANDIT METHOD")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Número de tareas: {len(TASK_REGISTRY)}")
    print(f"Temperatura Boltzmann: {temperature}")
    print(f"Ventana Learning Progress: {window_size}")
    print("=" * 70 + "\n")
    
    # Crear entorno de entrenamiento (comienza en tarea fácil)
    initial_task = TASK_REGISTRY[0]
    
    def make_train_env():
        return BreakoutEnv(
            ball_speed=initial_task["ball_speed"],
            paddle_width=initial_task["paddle_width"],
            brick_rows=initial_task["brick_rows"],
            brick_cols=initial_task["brick_cols"],
            max_steps=initial_task["max_steps"],
            reward_shaping=initial_task.get("reward_shaping", False),
        )
    
    train_env = DummyVecEnv([make_train_env])
    train_env = VecTransposeImage(train_env)
    
    # Crear entorno de evaluación (siempre en configuración estándar)
    def make_eval_env():
        return BreakoutEnv(**STANDARD_ENV)
    
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecTransposeImage(eval_env)
    
    # Crear el Bandit Teacher
    teacher = BanditTeacher(
        n_tasks=len(TASK_REGISTRY),
        window_size=window_size,
        recent_window=5,
        temperature=temperature,
        min_samples=3,
        epsilon_exploration=0.1,
    )
    
    # Crear el modelo DQN (Student)
    dqn_params = DQN_PARAMS.copy()
    dqn_params["tensorboard_log"] = log_dir
    dqn_params["seed"] = seed
    
    model = DQN(
        env=train_env,
        **dqn_params,
    )
    
    # Crear el callback del curriculum
    curriculum_callback = BanditCurriculumCallback(
        teacher=teacher,
        task_registry=TASK_REGISTRY,
        log_freq=10,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        verbose=verbose,
    )
    
    # Entrenar
    print("\n🎮 Iniciando entrenamiento con Bandit Curriculum...\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=curriculum_callback,
        progress_bar=True,
    )
    
    # Guardar modelo final
    final_model_path = os.path.join(model_dir, "dqn_bandit_final.zip")
    model.save(final_model_path)
    print(f"\n✓ Modelo final guardado: {final_model_path}")
    
    # Guardar métricas
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    metrics_data = {
        "config": {
            "total_timesteps": total_timesteps,
            "temperature": temperature,
            "window_size": window_size,
            "n_tasks": len(TASK_REGISTRY),
            "seed": seed,
        },
        "teacher_stats": teacher.get_statistics(),
        "curriculum_history": curriculum_callback.metrics_history,
        "eval_history": curriculum_callback.eval_history,
        "task_registry": {str(k): v for k, v in TASK_REGISTRY.items()},
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2, default=str)
    print(f"✓ Métricas guardadas: {metrics_path}")
    
    # Evaluación final
    print("\n📊 Evaluación final en entorno estándar...")
    final_mean, final_std = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Reward final: {final_mean:.2f} ± {final_std:.2f}")
    
    # Limpiar
    train_env.close()
    eval_env.close()
    
    return {
        "model_path": final_model_path,
        "metrics_path": metrics_path,
        "final_reward_mean": final_mean,
        "final_reward_std": final_std,
        "teacher_stats": teacher.get_statistics(),
    }


# =============================================================================
# CONFIGURACIÓN DE EXPERIMENTO (para experiment_configs.py)
# =============================================================================

def get_bandit_curriculum_config() -> Dict:
    """
    Configuración del experimento Bandit Curriculum Learning.
    Compatible con el formato de experiment_configs.py
    """
    return {
        "name": "bandit_curriculum",
        "description": "Automatic Curriculum Learning usando Multi-Armed Bandit con Learning Progress",
        "total_timesteps": 1_500_000,
        "method": "bandit",
        "bandit_config": {
            "temperature": 0.5,
            "window_size": 20,
            "recent_window": 5,
            "epsilon_exploration": 0.1,
            "min_samples": 3,
        },
        "task_registry": TASK_REGISTRY,
        "eval_freq": 10_000,
        "n_eval_episodes": 5,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bandit Curriculum Learning para Breakout")
    parser.add_argument("--timesteps", type=int, default=1_500_000, help="Total timesteps")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperatura Boltzmann")
    parser.add_argument("--window", type=int, default=20, help="Tamaño ventana LP")
    parser.add_argument("--output", type=str, default="results/bandit_curriculum", help="Output dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Eval frequency")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="Verbosity")
    
    args = parser.parse_args()
    
    results = train_with_bandit_curriculum(
        total_timesteps=args.timesteps,
        output_dir=args.output,
        temperature=args.temperature,
        window_size=args.window,
        eval_freq=args.eval_freq,
        seed=args.seed,
        verbose=args.verbose,
    )
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"Modelo: {results['model_path']}")
    print(f"Métricas: {results['metrics_path']}")
    print(f"Reward final: {results['final_reward_mean']:.2f} ± {results['final_reward_std']:.2f}")
    print("=" * 70)
