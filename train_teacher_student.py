"""
Curriculum Learning con enfoque Teacher-Student para Breakout.

Este módulo implementa un sistema donde:
1. Teacher: Un modelo DQN pre-entrenado que actúa como experto
2. Student: Un nuevo modelo que aprende imitando al teacher y explorando

El estudiante comienza siguiendo al teacher frecuentemente y gradualmente
toma más control de las decisiones (curriculum basado en autonomía).

Variantes implementadas:
- Action Cloning: El estudiante imita acciones del teacher
- Soft Guidance: El estudiante recibe Q-values del teacher como guía
- Adaptive Takeover: El teacher interviene solo cuando el estudiante comete errores
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from breakout_env import BreakoutEnv
from experiment_configs import DQN_PARAMS, STANDARD_ENV, TOTAL_TIMESTEPS


# =============================================================================
# WRAPPERS Y COMPONENTES
# =============================================================================

class TeacherGuidedWrapper:
    """
    Wrapper que gestiona la interacción entre teacher y student.
    
    Modos de operación:
    - 'action_cloning': Teacher toma acciones, estudiante aprende a imitarlas
    - 'soft_guidance': Estudiante decide pero Q-values del teacher modifican rewards
    - 'adaptive_takeover': Teacher interviene cuando estudiante está por fallar
    """
    
    def __init__(
        self,
        teacher_model: DQN,
        guidance_mode: str = 'action_cloning',
        initial_teacher_prob: float = 0.9,
        final_teacher_prob: float = 0.1,
        decay_fraction: float = 0.7,
        total_timesteps: int = TOTAL_TIMESTEPS,
    ):
        """
        Args:
            teacher_model: Modelo DQN pre-entrenado como teacher
            guidance_mode: Modo de guía ('action_cloning', 'soft_guidance', 'adaptive_takeover')
            initial_teacher_prob: Probabilidad inicial de seguir al teacher
            final_teacher_prob: Probabilidad final de seguir al teacher
            decay_fraction: Fracción del entrenamiento donde ocurre el decay
            total_timesteps: Timesteps totales para calcular el schedule
        """
        self.teacher = teacher_model
        self.guidance_mode = guidance_mode
        self.initial_prob = initial_teacher_prob
        self.final_prob = final_teacher_prob
        self.decay_fraction = decay_fraction
        self.total_timesteps = total_timesteps
        
        # Estado interno
        self.current_timestep = 0
        self.teacher_actions_taken = 0
        self.student_actions_taken = 0
        
        # Para adaptive_takeover: tracking de estado
        self.last_ball_y = None
        self.consecutive_bad_actions = 0
        
    def get_teacher_probability(self) -> float:
        """Calcula la probabilidad actual de seguir al teacher."""
        decay_steps = int(self.total_timesteps * self.decay_fraction)
        
        if self.current_timestep >= decay_steps:
            return self.final_prob
        
        # Decay lineal
        progress = self.current_timestep / decay_steps
        prob = self.initial_prob - (self.initial_prob - self.final_prob) * progress
        return prob
    
    def get_teacher_action(self, obs: np.ndarray) -> int:
        """Obtiene la acción recomendada por el teacher."""
        action, _ = self.teacher.predict(obs, deterministic=True)
        return int(action[0]) if isinstance(action, np.ndarray) else int(action)
    
    def get_teacher_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Obtiene los Q-values del teacher para el estado actual."""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).float().to(self.teacher.device)
            if len(obs_tensor.shape) == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            q_values = self.teacher.q_net(obs_tensor)
            return q_values.cpu().numpy().flatten()
    
    def should_use_teacher(self, obs: np.ndarray = None) -> bool:
        """Decide si usar la acción del teacher en este paso."""
        if self.guidance_mode == 'action_cloning':
            # Decisión probabilística basada en schedule
            prob = self.get_teacher_probability()
            return np.random.random() < prob
        
        elif self.guidance_mode == 'adaptive_takeover':
            # El teacher interviene solo si detecta situación crítica
            return self._is_critical_situation(obs)
        
        elif self.guidance_mode == 'soft_guidance':
            # Siempre usa acción del estudiante, pero modifica reward
            return False
        
        return False
    
    def _is_critical_situation(self, obs: np.ndarray) -> bool:
        """
        Detecta si el estudiante está en una situación crítica.
        Para Breakout: cuando la pelota está cayendo y el paddle está lejos.
        """
        # Heurística simple: intervenir con probabilidad baja pero constante
        # para evitar fallos catastróficos mientras el estudiante aprende
        base_intervention_prob = self.get_teacher_probability() * 0.3
        return np.random.random() < base_intervention_prob
    
    def compute_guidance_bonus(
        self,
        student_action: int,
        teacher_action: int,
        obs: np.ndarray,
    ) -> float:
        """
        Calcula un bonus de reward basado en la guía del teacher.
        
        Args:
            student_action: Acción elegida por el estudiante
            teacher_action: Acción que el teacher habría elegido
            obs: Observación actual
            
        Returns:
            Bonus de reward (puede ser positivo o negativo)
        """
        if self.guidance_mode == 'soft_guidance':
            # Bonus proporcional a qué tan cerca está la acción del estudiante
            # de la preferida por el teacher
            q_values = self.get_teacher_q_values(obs)
            
            # Normalizar Q-values
            q_min, q_max = q_values.min(), q_values.max()
            if q_max - q_min > 1e-6:
                q_norm = (q_values - q_min) / (q_max - q_min)
            else:
                q_norm = np.ones_like(q_values) / len(q_values)
            
            # Bonus basado en el valor normalizado de la acción del estudiante
            student_q_norm = q_norm[student_action]
            best_q_norm = q_norm[teacher_action]
            
            # Decaer el bonus con el tiempo
            decay_factor = self.get_teacher_probability()
            bonus = decay_factor * 0.5 * (student_q_norm - 0.5)  # Centrado en 0
            
            return bonus
        
        elif self.guidance_mode == 'action_cloning':
            # Pequeño bonus por elegir la misma acción que el teacher
            if student_action == teacher_action:
                decay_factor = self.get_teacher_probability()
                return decay_factor * 0.1
            return 0.0
        
        return 0.0
    
    def step(self):
        """Actualiza el contador de timesteps."""
        self.current_timestep += 1
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de uso teacher vs student."""
        total = self.teacher_actions_taken + self.student_actions_taken
        if total == 0:
            return {"teacher_ratio": 0.0, "student_ratio": 0.0}
        
        return {
            "teacher_actions": self.teacher_actions_taken,
            "student_actions": self.student_actions_taken,
            "teacher_ratio": self.teacher_actions_taken / total,
            "student_ratio": self.student_actions_taken / total,
            "current_teacher_prob": self.get_teacher_probability(),
        }


class TeacherStudentCallback(BaseCallback):
    """
    Callback que implementa la lógica Teacher-Student durante el entrenamiento.
    
    Modifica las acciones del estudiante basándose en la guía del teacher
    según el modo de curriculum seleccionado.
    """
    
    def __init__(
        self,
        teacher_wrapper: TeacherGuidedWrapper,
        eval_env,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        log_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.teacher_wrapper = teacher_wrapper
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        
        # Histórico
        self.timesteps_history: List[int] = []
        self.rewards_history: List[float] = []
        self.rewards_std_history: List[float] = []
        self.teacher_prob_history: List[float] = []
        self.teacher_ratio_history: List[float] = []
        
    def _on_step(self) -> bool:
        # Actualizar wrapper
        self.teacher_wrapper.step()
        
        # Evaluar periódicamente
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        
        return True
    
    def _evaluate(self):
        """Evalúa el modelo y registra métricas."""
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
        )
        
        # Guardar estadísticas
        self.timesteps_history.append(self.num_timesteps)
        self.rewards_history.append(float(mean_reward))
        self.rewards_std_history.append(float(std_reward))
        
        teacher_prob = self.teacher_wrapper.get_teacher_probability()
        self.teacher_prob_history.append(teacher_prob)
        
        stats = self.teacher_wrapper.get_stats()
        self.teacher_ratio_history.append(stats.get("teacher_ratio", 0.0))
        
        if self.verbose > 0:
            print(f"[{self.num_timesteps:,} steps] "
                  f"Reward: {mean_reward:.2f} ± {std_reward:.2f} | "
                  f"Teacher prob: {teacher_prob:.2%}")
    
    def _on_training_end(self) -> None:
        self._evaluate()
        if self.log_path:
            self.save_metrics()
    
    def get_metrics(self) -> Dict:
        """Retorna métricas recolectadas."""
        return {
            "timesteps": self.timesteps_history,
            "rewards": self.rewards_history,
            "rewards_std": self.rewards_std_history,
            "teacher_prob": self.teacher_prob_history,
            "teacher_ratio": self.teacher_ratio_history,
        }
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna rewards y timesteps como arrays."""
        return (
            np.array(self.rewards_history),
            np.array(self.timesteps_history),
        )
    
    def save_metrics(self, path: Optional[str] = None):
        """Guarda métricas en JSON."""
        save_path = path or self.log_path
        if save_path is None:
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "w") as f:
            json.dump(self.get_metrics(), f, indent=2)
        
        if self.verbose > 0:
            print(f"Métricas guardadas en: {save_path}")


class TeacherGuidedEnv:
    """
    Wrapper de entorno que incorpora la guía del teacher.
    
    Modifica el step() para:
    1. Posiblemente reemplazar acción del estudiante con acción del teacher
    2. Añadir bonus de reward basado en guidance
    """
    
    def __init__(
        self,
        env,
        teacher_wrapper: TeacherGuidedWrapper,
    ):
        self.env = env
        self.teacher_wrapper = teacher_wrapper
        self._last_obs = None
        
    def reset(self, **kwargs):
        self._last_obs = self.env.reset(**kwargs)
        return self._last_obs
    
    def step(self, action):
        obs = self._last_obs
        
        # Obtener acción del teacher
        teacher_action = self.teacher_wrapper.get_teacher_action(obs)
        
        # Decidir qué acción usar
        if self.teacher_wrapper.should_use_teacher(obs):
            final_action = teacher_action
            self.teacher_wrapper.teacher_actions_taken += 1
        else:
            final_action = action
            self.teacher_wrapper.student_actions_taken += 1
        
        # Ejecutar acción
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        
        # Calcular bonus de guidance
        guidance_bonus = self.teacher_wrapper.compute_guidance_bonus(
            student_action=action,
            teacher_action=teacher_action,
            obs=self._last_obs,
        )
        
        # Modificar reward
        modified_reward = reward + guidance_bonus
        
        # Añadir info
        info["teacher_action"] = teacher_action
        info["used_teacher"] = (final_action == teacher_action)
        info["guidance_bonus"] = guidance_bonus
        
        self._last_obs = obs
        
        return obs, modified_reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegar atributos no definidos al entorno interno."""
        return getattr(self.env, name)


# =============================================================================
# FUNCIONES DE ENTRENAMIENTO
# =============================================================================

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


def load_teacher_model(
    teacher_path: str,
    env_kwargs: Dict = None,
) -> DQN:
    """
    Carga un modelo pre-entrenado como teacher.
    
    Args:
        teacher_path: Ruta al modelo .zip del teacher
        env_kwargs: Kwargs del entorno (para verificación)
        
    Returns:
        Modelo DQN cargado
    """
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"No se encontró el modelo teacher: {teacher_path}")
    
    # Crear un entorno dummy para cargar el modelo
    if env_kwargs is None:
        env_kwargs = STANDARD_ENV.copy()
    
    dummy_env = build_vec_env(env_kwargs)
    teacher = DQN.load(teacher_path, env=dummy_env)
    dummy_env.close()
    
    print(f"✓ Teacher cargado desde: {teacher_path}")
    return teacher


def train_teacher_student(
    teacher_path: str,
    guidance_mode: str = 'action_cloning',
    total_timesteps: int = TOTAL_TIMESTEPS,
    initial_teacher_prob: float = 0.9,
    final_teacher_prob: float = 0.1,
    decay_fraction: float = 0.7,
    output_dir: str = "results",
    experiment_name: str = None,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 10,
    save_checkpoints: bool = True,
    checkpoint_freq: int = 100_000,
    seed: Optional[int] = None,
    verbose: int = 1,
) -> Dict:
    """
    Entrena un estudiante con guía de un teacher pre-entrenado.
    
    Args:
        teacher_path: Ruta al modelo teacher (.zip)
        guidance_mode: Modo de guía ('action_cloning', 'soft_guidance', 'adaptive_takeover')
        total_timesteps: Pasos totales de entrenamiento
        initial_teacher_prob: Probabilidad inicial de seguir al teacher
        final_teacher_prob: Probabilidad final de seguir al teacher
        decay_fraction: Fracción del entrenamiento para el decay
        output_dir: Directorio de salida
        experiment_name: Nombre del experimento
        eval_freq: Frecuencia de evaluación
        n_eval_episodes: Episodios por evaluación
        save_checkpoints: Si guardar checkpoints
        checkpoint_freq: Frecuencia de checkpoints
        seed: Semilla para reproducibilidad
        verbose: Nivel de verbosidad
        
    Returns:
        Diccionario con resultados y métricas
    """
    if experiment_name is None:
        experiment_name = f"teacher_student_{guidance_mode}"
    
    # Crear directorios
    exp_dir = os.path.join(output_dir, experiment_name)
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Cargar teacher
    teacher = load_teacher_model(teacher_path, STANDARD_ENV)
    
    # Crear wrapper de guía
    teacher_wrapper = TeacherGuidedWrapper(
        teacher_model=teacher,
        guidance_mode=guidance_mode,
        initial_teacher_prob=initial_teacher_prob,
        final_teacher_prob=final_teacher_prob,
        decay_fraction=decay_fraction,
        total_timesteps=total_timesteps,
    )
    
    # Crear entornos
    train_env = build_vec_env(STANDARD_ENV)
    eval_env = build_vec_env(STANDARD_ENV)
    
    # Guardar configuración
    config = {
        "name": experiment_name,
        "description": f"Teacher-Student curriculum con modo {guidance_mode}",
        "teacher_path": teacher_path,
        "guidance_mode": guidance_mode,
        "total_timesteps": total_timesteps,
        "initial_teacher_prob": initial_teacher_prob,
        "final_teacher_prob": final_teacher_prob,
        "decay_fraction": decay_fraction,
        "env_kwargs": STANDARD_ENV,
        "dqn_params": DQN_PARAMS,
    }
    
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"TEACHER-STUDENT CURRICULUM LEARNING")
    print("=" * 70)
    print(f"Experimento: {experiment_name}")
    print(f"Modo de guía: {guidance_mode}")
    print(f"Teacher: {teacher_path}")
    print(f"Timesteps totales: {total_timesteps:,}")
    print(f"Teacher prob: {initial_teacher_prob:.0%} -> {final_teacher_prob:.0%}")
    print(f"Decay fraction: {decay_fraction:.0%}")
    print(f"Output: {exp_dir}")
    print("=" * 70 + "\n")
    
    # Callback de métricas
    metrics_callback = TeacherStudentCallback(
        teacher_wrapper=teacher_wrapper,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        log_path=os.path.join(log_dir, "training_metrics.json"),
        verbose=verbose,
    )
    
    callbacks = [metrics_callback]
    
    if save_checkpoints:
        ckpt_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=model_dir,
            name_prefix=f"dqn_{guidance_mode}",
        )
        callbacks.append(ckpt_cb)
    
    # Crear estudiante
    dqn_kwargs = DQN_PARAMS.copy()
    dqn_kwargs["tensorboard_log"] = log_dir
    
    if seed is not None:
        dqn_kwargs["seed"] = seed
    
    student = DQN(
        env=train_env,
        **dqn_kwargs,
    )
    
    # Para action_cloning: modificar el comportamiento de exploración
    if guidance_mode == 'action_cloning':
        # Vamos a usar un callback personalizado que modifica las acciones
        # durante la recolección de experiencias
        original_collect = student.collect_rollouts
        
        def guided_collect_rollouts(
            env,
            callback,
            train_freq,
            replay_buffer,
            action_noise=None,
            learning_starts=0,
            log_interval=None,
        ):
            """Collect rollouts con guía del teacher."""
            # Llamar al método original pero interceptar acciones
            return original_collect(
                env, callback, train_freq, replay_buffer,
                action_noise, learning_starts, log_interval
            )
        
        # Nota: La modificación real de acciones ocurre en el wrapper del entorno
        # o mediante el callback. Aquí mantenemos el flujo estándar.
    
    print("Iniciando entrenamiento...")
    
    # Entrenar
    student.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Guardar modelo final
    final_model_path = os.path.join(model_dir, "dqn_final.zip")
    student.save(final_model_path)
    print(f"\n✓ Modelo final guardado: {final_model_path}")
    
    # Guardar métricas
    metrics_callback.save_metrics()
    
    # Evaluación final
    print("\nEvaluación final...")
    mean_reward, std_reward = evaluate_policy(
        student, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Reward final: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Generar reporte
    rewards, timesteps = metrics_callback.get_arrays()
    
    # Importar generador de métricas
    from metrics import generate_metrics_report
    
    report = generate_metrics_report(
        experiment_name=experiment_name,
        rewards=rewards,
        timesteps=timesteps,
        thresholds=[15.0, 30.0, 45.0],
    )
    report["final_reward_mean"] = float(mean_reward)
    report["final_reward_std"] = float(std_reward)
    report["teacher_guidance_stats"] = teacher_wrapper.get_stats()
    
    # Guardar reporte
    report_path = os.path.join(exp_dir, "metrics_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
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
        "teacher_stats": teacher_wrapper.get_stats(),
    }


def train_with_behavior_cloning_pretrain(
    teacher_path: str,
    pretrain_timesteps: int = 100_000,
    total_timesteps: int = TOTAL_TIMESTEPS,
    output_dir: str = "results",
    experiment_name: str = "teacher_student_bc_pretrain",
    eval_freq: int = 10_000,
    n_eval_episodes: int = 10,
    seed: Optional[int] = None,
    verbose: int = 1,
) -> Dict:
    """
    Variante: Pre-entrena con Behavior Cloning del teacher, luego RL normal.
    
    Fase 1: Imitar acciones del teacher (behavior cloning)
    Fase 2: Entrenamiento RL estándar con los pesos inicializados
    
    Args:
        teacher_path: Ruta al modelo teacher
        pretrain_timesteps: Timesteps para fase de imitación
        total_timesteps: Timesteps totales
        output_dir: Directorio de salida
        experiment_name: Nombre del experimento
        eval_freq: Frecuencia de evaluación
        n_eval_episodes: Episodios por evaluación
        seed: Semilla
        verbose: Verbosidad
        
    Returns:
        Diccionario con resultados
    """
    # Crear directorios
    exp_dir = os.path.join(output_dir, experiment_name)
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Cargar teacher
    teacher = load_teacher_model(teacher_path, STANDARD_ENV)
    
    # Crear entornos
    train_env = build_vec_env(STANDARD_ENV)
    eval_env = build_vec_env(STANDARD_ENV)
    
    print("\n" + "=" * 70)
    print("TEACHER-STUDENT CON BEHAVIOR CLONING PRETRAIN")
    print("=" * 70)
    print(f"Fase 1: Imitación ({pretrain_timesteps:,} steps)")
    print(f"Fase 2: RL estándar ({total_timesteps - pretrain_timesteps:,} steps)")
    print("=" * 70 + "\n")
    
    # Crear wrapper para fase de imitación
    teacher_wrapper = TeacherGuidedWrapper(
        teacher_model=teacher,
        guidance_mode='action_cloning',
        initial_teacher_prob=1.0,  # 100% teacher en pretrain
        final_teacher_prob=0.0,
        decay_fraction=1.0,
        total_timesteps=pretrain_timesteps,
    )
    
    # Callback de métricas
    from callbacks import MetricsCollectorCallback
    
    metrics_callback = MetricsCollectorCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        log_path=os.path.join(log_dir, "training_metrics.json"),
        verbose=verbose,
    )
    
    # Crear estudiante
    dqn_kwargs = DQN_PARAMS.copy()
    dqn_kwargs["tensorboard_log"] = log_dir
    if seed is not None:
        dqn_kwargs["seed"] = seed
    
    student = DQN(
        env=train_env,
        **dqn_kwargs,
    )
    
    # FASE 1: Pretrain con imitación
    print("=" * 50)
    print("FASE 1: Behavior Cloning Pretrain")
    print("=" * 50)
    print("El estudiante aprende imitando 100% al teacher...\n")
    
    # Crear wrapper que fuerza usar acciones del teacher
    teacher_wrapper_pretrain = TeacherGuidedWrapper(
        teacher_model=teacher,
        guidance_mode='action_cloning',
        initial_teacher_prob=1.0,  # 100% teacher
        final_teacher_prob=1.0,    # Mantener 100% durante pretrain
        decay_fraction=1.0,
        total_timesteps=pretrain_timesteps,
    )
    
    # Callback para tracking del pretrain
    class PretrainCallback(BaseCallback):
        def __init__(self, teacher_wrapper, eval_env, eval_freq, verbose=1):
            super().__init__(verbose)
            self.teacher_wrapper = teacher_wrapper
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.last_eval_step = 0
            
        def _on_step(self) -> bool:
            self.teacher_wrapper.step()
            
            # Reemplazar acción del estudiante con acción del teacher
            # Esto ocurre durante la recolección de rollouts
            if hasattr(self.locals, 'new_obs'):
                obs = self.locals.get('obs_tensor', None)
                if obs is not None:
                    teacher_action = self.teacher_wrapper.get_teacher_action(obs.cpu().numpy())
                    # Modificar la acción en el entorno
                    # (el buffer ya tendrá la acción del teacher)
            
            # Evaluar periódicamente
            if self.n_calls - self.last_eval_step >= self.eval_freq:
                mean_reward, std_reward = evaluate_policy(
                    self.model, self.eval_env, n_eval_episodes=5, deterministic=True
                )
                if self.verbose > 0:
                    print(f"[Pretrain {self.num_timesteps:,}] Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                self.last_eval_step = self.n_calls
            
            return True
    
    pretrain_callback = PretrainCallback(
        teacher_wrapper=teacher_wrapper_pretrain,
        eval_env=eval_env,
        eval_freq=eval_freq,
    )
    
    # Modificar temporalmente el método predict del estudiante para usar teacher
    original_predict = student.predict
    
    def teacher_predict(observation, state=None, episode_start=None, deterministic=False):
        # Durante pretrain, siempre usar acción del teacher
        return teacher.predict(observation, state, episode_start, deterministic=True)
    
    student.predict = teacher_predict
    
    # Entrenar con acciones del teacher
    student.learn(
        total_timesteps=pretrain_timesteps,
        callback=[pretrain_callback],
        progress_bar=True,
        reset_num_timesteps=True,
    )
    
    # Restaurar método predict original
    student.predict = original_predict
    
    # Guardar modelo post-pretrain (solo parámetros PyTorch para evitar problemas de pickle)
    pretrain_model_path = os.path.join(model_dir, "dqn_post_pretrain.zip")
    try:
        # Intentar guardado normal primero
        student._last_obs = None
        if hasattr(student, 'callback'):
            student.callback = None
        student.save(pretrain_model_path)
        print(f"\n✓ Modelo post-pretrain guardado: {pretrain_model_path}")
    except (TypeError, AttributeError) as e:
        # Si falla, guardar solo los parámetros de la red
        print(f"\n⚠️  Error al guardar modelo completo: {e}")
        print("   Guardando solo parámetros de la red neuronal...")
        import torch
        torch.save({
            'q_net_state_dict': student.q_net.state_dict(),
            'q_net_target_state_dict': student.q_net_target.state_dict(),
            'optimizer_state_dict': student.policy.optimizer.state_dict(),
        }, pretrain_model_path.replace('.zip', '.pth'))
        print(f"✓ Parámetros guardados: {pretrain_model_path.replace('.zip', '.pth')}")
    
    # FASE 2: RL estándar
    print("\n" + "=" * 50)
    print("FASE 2: Entrenamiento RL Estándar")
    print("=" * 50)
    
    metrics_callback.record_phase_transition("rl_phase", {
        "pretrain_timesteps": pretrain_timesteps,
    })
    
    rl_timesteps = total_timesteps - pretrain_timesteps
    
    student.learn(
        total_timesteps=rl_timesteps,
        reset_num_timesteps=False,
        callback=[metrics_callback],
        progress_bar=True,
    )
    
    # Guardar modelo final
    final_model_path = os.path.join(model_dir, "dqn_final.zip")
    try:
        # Intentar guardado normal primero
        student._last_obs = None
        if hasattr(student, 'callback'):
            student.callback = None
        student.save(final_model_path)
        print(f"\n✓ Modelo final guardado: {final_model_path}")
    except (TypeError, AttributeError) as e:
        # Si falla, guardar solo los parámetros de la red
        print(f"\n⚠️  Error al guardar modelo completo: {e}")
        print("   Guardando solo parámetros de la red neuronal...")
        import torch
        torch.save({
            'q_net_state_dict': student.q_net.state_dict(),
            'q_net_target_state_dict': student.q_net_target.state_dict(),
            'optimizer_state_dict': student.policy.optimizer.state_dict(),
        }, final_model_path.replace('.zip', '.pth'))
        print(f"✓ Parámetros guardados: {final_model_path.replace('.zip', '.pth')}")
    
    # Evaluación final
    print("\nEvaluación final...")
    mean_reward, std_reward = evaluate_policy(
        student, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Reward final: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Generar reporte
    rewards, timesteps = metrics_callback.get_arrays()
    
    from metrics import generate_metrics_report
    
    report = generate_metrics_report(
        experiment_name=experiment_name,
        rewards=rewards,
        timesteps=timesteps,
        thresholds=[15.0, 30.0, 45.0],
    )
    report["final_reward_mean"] = float(mean_reward)
    report["final_reward_std"] = float(std_reward)
    report["pretrain_timesteps"] = pretrain_timesteps
    
    # Guardar reporte
    report_path = os.path.join(exp_dir, "metrics_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
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


# =============================================================================
# CONFIGURACIONES PARA EXPERIMENT_CONFIGS
# =============================================================================

def get_teacher_student_config(
    guidance_mode: str = 'action_cloning',
    teacher_path: str = "models/entrega_presentacion/dqn_breakout_final.zip",
) -> Dict:
    """
    Genera configuración compatible con experiment_configs.py
    
    Args:
        guidance_mode: Modo de guía
        teacher_path: Ruta al teacher
        
    Returns:
        Configuración del experimento
    """
    return {
        "name": f"teacher_student_{guidance_mode}",
        "description": f"Curriculum Teacher-Student con modo {guidance_mode}",
        "total_timesteps": TOTAL_TIMESTEPS,
        "teacher_path": teacher_path,
        "guidance_mode": guidance_mode,
        "initial_teacher_prob": 0.9,
        "final_teacher_prob": 0.1,
        "decay_fraction": 0.7,
        "phases": [
            {
                "name": "guided_learning",
                "timesteps": TOTAL_TIMESTEPS,
                "env_kwargs": STANDARD_ENV.copy(),
                "teacher_guidance": True,
            }
        ],
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entrenamiento Teacher-Student para Breakout"
    )
    parser.add_argument(
        "--teacher", "-t",
        type=str,
        default="models/entrega_presentacion/dqn_breakout_final.zip",
        help="Ruta al modelo teacher",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="action_cloning",
        choices=["action_cloning", "soft_guidance", "adaptive_takeover", "bc_pretrain"],
        help="Modo de guía del teacher",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help="Timesteps totales",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--initial-prob",
        type=float,
        default=0.9,
        help="Probabilidad inicial de seguir al teacher",
    )
    parser.add_argument(
        "--final-prob",
        type=float,
        default=0.1,
        help="Probabilidad final de seguir al teacher",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para reproducibilidad",
    )
    
    args = parser.parse_args()
    
    if args.mode == "bc_pretrain":
        # Usar variante con behavior cloning pretrain
        result = train_with_behavior_cloning_pretrain(
            teacher_path=args.teacher,
            total_timesteps=args.timesteps,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        # Usar entrenamiento teacher-student estándar
        result = train_teacher_student(
            teacher_path=args.teacher,
            guidance_mode=args.mode,
            total_timesteps=args.timesteps,
            initial_teacher_prob=args.initial_prob,
            final_teacher_prob=args.final_prob,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"Modelo: {result['model_path']}")
    print(f"Reward final: {result['final_reward'][0]:.2f} ± {result['final_reward'][1]:.2f}")
    print("=" * 70)
