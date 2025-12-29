"""
Callbacks personalizados para recolectar métricas durante el entrenamiento.
"""
import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Optional, Dict, List


class MetricsCollectorCallback(BaseCallback):
    """
    Callback que recolecta métricas detalladas durante el entrenamiento.
    
    IMPORTANTE: Este callback SIEMPRE evalúa en el entorno que se le pasa
    en el constructor (típicamente STANDARD_ENV), independientemente de
    la fase de curriculum en la que esté entrenando el modelo.
    
    Esto permite medir el progreso real en la tarea final durante todo
    el entrenamiento, incluso cuando el agente está entrenando en fases
    más fáciles.
    
    Evalúa el modelo periódicamente y guarda estadísticas para
    calcular Time to Threshold, Jumpstart y Asymptotic Performance.
    """
    
    def __init__(
        self,
        eval_env,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        log_path: Optional[str] = None,
        verbose: int = 0,
    ):
        """
        Args:
            eval_env: Entorno de evaluación (debe ser STANDARD_ENV para comparabilidad)
            eval_freq: Frecuencia de evaluación (en timesteps)
            n_eval_episodes: Número de episodios por evaluación
            log_path: Ruta donde guardar las métricas (opcional)
            verbose: Nivel de verbosidad
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        
        # Histórico
        self.timesteps_history: List[int] = []
        self.rewards_history: List[float] = []
        self.rewards_std_history: List[float] = []
        self.episode_lengths_history: List[float] = []
        
        # Tracking de fase (para curriculum)
        self.phase_transitions: List[Dict] = []
        self.current_phase: str = "initial"
        
        if verbose > 0:
            print(f"\n✓ MetricsCollectorCallback inicializado:")
            print(f"  - Evalúa cada {eval_freq:,} timesteps")
            print(f"  - Usa {n_eval_episodes} episodios por evaluación")
            print(f"  - ⚠️  IMPORTANTE: Siempre evalúa en el MISMO entorno (típicamente estándar)")
            print(f"                   para garantizar comparabilidad entre fases\n")
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True
    
    def _on_training_end(self) -> None:
        """Se ejecuta al final del entrenamiento."""
        # Evaluación final
        self._evaluate()
        
        # Guardar métricas si hay path
        if self.log_path:
            self.save_metrics()
    
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
        
        if self.verbose > 0:
            phase_info = f" [Fase: {self.current_phase}]" if self.current_phase != "initial" else ""
            print(f"[{self.num_timesteps:,} steps]{phase_info} "
                  f"Reward en entorno ESTÁNDAR: {mean_reward:.2f} ± {std_reward:.2f}")
    
    def record_phase_transition(self, phase_name: str, info: Dict = None):
        """
        Registra una transición de fase del curriculum.
        
        Args:
            phase_name: Nombre de la nueva fase
            info: Información adicional sobre la transición
        """
        transition = {
            "timestep": self.num_timesteps,
            "from_phase": self.current_phase,
            "to_phase": phase_name,
            "reward_at_transition": self.rewards_history[-1] if self.rewards_history else 0,
            "info": info or {},
        }
        self.phase_transitions.append(transition)
        self.current_phase = phase_name
        
        if self.verbose > 0:
            print(f"[Phase Transition] {transition['from_phase']} -> {phase_name} "
                  f"at timestep {self.num_timesteps:,}")
    
    def get_metrics(self) -> Dict:
        """Retorna todas las métricas recolectadas como diccionario."""
        return {
            "timesteps": np.array(self.timesteps_history),
            "rewards": np.array(self.rewards_history),
            "rewards_std": np.array(self.rewards_std_history),
            "phase_transitions": self.phase_transitions,
        }
    
    def get_arrays(self):
        """Retorna rewards y timesteps como arrays numpy."""
        return (
            np.array(self.rewards_history),
            np.array(self.timesteps_history),
        )
    
    def save_metrics(self, path: Optional[str] = None):
        """
        Guarda las métricas en un archivo JSON.
        
        Args:
            path: Ruta del archivo (usa log_path si no se especifica)
        """
        save_path = path or self.log_path
        if save_path is None:
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        metrics = {
            "timesteps": self.timesteps_history,
            "rewards": self.rewards_history,
            "rewards_std": self.rewards_std_history,
            "phase_transitions": self.phase_transitions,
        }
        
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        if self.verbose > 0:
            print(f"Métricas guardadas en: {save_path}")


class PhaseTransitionCallback(BaseCallback):
    """
    Callback para curriculum con transición basada en rendimiento (Time to Threshold).
    
    Cambia automáticamente de fase cuando el agente alcanza un umbral de rendimiento.
    """
    
    def __init__(
        self,
        phases: List[Dict],
        build_env_fn,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        promotion_threshold: float = 0.7,  # Ratio del máximo teórico
        min_timesteps_per_phase: int = 50_000,
        verbose: int = 1,
    ):
        """
        Args:
            phases: Lista de fases con 'name', 'env_kwargs'
            build_env_fn: Función para crear entornos
            eval_freq: Frecuencia de evaluación
            n_eval_episodes: Episodios por evaluación
            promotion_threshold: Ratio de reward vs máximo para promover
            min_timesteps_per_phase: Mínimo de steps antes de poder cambiar de fase
            verbose: Nivel de verbosidad
        """
        super().__init__(verbose)
        self.phases = phases
        self.build_env_fn = build_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.promotion_threshold = promotion_threshold
        self.min_timesteps_per_phase = min_timesteps_per_phase
        
        self.current_phase_idx = 0
        self.phase_start_timestep = 0
        self.phase_history: List[Dict] = []
        self.eval_env = None
        
    def _on_training_start(self):
        """Inicializa el entorno de evaluación."""
        self.eval_env = self.build_env_fn(self.phases[0]["env_kwargs"])
        self.phase_start_timestep = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_maybe_promote()
        return True
    
    def _evaluate_and_maybe_promote(self):
        """Evalúa y decide si promover a la siguiente fase."""
        if self.current_phase_idx >= len(self.phases) - 1:
            return  # Ya en la última fase
        
        # Verificar mínimo de timesteps en la fase
        steps_in_phase = self.num_timesteps - self.phase_start_timestep
        if steps_in_phase < self.min_timesteps_per_phase:
            return
        
        current_phase = self.phases[self.current_phase_idx]
        env_kwargs = current_phase["env_kwargs"]
        
        # Evaluar rendimiento actual
        mean_reward, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
        )
        
        # Calcular reward máximo teórico para esta fase
        max_bricks = env_kwargs["brick_rows"] * env_kwargs["brick_cols"]
        max_theoretical_reward = max_bricks  # 1 punto por ladrillo
        
        performance_ratio = mean_reward / max_theoretical_reward
        
        if self.verbose > 0:
            print(f"\n[Phase {self.current_phase_idx + 1}] "
                  f"Reward: {mean_reward:.2f} / {max_theoretical_reward} "
                  f"({performance_ratio:.1%})")
        
        # Decidir si promover
        if performance_ratio >= self.promotion_threshold:
            self._promote_to_next_phase(mean_reward)
    
    def _promote_to_next_phase(self, reward_at_promotion: float):
        """Avanza a la siguiente fase del curriculum."""
        # Guardar historial
        self.phase_history.append({
            "phase_idx": self.current_phase_idx,
            "phase_name": self.phases[self.current_phase_idx]["name"],
            "start_timestep": self.phase_start_timestep,
            "end_timestep": self.num_timesteps,
            "duration": self.num_timesteps - self.phase_start_timestep,
            "final_reward": reward_at_promotion,
        })
        
        self.current_phase_idx += 1
        self.phase_start_timestep = self.num_timesteps
        
        new_phase = self.phases[self.current_phase_idx]
        
        if self.verbose > 0:
            print(f"\n{'='*50}")
            print(f"¡PROMOCIÓN! Avanzando a fase {self.current_phase_idx + 1}: {new_phase['name']}")
            print(f"{'='*50}\n")
        
        # Cambiar entorno del modelo
        new_train_env = self.build_env_fn(new_phase["env_kwargs"])
        self.model.set_env(new_train_env)
        
        # Actualizar entorno de evaluación
        self.eval_env = self.build_env_fn(new_phase["env_kwargs"])
    
    def get_phase_history(self) -> List[Dict]:
        """Retorna el historial de fases."""
        # Añadir fase actual si no está ya
        if not self.phase_history or self.phase_history[-1]["phase_idx"] != self.current_phase_idx:
            current = {
                "phase_idx": self.current_phase_idx,
                "phase_name": self.phases[self.current_phase_idx]["name"],
                "start_timestep": self.phase_start_timestep,
                "end_timestep": self.num_timesteps,
                "duration": self.num_timesteps - self.phase_start_timestep,
                "final_reward": None,  # Aún no terminado
            }
            return self.phase_history + [current]
        return self.phase_history
