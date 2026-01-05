"""
Configuración de experimentos para comparación de curriculum learning.

Define todos los curriculums a comparar:
- Baseline (sin curriculum)
- Progresión por velocidad de bola
- Progresión por tamaño de paddle
- Progresión por layout de ladrillos
- Progresión combinada
"""
from typing import Dict, List


# =============================================================================
# CONFIGURACIÓN COMÚN
# =============================================================================

# Parámetros DQN comunes a todos los experimentos
DQN_PARAMS = {
    "policy": "CnnPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 100_000,  # Aumentado para más diversidad de experiencias
    "learning_starts": 1_000,
    "batch_size": 32,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 2_000,  # Aumentado para mayor estabilidad
    "exploration_fraction": 0.15,  # Más tiempo explorando
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.01,  # Reducido para menos ruido al final
    "verbose": 1,
}

# Configuración estándar del entorno (tarea final / dificultad máxima)
STANDARD_ENV = {
    "ball_speed": 1.0,
    "paddle_width": 1.0,
    "brick_rows": 6,
    "brick_cols": 10,
    "max_steps": 7_000,
    "reward_shaping": False,
}

# Total de timesteps para todos los experimentos (para comparación justa)
TOTAL_TIMESTEPS = 1_500_000


# =============================================================================
# DEFINICIÓN DE CURRICULUMS
# =============================================================================

def get_baseline_config() -> Dict:
    """
    Baseline: Sin curriculum, entrena directamente en configuración estándar.
    """
    return {
        "name": "baseline",
        "description": "Sin curriculum learning - entrenamiento directo",
        "total_timesteps": TOTAL_TIMESTEPS,
        "phases": [
            {
                "name": "standard",
                "timesteps": TOTAL_TIMESTEPS,
                "env_kwargs": STANDARD_ENV.copy(),
            }
        ],
    }


def get_curriculum_ball_speed() -> Dict:
    """
    Curriculum por velocidad de bola: de lenta a normal.
    Fases menos triviales para mejor transferencia.
    """
    return {
        "name": "curriculum_ball_speed",
        "description": "Progresión por velocidad de bola (0.75 -> 0.85 -> 0.95 -> 1.0)",
        "total_timesteps": TOTAL_TIMESTEPS,
        "phases": [
            {
                "name": "phase1_slow",
                "timesteps": 200_000,  # 13%
                "env_kwargs": {
                    "ball_speed": 0.75,  # Menos trivial que 0.5
                    "paddle_width": 1.0,
                    "brick_rows": 6,
                    "brick_cols": 10,
                    "max_steps": 5_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase2_medium_slow",
                "timesteps": 300_000,  # 20%
                "env_kwargs": {
                    "ball_speed": 0.85,
                    "paddle_width": 1.0,
                    "brick_rows": 6,
                    "brick_cols": 10,
                    "max_steps": 6_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase3_almost_standard",
                "timesteps": 400_000,  # 27%
                "env_kwargs": {
                    "ball_speed": 0.95,  # Muy cerca de la velocidad final
                    "paddle_width": 1.0,
                    "brick_rows": 6,
                    "brick_cols": 10,
                    "max_steps": 6_500,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase4_standard",
                "timesteps": 600_000,  # 40% en tarea final
                "env_kwargs": STANDARD_ENV.copy(),
            },
        ],
    }


def get_curriculum_paddle_width() -> Dict:
    """
    Curriculum por tamaño de paddle: de grande a normal.
    Fases menos triviales para mejor transferencia.
    """
    return {
        "name": "curriculum_paddle_width",
        "description": "Progresión por tamaño de paddle (1.4 -> 1.25 -> 1.1 -> 1.0)",
        "total_timesteps": TOTAL_TIMESTEPS,
        "phases": [
            {
                "name": "phase1_wide",
                "timesteps": 200_000,  # 13%
                "env_kwargs": {
                    "ball_speed": 1.0,
                    "paddle_width": 1.4,  # Menos trivial que 2.0
                    "brick_rows": 6,
                    "brick_cols": 10,
                    "max_steps": 5_500,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase2_medium_wide",
                "timesteps": 300_000,  # 20%
                "env_kwargs": {
                    "ball_speed": 1.0,
                    "paddle_width": 1.25,
                    "brick_rows": 6,
                    "brick_cols": 10,
                    "max_steps": 6_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase3_almost_standard",
                "timesteps": 400_000,  # 27%
                "env_kwargs": {
                    "ball_speed": 1.0,
                    "paddle_width": 1.1,  # Muy cerca del tamaño final
                    "brick_rows": 6,
                    "brick_cols": 10,
                    "max_steps": 6_500,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase4_standard",
                "timesteps": 600_000,  # 40% en tarea final
                "env_kwargs": STANDARD_ENV.copy(),
            },
        ],
    }


def get_curriculum_layout() -> Dict:
    """
    Curriculum por layout de ladrillos: de pocos a muchos.
    NOTA: Este curriculum tiene problemas de comparabilidad en métricas
    como Time to Threshold debido a que las fases tienen diferente número
    de ladrillos. Se mantiene para experimentación pero se recomienda usar
    los otros curriculums para el paper.
    """
    return {
        "name": "curriculum_layout",
        "description": "Progresión por densidad de ladrillos (2x5 -> 3x7 -> 5x9 -> 6x10)",
        "total_timesteps": TOTAL_TIMESTEPS,
        "phases": [
            {
                "name": "phase1_sparse",
                "timesteps": 200_000,  # 13%
                "env_kwargs": {
                    "ball_speed": 1.0,
                    "paddle_width": 1.0,
                    "brick_rows": 2,
                    "brick_cols": 5,
                    "max_steps": 3_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase2_light",
                "timesteps": 300_000,  # 20%
                "env_kwargs": {
                    "ball_speed": 1.0,
                    "paddle_width": 1.0,
                    "brick_rows": 3,
                    "brick_cols": 7,
                    "max_steps": 4_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase3_medium",
                "timesteps": 400_000,  # 27%
                "env_kwargs": {
                    "ball_speed": 1.0,
                    "paddle_width": 1.0,
                    "brick_rows": 5,
                    "brick_cols": 9,
                    "max_steps": 6_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase4_standard",
                "timesteps": 600_000,  # 40%
                "env_kwargs": STANDARD_ENV.copy(),
            },
        ],
    }


def get_curriculum_combined() -> Dict:
    """
    Curriculum combinado: ajusta velocidad y paddle (SIN modificar layout de ladrillos).
    Transiciones más suaves y más tiempo en fase final.
    """
    return {
        "name": "curriculum_combined",
        "description": "Progresión combinada: velocidad + paddle (layout fijo 6x10)",
        "total_timesteps": TOTAL_TIMESTEPS,
        "phases": [
            {
                "name": "phase1_easy",
                "timesteps": 200_000,  # 13%
                "env_kwargs": {
                    "ball_speed": 0.75,
                    "paddle_width": 1.3,
                    "brick_rows": 6,      # Layout estándar fijo
                    "brick_cols": 10,
                    "max_steps": 4_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase2_medium",
                "timesteps": 300_000,  # 20%
                "env_kwargs": {
                    "ball_speed": 0.85,
                    "paddle_width": 1.2,
                    "brick_rows": 6,      # Layout estándar fijo
                    "brick_cols": 10,
                    "max_steps": 5_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase3_almost_standard",
                "timesteps": 400_000,  # 27%
                "env_kwargs": {
                    "ball_speed": 0.95,
                    "paddle_width": 1.1,
                    "brick_rows": 6,      # Layout estándar fijo
                    "brick_cols": 10,
                    "max_steps": 6_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase4_standard",
                "timesteps": 600_000,  # 40% en tarea final
                "env_kwargs": STANDARD_ENV.copy(),
            },
        ],
    }


def get_curriculum_layout_v2() -> Dict:
    """
    Curriculum por disposición de ladrillos: diferentes patrones espaciales.
    Progresión libre que experimenta con cantidad y disposición de ladrillos.
    Permite explorar si patrones espaciales ayudan al aprendizaje.
    """
    return {
        "name": "curriculum_layout_v2",
        "description": "Progresión experimental por patrones espaciales de ladrillos",
        "total_timesteps": TOTAL_TIMESTEPS,
        "phases": [
            {
                "name": "phase1_sparse_columns",
                "timesteps": 200_000,  # 13%
                "env_kwargs": {
                    "ball_speed": 1.0,
                    "paddle_width": 1.0,
                    "brick_rows": 3,
                    "brick_cols": 6,
                    "brick_layout": "columns",  # Columnas espaciadas (18 ladrillos)
                    "max_steps": 4_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase2_checkerboard",
                "timesteps": 300_000,  # 20%
                "env_kwargs": {
                    "ball_speed": 1.0,
                    "paddle_width": 1.0,
                    "brick_rows": 4,
                    "brick_cols": 8,
                    "brick_layout": "checkerboard",  # Patrón ajedrez (~16 ladrillos)
                    "max_steps": 5_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase3_clusters",
                "timesteps": 400_000,  # 27%
                "env_kwargs": {
                    "ball_speed": 1.0,
                    "paddle_width": 1.0,
                    "brick_rows": 5,
                    "brick_cols": 9,
                    "brick_layout": "clusters",  # Grupos de ladrillos (~30 ladrillos)
                    "max_steps": 6_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase4_standard",
                "timesteps": 600_000,  # 40% en tarea final
                "env_kwargs": STANDARD_ENV.copy(),  # 60 ladrillos densos
            },
        ],
    }


def get_curriculum_combined_v2() -> Dict:
    """
    Curriculum combinado versión 2: velocidad + paddle + patrones espaciales.
    Combina la progresión de dificultad mecánica (velocidad y paddle)
    con la progresión de patrones espaciales de ladrillos.
    """
    return {
        "name": "curriculum_combined_v2",
        "description": "Progresión combinada v2: velocidad + paddle + patrones espaciales",
        "total_timesteps": TOTAL_TIMESTEPS,
        "phases": [
            {
                "name": "phase1_easy_columns",
                "timesteps": 200_000,  # 13%
                "env_kwargs": {
                    "ball_speed": 0.75,
                    "paddle_width": 1.3,
                    "brick_rows": 3,
                    "brick_cols": 6,
                    "brick_layout": "columns",  # Columnas espaciadas
                    "max_steps": 4_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase2_medium_checkerboard",
                "timesteps": 300_000,  # 20%
                "env_kwargs": {
                    "ball_speed": 0.85,
                    "paddle_width": 1.2,
                    "brick_rows": 4,
                    "brick_cols": 8,
                    "brick_layout": "checkerboard",  # Patrón ajedrez
                    "max_steps": 5_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase3_almost_standard_clusters",
                "timesteps": 400_000,  # 27%
                "env_kwargs": {
                    "ball_speed": 0.95,
                    "paddle_width": 1.1,
                    "brick_rows": 5,
                    "brick_cols": 9,
                    "brick_layout": "clusters",  # Grupos de ladrillos
                    "max_steps": 6_000,
                    "reward_shaping": False,
                },
            },
            {
                "name": "phase4_standard",
                "timesteps": 600_000,  # 40% en tarea final
                "env_kwargs": STANDARD_ENV.copy(),
            },
        ],
    }


# =============================================================================
# TEACHER-STUDENT CURRICULUM CONFIGURATIONS
# =============================================================================

# Ruta por defecto al modelo teacher pre-entrenado
DEFAULT_TEACHER_PATH = "results/baseline/models/dqn_standard.zip"


def get_teacher_student_action_cloning() -> Dict:
    """
    Teacher-Student con Action Cloning.
    
    El estudiante imita las acciones del teacher con probabilidad decreciente.
    Comienza siguiendo al teacher 90% del tiempo y termina con 10%.
    """
    return {
        "name": "teacher_student_action_cloning",
        "description": "Teacher-Student con imitación de acciones (90% -> 10%)",
        "total_timesteps": TOTAL_TIMESTEPS,
        "is_teacher_student": True,
        "teacher_path": DEFAULT_TEACHER_PATH,
        "guidance_mode": "action_cloning",
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


def get_teacher_student_soft_guidance() -> Dict:
    """
    Teacher-Student con Soft Guidance.
    
    El estudiante siempre toma sus propias decisiones, pero recibe
    un bonus de reward proporcional a qué tan bien se alinea con el teacher.
    """
    return {
        "name": "teacher_student_soft_guidance",
        "description": "Teacher-Student con guía suave via reward shaping",
        "total_timesteps": TOTAL_TIMESTEPS,
        "is_teacher_student": True,
        "teacher_path": DEFAULT_TEACHER_PATH,
        "guidance_mode": "soft_guidance",
        "initial_teacher_prob": 0.8,
        "final_teacher_prob": 0.05,
        "decay_fraction": 0.6,
        "phases": [
            {
                "name": "soft_guided_learning",
                "timesteps": TOTAL_TIMESTEPS,
                "env_kwargs": STANDARD_ENV.copy(),
                "teacher_guidance": True,
            }
        ],
    }


def get_teacher_student_adaptive_takeover() -> Dict:
    """
    Teacher-Student con Adaptive Takeover.
    
    El teacher solo interviene cuando detecta que el estudiante
    está en una situación crítica (a punto de perder).
    """
    return {
        "name": "teacher_student_adaptive",
        "description": "Teacher-Student con intervención adaptativa del teacher",
        "total_timesteps": TOTAL_TIMESTEPS,
        "is_teacher_student": True,
        "teacher_path": DEFAULT_TEACHER_PATH,
        "guidance_mode": "adaptive_takeover",
        "initial_teacher_prob": 0.7,
        "final_teacher_prob": 0.05,
        "decay_fraction": 0.5,
        "phases": [
            {
                "name": "adaptive_learning",
                "timesteps": TOTAL_TIMESTEPS,
                "env_kwargs": STANDARD_ENV.copy(),
                "teacher_guidance": True,
            }
        ],
    }


def get_teacher_student_bc_pretrain() -> Dict:
    """
    Teacher-Student con Behavior Cloning Pre-training.
    
    Fase 1: Imitar al teacher 100% (Behavior Cloning) - 200k steps
    Fase 2: Entrenamiento RL estándar con pesos inicializados - resto
    """
    pretrain_steps = 200_000
    rl_steps = TOTAL_TIMESTEPS - pretrain_steps
    
    return {
        "name": "teacher_student_bc_pretrain",
        "description": f"BC pretrain ({pretrain_steps//1000}k) + RL estándar",
        "total_timesteps": TOTAL_TIMESTEPS,
        "is_teacher_student": True,
        "teacher_path": DEFAULT_TEACHER_PATH,
        "guidance_mode": "bc_pretrain",
        "pretrain_timesteps": pretrain_steps,
        "phases": [
            {
                "name": "behavior_cloning",
                "timesteps": pretrain_steps,
                "env_kwargs": STANDARD_ENV.copy(),
                "teacher_guidance": True,
                "guidance_prob": 1.0,
            },
            {
                "name": "rl_finetuning",
                "timesteps": rl_steps,
                "env_kwargs": STANDARD_ENV.copy(),
                "teacher_guidance": False,
            }
        ],
    }


# =============================================================================
# BANDIT CURRICULUM LEARNING (Automatic Curriculum)
# =============================================================================

# Task Registry para Bandit Curriculum (de fácil a difícil)
BANDIT_TASK_REGISTRY = {
    0: {"name": "trivial", "ball_speed": 0.5, "paddle_width": 2.0, "brick_rows": 2, "brick_cols": 10, "max_steps": 5000, "reward_shaping": False},
    1: {"name": "very_easy", "ball_speed": 0.5, "paddle_width": 1.8, "brick_rows": 3, "brick_cols": 10, "max_steps": 5000, "reward_shaping": False},
    2: {"name": "easy", "ball_speed": 0.65, "paddle_width": 1.5, "brick_rows": 4, "brick_cols": 10, "max_steps": 5500, "reward_shaping": False},
    3: {"name": "easy_medium", "ball_speed": 0.75, "paddle_width": 1.3, "brick_rows": 4, "brick_cols": 10, "max_steps": 6000, "reward_shaping": False},
    4: {"name": "medium", "ball_speed": 0.85, "paddle_width": 1.15, "brick_rows": 5, "brick_cols": 10, "max_steps": 6500, "reward_shaping": False},
    5: {"name": "medium_hard", "ball_speed": 0.92, "paddle_width": 1.05, "brick_rows": 5, "brick_cols": 10, "max_steps": 7000, "reward_shaping": False},
    6: {"name": "hard", "ball_speed": 0.97, "paddle_width": 1.0, "brick_rows": 6, "brick_cols": 10, "max_steps": 7000, "reward_shaping": False},
    7: {"name": "standard", "ball_speed": 1.0, "paddle_width": 1.0, "brick_rows": 6, "brick_cols": 10, "max_steps": 7000, "reward_shaping": False},
    8: {"name": "god_mode", "ball_speed": 1.15, "paddle_width": 0.85, "brick_rows": 6, "brick_cols": 10, "max_steps": 7000, "reward_shaping": False},
}


def get_bandit_curriculum() -> Dict:
    """
    Automatic Curriculum Learning usando Multi-Armed Bandit con Learning Progress.
    
    El "Teacher" es un algoritmo Bandit que selecciona tareas basándose en
    el progreso de aprendizaje del estudiante (cambio en rewards).
    """
    return {
        "name": "bandit_curriculum",
        "description": "Automatic Curriculum Learning - Multi-Armed Bandit con Learning Progress",
        "total_timesteps": TOTAL_TIMESTEPS,
        "is_bandit_curriculum": True,
        "bandit_config": {
            "temperature": 0.5,  # τ para Boltzmann (menor = más greedy)
            "window_size": 20,   # Historial de rewards por tarea
            "recent_window": 5,  # Ventana reciente para LP
            "epsilon_exploration": 0.1,  # Prob. exploración aleatoria
            "min_samples": 3,    # Mínimo samples antes de calcular LP
        },
        "task_registry": BANDIT_TASK_REGISTRY,
        "eval_freq": 10_000,
        "n_eval_episodes": 5,
    }


# =============================================================================
# REGISTRO DE TODOS LOS EXPERIMENTOS
# =============================================================================

EXPERIMENTS = {
    # "baseline": get_baseline_config,
    # "curriculum_ball_speed": get_curriculum_ball_speed,
    # "curriculum_paddle_width": get_curriculum_paddle_width,
    # "curriculum_layout": get_curriculum_layout,
    "curriculum_layout_v2": get_curriculum_layout_v2,
    "curriculum_combined": get_curriculum_combined,
    "curriculum_combined_v2": get_curriculum_combined_v2,
    # Teacher-Student experiments
    # "teacher_student_action_cloning": get_teacher_student_action_cloning,
    # "teacher_student_soft_guidance": get_teacher_student_soft_guidance,
    "teacher_student_adaptive": get_teacher_student_adaptive_takeover,
    # "teacher_student_bc_pretrain": get_teacher_student_bc_pretrain,
    # Bandit Curriculum (Automatic Curriculum Learning)
    # "bandit_curriculum": get_bandit_curriculum,
}


def get_experiment(name: str) -> Dict:
    """
    Obtiene la configuración de un experimento por nombre.
    
    Args:
        name: Nombre del experimento
        
    Returns:
        Diccionario con la configuración
    """
    if name not in EXPERIMENTS:
        raise ValueError(f"Experimento '{name}' no encontrado. "
                        f"Disponibles: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name]()


def get_all_experiments() -> Dict[str, Dict]:
    """Retorna todos los experimentos disponibles."""
    return {name: fn() for name, fn in EXPERIMENTS.items()}


def print_experiment_summary():
    """Imprime un resumen de todos los experimentos disponibles."""
    print("\n" + "=" * 70)
    print("EXPERIMENTOS DISPONIBLES")
    print("=" * 70)
    
    # Separar experimentos normales y teacher-student
    normal_experiments = []
    teacher_student_experiments = []
    
    for name, config_fn in EXPERIMENTS.items():
        config = config_fn()
        if config.get("is_teacher_student", False):
            teacher_student_experiments.append((name, config))
        else:
            normal_experiments.append((name, config))
    
    # Mostrar experimentos de curriculum clásico
    print("\n📚 CURRICULUM LEARNING CLÁSICO:")
    print("-" * 50)
    
    for name, config in normal_experiments:
        print(f"\n  {name.upper()}")
        print(f"    Descripción: {config['description']}")
        print(f"    Timesteps totales: {config['total_timesteps']:,}")
        print(f"    Número de fases: {len(config['phases'])}")
        
        for i, phase in enumerate(config['phases'], 1):
            env = phase['env_kwargs']
            print(f"      Fase {i}: {phase['name']}")
            print(f"        - Timesteps: {phase['timesteps']:,}")
            print(f"        - Ball speed: {env['ball_speed']}, "
                  f"Paddle: {env['paddle_width']}, "
                  f"Bricks: {env['brick_rows']}x{env['brick_cols']}")
    
    # Mostrar experimentos Teacher-Student
    print("\n\n🎓 TEACHER-STUDENT CURRICULUM:")
    print("-" * 50)
    
    for name, config in teacher_student_experiments:
        print(f"\n  {name.upper()}")
        print(f"    Descripción: {config['description']}")
        print(f"    Timesteps totales: {config['total_timesteps']:,}")
        print(f"    Modo de guía: {config.get('guidance_mode', 'N/A')}")
        
        if config.get('guidance_mode') != 'bc_pretrain':
            print(f"    Prob. teacher: {config.get('initial_teacher_prob', 0):.0%} -> "
                  f"{config.get('final_teacher_prob', 0):.0%}")
            print(f"    Decay fraction: {config.get('decay_fraction', 0):.0%}")
        else:
            print(f"    Pretrain steps: {config.get('pretrain_timesteps', 0):,}")
        
        print(f"    Teacher: {config.get('teacher_path', 'N/A')}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_experiment_summary()
