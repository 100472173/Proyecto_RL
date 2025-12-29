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
    Curriculum combinado mejorado: ajusta velocidad, paddle y ladrillos simultáneamente.
    Transiciones más suaves y más tiempo en fase final.
    """
    return {
        "name": "curriculum_combined",
        "description": "Progresión combinada mejorada (4 fases con transiciones suaves)",
        "total_timesteps": TOTAL_TIMESTEPS,
        "phases": [
            {
                "name": "phase1_easy",
                "timesteps": 200_000,  # 13%
                "env_kwargs": {
                    "ball_speed": 0.75,  # Menos trivial que 0.6
                    "paddle_width": 1.3,  # Menos trivial que 1.5
                    "brick_rows": 4,      # Más ladrillos que antes
                    "brick_cols": 8,
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
                    "brick_rows": 5,
                    "brick_cols": 9,
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
                    "brick_rows": 6,
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


# =============================================================================
# REGISTRO DE TODOS LOS EXPERIMENTOS
# =============================================================================

EXPERIMENTS = {
    "baseline": get_baseline_config,
    "curriculum_ball_speed": get_curriculum_ball_speed,
    "curriculum_paddle_width": get_curriculum_paddle_width,
    "curriculum_layout": get_curriculum_layout,
    "curriculum_combined": get_curriculum_combined,
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
    
    for name, config_fn in EXPERIMENTS.items():
        config = config_fn()
        print(f"\n{name.upper()}")
        print(f"  Descripción: {config['description']}")
        print(f"  Timesteps totales: {config['total_timesteps']:,}")
        print(f"  Número de fases: {len(config['phases'])}")
        
        for i, phase in enumerate(config['phases'], 1):
            env = phase['env_kwargs']
            print(f"    Fase {i}: {phase['name']}")
            print(f"      - Timesteps: {phase['timesteps']:,}")
            print(f"      - Ball speed: {env['ball_speed']}, "
                  f"Paddle: {env['paddle_width']}, "
                  f"Bricks: {env['brick_rows']}x{env['brick_cols']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_experiment_summary()
