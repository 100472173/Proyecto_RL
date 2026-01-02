"""
Métricas avanzadas para evaluar curriculum learning.

Implementa las tres métricas principales del paper:
- Time to Threshold: Timesteps necesarios para alcanzar un rendimiento dado
- Jumpstart: Ventaja inicial del curriculum vs baseline
- Asymptotic Performance: Rendimiento final estable
"""
import numpy as np
from typing import List, Tuple, Dict, Optional


def compute_time_to_threshold(
    rewards: np.ndarray,
    timesteps: np.ndarray,
    threshold: float,
    window: int = 5,
) -> int:
    """
    Calcula el número de timesteps necesarios para alcanzar un umbral de reward.
    
    Args:
        rewards: Array de rewards medios por evaluación
        timesteps: Array de timesteps correspondientes
        threshold: Umbral de reward a alcanzar (ej: 30.0 para Breakout)
        window: Tamaño de ventana para suavizar
        
    Returns:
        Timesteps necesarios para alcanzar el umbral (o -1 si nunca se alcanza)
    """
    if len(rewards) == 0:
        return -1
    
    # Suavizar para evitar fluctuaciones
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        valid_timesteps = timesteps[window-1:]
    else:
        smoothed = rewards
        valid_timesteps = timesteps
    
    # Encontrar primer punto donde se supera el umbral
    above_threshold = np.where(smoothed >= threshold)[0]
    
    if len(above_threshold) > 0:
        return int(valid_timesteps[above_threshold[0]])
    return -1  # Nunca alcanzó el umbral


def compute_jumpstart(
    curriculum_rewards_early: List[float],
    baseline_rewards_early: List[float]
) -> Tuple[float, float, float]:
    """
    Mide la ventaja inicial del curriculum learning (Jumpstart).
    
    Compara el rendimiento promedio en los primeros episodios de evaluación
    en la tarea final.
    
    Args:
        curriculum_rewards_early: Primeros N rewards del modelo con curriculum
        baseline_rewards_early: Primeros N rewards del baseline
        n_episodes: Número de episodios iniciales a considerar
        
    Returns:
        Tuple de (diferencia_absoluta, diferencia_porcentual, curriculum_mean)
    """
    
    curr_mean = np.mean(curriculum_rewards_early)
    base_mean = np.mean(baseline_rewards_early)
    
    diff_absolute = curr_mean - base_mean
    
    if base_mean == 0:
        diff_percent = float('inf') if curr_mean != 0 else 0.0
    else:
        diff_percent = (diff_absolute / abs(base_mean)) * 100

    return diff_absolute, diff_percent, curr_mean


def compute_asymptotic_performance(
    rewards: np.ndarray,
    last_n_percent: float = 0.2,
) -> Tuple[float, float]:
    """
    Calcula el rendimiento asintótico (estable al final del entrenamiento).
    
    Args:
        rewards: Array completo de rewards durante el entrenamiento
        last_n_percent: Porcentaje final del entrenamiento a considerar (ej: 20%)
        
    Returns:
        (media, desviación estándar) del rendimiento asintótico
    """
    if len(rewards) == 0:
        return 0.0, 0.0
    
    n_last = max(1, int(len(rewards) * last_n_percent))
    final_rewards = rewards[-n_last:]
    
    return float(np.mean(final_rewards)), float(np.std(final_rewards))


def compute_area_under_curve(
    rewards: np.ndarray,
    timesteps: np.ndarray,
) -> float:
    """
    Calcula el área bajo la curva de aprendizaje (sample efficiency global).
    
    Mayor AUC = mejor eficiencia de muestreo.
    
    Args:
        rewards: Array de rewards durante evaluaciones
        timesteps: Array de timesteps correspondientes
        
    Returns:
        AUC normalizado
    """
    if len(rewards) < 2:
        return 0.0
    
    # Normalizar timesteps a [0, 1]
    normalized_steps = (timesteps - timesteps[0]) / (timesteps[-1] - timesteps[0])
    
    # Integración trapezoidal
    auc = np.trapz(rewards, normalized_steps)
    return float(auc)


def compute_stability(
    rewards: np.ndarray,
    window_size: int = 10,
) -> float:
    """
    Mide la estabilidad del entrenamiento mediante la varianza promedio.
    
    Menor valor = más estable.
    
    Args:
        rewards: Array de rewards durante el entrenamiento
        window_size: Tamaño de ventana para calcular varianza local
        
    Returns:
        Varianza media del entrenamiento
    """
    if len(rewards) < window_size:
        return float(np.std(rewards))
    
    variances = []
    for i in range(len(rewards) - window_size + 1):
        window = rewards[i:i+window_size]
        variances.append(np.var(window))
    
    return float(np.mean(variances))


def compute_sample_efficiency(
    rewards: np.ndarray,
    timesteps: np.ndarray,
    target_performance: float,
) -> float:
    """
    Calcula la eficiencia de muestreo: ratio entre performance y timesteps usados.
    
    Args:
        rewards: Array de rewards
        timesteps: Array de timesteps
        target_performance: Rendimiento objetivo
        
    Returns:
        Ratio de eficiencia (mayor = mejor)
    """
    ttt = compute_time_to_threshold(rewards, timesteps, target_performance)
    
    if ttt == -1:
        # No alcanzó el objetivo, usar timesteps totales
        return float(np.max(rewards)) / float(timesteps[-1]) if timesteps[-1] > 0 else 0.0
    
    return target_performance / float(ttt)


def generate_metrics_report(
    experiment_name: str,
    rewards: np.ndarray,
    timesteps: np.ndarray,
    baseline_rewards: Optional[np.ndarray] = None,
    baseline_timesteps: Optional[np.ndarray] = None,
    thresholds: List[float] = None,
    max_possible_reward: float = 60.0,  # 6 filas * 10 cols = 60 ladrillos
) -> Dict:
    """
    Genera un reporte completo de métricas para un experimento.
    
    Args:
        experiment_name: Nombre del experimento
        rewards: Array de rewards durante evaluaciones
        timesteps: Array de timesteps correspondientes
        baseline_rewards: Rewards del baseline (para comparar jumpstart)
        baseline_timesteps: Timesteps del baseline
        thresholds: Lista de umbrales para Time to Threshold
        max_possible_reward: Reward máximo teórico
        
    Returns:
        Diccionario con todas las métricas
    """
    if thresholds is None:
        # Umbrales por defecto: 25%, 50%, 75% del máximo
        thresholds = [
            max_possible_reward * 0.25,
            max_possible_reward * 0.50,
            max_possible_reward * 0.75,
        ]
    
    report = {
        "experiment": experiment_name,
        "total_timesteps": int(timesteps[-1]) if len(timesteps) > 0 else 0,
        "n_evaluations": len(rewards),
        
        # Asymptotic performance
        "asymptotic_mean": None,
        "asymptotic_std": None,
        
        # Time to threshold para múltiples umbrales
        "time_to_thresholds": {},
        
        # Sample efficiency
        "auc": None,
        "stability": None,
        
        # Jumpstart (si hay baseline)
        "jumpstart_absolute": None,
        "jumpstart_percent": None,
    }
    
    if len(rewards) == 0:
        return report
    
    # Asymptotic performance
    asymp_mean, asymp_std = compute_asymptotic_performance(rewards)
    report["asymptotic_mean"] = asymp_mean
    report["asymptotic_std"] = asymp_std
    
    # Time to threshold para múltiples umbrales
    for thresh in thresholds:
        ttt = compute_time_to_threshold(rewards, timesteps, thresh)
        report["time_to_thresholds"][f"threshold_{thresh:.1f}"] = ttt
    
    # Stability
    report["stability"] = compute_stability(rewards)
    
    # AUC
    report["auc"] = compute_area_under_curve(rewards, timesteps)
    
    # Jumpstart (si hay baseline)
    report["jumpstart_absolute"] = None
    report["jumpstart_percent"] = None
    
    return report


def compare_experiments(
    experiments: Dict[str, Tuple[np.ndarray, np.ndarray]],
    thresholds: List[float] = None,
    baseline_name: str = "baseline",
) -> Dict[str, Dict]:
    """
    Compara múltiples experimentos y genera un reporte consolidado.
    
    Args:
        experiments: Dict con {nombre: (rewards, timesteps)}
        thresholds: Umbrales para Time to Threshold
        baseline_name: Nombre del experimento baseline para jumpstart
        
    Returns:
        Dict con reportes de cada experimento
    """
    reports = {}
    
    # Obtener baseline si existe
    baseline_rewards = None
    baseline_timesteps = None
    if baseline_name in experiments:
        baseline_rewards, baseline_timesteps = experiments[baseline_name]
    
    for name, (rewards, timesteps) in experiments.items():
        base_r = baseline_rewards if name != baseline_name else None
        base_t = baseline_timesteps if name != baseline_name else None
        
        report = generate_metrics_report(
            experiment_name=name,
            rewards=rewards,
            timesteps=timesteps,
            baseline_rewards=base_r,
            baseline_timesteps=base_t,
            thresholds=thresholds,
        )
        reports[name] = report
    
    return reports


def print_comparison_table(reports: Dict[str, Dict]) -> str:
    """
    Imprime una tabla formateada con la comparación de experimentos.
    
    Args:
        reports: Dict con reportes de cada experimento
        
    Returns:
        String con la tabla formateada
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("COMPARACIÓN DE EXPERIMENTOS")
    lines.append("=" * 80)
    
    # Header
    header = f"{'Experimento':<25} | {'Asym. Mean':>10} | {'Asym. Std':>9} | {'AUC':>8} | {'Stability':>9}"
    lines.append(header)
    lines.append("-" * 80)
    
    # Rows
    for name, report in reports.items():
        asymp_mean = report.get('asymptotic_mean', 0) or 0
        asymp_std = report.get('asymptotic_std', 0) or 0
        auc = report.get('auc', 0) or 0
        stability = report.get('stability', 0) or 0
        
        row = f"{name:<25} | {asymp_mean:>10.2f} | {asymp_std:>9.2f} | {auc:>8.2f} | {stability:>9.2f}"
        lines.append(row)
    
    lines.append("=" * 80)
    
    # Time to Threshold section
    lines.append("\nTIME TO THRESHOLD (timesteps, -1 = no alcanzado)")
    lines.append("-" * 80)
    
    # Obtener todos los umbrales
    all_thresholds = set()
    for report in reports.values():
        all_thresholds.update(report.get('time_to_thresholds', {}).keys())
    all_thresholds = sorted(all_thresholds)
    
    if all_thresholds:
        thresh_header = f"{'Experimento':<25}"
        for t in all_thresholds:
            thresh_header += f" | {t.replace('threshold_', 'T='):>12}"
        lines.append(thresh_header)
        lines.append("-" * 80)
        
        for name, report in reports.items():
            row = f"{name:<25}"
            for t in all_thresholds:
                ttt = report.get('time_to_thresholds', {}).get(t, -1)
                if ttt == -1:
                    row += f" | {'-':>12}"
                else:
                    row += f" | {ttt:>12,}"
            lines.append(row)
    
    lines.append("=" * 80)
    
    # Jumpstart section
    lines.append("\nJUMPSTART (vs baseline)")
    lines.append("-" * 80)
    for name, report in reports.items():
        jump_abs = report.get('jumpstart_absolute')
        jump_pct = report.get('jumpstart_percent')
        if jump_abs is not None:
            lines.append(f"{name:<25} | Δ = {jump_abs:>+8.2f} ({jump_pct:>+6.1f}%)")
    
    lines.append("=" * 80 + "\n")
    
    result = "\n".join(lines)
    print(result)
    return result
