"""
Análisis y visualización de resultados de experimentos.

Genera:
- Curvas de aprendizaje comparativas
- Gráficos de Time to Threshold
- Comparación de Asymptotic Performance
- Análisis de Jumpstart
- Tablas LaTeX para el paper
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from metrics import (
    compute_time_to_threshold,
    compute_jumpstart,
    compute_asymptotic_performance,
    compute_area_under_curve,
    compute_stability,
    compare_experiments,
    print_comparison_table,
)


# Configuración de estilo para los gráficos
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
FIGSIZE = (12, 6)
DPI = 150


def load_experiment_data(results_dir: str) -> Dict[str, Dict]:
    """
    Carga los datos de todos los experimentos desde el directorio de resultados.
    
    Args:
        results_dir: Directorio raíz de resultados
        
    Returns:
        Dict con datos de cada experimento
    """
    experiments = {}
    results_path = Path(results_dir)
    
    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        metrics_file = exp_dir / "logs" / "training_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            
            experiments[exp_dir.name] = {
                "timesteps": np.array(data.get("timesteps", [])),
                "rewards": np.array(data.get("rewards", [])),
                "rewards_std": np.array(data.get("rewards_std", [])),
                "phase_transitions": data.get("phase_transitions", []),
            }
            
            # Cargar reporte si existe
            report_file = exp_dir / "metrics_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    experiments[exp_dir.name]["report"] = json.load(f)
    
    return experiments


def plot_learning_curves(
    experiments: Dict[str, Dict],
    output_path: str,
    title: str = "Curvas de Aprendizaje",
    smooth_window: int = 5,
    show_std: bool = True,
):
    """
    Genera gráfico comparativo de curvas de aprendizaje.
    
    Args:
        experiments: Dict con datos de experimentos
        output_path: Ruta donde guardar el gráfico
        title: Título del gráfico
        smooth_window: Ventana para suavizar curvas
        show_std: Mostrar banda de desviación estándar
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    for idx, (name, data) in enumerate(experiments.items()):
        timesteps = data["timesteps"]
        rewards = data["rewards"]
        rewards_std = data.get("rewards_std", np.zeros_like(rewards))
        
        if len(rewards) == 0:
            continue
        
        # Suavizar
        if len(rewards) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(rewards, kernel, mode='valid')
            smoothed_std = np.convolve(rewards_std, kernel, mode='valid')
            valid_timesteps = timesteps[smooth_window-1:]
        else:
            smoothed = rewards
            smoothed_std = rewards_std
            valid_timesteps = timesteps
        
        color = COLORS[idx % len(COLORS)]
        label = name.replace("_", " ").title()
        
        ax.plot(valid_timesteps / 1e6, smoothed, label=label, color=color, linewidth=2)
        
        if show_std and len(smoothed_std) == len(smoothed):
            ax.fill_between(
                valid_timesteps / 1e6,
                smoothed - smoothed_std,
                smoothed + smoothed_std,
                alpha=0.2,
                color=color,
            )
        
        # Marcar transiciones de fase
        for transition in data.get("phase_transitions", []):
            ts = transition.get("timestep", 0)
            if ts > 0:
                ax.axvline(ts / 1e6, color=color, linestyle='--', alpha=0.3)
    
    ax.set_xlabel("Timesteps (millones)", fontsize=12)
    ax.set_ylabel("Reward Medio", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardado: {output_path}")


def plot_time_to_threshold(
    experiments: Dict[str, Dict],
    output_path: str,
    thresholds: List[float] = None,
):
    """
    Genera gráfico de barras de Time to Threshold.
    
    Args:
        experiments: Dict con datos de experimentos
        output_path: Ruta donde guardar el gráfico
        thresholds: Lista de umbrales a evaluar
    """
    if thresholds is None:
        thresholds = [15.0, 30.0, 45.0]  # 25%, 50%, 75% de 60 ladrillos
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    exp_names = list(experiments.keys())
    x = np.arange(len(thresholds))
    width = 0.15
    
    for idx, name in enumerate(exp_names):
        data = experiments[name]
        rewards = data["rewards"]
        timesteps = data["timesteps"]
        
        ttts = []
        for thresh in thresholds:
            ttt = compute_time_to_threshold(rewards, timesteps, thresh)
            # Convertir a miles para visualizar
            ttts.append(ttt / 1000 if ttt != -1 else 0)
        
        offset = (idx - len(exp_names) / 2) * width + width / 2
        label = name.replace("_", " ").title()
        bars = ax.bar(x + offset, ttts, width, label=label, color=COLORS[idx % len(COLORS)])
        
        # Marcar con X si no alcanzó el umbral
        for j, ttt in enumerate(ttts):
            if ttt == 0:
                ax.text(x[j] + offset, 10, '✗', ha='center', va='bottom', 
                       fontsize=12, color='red')
    
    ax.set_xlabel("Umbral de Reward", fontsize=12)
    ax.set_ylabel("Timesteps (×1000)", fontsize=12)
    ax.set_title("Time to Threshold: Comparativa", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"R ≥ {t:.0f}" for t in thresholds])
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardado: {output_path}")


def plot_asymptotic_performance(
    experiments: Dict[str, Dict],
    output_path: str,
):
    """
    Genera gráfico de barras de Asymptotic Performance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = []
    means = []
    stds = []
    
    for name, data in experiments.items():
        rewards = data["rewards"]
        if len(rewards) == 0:
            continue
        
        mean, std = compute_asymptotic_performance(rewards, last_n_percent=0.2)
        names.append(name.replace("_", " ").title())
        means.append(mean)
        stds.append(std)
    
    x = np.arange(len(names))
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel("Reward Medio (último 20%)", fontsize=12)
    ax.set_title("Asymptotic Performance: Comparativa", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Valores sobre las barras
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(
            f'{mean:.1f}',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=10
        )
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardado: {output_path}")


def plot_jumpstart(
    experiments: Dict[str, Dict],
    output_path: str,
    baseline_name: str = "baseline",
    n_early: int = 5,
):
    """
    Genera gráfico de Jumpstart (ventaja inicial vs baseline).
    
    JUMPSTART mide: ¿El curriculum tiene ventaja cuando AMBOS están en la fase final?
    - Baseline: primeros n_early episodios (inicio del entrenamiento)
    - Curriculum: primeros n_early episodios DESDE QUE ENTRA EN LA FASE FINAL
    """
    if baseline_name not in experiments:
        print(f"Baseline '{baseline_name}' no encontrado. Saltando jumpstart.")
        return
    
    baseline_rewards = experiments[baseline_name]["rewards"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = []
    jumpstarts = []
    
    for name, data in experiments.items():
        if name == baseline_name:
            continue
        
        rewards = data["rewards"]
        phase_transitions = data.get("phase_transitions", [])
        
        if len(rewards) == 0:
            continue
        
        # Para curriculum: encontrar índice de inicio de la última fase
        if phase_transitions:
            # La última transición marca el inicio de la fase final
        
            last_transition = phase_transitions[-1]
            last_phase_idx = int((int(last_transition.get("timestep")) / 10000) +2)  # lo del +2 es porque se repiten las evaluaciones de las transiciones o no se que movida, hay que corregirlo asi
            print(last_phase_idx)
            print()
            # Tomar los primeros n_early episodios desde ese punto
            curriculum_rewards_early = rewards[last_phase_idx:last_phase_idx +  n_early]
        else:
            # Si no hay transiciones (ej: baseline), usar desde el inicio
            curriculum_rewards_early = rewards[:n_early]
        
        _, jump_pct, _ = compute_jumpstart(
            curriculum_rewards_early.tolist(),
            baseline_rewards[:n_early].tolist()
        )
        
        names.append(name.replace("_", " ").title())
        jumpstarts.append(jump_pct)
    
    if not names:
        print("No hay experimentos para comparar jumpstart.")
        return
    
    x = np.arange(len(names))
    colors = ['green' if j > 0 else 'red' for j in jumpstarts]
    
    bars = ax.barh(x, jumpstarts, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linewidth=1)
    
    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mejora vs Baseline (%)", fontsize=12)
    ax.set_title(f"Jumpstart: Ventaja al Entrar en Fase Final (primeras {n_early} evaluaciones)", fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Valores en las barras
    for bar, jump in zip(bars, jumpstarts):
        x_pos = bar.get_width() + (2 if jump >= 0 else -15)
        ax.annotate(
            f'{jump:+.1f}%',
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5 if jump >= 0 else -5, 0),
            textcoords="offset points",
            ha='left' if jump >= 0 else 'right',
            va='center',
            fontsize=10,
        )
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardado: {output_path}")


def plot_sample_efficiency(
    experiments: Dict[str, Dict],
    output_path: str,
):
    """
    Genera gráfico de eficiencia de muestreo (AUC normalizado).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = []
    aucs = []
    
    for name, data in experiments.items():
        rewards = data["rewards"]
        timesteps = data["timesteps"]
        
        if len(rewards) < 2:
            continue
        
        auc = compute_area_under_curve(rewards, timesteps)
        names.append(name.replace("_", " ").title())
        aucs.append(auc)
    
    x = np.arange(len(names))
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
    
    bars = ax.bar(x, aucs, color=colors, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel("AUC (Area Under Curve)", fontsize=12)
    ax.set_title("Eficiencia de Muestreo: AUC de Curvas de Aprendizaje", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardado: {output_path}")


def generate_latex_table(
    experiments: Dict[str, Dict],
    output_path: str,
    thresholds: List[float] = None,
):
    """
    Genera tabla LaTeX con todos los resultados.
    """
    if thresholds is None:
        thresholds = [15.0, 30.0, 45.0]
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparación de métodos de curriculum learning}",
        r"\label{tab:results}",
        r"\begin{tabular}{l" + "c" * (4 + len(thresholds)) + "}",
        r"\toprule",
    ]
    
    # Header
    header = r"Método & Asym. Mean & Asym. Std & AUC & Stability"
    for t in thresholds:
        header += f" & TTT$_{{{t:.0f}}}$"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Rows
    for name, data in experiments.items():
        rewards = data["rewards"]
        timesteps = data["timesteps"]
        
        if len(rewards) == 0:
            continue
        
        asymp_mean, asymp_std = compute_asymptotic_performance(rewards)
        auc = compute_area_under_curve(rewards, timesteps)
        stability = compute_stability(rewards)
        
        row = f"{name.replace('_', ' ').title()}"
        row += f" & {asymp_mean:.1f} & {asymp_std:.1f} & {auc:.1f} & {stability:.2f}"
        
        for t in thresholds:
            ttt = compute_time_to_threshold(rewards, timesteps, t)
            if ttt == -1:
                row += r" & --"
            else:
                row += f" & {ttt/1000:.0f}k"
        
        row += r" \\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"✓ Guardado: {output_path}")


def generate_all_plots(
    results_dir: str = "results",
    output_dir: str = "figures",
):
    """
    Genera todos los gráficos y tablas para el paper.
    
    Args:
        results_dir: Directorio con los resultados de experimentos
        output_dir: Directorio donde guardar figuras
    """
    print("\n" + "=" * 60)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 60)
    
    # Cargar datos
    experiments = load_experiment_data(results_dir)
    
    if not experiments:
        print(f"No se encontraron datos en {results_dir}")
        return
    
    print(f"Experimentos cargados: {list(experiments.keys())}")
    
    # Generar gráficos
    plot_learning_curves(
        experiments,
        os.path.join(output_dir, "learning_curves.png"),
        title="Curvas de Aprendizaje: Comparativa de Curriculum Learning",
    )
    
    plot_time_to_threshold(
        experiments,
        os.path.join(output_dir, "time_to_threshold.png"),
    )
    
    plot_asymptotic_performance(
        experiments,
        os.path.join(output_dir, "asymptotic_performance.png"),
    )
    
    plot_jumpstart(
        experiments,
        os.path.join(output_dir, "jumpstart.png"),
    )
    
    plot_sample_efficiency(
        experiments,
        os.path.join(output_dir, "sample_efficiency.png"),
    )
    
    # Generar tabla LaTeX
    generate_latex_table(
        experiments,
        os.path.join(output_dir, "results_table.tex"),
    )
    
    # Generar reporte en texto
    exp_tuples = {
        name: (data["rewards"], data["timesteps"])
        for name, data in experiments.items()
        if len(data["rewards"]) > 0
    }
    
    if exp_tuples:
        reports = compare_experiments(exp_tuples, baseline_name="baseline")
        table_str = print_comparison_table(reports)
        
        with open(os.path.join(output_dir, "comparison_report.txt"), "w") as f:
            f.write(table_str)
        print(f"✓ Guardado: {os.path.join(output_dir, 'comparison_report.txt')}")
    
    print("\n" + "=" * 60)
    print("✓ VISUALIZACIONES GENERADAS")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analizar resultados de experimentos")
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results",
        help="Directorio con resultados",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="figures",
        help="Directorio de salida para figuras",
    )
    
    args = parser.parse_args()
    generate_all_plots(args.results_dir, args.output_dir)
