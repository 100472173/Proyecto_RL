# Proyecto: Curriculum Learning en Breakout

Implementación de técnicas de Curriculum Learning y Teacher-Student para el juego Breakout con DQN para el proyecto.

## Estructura del Proyecto

```
├── breakout_env.py              # Entorno personalizado de Breakout
├── experiment_configs.py         # Definición de experimentos y curriculums
├── run_experiments.py           # Script principal para ejecutar experimentos
├── train_teacher_student.py     # Implementación de Teacher-Student learning
├── run_advanced_comparison.py   # Comparación con reward shaping
├── analyze_results.py           # Análisis y visualización de resultados
├── test.py                      # Evaluación de modelos entrenados
├── callbacks.py                 # Callbacks personalizados para métricas
├── metrics.py                   # Métricas de curriculum learning
└── requirements.txt             # Dependencias del proyecto
```

## Instalación

### 1. Crear y activar entorno conda

```bash
conda create -n rl_proyecto python=3.10
conda activate rl_proyecto
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

### 1. Ver experimentos disponibles

Para listar todos los experimentos configurados:

```bash
python experiment_configs.py
```

Los experimentos se activan/desactivan editando el diccionario `EXPERIMENTS` en `experiment_configs.py`.

### 2. Ejecutar experimentos individuales

Ejecutar un experimento específico:

```bash
python run_experiments.py --experiment baseline
python run_experiments.py --experiment curriculum_ball_speed
python run_experiments.py --experiment teacher_student_adaptive
```

Parámetros opcionales:
- `--output-dir`: Directorio para resultados (default: `results/`)
- `--eval-freq`: Frecuencia de evaluación en timesteps (default: 10000)
- `--n-eval-episodes`: Episodios por evaluación (default: 10)
- `--seed`: Semilla para reproducibilidad
- `--no-checkpoints`: Desactivar guardado de checkpoints intermedios

### 3. Ejecutar múltiples experimentos

Para ejecutar todos los experimentos activos en `experiment_configs.py`:

```bash
python run_experiments.py --all
```

Los resultados se guardan en `results/<nombre_experimento>/`.

### 4. Analizar resultados

Generar gráficos comparativos y reportes:

```bash
python analyze_results.py
```

Esto genera:
- Curvas de aprendizaje comparativas
- Métricas de Time to Threshold
- Análisis de Jumpstart y Asymptotic Performance
- Tablas LaTeX para publicación

Los gráficos se guardan en `figures/`.

Para controlar qué experimentos incluir en el análisis, editar la lista `EXPERIMENTS_TO_INCLUDE` en `analyze_results.py`.

### 5. Evaluar modelos entrenados

Probar un modelo específico:

```bash
python test.py results/baseline/models/dqn_standard.zip
```

Parámetros:
- `--episodes`: Número de episodios a evaluar (default: 5)
- `--no-render`: Ejecutar sin visualización

### 6. Comparación avanzada con reward shaping

Ejecutar experimentos con reward shaping:

```bash
python run_advanced_comparison.py
```

## Tipos de Experimentos

### Curriculum Learning Clásico

1. **Baseline**: Entrenamiento directo sin curriculum
2. **Curriculum por velocidad**: Progresión de pelota lenta a normal
3. **Curriculum por paddle**: Progresión de paddle ancho a normal
4. **Curriculum por layout**: Progresión de pocos a muchos ladrillos
5. **Curriculum por layout (segunda versión)**: Variación alternativa de layout, probando patrones de disposición por columnas, ajedrez o clúster/agrupaciones de ladrillos.
6. **Curriculum combinado**: Combina velocidad + paddle.
7. **Curriculum combinado (segunda versión)**: Combina velocidad + layout (en su segunda versión) + paddle.

### Teacher-Student Learning

1. **Action Cloning**: Imita acciones del teacher con probabilidad decreciente
2. **Soft Guidance**: Recibe Q-values del teacher como guía
3. **Adaptive Takeover**: Teacher interviene solo en situaciones críticas
4. **BC Pretrain**: Pre-entrenamiento con imitación + RL posterior

## Métricas Evaluadas

El sistema calcula tres métricas principales:

1. **Time to Threshold**: Timesteps necesarios para alcanzar un umbral de rendimiento
2. **Jumpstart**: Ventaja inicial del curriculum vs baseline
3. **Asymptotic Performance**: Rendimiento final estabilizado

Todas las métricas se evalúan en el entorno estándar para garantizar comparabilidad.

## Estructura de Resultados

Cada experimento genera:

```
results/<nombre_experimento>/
├── config.json                  # Configuración del experimento
├── metrics_report.json          # Métricas numéricas
├── logs/
│   └── training_metrics.json    # Histórico de entrenamiento
└── models/
    ├── dqn_standard.zip         # Modelo final
    └── checkpoints/             # Checkpoints intermedios
```

## Configuración de Experimentos

Los experimentos se definen en `experiment_configs.py`. Para crear un nuevo experimento:

1. Definir función que retorna configuración:
```python
def get_mi_curriculum() -> Dict:
    return {
        "name": "mi_curriculum",
        "description": "Descripción del experimento",
        "total_timesteps": 1_500_000,
        "phases": [
            {
                "name": "fase1",
                "timesteps": 500_000,
                "env_kwargs": {
                    "ball_speed": 0.8,
                    "paddle_width": 1.2,
                    "brick_rows": 6,
                    "brick_cols": 10,
                    "max_steps": 5000,
                },
            },
            # ... más fases
        ],
    }
```

2. Registrar en el diccionario `EXPERIMENTS`:
```python
EXPERIMENTS = {
    "mi_curriculum": get_mi_curriculum,
}
```

## Parámetros del Entorno

El entorno Breakout personalizado acepta:

- `ball_speed`: Velocidad de la pelota (1.0 = normal)
- `paddle_width`: Ancho de la pala (1.0 = normal)
- `brick_rows`: Filas de ladrillos
- `brick_cols`: Columnas de ladrillos
- `max_steps`: Máximo de pasos por episodio
- `reward_shaping`: Activar/desactivar reward shaping

## Notas Técnicas

### Teacher Model

Los experimentos Teacher-Student requieren un modelo pre-entrenado. Por defecto buscan:
```
results/baseline/models/dqn_standard.zip
```

Entrenar el teacher primero con:
```bash
python run_experiments.py --experiment baseline
```

### Reproducibilidad

Para resultados reproducibles usar semilla fija:
```bash
python run_experiments.py --experiment baseline --seed 42
```

### Consideraciones de tiempo

**Sólo como recordatorio!!** Nuestro entrenamiento completo (1.5M timesteps) tomó aproximadamente:
- Con GPU: casi 4h horas por experimento, el reward shaping casi 8h.