# Proyecto de Reinforcement Learning - Breakout con DQN

Entorno Breakout custom y entrenamiento DQN con Stable Baselines3.

## Estructura

```
├── breakout_env.py    # Entorno Breakout customizable
├── train.py           # Entrenamiento DQN
├── test.py            # Evaluación y modo jugar
├── requirements.txt   # Dependencias
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Entrenar
```bash
python train.py
```

### Evaluar modelo
```bash
python test.py models/best_model.zip 5
```

### Jugar como humano
```bash
python test.py --play
```

## Parámetros del entorno (para curriculum learning)

El entorno `BreakoutEnv` acepta estos parámetros:

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `ball_speed` | 1.0 | Multiplicador velocidad pelota |
| `paddle_width` | 1.0 | Multiplicador ancho pala |
| `brick_rows` | 6 | Filas de ladrillos |
| `brick_cols` | 10 | Columnas de ladrillos |
| `max_steps` | 10000 | Máximo pasos por episodio |
| `reward_shaping` | False | Recompensa extra por golpear pelota |

### Ejemplo fase fácil (curriculum):
```python
env = BreakoutEnv(
    ball_speed=0.5,      # Pelota lenta
    paddle_width=2.0,    # Pala doble de ancha
    brick_rows=2,        # Pocos ladrillos
    reward_shaping=True  # Recompensa por golpear
)
```
