"""
Evalúa un agente DQN entrenado en el entorno Breakout custom.
"""
import sys
import os
import time
import argparse
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from breakout_env import BreakoutEnv


def make_env(render_mode=None, **kwargs):
    """Crea el entorno Breakout custom."""
    def _init():
        env = BreakoutEnv(render_mode=render_mode, **kwargs)
        env = Monitor(env)
        return env
    return _init


def parse_env_kwargs(args):
    """Construye kwargs del entorno a partir de argumentos CLI."""
    return {
        "ball_speed": args.ball_speed,
        "paddle_width": args.paddle_width,
        "brick_rows": args.brick_rows,
        "brick_cols": args.brick_cols,
        "max_steps": args.max_steps,
        "reward_shaping": args.reward_shaping,
    }


def test_agent(model_path, n_episodes=5, render=True, env_kwargs=None):
    """Evalúa un agente entrenado y reporta más métricas útiles."""
    # Crear entorno
    render_mode = 'human' if render else None
    env = DummyVecEnv([make_env(render_mode=render_mode, **(env_kwargs or {}))])
    env = VecFrameStack(env, n_stack=4)

    # Cargar modelo
    model = DQN.load(model_path)
    print(f"Modelo cargado desde {model_path}")
    print(f"\nEvaluando durante {n_episodes} episodios...\n")

    episode_rewards = []
    episode_scores = []
    episode_steps = []
    episode_lives_left = []
    results = []  # 'win' | 'loss' | 'timeout'

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        truncated = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1

            if render:
                env.envs[0].render()
                time.sleep(0.016)  # ~60 FPS

            truncated = info[0].get('TimeLimit.truncated', False)

        score = info[0].get('score', episode_reward)
        lives_left = info[0].get('lives', 0)

        # Determinar resultado
        if truncated:
            result = 'timeout'
        elif lives_left > 0:
            result = 'win'
        else:
            result = 'loss'

        episode_rewards.append(episode_reward)
        episode_scores.append(score)
        episode_steps.append(steps)
        episode_lives_left.append(lives_left)
        results.append(result)

        print(
            f"Episode {episode + 1}: "
            f"Result={result.upper():7s} | "
            f"Score={score:.0f} | Reward={episode_reward:.1f} | Steps={steps} | Lives={lives_left}"
        )

    env.close()

    wins = results.count('win')
    losses = results.count('loss')
    timeouts = results.count('timeout')

    # Estadísticas
    print(f"\n{'='*60}")
    print(f"RESULTADOS ({n_episodes} episodios):")
    print(f"{'='*60}")
    print(f"Score medio:        {np.mean(episode_scores):.2f} +/- {np.std(episode_scores):.2f}")
    print(f"Recompensa media:   {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Pasos medios:       {np.mean(episode_steps):.1f} +/- {np.std(episode_steps):.1f}")
    print(f"Vidas restantes:    {np.mean(episode_lives_left):.2f} +/- {np.std(episode_lives_left):.2f}")
    print(f"Win rate:           {wins / n_episodes:.2%}  ({wins}/{n_episodes})")
    print(f"Timeout rate:       {timeouts / n_episodes:.2%}  ({timeouts}/{n_episodes})")
    print(f"Loss rate:          {losses / n_episodes:.2%}  ({losses}/{n_episodes})")
    print(f"Mejor score:        {np.max(episode_scores):.0f}")
    print(f"Peor score:         {np.min(episode_scores):.0f}")


def play_human(env_kwargs=None):
    """Modo para jugar como humano (probando config personalizada)."""
    import pygame

    env = BreakoutEnv(render_mode='human', **(env_kwargs or {}))
    obs, info = env.reset()

    print("Controles: ← → para mover, Q para salir")

    running = True
    total_reward = 0

    while running:
        env.render()

        action = 0  # NOOP por defecto

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 2
        if keys[pygame.K_q]:
            running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}, Total Reward: {total_reward:.1f}")
            obs, info = env.reset()
            total_reward = 0

    env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluar o jugar Breakout custom.")
    parser.add_argument('--play', action='store_true', help='Modo humano (no requiere modelo)')
    parser.add_argument('model_path', nargs='?', help='Ruta al modelo .zip (para evaluar)')
    parser.add_argument('-n', '--n-episodes', type=int, default=5, help='Episodios de evaluación')
    parser.add_argument('--no-render', action='store_true', help='Desactiva render en evaluación')
    # Parámetros de entorno
    parser.add_argument('--ball-speed', type=float, default=1.0, help='Multiplicador velocidad pelota')
    parser.add_argument('--paddle-width', type=float, default=1.0, help='Multiplicador ancho pala')
    parser.add_argument('--brick-rows', type=int, default=6, help='Filas de ladrillos')
    parser.add_argument('--brick-cols', type=int, default=10, help='Columnas de ladrillos')
    parser.add_argument('--max-steps', type=int, default=10_000, help='Pasos máximos por episodio')
    parser.add_argument('--reward-shaping', action='store_true', help='Activa reward shaping')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    env_kwargs = parse_env_kwargs(args)

    if args.play:
        play_human(env_kwargs=env_kwargs)
        sys.exit(0)

    if not args.model_path:
        parser.error('Debes pasar la ruta al modelo o usar --play')

    model_path = args.model_path
    if not model_path.endswith('.zip'):
        model_path += '.zip'

    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el modelo en {model_path}")
        sys.exit(1)

    test_agent(
        model_path,
        n_episodes=args.n_episodes,
        render=not args.no_render,
        env_kwargs=env_kwargs,
    )
