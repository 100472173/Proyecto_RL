"""
Evalúa un agente DQN entrenado en el entorno Breakout custom.
"""
import sys
import os
import time
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


def test_agent(model_path, n_episodes=5, render=True):
    """Evalúa un agente entrenado y reporta más métricas útiles."""
    # Crear entorno
    render_mode = 'human' if render else None
    env = DummyVecEnv([make_env(render_mode=render_mode)])
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


def play_human():
    """Modo para jugar como humano (para testing)."""
    import pygame
    
    env = BreakoutEnv(render_mode='human')
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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python test.py <path_al_modelo> [n_episodes]  - Evaluar modelo")
        print("  python test.py --play                         - Jugar como humano")
        print("\nEjemplos:")
        print("  python test.py models/best_model.zip 5")
        print("  python test.py models/dqn_breakout_final.zip")
        print("  python test.py --play")
        sys.exit(1)
    
    if sys.argv[1] == '--play':
        play_human()
    else:
        model_path = sys.argv[1]
        if not model_path.endswith('.zip'):
            model_path += '.zip'
        
        if not os.path.exists(model_path):
            print(f"Error: No se encuentra el modelo en {model_path}")
            sys.exit(1)
        
        n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        test_agent(model_path, n_episodes=n_episodes, render=True)
