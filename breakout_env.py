"""
Entorno Breakout personalizado usando Gymnasium.
Permite modificar: velocidad de pelota, tamaño de pala, número de ladrillos, etc.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class BreakoutEnv(gym.Env):
    """
    Entorno Breakout customizable para curriculum learning.
    
    Parámetros modificables:
        - ball_speed: velocidad de la pelota (1.0 = normal)
        - paddle_width: ancho de la pala (1.0 = normal)
        - brick_rows: número de filas de ladrillos
        - brick_cols: número de columnas de ladrillos
        - max_steps: máximo de pasos por episodio
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        render_mode=None,
        ball_speed=1.0,
        paddle_width=1.0,
        brick_rows=6,
        brick_cols=10,
        max_steps=10000,
        reward_shaping=False
    ):
        super().__init__()
        
        # Configuración del juego
        self.screen_width = 160
        self.screen_height = 210
        self.render_mode = render_mode
        
        # Parámetros modificables (para curriculum learning)
        self.ball_speed_multiplier = ball_speed
        self.paddle_width_multiplier = paddle_width
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        
        # Dimensiones base
        self.paddle_height = 4
        self.paddle_base_width = 24
        self.paddle_width = int(self.paddle_base_width * self.paddle_width_multiplier)
        self.ball_size = 4
        self.brick_width = self.screen_width // self.brick_cols
        self.brick_height = 8
        self.brick_top_offset = 50
        
        # Velocidad base de la pelota
        self.ball_base_speed = 3.0
        
        # Espacio de acciones: 0=NOOP, 1=LEFT, 2=RIGHT
        self.action_space = spaces.Discrete(3)
        
        # Espacio de observaciones: imagen en escala de grises 84x84x1 (channel-last)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
        
        # Pygame
        self.screen = None
        self.clock = None
        self.font = None
        
        # Estado del juego
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Recalcular paddle width (por si cambia entre episodios)
        self.paddle_width = int(self.paddle_base_width * self.paddle_width_multiplier)
        
        # Posición de la pala (centro inferior)
        self.paddle_x = self.screen_width // 2 - self.paddle_width // 2
        self.paddle_y = self.screen_height - 20
        
        # Posición de la pelota (más arriba, cerca de los ladrillos)
        self.ball_x = float(self.screen_width // 2)
        self.ball_y = float(self.brick_top_offset + self.brick_rows * self.brick_height + 30)
        
        # Velocidad inicial (hacia abajo con ángulo aleatorio)
        angle = self.np_random.uniform(-0.5, 0.5)
        speed = self.ball_base_speed * self.ball_speed_multiplier
        self.ball_vx = speed * np.sin(angle)
        self.ball_vy = abs(speed * np.cos(angle))  # Siempre hacia abajo al inicio
        
        # Frames de gracia antes de que la pelota se mueva
        self.grace_frames = 30  # ~0.5 segundos a 60 FPS
        
        # Crear ladrillos
        self._create_bricks()
        
        # Contadores
        self.steps = 0
        self.score = 0
        self.lives = 5
        
        # Para reward shaping
        self.last_ball_y = self.ball_y
        self.hit_paddle_this_step = False
        
        observation = self._get_observation()
        info = {"score": self.score, "lives": self.lives}
        
        return observation, info
    
    def _create_bricks(self):
        """Crea la matriz de ladrillos."""
        self.bricks = []
        self.brick_width = self.screen_width // self.brick_cols
        
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                brick = {
                    "x": col * self.brick_width,
                    "y": self.brick_top_offset + row * self.brick_height,
                    "width": self.brick_width - 2,
                    "height": self.brick_height - 2,
                    "alive": True,
                    "color": self._get_brick_color(row)
                }
                self.bricks.append(brick)
    
    def _get_brick_color(self, row):
        """Devuelve el color del ladrillo según la fila."""
        colors = [
            (255, 0, 0),      # Rojo
            (255, 128, 0),    # Naranja
            (255, 255, 0),    # Amarillo
            (0, 255, 0),      # Verde
            (0, 128, 255),    # Azul claro
            (128, 0, 255),    # Morado
        ]
        return colors[row % len(colors)]
    
    def step(self, action):
        self.steps += 1
        self.hit_paddle_this_step = False
        reward = 0.0
        
        # Mover pala (siempre se puede mover, incluso en grace period)
        paddle_speed = 4
        if action == 1:  # LEFT
            self.paddle_x = max(0, self.paddle_x - paddle_speed)
        elif action == 2:  # RIGHT
            self.paddle_x = min(self.screen_width - self.paddle_width, self.paddle_x + paddle_speed)
        
        # Durante grace period, la pelota no se mueve
        if self.grace_frames > 0:
            self.grace_frames -= 1
        else:
            # Mover pelota
            self.ball_x += self.ball_vx
            self.ball_y += self.ball_vy
            
            # Colisiones con paredes laterales
            if self.ball_x <= 0:
                self.ball_x = 0
                self.ball_vx = -self.ball_vx
            elif self.ball_x >= self.screen_width - self.ball_size:
                self.ball_x = self.screen_width - self.ball_size
                self.ball_vx = -self.ball_vx
            
            # Colisión con techo
            if self.ball_y <= 0:
                self.ball_y = 0
                self.ball_vy = -self.ball_vy
            
            # Colisión con pala
            if self._check_paddle_collision():
                self.hit_paddle_this_step = True
                self.ball_vy = -abs(self.ball_vy)  # Siempre hacia arriba
                
                # Ajustar ángulo según dónde golpea
                paddle_center = self.paddle_x + self.paddle_width / 2
                hit_pos = (self.ball_x + self.ball_size / 2 - paddle_center) / (self.paddle_width / 2)
                hit_pos = np.clip(hit_pos, -1, 1)
                
                speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
                self.ball_vx = speed * hit_pos * 0.8
                self.ball_vy = -np.sqrt(speed**2 - self.ball_vx**2)
                
                # Reward shaping: pequeña recompensa por golpear la pelota
                if self.reward_shaping:
                    reward += 0.1
            
            # Colisión con ladrillos
            brick_reward = self._check_brick_collisions()
            reward += brick_reward
            self.score += int(brick_reward)
            
            # Pelota cae abajo (pierde vida)
            if self.ball_y >= self.screen_height:
                self.lives -= 1
                if self.lives <= 0:
                    reward -= 1.0  # Penalización por perder
                else:
                    self._reset_ball()
        
        # Comprobar terminación
        terminated = False
        if self.lives <= 0:
            terminated = True
        
        # Ganar (todos los ladrillos destruidos)
        if all(not b["alive"] for b in self.bricks):
            terminated = True
            reward += 10.0  # Bonus por ganar
        
        # Máximo de pasos
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        info = {"score": self.score, "lives": self.lives}
        
        return observation, reward, terminated, truncated, info
    
    def _reset_ball(self):
        """Resetea la pelota después de perder una vida."""
        # Pelota aparece arriba, cerca de los ladrillos
        self.ball_x = float(self.screen_width // 2)
        self.ball_y = float(self.brick_top_offset + self.brick_rows * self.brick_height + 30)
        
        # Velocidad hacia abajo con ángulo aleatorio
        angle = self.np_random.uniform(-0.5, 0.5)
        speed = self.ball_base_speed * self.ball_speed_multiplier
        self.ball_vx = speed * np.sin(angle)
        self.ball_vy = abs(speed * np.cos(angle))  # Siempre hacia abajo
        
        # Frames de gracia para que el agente reaccione
        self.grace_frames = 30
    
    def _check_paddle_collision(self):
        """Comprueba colisión entre pelota y pala."""
        if self.ball_vy < 0:  # Solo si la pelota va hacia abajo
            return False
        
        ball_bottom = self.ball_y + self.ball_size
        ball_right = self.ball_x + self.ball_size
        
        if (ball_bottom >= self.paddle_y and 
            self.ball_y <= self.paddle_y + self.paddle_height and
            ball_right >= self.paddle_x and 
            self.ball_x <= self.paddle_x + self.paddle_width):
            return True
        return False
    
    def _check_brick_collisions(self):
        """Comprueba colisión con ladrillos y devuelve recompensa."""
        reward = 0.0
        ball_rect = (self.ball_x, self.ball_y, self.ball_size, self.ball_size)
        
        for brick in self.bricks:
            if not brick["alive"]:
                continue
            
            brick_rect = (brick["x"], brick["y"], brick["width"], brick["height"])
            
            if self._rects_collide(ball_rect, brick_rect):
                brick["alive"] = False
                reward += 1.0
                
                # Rebotar pelota
                ball_center_x = self.ball_x + self.ball_size / 2
                ball_center_y = self.ball_y + self.ball_size / 2
                brick_center_x = brick["x"] + brick["width"] / 2
                brick_center_y = brick["y"] + brick["height"] / 2
                
                dx = ball_center_x - brick_center_x
                dy = ball_center_y - brick_center_y
                
                if abs(dx / brick["width"]) > abs(dy / brick["height"]):
                    self.ball_vx = -self.ball_vx
                else:
                    self.ball_vy = -self.ball_vy
                
                break  # Solo un ladrillo por frame
        
        return reward
    
    def _rects_collide(self, rect1, rect2):
        """Comprueba si dos rectángulos colisionan."""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2)
    
    def _get_observation(self):
        """Genera la observación (imagen 84x84 en escala de grises)."""
        # Crear imagen
        obs = np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
        
        # Dibujar ladrillos
        for brick in self.bricks:
            if brick["alive"]:
                x, y = int(brick["x"]), int(brick["y"])
                w, h = int(brick["width"]), int(brick["height"])
                # Usar diferentes tonos de gris para las filas
                gray_value = 100 + (brick["y"] - self.brick_top_offset) // self.brick_height * 20
                gray_value = min(200, gray_value)
                obs[y:y+h, x:x+w] = gray_value
        
        # Dibujar pala
        px, py = int(self.paddle_x), int(self.paddle_y)
        obs[py:py+self.paddle_height, px:px+self.paddle_width] = 255
        
        # Dibujar pelota
        bx, by = int(self.ball_x), int(self.ball_y)
        bx = np.clip(bx, 0, self.screen_width - self.ball_size)
        by = np.clip(by, 0, self.screen_height - self.ball_size)
        obs[by:by+self.ball_size, bx:bx+self.ball_size] = 255
        
        # Redimensionar a 84x84
        import cv2
        obs_resized = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Añadir dimensión del canal (84, 84) -> (84, 84, 1)
        obs_resized = np.expand_dims(obs_resized, axis=-1)
        
        return obs_resized
    
    def render(self):
        if self.render_mode is None:
            return None
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width * 3, self.screen_height * 3)
            )
            pygame.display.set_caption("Breakout - Custom Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        # Limpiar pantalla
        self.screen.fill((0, 0, 0))
        
        # Dibujar ladrillos
        for brick in self.bricks:
            if brick["alive"]:
                rect = pygame.Rect(
                    brick["x"] * 3, brick["y"] * 3,
                    brick["width"] * 3, brick["height"] * 3
                )
                pygame.draw.rect(self.screen, brick["color"], rect)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)
        
        # Dibujar pala
        paddle_rect = pygame.Rect(
            self.paddle_x * 3, self.paddle_y * 3,
            self.paddle_width * 3, self.paddle_height * 3
        )
        pygame.draw.rect(self.screen, (200, 200, 200), paddle_rect)
        
        # Dibujar pelota
        ball_rect = pygame.Rect(
            int(self.ball_x * 3), int(self.ball_y * 3),
            self.ball_size * 3, self.ball_size * 3
        )
        pygame.draw.rect(self.screen, (255, 255, 255), ball_rect)
        
        # Dibujar score y vidas
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        lives_text = self.font.render(f"Lives: {self.lives}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (self.screen_width * 3 - 100, 10))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


# Registrar el entorno en Gymnasium
gym.register(
    id='CustomBreakout-v0',
    entry_point='breakout_env:BreakoutEnv',
)
