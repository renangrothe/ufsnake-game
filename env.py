"""
env.py — Ambiente Snake compatível com Gymnasium
Ponto de partida: implementação BFS em C (UFSKernel/snake)

Espaço de estados : tensor 3D (7, 10, 10)
Espaço de ações   : Discrete(3) — reto, direita, esquerda (relativo)
Recompensas       : +30 maçã | -100 colisão | +0.1/-0.1 aproximação/afastamento | -0.1 por passo
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
from enum import IntEnum


class Direction(IntEnum):
    RIGHT = 0
    DOWN  = 1
    LEFT  = 2
    UP    = 3

# Deslocamento (delta_row, delta_col) por direção, row cresce para baixo
DIR_VEC = {
    Direction.RIGHT: ( 0,  1),
    Direction.DOWN: ( 1,  0),
    Direction.LEFT: ( 0, -1),
    Direction.UP: (-1,  0),
}

# Ordem horária usado para calcular viradas relativas
CLOCKWISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

class SnakeEnv(gym.Env):
    """
    Snake 10×10 compatível com Gymnasium.

    Ações (relativas à direção atual):
        0 = reto          (mantém direção)
        1 = virar direita (próximo em sentido horário)
        2 = virar esquerda

    Recompensas:
        +10  : cabeça alcança a maçã
        -10  : colisão com parede ou corpo
        +0.1 : passo que diminui distância de Manhattan até a maçã
        -0.1 : passo que aumenta distância de Manhattan
    """

    metadata   = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    BOARD_SIZE = 10   # área jogável (sem bordas extras)
    CELL_PX    = 50   # pixels por célula no render

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space      = spaces.Discrete(3)
        # 7 canais: cabeça, corpo, maçã, + 4 one‑hot para direção
        self.observation_space = spaces.Box(0.0, 1.0, shape=(7, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        self._window = None
        self._clock  = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        c = self.BOARD_SIZE // 2
        self.direction = Direction.RIGHT

        # Cobra inicial começa com 3 células no centro, orientada para a direita
        self.snake = deque([(c, c), (c, c - 1), (c, c - 2)])

        self._place_food()
        self._steps_no_food = 0
        self.score = 0

        return self._state(), {}

    def step(self, action: int):
        self._steps_no_food += 1

        # Atualiza direção com base na ação relativa
        idx = CLOCKWISE.index(self.direction)
        if   action == 1: self.direction = CLOCKWISE[(idx + 1) % 4] # direita
        elif action == 2: self.direction = CLOCKWISE[(idx - 1) % 4] # esquerda
        # action == 0 mantém a direção atual

        dr, dc   = DIR_VEC[self.direction]
        head_r, head_c = self.snake[0] # extrai a cabeça
        new_head = (head_r + dr, head_c + dc)

        # Distância de Manhattan antes do movimento (reward shaping)
        d_before = self._manhattan(self.snake[0])

        # Verifica colisão com parede ou corpo
        r, c = new_head
        if (r < 0 or r >= self.BOARD_SIZE or
                c < 0 or c >= self.BOARD_SIZE or
                new_head in set(self.snake)):
            return self._state(), -100.0, True, False, {"score": self.score}

        # Avança a cobra
        self.snake.appendleft(new_head)
        d_after = self._manhattan(new_head)

        if new_head == self.food:       # comeu a maçã
            self.score          += 1
            self._steps_no_food  = 0
            reward = 30.0
            self._place_food()
        else:                           # movimento normal
            self.snake.pop()            # remove a cauda
            reward = 0.1 if d_after < d_before else -0.1
            reward -= 0.01

        # Truncate previne loops infinitos (cobra presa em ciclo)
        truncated = self._steps_no_food > 300 * len(self.snake)

        return self._state(), reward, False, truncated, {"score": self.score}

    def render(self):
        if   self.render_mode == "human":     self._render_human()
        elif self.render_mode == "rgb_array": return self._render_frame()

    def close(self):
        if self._window:
            import pygame
            pygame.quit()
            self._window = None

    def _place_food(self):
        """Posiciona a maçã em uma célula não ocupada pela cobra."""
        occupied = set(self.snake)
        while True:
            pos = (
                int(self.np_random.integers(0, self.BOARD_SIZE)),
                int(self.np_random.integers(0, self.BOARD_SIZE)),
            )
            if pos not in occupied:
                self.food = pos
                return

    def _manhattan(self, pos):
        """Distância de Manhattan entre uma posição (tupla) e a maçã."""
        return abs(pos[0] - self.food[0]) + abs(pos[1] - self.food[1])   # ← CORREÇÃO

    def _state(self) -> np.ndarray:
        """
        Retorna o tabuleiro como um tensor 3D de shape (7, BOARD_SIZE, BOARD_SIZE)
        Canal 0: Posição da Cabeça
        Canal 1: Posição do Corpo
        Canal 2: Posição da Maçã
        Canal 3,4,5,6: Direção (One-hot preenchendo o canal inteiro)
        """
        state = np.zeros((7, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        
        # Canal 0: Cabeça
        head_r, head_c = self.snake[0]
        state[0, head_r, head_c] = 1.0
        
        # Canal 1: Corpo (ignorando a cabeça)
        for i in range(1, len(self.snake)):
            r, c = self.snake[i]
            state[1, r, c] = 1.0
            
        # Canal 2: Maçã
        food_r, food_c = self.food
        state[2, food_r, food_c] = 1.0

        # Canais 3, 4, 5, 6: Direção atual (One-hot preenchendo o canal inteiro)
        dir_channel = 3 + int(self.direction)
        state[dir_channel, :, :] = 1.0
        
        return state
    
    def _render_frame(self) -> np.ndarray:
        """Renderiza o tabuleiro como array NumPy (H, W, 3) — chamado pelo play.py."""
        sz    = self.BOARD_SIZE * self.CELL_PX
        frame = np.full((sz, sz, 3), 30, dtype=np.uint8)   # fundo escuro

        # Cores
        BG_COLOR    = (30, 30, 30)
        HEAD_COLOR  = (0, 255, 0)      # verde vivo
        BODY_COLOR  = (0, 200, 0)      # verde mais escuro
        FOOD_COLOR  = (255, 0, 0)      # vermelho
        GRID_COLOR  = (40, 40, 40)

        # Desenha as células (grid)
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                y0, x0 = r * self.CELL_PX, c * self.CELL_PX
                # Preenche com a cor de fundo
                frame[y0:y0 + self.CELL_PX - 1,
                      x0:x0 + self.CELL_PX - 1] = BG_COLOR

        # Corpo da cobra
        for i, (r, c) in enumerate(self.snake):
            color = HEAD_COLOR if i == 0 else BODY_COLOR
            y0 = r * self.CELL_PX + 2
            x0 = c * self.CELL_PX + 2
            frame[y0:y0 + self.CELL_PX - 4,
                  x0:x0 + self.CELL_PX - 4] = color

        # Maçã
        fr, fc = self.food
        y0 = fr * self.CELL_PX + 4
        x0 = fc * self.CELL_PX + 4
        frame[y0:y0 + self.CELL_PX - 8,
              x0:x0 + self.CELL_PX - 8] = FOOD_COLOR

        return frame

    def _render_human(self):
        """Renderização via Pygame para play.py."""
        import pygame
        if self._window is None:
            pygame.init()
            sz = self.BOARD_SIZE * self.CELL_PX
            self._window = pygame.display.set_mode((sz, sz + 40))
            pygame.display.set_caption("Snake RL")
            self._clock = pygame.time.Clock()

        frame = self._render_frame()
        # Pygame espera (W, H, 3), NumPy produz (H, W, 3) -> transpõe
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        self._window.fill((20, 20, 20))
        self._window.blit(surf, (0, 40))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
