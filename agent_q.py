"""
agent_q.py — Agente Q-Learning Tabular

Como o vetor de estado é binário (11 bits → 2048 combinações possíveis),
a tabela Q pode ser representada de forma exata como um dicionário.

Tabela Q: estado (tupla de 11 ints) → array NumPy de 3 Q-valores


Tem que refazer com o o novo tensor de input
"""

import numpy as np
import pickle
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning off-policy com política ε-greedy.

    Atualização (Bellman):
        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s', a') - Q(s,a)]

    Parâmetros:
        lr            : taxa de aprendizado (α)
        gamma         : fator de desconto (γ) — quanto valoriza recompensas futuras
        epsilon       : probabilidade de exploração (ação aleatória)
        epsilon_min   : piso mínimo de epsilon
        epsilon_decay : multiplicador de decaimento por episódio
    """

    def __init__(
        self,
        state_size:    int   = 11,
        action_size:   int   = 3,
        lr:            float = 0.1,
        gamma:         float = 0.9,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.state_size    = state_size
        self.action_size   = action_size
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # defaultdict: estado nunca visto → Q-valores zerados para todas as ações
        self.q_table: dict = defaultdict(
            lambda: np.zeros(action_size, dtype=np.float32)
        )

    # ------------------------------------------------------------------ #

    def act(self, state: np.ndarray) -> int:
        """ε-greedy: explora aleatoriamente com prob. ε, greedy caso contrário."""
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))

        # Converte para tupla de inteiros — chave do dicionário
        key = tuple(state.astype(np.int8))
        return int(np.argmax(self.q_table[key]))

    def learn(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Um passo de atualização Q-Learning."""
        s  = tuple(state.astype(np.int8))
        s_ = tuple(next_state.astype(np.int8))

        q_cur = self.q_table[s][action]

        # Se terminal, não há estado futuro
        q_target = (
            reward if done
            else reward + self.gamma * float(np.max(self.q_table[s_]))
        )

        # Atualização incremental
        self.q_table[s][action] += self.lr * (q_target - q_cur)

    def decay_epsilon(self) -> None:
        """Decai epsilon após cada episódio (chamar no final do episódio)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str = "q_table.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"[Q-Learning] Tabela salva → {path}  "
              f"({len(self.q_table)} estados descobertos)")

    def load(self, path: str = "q_table.pkl") -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(
            lambda: np.zeros(self.action_size, dtype=np.float32), data
        )
        self.epsilon = self.epsilon_min # modo exploração mínima após carregar
        print(f"[Q-Learning] Tabela carregada ← {path}  "
              f"({len(self.q_table)} estados)")
