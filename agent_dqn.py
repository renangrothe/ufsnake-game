"""
agent_dqn.py — Deep Q-Network (Double DQN) otimizado para RTX 3050 Mobile

Otimizações CUDA:
  • ReplayBuffer pré-alocado diretamente na GPU — elimina alocações dinâmicas
    e minimiza transferências CPU→GPU (1 push = 1 tensor.to(device) por campo)
  • Batch size grande (1024) para saturar os CUDA cores do Ampere
  • Inferência com torch.no_grad() nas chamadas de act() e no target_net
  • Double DQN: policy_net escolhe ação; target_net avalia Q(s',a')
    → reduz superestimação sistemática do Q-valor
  • Huber Loss (SmoothL1) mais robusta a outliers do que MSE puro
  • Gradient clipping (norm ≤ 1.0) para estabilidade numérica
"""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ═══════════════════════════════════════════════════════════════════════ #
#  Rede Neural                                                             #
# ═══════════════════════════════════════════════════════════════════════ #

class QNetworkCNN(nn.Module):
    """
    Arquitetura Convolucional para visão global do grid 10x10.
    Input: Tensor (Batch, 7, 10, 10)
    """
    def __init__(self, action_size: int = 3): # Mantém as 3 ações (reto, dir, esq)
        super().__init__()
        
        # Extração de características espaciais
        self.conv = nn.Sequential(
            # Entrada: 7 canais, 10x10. Saída: 16 canais, 8x8 (sem padding)
            nn.Conv2d(in_channels=7, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            # Entrada: 16 canais, 8x8. Saída: 32 canais, 6x6
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        flatten_size = 64 * 6 * 6
        
        # Tomada de decisão
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        features = self.conv(x)
        features = features.reshape(features.size(0), -1) # Flatten (Batch, 1152)
        q_values = self.fc(features)
        
        return q_values

# ═══════════════════════════════════════════════════════════════════════ #
#  Replay Buffer pré-alocado na GPU                                        #
# ═══════════════════════════════════════════════════════════════════════ #

class ReplayBuffer:
    """
    Buffer circular de experiências (s, a, r, s', done) pré-alocado na GPU.

    Por que pré-alocar?

    A abordagem convencional (deque + stack no batch) faz N alocações de
    tensor por passo e uma grande transferência CPU para GPU por batch.
    Aqui, toda a memória já está na GPU; cada push() faz apenas uma
    transferência de 1 vetor/escalar por campo — ~5× menos overhead.
    """

    def __init__(self, capacity: int, state_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self._ptr     = 0
        self._size    = 0

        self.s  = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.a  = torch.zeros( capacity, dtype=torch.long, device=device)
        self.r  = torch.zeros( capacity, dtype=torch.float32, device=device)
        self.s_ = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.d  = torch.zeros( capacity, dtype=torch.float32, device=device)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Armazena uma transição no buffer circular. Uma transferência por campo."""
        i = self._ptr % self.capacity
        self.s [i] = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        self.a [i] = action
        self.r [i] = reward
        self.s_[i] = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
        self.d [i] = float(done)
        self._ptr  += 1
        self._size  = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Amostragem uniforme diretamente nos tensores GPU."""
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return (
            self.s [idx],
            self.a [idx],
            self.r [idx],
            self.s_[idx],
            self.d [idx],
        )

    def __len__(self) -> int:
        return self._size


# ═══════════════════════════════════════════════════════════════════════ #
#  Agente DQN                                                              #
# ═══════════════════════════════════════════════════════════════════════ #

class DQNAgent:
    """
    Double DQN para o ambiente Snake.

    Double DQN — motivação:
        DQN padrão usa target_net tanto para escolher quanto avaliar ação
        no estado seguinte, o que introduz viés positivo (overestimation).
        Double DQN separa os papéis:
            ação*  = argmax_a  policy_net(s') escolhe
            target = r + γ · target_net(s', ação*) target_net avalia
    """

    def __init__(
        self,
        state_shape: tuple = (7, 10, 10),
        action_size: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon:  float = 1.0,
        epsilon_min:  float = 0.05,
        epsilon_decay: float = 0.999,
        buffer_capacity: int = 100_000,
        batch_size: int = 1024,
        target_update_freq: int = 500,   # passos de treino entre sync
        device: Optional[str] = None,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._train_steps = 0

        # Seleciona CUDA se disponível
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[DQN] Device: {self.device}")
        if self.device.type == "cuda":
            print(f"[DQN] GPU: {torch.cuda.get_device_name(self.device)}")

        # Redes
        self.policy_net = QNetworkCNN(action_size).to(self.device)
        self.target_net = QNetworkCNN(action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # target_net: apenas inferência, nunca acumula grad

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # Huber Loss (SmoothL1)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(buffer_capacity, state_shape, self.device)

    def act(self, state: np.ndarray) -> int:
        """epsilon-greedy com inferência na GPU (torch.no_grad para evitar grad desnecessário)."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            s_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)   # (7, 10, 10) -> (1, 7, 10, 10)
            return int(self.policy_net(s_t).argmax().item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        """
        Um passo de treino (backpropagation).

        Retorna o valor da loss ou None se o buffer ainda não tem amostras suficientes.
        """
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s_, d = self.buffer.sample(self.batch_size)

        # Q(s, a) — valores previstos pela policy_net
        # gather: seleciona a coluna correspondente à ação tomada -> shape [batch]
        q_pred = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target sem grad (target_net é read-only durante treino)
        with torch.no_grad():
            # policy_net escolhe qual ação tomar em s'
            best_actions = self.policy_net(s_).argmax(1, keepdim=True)
            # target_net avalia o Q dessa ação
            q_next   = self.target_net(s_).gather(1, best_actions).squeeze(1)
            q_target = r + self.gamma * q_next * (1.0 - d)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping previne explosão de gradientes em episódios longos
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Sincroniza target_net a cada N passos de treino
        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decai epsilon após cada episódio (chamar no final do episódio)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str = "dqn.pth") -> None:
        torch.save({
            "policy":      self.policy_net.state_dict(),
            "target":      self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "train_steps": self._train_steps,
        }, path)
        print(f"[DQN] Checkpoint salvo → {path}")

    def load(self, path: str = "dqn.pth") -> None:
        # weights_only=False: checkpoint contém dicts com state_dicts + metadados
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon",     self.epsilon_min)
        self._train_steps = ckpt.get("train_steps", 0)
        self.target_net.eval()
        print(f"[DQN] Checkpoint carregado ← {path}  "
              f"(passo {self._train_steps}, ε={self.epsilon:.4f})")
