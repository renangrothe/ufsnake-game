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
from cpprb import PrioritizedReplayBuffer


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

    # inicializacao substitui o replay buffer antigo pelo do PER
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
        batch_size: int = 128,
        target_update_freq: int = 500,
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

        # Parâmetros do PER
        self.per_alpha = 0.6  # Quão agressiva é a priorização
        self.per_beta = 0.4   # Peso inicial de correção (sobe para 1.0 ao longo do treino)
        self.per_beta_increment = 0.001

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[DQN] Device: {self.device}")

        # Redes
        self.policy_net = QNetworkCNN(action_size).to(self.device)
        self.target_net = QNetworkCNN(action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Configuração do Dicionário para a cpprb
        env_dict = {
            "obs": {"shape": state_shape},
            "act": {"shape": 1, "dtype": int}, # Ação é escalar
            "rew": {},
            "next_obs": {"shape": state_shape},
            "done": {}
        }
        
        # Inicializa o Prioritized Replay Buffer em C++
        self.buffer = PrioritizedReplayBuffer(buffer_capacity, env_dict, alpha=self.per_alpha)

    def act(self, state: np.ndarray) -> int:
        """epsilon-greedy com inferência na GPU (torch.no_grad para evitar grad desnecessário)."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            s_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)   # (7, 10, 10) -> (1, 7, 10, 10)
            return int(self.policy_net(s_t).argmax().item())

    def remember(self, state, action, reward, next_state, done):
        # cpprb espera a adição em formato de dicionário
        self.buffer.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)   

    def learn(self) -> Optional[float]:
        # get_stored_size() pega o número de elementos atuais no buffer da cpprb
        if self.buffer.get_stored_size() < self.batch_size:
            return None

        # Amostragem pela SumTree
        # O beta corrige o viés de amostragem viciado do PER
        sample = self.buffer.sample(self.batch_size, beta=self.per_beta)
        
        # Aumenta o beta gradativamente em direção a 1.0
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        # Transferência para a GPU
        s = torch.as_tensor(sample["obs"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(sample["act"], dtype=torch.long, device=self.device)
        r = torch.as_tensor(sample["rew"], dtype=torch.float32, device=self.device).squeeze(-1)
        s_ = torch.as_tensor(sample["next_obs"], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(sample["done"], dtype=torch.float32, device=self.device).squeeze(-1)
        
        # Pesos de correção (Importance Sampling) para a Loss
        weights = torch.as_tensor(sample["weights"], dtype=torch.float32, device=self.device).squeeze(-1)

        # Matemática do Double DQN
        q_pred = self.policy_net(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            best_actions = self.policy_net(s_).argmax(1, keepdim=True)
            q_next = self.target_net(s_).gather(1, best_actions).squeeze(1)
            q_target = r + self.gamma * q_next * (1.0 - d)

        # Cálculo do TD Error absoluto para atualizar a SumTree
        # Fazemos detach() e movemos para CPU pois a cpprb precisa disso
        td_errors = torch.abs(q_target - q_pred).detach().cpu().numpy()

        # Cálculo da Loss com os pesos do PER
        # Em vez de SmoothL1Loss substituimos manualmente para aplicar o peso por amostra
        elementwise_loss = torch.nn.functional.smooth_l1_loss(q_pred, q_target, reduction="none")
        loss = torch.mean(elementwise_loss * weights)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # quão "surpresa" a rede ficou
        self.buffer.update_priorities(sample["indexes"], td_errors)

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decai epsilon após cada episódio (chamar no final do episódio)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str = "dqn.pth", episode: int = 0) -> None:
        torch.save({
            "policy":      self.policy_net.state_dict(),
            "target":      self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "train_steps": self._train_steps,
            "episode":     episode, # Salvando o episódio atual
        }, path)
        print(f"[DQN] Checkpoint salvo → {path}")

    def load(self, path: str = "dqn.pth") -> int:
        # weights_only=False: checkpoint contém dicts com state_dicts + metadados
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon",     self.epsilon_min)
        self._train_steps = ckpt.get("train_steps", 0)
        self.target_net.eval()
        
        episode = ckpt.get("episode", 0) # Recuperando o episódio
        
        print(f"[DQN] Checkpoint carregado ← {path}  "
              f"(passo {self._train_steps}, ε={self.epsilon:.4f}, episódio {episode})")
        
        return episode # Retornando para o loop de treino
