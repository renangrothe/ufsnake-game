"""
train.py — Loop de treinamento acelerado (Headless)
"""
import numpy as np
import argparse
import os
from env import SnakeEnv
from agent_dqn import DQNAgent

def train_dqn(episodes=10000, resume_path=None):
    env = SnakeEnv(render_mode=None)
    agent = DQNAgent(
        state_shape=(7, 10, 10),
        action_size=3,
        batch_size=256,
        epsilon_decay=0.995
    )

    start_episode = 1
    scores = []
    episode_lengths = []
    best_avg = -float('inf')   # melhor média já observada

    # Parâmetros de reset
    CYCLE_LENGTH = 2000
    STAGNATION_WINDOW = 1000 # em episódios
    IMPROVEMENT_THRESHOLD = 0.5 # aumento mínimo para considerar melhora

    # Histórico para detecção de estagnação
    avg_history = []
    stagnant_counter = 0

    # Lógica de carregamento
    if resume_path and os.path.exists(resume_path):
        print(f"Carregando checkpoint de '{resume_path}'...")
        start_episode = agent.load(resume_path) + 1
        print(f"Treinamento retomado do episódio {start_episode}.")
    elif resume_path:
        print(f"Aviso: Arquivo '{resume_path}' não encontrado. Iniciando do zero.")

    print("Iniciando treinamento DQN...")

    warmup = False  # buffer ainda não cheio

    for e in range(start_episode, episodes + 1):
        state, _ = env.reset()
        done = False
        truncated = False
        steps = 0

        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            steps += 1

            agent.remember(state, action, reward, next_state, done)

            if agent.buffer.get_stored_size() > 10_000:
                if not warmup:
                    print(f"\n    Inicialização do buffer concluída (Episódio {e}) Iniciando backpropagation...")
                    warmup = True
                agent.learn()

            state = next_state

        # Decaimento da temperatura
        agent.decay_temperature()

        scores.append(info['score'])
        episode_lengths.append(steps)

        # LOG E RESETS a cada 50 episódios)
        if e % 50 == 0:
            avg_score = np.mean(scores[-50:])
            avg_length = np.mean(episode_lengths[-50:]) if episode_lengths else 0
            current_lr = agent.optimizer.param_groups[0]['lr']
            current_beta = agent.per_beta
            current_temp = agent.temperature

            print(f"Ep: {e:6d}/{episodes} | "
                  f"Score: {avg_score:5.2f} | "
                  f"Steps: {avg_length:6.1f} | "
                  f"Temp: {current_temp:.4f} | "
                  f"Beta: {current_beta:.3f} | "
                  f"LR: {current_lr:.2e}")

            # Detecção de estagnação, aumenta 'curiosidade'
            avg_history.append(avg_score)
            if len(avg_history) > STAGNATION_WINDOW // 50:
                avg_history.pop(0)

            if avg_score > best_avg + IMPROVEMENT_THRESHOLD:
                best_avg = avg_score
                stagnant_counter = 0
            else:
                stagnant_counter += 1

            # Reset por estagnação
            if stagnant_counter >= (STAGNATION_WINDOW // 50):
                if agent.temperature < 0.5:
                    agent.temperature = 1.0
                    print(f"* Reset de temperatura por estagnação no ep {e} *")
                    stagnant_counter = 0

                    # Salva o melhor modelo se for recorde
                    if warmup and avg_score > best_avg:
                        best_avg = avg_score
                        agent.save("dqn_best.pth", episode=e)
                        print(f" -> Novo recorde alcançado! Modelo salvo em 'dqn_best.pth'")

        # Salvamento periódico
        if e % 200 == 0:
            agent.save("dqn_latest.pth", episode=e)

        # Reset por ciclo
        if e % CYCLE_LENGTH == 0 and agent.temperature <= agent.temp_min + 0.05:
            agent.temperature = 1.0
            print(f"*** Reset de temperatura por ciclo no ep {e} ***")

    agent.save("dqn_final.pth", episode=episodes)
    print("Treinamento finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento DQN para Snake")
    parser.add_argument("--episodes", type=int, default=10000, help="Número total de episódios a treinar")
    parser.add_argument("--resume", type=str, default=None, help="Caminho do arquivo .pth para retomar (ex: dqn_latest.pth)")
    args = parser.parse_args()
    train_dqn(episodes=args.episodes, resume_path=args.resume)
