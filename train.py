"""
train.py — Loop de treinamento acelerado (Headless)
"""
import numpy as np
import argparse
import os
from env import SnakeEnv
from agent_dqn import DQNAgent

def train_dqn(episodes=10000, resume_path=None):
    env = SnakeEnv(render_mode=None) # Sem renderização para maximizar uso da GPU
    agent = DQNAgent(
        state_shape=(7, 10, 10),
        action_size=3, 
        batch_size=128,
        epsilon_decay=0.995 
    )
    
    start_episode = 1
    scores = []
    steps = 0
    episode_lengths = [] #armazena quantos passos durou cada ep
    best_avg_score = -float('inf')
    CYCLE_LENGTH = 2000
    
    # Lógica de carregamento
    if resume_path and os.path.exists(resume_path):
        print(f"Carregando checkpoint de '{resume_path}'...")
        start_episode = agent.load(resume_path) + 1
        print(f"Treinamento retomado do episódio {start_episode}.")
    elif resume_path:
        print(f"Aviso: Arquivo '{resume_path}' não encontrado. Iniciando do zero.")

    print("Iniciando treinamento DQN...")

    warmup = False # nao preencher o buffer com jogadas ruins no comeco, e gerenciar exploracao inicial

    for e in range(start_episode, episodes + 1):
        state, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            steps += 1
            
            # Armazena na memória da sumtree e treina
            agent.remember(state, action, reward, next_state, done)

            # otimizar os pesos
            if agent.buffer.get_stored_size() > 10_000:
                if not warmup:
                    print(f"\n    Inicialização do buffer concluída (Episódio {e}) Iniciando backpropagation...")
                    warmup= True

                agent.learn()
            
            state = next_state
            
        # agora so gasta a taxa de exploracao se a rede ja esta aprendendo 
        if agent.buffer.get_stored_size() > 10_000:
            #agent.decay_epsilon()
            agent.decay_temperature()

        scores.append(info['score'])
        episode_lengths.append(steps)
        
        # Logs de progresso e salvamento
        if e % 50 == 0:
            avg_score = np.mean(scores[-50:])
            avg_length = np.mean(episode_lengths[-50:]) if episode_lengths else 0

            # Pega a LR atual do otimizador
            current_lr = agent.optimizer.param_groups[0]['lr']

            # Pega o beta do PER (correção de viés)
            current_beta = agent.per_beta

            # Pega a temperatura (se você migrou para Boltzmann)
            current_temp = agent.temperature

            print(f"Ep: {e:6d}/{episodes} | "
                f"Score: {avg_score:5.2f} | "
                f"Steps: {avg_length:6.1f} | "
                f"Temp: {current_temp:.4f} | "
                f"Beta: {current_beta:.3f} | "
                f"LR: {current_lr:.2e}")

            # Salva apenas se for o melhor modelo até agora
            if warmup and avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save("dqn_best.pth", episode=e)
                print(f" -> Novo recorde alcançado! Modelo salvo em 'dqn_best.pth'")
            
        # Sobrescreve o mesmo arquivo periódico
        if e % 200 == 0:
            agent.save("dqn_latest.pth", episode=e)

        # if e % CYCLE_LENGTH == 0 and agent.epsilon <= 0.1: # favorece a exploracao, visando remover de possiveis minimos locais
        #     agent.epsilon = 0.5
        if e % CYCLE_LENGTH == 0 and agent.temperature <= agent.temp_min + 0.05:
            agent.temperature = 1.0   # reseta para explorar novamente
            
    agent.save("dqn_final.pth", episode=episodes)
    print("Treinamento finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento DQN para Snake")
    parser.add_argument("--episodes", type=int, default=10000, help="Número total de episódios a treinar")
    parser.add_argument("--resume", type=str, default=None, help="Caminho do arquivo .pth para retomar (ex: dqn_latest.pth)")
    
    args = parser.parse_args()
    
    train_dqn(episodes=args.episodes, resume_path=args.resume)
