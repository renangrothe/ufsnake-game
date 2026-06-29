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
    best_avg_score = -float('inf')
    
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
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            
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
            agent.decay_epsilon()

        scores.append(info['score'])
        
        # Logs de progresso e salvamento
        if e % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episódio: {e}/{episodes} | Score Médio (ultimos 50): {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")
            
            # Salva apenas se for o melhor modelo até agora
            if warmup and avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save("dqn_best.pth", episode=e)
                print(f" -> Novo recorde alcançado! Modelo salvo em 'dqn_best.pth'")
            
        # Sobrescreve o mesmo arquivo periódico
        if e % 200 == 0:
            agent.save("dqn_latest.pth", episode=e)
            
    agent.save("dqn_final.pth", episode=episodes)
    print("Treinamento finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento DQN para Snake")
    parser.add_argument("--episodes", type=int, default=10000, help="Número total de episódios a treinar")
    parser.add_argument("--resume", type=str, default=None, help="Caminho do arquivo .pth para retomar (ex: dqn_latest.pth)")
    
    args = parser.parse_args()
    
    train_dqn(episodes=args.episodes, resume_path=args.resume)
