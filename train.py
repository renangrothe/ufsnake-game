"""
train.py — Loop de treinamento acelerado (Headless)
"""
import numpy as np
from env import SnakeEnv
from agent_dqn import DQNAgent
# from agent_q import QLearningAgent # TODO: corrigir e treinar com o QLearningagent

def train_dqn(episodes=2000):
    env = SnakeEnv(render_mode=None) # Sem renderização para maximizar uso da GPU
    agent = DQNAgent(
        state_shape=(7, 10, 10),
        action_size=3, 
        batch_size=128, # Lote menor funcionou melhor aqui, lote maior funcionou melhor pra processamento de imagens
        epsilon_decay=0.995 # Decaimento leve fez explorar melhor o tabuleiro
    )
    
    scores = []
    
    print("Iniciando treinamento DQN...")
    for e in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Armazena na memória GPU e treina
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            
        agent.decay_epsilon()
        scores.append(info['score'])
        
        # Logs de progresso
        if e % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episódio: {e}/{episodes} | Score Médio (ultimos 50): {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")
            
        # Salva o modelo periodicamente
        if e % 500 == 0:
            agent.save(f"dqn_checkpoint_ep{e}.pth")
            
    agent.save("dqn_final.pth")
    print("Treinamento finalizado.")

if __name__ == "__main__":
    train_dqn(episodes=10000)
