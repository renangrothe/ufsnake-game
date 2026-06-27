"""
play.py — Visualizador do modelo treinado
"""
import time
import os
import glob
import re
from env import SnakeEnv
from agent_dqn import DQNAgent

def get_episode_number(filename):
    """Extrai de forma segura o número do episódio do nome do arquivo."""
    match = re.search(r'ep(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def play_game(model_path=None):
    # fallback para encontrar o melhor modelo
    if model_path is None:
        if os.path.exists("dqn_final.pth"):
            model_path = "dqn_final.pth"
        else:
            checkpoints = glob.glob("dqn_checkpoint_ep*.pth")
            if checkpoints:
                # determinar o maior número
                model_path = max(checkpoints, key=get_episode_number)
            else:
                print("Erro: Nenhum modelo encontrado (nem final, nem checkpoints). Execute train.py primeiro.")
                return

    print(f"Carregando pesos de: {model_path}")

    # Modo 'human' renderiza o Pygame a 30 FPS
    env = SnakeEnv(render_mode="human")
    
    # state_shape atualizado para (7, 10, 10)
    agent = DQNAgent(state_shape=(7, 10, 10), action_size=3, epsilon=0.0)
    
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    print("Iniciando visualização...")
    
    for episode in range(5): # Joga 5 partidas para demonstração
        state, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            env.render()
            
            # Ação escolhida inteiramente pela rede neural
            action = agent.act(state)
            state, reward, done, truncated, info = env.step(action)
            
        print(f"Fim da partida {episode + 1} | Score Final: {info['score']}")
        time.sleep(1) # Pausa dramática antes do próximo jogo
        
    env.close()

if __name__ == "__main__":
    play_game()
