import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import torch
import numpy as np
from types import SimpleNamespace
from basic_config import config as config_dict
from collections import deque
from traffic_env import SumoTrafficEnv
from Agent import Agent
def main():
    config = SimpleNamespace(**config_dict)
    config.device = "cuda" if torch.cuda.is_available() and config.cuda else "cpu"
    print(f"Using device: {config.device}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    env = SumoTrafficEnv(
    sumo_cfg_path="C:///Users//adity//Projects_of_Aditya//Research_Paper_Project//sumo_files//bengaluru_junction.sumocfg",
    tls_id="10043357419",
    incoming_lanes=[
            "638928399#0_0", "638928399#0_1", "638928399#0_2",
            "111814614#4_0", "111814614#4_1", #"111814614#4_2",
        ],
    phases=[0, 2],
    steps_per_action=15,
    use_gui=False
    )
    agent = Agent(env, config)
    scores_deque = deque(maxlen=100)
    scores = []
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print("Starting training...")
    for i_episode in range(1, config.num_episodes + 1):
        state, _ = env.reset(seed=config.seed + i_episode) # Vary seed per episode
        score = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)   
            state = next_state
            score += reward           
        scores_deque.append(score)
        scores.append(score)
        avg_score_100_ep = np.mean(scores_deque)
        print(f'Episode {i_episode}	Average Score: {avg_score_100_ep:.2f}	Score: {score:.2f}	Epsilon: {agent.epsilon:.4f}', end="")
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}	Average Score: {avg_score_100_ep:.2f}')
    print(f'\nTraining finished. Final model saved.')
    agent.save_model_weights(model_dir, f"{config.run_name}_final.pth")
    env.close()
if __name__ == "__main__":
    main()

 
