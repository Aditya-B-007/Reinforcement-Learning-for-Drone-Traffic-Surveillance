config = {
    "project_name": "Rainbow-Framework",
    "run_name": "dqn_baseline",
    "seed": 42,
    "num_episodes": 2000, # Number of episodes to train for
    "cuda": True,
    "env_name": "LunarLander-v3",
    "gamma": 0.99, # Discount factor
    "batch_size": 64,
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "target_update_frequency": 1000, # In steps
    "tau": 1.0, # For hard target network updates
    "norm_clip": 10.0, # For gradient clipping
    "atoms": 51,
    "V_min": -1000.0,
    "V_max": 200.0,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 30000,
    "use_double_dqn": True,
    "use_prioritized_replay": True,
    "use_dueling_network": True, # Dueling Network Architecture
    "multi_step": 3, # N-step returns. 1 means standard 1-step returns.
    "use_distributional": True,
    "use_noisy_nets": True,
}