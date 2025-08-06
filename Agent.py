#This is the main python file where the agent would be coded up
#Each step has an explanation
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
from network import DQN
from memory import ReplayBuffer

class Agent():
    """A Rainbow-style DQN Agent that can be configured for different components."""
    def __init__(self, env,args):
        self.env = env
        # Stores the environment's action space, which defines all possible actions.
        # For a Bengaluru traffic junction
        self.action_space=env.action_space
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = args.device
        self.args = args # Store args for easy access
        self.atoms=args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1) 
        self.batch_size = args.batch_size
        self.multi_step = args.multi_step
        self.gamma = args.gamma
        self.norm_clip = args.norm_clip
        self.online_net = DQN(self.state_dim, self.action_dim, self.atoms, self.args).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim, self.atoms, self.args).to(self.device)
        self.update_target_net() # Copy weights from online to target
        self.target_net.eval() # Target network is not trained, only for evaluation
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=args.learning_rate)
        self.memory = ReplayBuffer(self.args, self.device)
        self.learn_step_counter = 0
        self.epsilon = args.epsilon_start

    def act(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist = self.online_net(state_tensor)
            q_values = (dist * self.support).sum(dim=2)
            return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample()

        with torch.no_grad():
            next_q_values = (self.online_net(next_states) * self.support).sum(dim=2)
            best_next_actions = next_q_values.argmax(dim=1)
            next_dist = self.target_net(next_states)[range(self.batch_size), best_next_actions]
            Tz = rewards + (1 - dones) * (self.gamma ** self.multi_step) * self.support
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  
            b = (Tz - self.Vmin) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            l[(u > 0) & (l == u)] -= 1
            u[(l < (self.atoms - 1)) & (l == u)] += 1
            m = torch.zeros_like(next_dist)
            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(l)
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        log_ps = torch.log(self.online_net(states)[range(self.batch_size), actions.squeeze(1)])
        elementwise_loss = -(m * log_ps).sum(1)
        if self.args.use_prioritized_replay:
            loss = (torch.tensor(weights, device=self.device).squeeze(1) * elementwise_loss).mean()
            new_priorities = elementwise_loss.detach().cpu().numpy()
            self.memory.update_priorities(indices, new_priorities)
        else:
            loss = elementwise_loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimizer.step()
        self.epsilon = max(self.args.epsilon_end, self.epsilon - (self.args.epsilon_start - self.args.epsilon_end) / self.args.epsilon_decay)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.args.target_update_frequency == 0:
            self.update_target_net()
        
        self.learn()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_model_weights(self,path,name='model_weights.pth'):
        torch.save(self.online_net.state_dict(),path+name)