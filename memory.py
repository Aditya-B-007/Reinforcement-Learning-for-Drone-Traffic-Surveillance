import numpy as np
import torch
import random
from collections import namedtuple, deque

class SumTree:
    """
    A SumTree data structure for efficient priority-based sampling.
    The value of any node is the sum of its children. The leaves store the priorities.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

class ReplayBuffer:
    """A Replay Buffer that supports Uniform, Prioritized, and N-Step sampling."""

    def __init__(self, args, device):
        self.device = device
        self.use_per = args.use_prioritized_replay
        self.batch_size = args.batch_size
        self.n_steps = args.multi_step
        self.gamma = args.gamma

        if self.use_per:
            self.alpha = 0.5  # Priority exponent
            self.beta = 0.4   # Importance-sampling exponent
            self.beta_increment_per_sampling = 0.001
            self.abs_err_upper = 1.0  # Clipped abs error
            self.tree = SumTree(args.buffer_size)
        else:
            self.memory = deque(maxlen=args.buffer_size)
        # Buffer for n-step returns
        self.n_step_buffer = deque(maxlen=self.n_steps)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience, processing n-step returns."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_steps:
            return

        # Get the experience to store (the one from n-steps ago)
        R, next_s, is_done = self._get_n_step_info()
        s, a, _, _, _ = self.n_step_buffer[0]

        exp = self.experience(s, a, R, next_s, is_done)

        if self.use_per:
            max_p = np.max(self.tree.tree[-self.tree.capacity:])
            if max_p == 0:
                max_p = self.abs_err_upper
            self.tree.add(max_p, exp)
        else:
            self.memory.append(exp)

    def _get_n_step_info(self):
        """Calculate the n-step return and identify the final next_state."""
        reward, next_state, done = 0, None, True
        for i in range(self.n_steps):
            s, a, r, next_s, is_done = self.n_step_buffer[i]
            reward += r * (self.gamma ** i)
            if not is_done:
                next_state = next_s
                done = False
        return reward, next_state, done

    def sample(self):
        """Sample a batch of experiences from memory."""
        if self.use_per:
            b_idx, b_memory, ISWeights = np.empty((self.batch_size,), dtype=np.int32), np.empty((self.batch_size,), dtype=object), np.empty((self.batch_size, 1))
            pri_seg = self.tree.total_priority / self.batch_size
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

            for i in range(self.batch_size):
                a, b = pri_seg * i, pri_seg * (i + 1)
                v = np.random.uniform(a, b)
                idx, p, data = self.tree.get_leaf(v)
                if data==0:
                    v=np.random.uniform(self.tree.total_capacity)
                    idx, p, data = self.tree.get_leaf(v)
                prob = p / self.tree.total_priority
                ISWeights[i, 0] = np.power(prob * self.tree.n_entries, -self.beta)
                b_idx[i], b_memory[i] = idx, data
            experiences = b_memory
        else:
            experiences = random.sample(self.memory, k=self.batch_size)
            b_idx, ISWeights = None, None

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, actions, rewards, next_states, dones, b_idx, ISWeights

    def update_priorities(self, tree_idx, abs_errors):
        """Update priorities of sampled transitions."""
        if not self.use_per:
            return
        abs_errors += 1e-6  # avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.tree.n_entries if self.use_per else len(self.memory)