import numpy as np, torch
from collections import namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "priority"])

class ReplayBuffer:
    def __init__(self, num_actions, memory_len=100000, alpha=0.6, epsilon=0.01):
        self.memory_len = memory_len
        self.transition = []
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        if self.length() > self.memory_len:
            self.remove()
        self.transition.append(Transition(state, action, reward, next_state, done, self.max_priority))

    def sample_batch(self, batch_size=32, device='cuda'):
        if self.length() <= batch_size:
            minibatch = self.transition
        else:
            priorities = np.array([t.priority for t in self.transition])
            probs = priorities ** self.alpha / sum(priorities ** self.alpha)
            
            indices = np.random.choice(len(self.transition), batch_size, p=probs)
            minibatch = [self.transition[idx] for idx in indices]
            
        states_mb, a_, reward_mb, next_states_mb, done_mb, _ = map(np.array, zip(*minibatch))

        mb_reward = torch.from_numpy(reward_mb).to(device=device, dtype=torch.float32)
        mb_done = torch.from_numpy(done_mb.astype(int)).to(device=device)
        a_ = a_.astype(int)
        a_mb = np.zeros((a_.size, self.num_actions), dtype=np.float32)
        a_mb[np.arange(a_.size), a_] = 1
        mb_a = torch.from_numpy(a_mb).to(device=device)
        return states_mb, mb_a, mb_reward, next_states_mb, mb_done

    def length(self):
        return len(self.transition)

    def remove(self):
        self.transition.pop(0)
        
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            if idx < len(self.transition):
                priority = (abs(error) + self.epsilon) ** self.alpha
                old_transition = self.transition[idx]
                self.transition[idx] = Transition(
                    old_transition.state,
                    old_transition.action,
                    old_transition.reward,
                    old_transition.next_state,
                    old_transition.done,
                    priority
                )
                self.max_priority = max(self.max_priority, priority)