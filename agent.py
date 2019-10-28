import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import ActorNetwork, CriticNetwork


BUFFER_SIZE = int(1e6)   # replay buffer size
BATCH_SIZE = 1024        # minibatch size
GAMMA = 0.995            # discount factor
TAU = 1e-3               # for soft update of target parameters
LR_ACTOR = 1e-4          # learning rate of the actor 
LR_CRITIC = 1e-3         # learning rate of the critic
WEIGHT_DECAY = 1e-6      # L2 weight decay


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, agent_size=1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.agent_size = agent_size

        self.local_actor = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.target_actor = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.local_critic = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.target_critic = CriticNetwork(state_size, action_size, random_seed).to(device)

        self.opt_actor = optim.Adam(self.local_actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = optim.Adam(self.local_critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def save_experience(self, state, action, reward, next_state, done):                               
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience
        self.memory.add(state, action, reward, next_state, done)
                          
    def multi_step(self, t):
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            if t % 20 == 0:
                    for i in range(0,10):
                        self.learn(self.memory.sample(), GAMMA)
            else:
                pass

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()
        if add_noise:
            for a in range(0,self.agent_size):
                action[a] += self.noise.sample()
        return np.clip(action, -1, 1)   # all actions between -1 and 1 

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * target_critic(next_state, target_actor(next_state))
        where:
            target_actor(state) -> action
            target_critic(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.local_critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.opt_critic.zero_grad()
        critic_loss.backward()
        #use gradient clipping when training the critic network
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), 1)
        self.opt_critic.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.local_actor(states)
        actor_loss = -self.local_critic(states, actions_pred).mean()
        # Minimize the loss
        self.opt_actor.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), 1)
        actor_loss.backward()
        self.opt_actor.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.local_critic, self.target_critic, TAU)
        self.soft_update(self.local_actor, self.target_actor, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)