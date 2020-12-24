import copy
import random
import tqdm

import gym
import torch

from replay_buffer import memory
from replay_buffer.transition import Transition
from replay_buffer.utilities import unpack_transition_list
from utils import np_to_unsq_tensor, squeeze_np


class CategoricalDQN:

    def __init__(self, z_net, n_atoms, v_min, v_max, discount_factor=0.99, buffer_len=1e6, batch_size=32,
                 lr=0.5e-3, update_mode='hard', update_freq=10, tau=0.05, epsilon=0.1,
                 start_train_at=2000):
        """[summary]

        Args:
            z_net (torch.nn.Module): ANN which inputs state and outputs action-return distribution (batch_size, actions, n_atoms)
            n_atoms (int): Number of bins of return categorical distribution
            v_min (float): Minimum reward
            v_max (float): Maximum reward
            discount_factor (float, optional): Discount factor in [0, 1]. Defaults to 0.99.
            buffer_len (int, optional): Maximum length of the replay buffer. Defaults to 1e6.
            batch_size (int, optional): Number of samples used for training step. Defaults to 32.
            lr (float, optional): Learning rate of the optimizer. Defaults to 0.5e-3.
            update_mode (str, optional): How to update the target network weights. Use 'hard' for copying the learning
                network weights into the target network, 'soft' for Polyak averaging. Defaults to 'hard'.
            update_freq (int, optional): How many steps between target network update. Defaults to 10.
            tau (float, optional): Polyak averaging parameter. Defaults to 0.05.
            epsilon (float, optional): Probability of a greedy action. Defaults to 0.1.
            start_train_at (int, optional): Number of initial steps in the environment taken before starting training. Defaults to 2000.
        """
        self.z_net = z_net
        
        # Discretization parameters
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        self._delta = (v_max - v_min) / n_atoms
        self._atoms = torch.arange(self._v_min, self._v_max, self._delta).unsqueeze(0)

        # DQN parameters
        self._discount_factor = discount_factor
        self._batch_size = batch_size
        if update_mode != "soft" and update_mode != "hard":
            raise ValueError("update mode must be either soft or hard")
        self._update_mode = update_mode
        self._update_freq = update_freq
        self._tau = tau
        self._epsilon = epsilon
        self._start_train_at = start_train_at
        self._replay_buffer = memory.TransitionReplayBuffer(maxlen=buffer_len)
        self._target_net = copy.deepcopy(z_net)
        self._optimizer = torch.optim.Adam(self.z_net.parameters(), lr=lr)

    def train(self, env: gym.Env, n_steps):
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0  # Keeps the rewards of the current episode
        current_episode_len = 0  # Keeps the rewards of the current episode
        state = np_to_unsq_tensor(env.reset())  # Shape (1, *state_shape)
        loop_range = tqdm.tqdm(range(n_steps))
        for step in loop_range:
            with torch.no_grad():
                z = self.z_net(state)  # (1, n_actions, n_atoms)
            if random.random() < self._epsilon:  # Random action
                action = torch.LongTensor([[env.action_space.sample()]])
            else:
                action = self._select_argmax_action(z, self._atoms)
            next_state, reward, done, info = env.step(squeeze_np(action))
            next_state = np_to_unsq_tensor(next_state) if not done else None
            self._replay_buffer.remember(
                Transition(state, action, torch.tensor([[reward]]), next_state))
            state = next_state

            # Perform training step
            self._train_step(step)

            # Update episode stats
            current_episode_reward += reward
            current_episode_len += 1
            if done:
                state = np_to_unsq_tensor(env.reset())
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_len)
                current_episode_reward = 0.
                current_episode_len = 0.
                loop_range.set_description(f'Reward {episode_rewards[-1]}')
    
    def _select_argmax_action(z, atoms):
        # Take state-action distribution z, which is a (batch_size, action_size, n_atoms) and
        # returns a tensor of shape (batch_size, 1) with the greedy actions for each state
        q_values = (z * (atoms[:, None, :] + self._delta/2)).sum(dim=-1)
        return q_values.argmax(dim=-1).unsqueeze(1)

    def _train_step(self, step):
        if step < self._start_train_at or self._replay_buffer.size() < self._batch_size:
            return
        batch = self._replay_buffer.sample(self._batch_size)
        states, actions, rewards, next_states, mask, _, _ = unpack_transition_list(batch)
        targets = self._compute_targets(rewards, next_states, mask)
        self._train_net(states, actions, targets, update=(step % self._update_freq) == 0)

    def _train_net(self, states, actions, targets, update):
        self._optimizer.zero_grad()
        z = self.z_net(states)
        z = torch.cat([z[i, actions[i]] for i in range(z.shape[0])])
        # Compute cross-entropy loss
        loss = -(targets * z.log()).sum(dim=-1).mean()
        loss.backward()
        self._optimizer.step()
        if update:
            self._update_target_net()

    def _update_target_net(self):
        # Mode can be 'hard' or 'soft'
        if self._update_mode == 'hard':
            self._target_net.load_state_dict(self.z_net.state_dict())
        else:
            for param, target_param in zip(self.z_net.parameters(), self._target_net.parameters()):
                target_param.copy_(self._tau * param + (1 - self._tau) * target_param)

    def _compute_targets(self, rewards, next_states, mask):
        """Compute the target distributions for the given transitions.

        """
        # rewards = (batch_size, 1) 
        # next_states = (n, *state_shape)
        # mask = (batch_size) of booleans
        # All these are (batch_size, *shape) tensors
        atoms = torch.arange(self._v_min, self._v_max, self._delta)
        atoms = (rewards + self._discount_factor * mask[:, None] * atoms).clamp(min=self._v_min, max=self._v_max)
        b = (atoms - self._v_min) / self._delta
        l = torch.floor(b).long()
        u = torch.ceil(b).clamp(max=self._n_atoms - 1).long()  # Prevent out of bounds
        # Predict next state return distribution for each action
        with torch.no_grad():
            z_prime = self._target_net(next_states)
        target_actions = select_argmax_action(z_prime, atoms[mask])
        # TODO: Do this with gather or similar
        z_prime = torch.cat([z_prime[i, target_actions[i]] for i in range(z_prime.shape[0])])

        # For elements that do not have a next state, atoms are all equal to reward and we set a
        # uniform distribution (it will collapse to the same atom in any case)
        probabilities = torch.ones((self._batch_size, self._n_atoms)) / self._n_atoms
        probabilities[mask] = z_prime
        # Compute partitions of atoms
        lower = probabilities * (u - b)
        upper = probabilities * (b - l)
        z_projected = torch.zeros_like(probabilities)
        z_projected.scatter_add_(1, l, lower)
        z_projected.scatter_add_(1, u, upper)
        return z_projected


if __name__ == '__main__':
    import networks

    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    n_atoms = 50
    z_net = networks.DistributionalNetwork(inputs=state_dim, n_actions=act_dim, n_atoms=n_atoms,
                                           n_hidden_units=64, n_hidden_layers=2)

    DDQN = CategoricalDQN(z_net=z_net, n_atoms=n_atoms, v_min=0, v_max=100, start_train_at=32,
                          update_freq=5)
    DDQN.train(env=env, n_steps=30000)