"""This module contain the workers code, that takes care of running the policy in the given
environment and performing asynchronous updates on the shared model.

Code is freely taken or inspired from https://github.com/Kaixhin/ACER or
https://github.com/dchetelat/acer

"""

import copy
import math
import random

import gym

import torch
from typing import List

import memory
import utils


def _poisson(lmbd):
    # Knuth's algorithm for generating Poisson samples
    # from https://github.com/Kaixhin/ACER/blob/master/train.py
    l, k, p = math.exp(-lmbd), 0, 1
    while p > l:
        k += 1
        p *= random.uniform(0, 1)
    return max(k - 1, 0)


class TraceTrainingData:
    """Container class that keeps data used for training.

    Attributes:
        actions (list): List of tensors of shape (1, 1). Each tensor must be attached to graph.
        policies (list): List of tensors of shape (1, K) containing policy predictions of the
            episode states computed with the current model. Each tensor must be attached to graph.
        q_values (list): List of tensors of shape (1, K). Each tensor must be attached to graph.
        values (list): List of tensors of shape (1, 1). Each tensor must be attached to graph.
        rewards (list): List of tensors of shape (1, 1).
        avg_policies (list): List of tensors of shape (1, K). Each tensor must be attached to graph.
        old_policies  (list): List of tensors of shape (1, K), or None if on-policy. Each tensor
            must be attached to graph.
        last_state (torch.Tensor): Tensor of shape (1, *state_dim) with the last state of the
            trace, or None if episode ended.
    """

    def __init__(self):
        self.actions = []
        self.policies = []
        self.q_values = []
        self.values = []
        self.rewards = []
        self.avg_policies = []
        self.old_policies = []
        self.last_state = None

    def append(self, action, policy, q_values, value, reward, average_policy, last_state=None,
               old_policies=None):
        # NOTE: policy is always the policy for the current model, old_policy the one in the
        # replay buffer.
        self.actions.append(action)
        self.policies.append(policy)
        self.q_values.append(q_values)
        self.values.append(value)
        self.rewards.append(reward)
        self.avg_policies.append(average_policy)
        self.last_state = last_state
        self.old_policies = old_policies

    def init_from(self, actions, policies, q_values, values, rewards, avg_policies,
                  old_policies=None, last_state=None):
        self.actions = actions
        self.policies = policies
        self.q_values = q_values
        self.values = values
        self.rewards = rewards
        self.avg_policies = avg_policies
        self.old_policies = old_policies if old_policies is not None else []
        self.last_state = last_state

    def length(self):
        return len(self.actions)


def extract(training_data, valid_indices, t, off_policy, act_dim):
    """Extract from the training data all the elements for the given timestep as a tuple of
    stacked tensors.

    Args:
        training_data (list): List of TraceTrainingData objects from which to extract.
        valid_indices (list): List of indices corresponding to train_data object for which t is
            a valid timestep.
        t (int): Time-step for which the data has to be extracted.
        off_policy (bool): Whether the training_data objects come from old policies.
        act_dim (int): Dimension of actions.

    Returns:
        A tuple (actions, policies, rewards, q_values, values, rhos, avg_p) where each element is
        a tensor of stacked elements from training_data[valid_indices] at time-step t.
        rhos contain the importance samples computed, or ones if not off_policy.
    """
    # Extract elements at position -t from training data
    actions = torch.cat(tuple(training_data[i].actions[-t] for i in valid_indices), dim=0)
    policies = torch.cat(tuple(training_data[i].policies[-t] for i in valid_indices), dim=0)
    q_values = torch.cat(tuple(training_data[i].q_values[-t] for i in valid_indices), dim=0)
    rewards = torch.cat(tuple(training_data[i].rewards[-t] for i in valid_indices), dim=0)
    values = torch.cat(tuple(training_data[i].values[-t] for i in valid_indices), dim=0)
    avg_p = torch.cat(tuple(training_data[i].avg_policies[-t] for i in valid_indices), dim=0)
    if off_policy:
        old_policies = torch.cat(
            tuple(training_data[i].old_policies[-t] for i in valid_indices), dim=0)
        rhos = policies.detach() / old_policies
    else:
        rhos = torch.ones(actions.shape[0], act_dim)
    return actions, policies, rewards, q_values, values, rhos, avg_p


class Worker:
    """Implementation of a worker.
    """

    def __init__(self, worker_id, env_name, n_steps, max_steps, shared_model, shared_avg_model,
                 shared_optimizer, shared_counter, df, c, entropy_weight, tau, buffer_len,
                 replay_ratio, batch_size, start_train_at, grad_norm_clip,
                 trust_region, shared_model_lock, use_lock_update, summary_queue=None):
        """Create the worker, that collects trajectories from the environment, stores them in the
        replay buffer, performs local training and updates the shared model parameters.

        Args:
            worker_id (int): Unique ID of the worker. Ids must be incremental starting from 0.
            env_name (str): Name of the environment to create.
            n_steps (int): Maximum length of trajectories collected at each iteration.
            max_steps (int): Maximum number of steps to perform across all threads.
            shared_model (torch.nn.Module): Shared model that will be updated by the worker.
            shared_avg_model (torch.nn.Module): Shared running average of shared_model.
            shared_counter (utils.Counter): Shared counter that keeps track of the total step.
            df (float): Discount factor.
            c (float): Maximum value for importance weights.
            entropy_weight (float): Weight of the entropy term in policy loss.
            tau (float): Decay parameter for average policy network update.
            replay_ratio (int): Expected value of off-policy trainings for each on-policy training.
            buffer_len (int): Maximum capacity (in trajectories) of the episodic replay buffer.
            batch_size (int): Batch size for the off-policy step.
            start_train_at (int): Number of steps to take before starting to train.
            grad_norm_clip (float): Clipping gradient norm value. None for no clip.
            trust_region (float): KL Divergence threshold between updates.
            shared_model_lock (torch.multiprocessing.Lock): If not None, the lock is used to
                safely pass gradients to the shared model.
            use_lock_update(bool): Whether to perform the shared model update inside the lock block.
            summary_queue (torch.multiprocessing.Queue): Queue used to pass data to tensorboard.
        """
        self.worker_id = worker_id
        self.n_steps = n_steps
        self.max_steps = max_steps
        self.shared_model = shared_model
        self.shared_avg_model = shared_avg_model
        self.shared_optimizer = shared_optimizer
        self.shared_counter = shared_counter
        self.df = df
        self.c = c
        self.entropy_weight = entropy_weight
        self.tau = tau
        self.start_train_at = start_train_at
        self.replay_ratio = replay_ratio
        self.batch_size = batch_size
        self.grad_norm_clip = grad_norm_clip
        self.trust_region = trust_region
        self.shared_model_lock = shared_model_lock
        self.use_lock_update = use_lock_update
        if self.use_lock_update and self.shared_model_lock is None:
            raise ValueError('Lock is not passed but use_lock_update is True.')
        self.summary_queue = summary_queue
        self.env = gym.make(env_name)
        self.replay_buffer = memory.EpisodicReplayBuffer(maxlen=buffer_len)
        self.model = copy.deepcopy(self.shared_model)
        self.cur_state = None
        self.done = True
        self.episode_rewards = []  # Each element is a reward of the current episode
        self.rewards = []  # Each element is the total reward of an episode
        self.episode_lengths = []  # Each element length of correspondent in self.rewards

    def run(self):
        while self.shared_counter.value() <= self.max_steps:
            self.model.load_state_dict(self.shared_model.state_dict())
            training_data = self.on_policy()  # Collect n_steps on policy
            kl_div, h_loss, h_mean, p_loss, v_loss = self._train(training_data, off_policy=False)
            if self.replay_buffer.length() > self.start_train_at \
                    and self.replay_buffer.length() > self.batch_size:
                self.off_policy()
            self._log_to_queue(rewards=self.rewards, episode_lengths=self.episode_lengths,
                               p_loss=p_loss, v_loss=v_loss, on_kl_div=kl_div,
                               entropy_loss=h_loss, entropy_mean=h_mean, off_kl_div=None)
            # Reset rewards and episode_lengths but not episode_rewards
            self.rewards = []
            self.episode_lengths = []

    def on_policy(self):
        """Perform n_steps on-policy, and return the data necessary for on-policy update,
        and updates shared_counter.

        Returns:
            training_data (list): A list of TraceTrainingData objects, one for each episode run.
                Only the last object may contain a last_state attribute corresponding to the
                state at which the last episode was cut.
        """
        t = 0
        training_data = [] if self.done else [TraceTrainingData()]
        while t < self.n_steps:
            if self.done:  # Re-initialize objects for new episode
                self.cur_state = utils.np_to_unsq_tensor(self.env.reset())
                self.done = False
                training_data.append(TraceTrainingData())
                if len(self.episode_rewards) > 0:
                    self.rewards.append(sum(self.episode_rewards))
                    self.episode_lengths.append(len(self.episode_rewards))
                    self.episode_rewards = []

            # Compute policy and q_values. Note that we do not detach elements used in training,
            # as this saves us computations in _train()
            policy, q_values = self.model(self.cur_state)
            value = (policy * q_values).sum(dim=1, keepdim=True)
            with torch.no_grad():
                avg_policy, _ = self.shared_avg_model(self.cur_state)
            action = torch.multinomial(policy, num_samples=1)[0, 0]

            next_state, reward, done, _ = self.env.step(action.item())
            next_state = utils.np_to_unsq_tensor(next_state)

            # Save transition in replay buffer
            self.replay_buffer.append_transition(
                (self.cur_state, torch.LongTensor([[action.item()]]), policy.detach(),
                 torch.LongTensor([[reward]]), done))
            # Save data for training (all tensors have first dimension 1)
            training_data[-1].append(
                action=torch.LongTensor([[action]]), policy=policy, q_values=q_values,
                value=value, reward=torch.Tensor([[reward]]), average_policy=avg_policy)

            # Update loop data
            t += 1
            self.done = done
            self.cur_state = next_state
            self.episode_rewards.append(reward)

        if not self.done:
            training_data[-1].last_state = self.cur_state
            self.replay_buffer.cutoff(self.cur_state)  # Notify termination to the replay buffer
        self.shared_counter.increment(t)
        return training_data

    def off_policy(self):
        """Perform n off-policy training steps, where n ~ Poisson(self.lambda).
        """
        for i in range(_poisson(self.replay_ratio)):
            batch = self.replay_buffer.sample(batch_size=self.batch_size)
            training_data = [self._get_training_data(trajectory) for trajectory in batch]
            self._train(training_data, off_policy=True)

    def _train(self, training_data: List[TraceTrainingData], off_policy):
        # Perform training step. Code mainly inspired from
        # https://github.com/Kaixhin/ACER/blob/c711b911baf34b7acf1dbaf0cfeccc6d78277134/train.py
        # NOTE: training_data contains data from trajectories that do not have the same length
        act_dim = training_data[0].policies[0].size(1)
        total_loss = torch.tensor([0.])

        # Iterate all training data from last step backwards, so that they are aligned w.r.t. the
        # backward retrace recursive target computation even when they have different lengths.
        kl_divs, entropy_losses, entropy_values, policy_losses, value_losses = [], [], [], [], []
        n_episodes = len(training_data)
        max_length = max(len(data.rewards) for data in training_data)
        q_rets = None  # retrace action values for each trajectory, shape (n_episodes, 1)
        for t in range(1, max_length):
            # Indices of trajectories that have more than t steps, used as mask
            mask = [i for i in range(n_episodes) if training_data[i].length() - t >= 0]
            actions, policies, rewards, q_values, values, rhos, avg_p = extract(
                training_data, mask, t, off_policy, act_dim)
            if t == 1:  # Last time step of trajectories, initialize q_rets
                q_rets = self._initial_q_ret(training_data)
            q_rets[mask] = rewards + self.df * q_rets[mask]
            adv = q_rets[mask] - values
            log_prob = policies.gather(1, actions).log()
            step_policy_loss = -(rhos.gather(1, actions).clamp(max=self.c) * log_prob *
                                 adv.detach()).mean(0)  # Average over batch
            if off_policy:  # Applying bias correction (second term of Eq. 9 in the paper)
                # Multiply by policies and sum(1) in the end = expected value
                bias_weight = (1 - self.c / rhos).clamp(min=0) * policies
                step_policy_loss -= (bias_weight * policies.log() *
                                     (q_values.detach() - values.expand_as(q_values).detach())
                                     ).sum(1).mean(0)  # Average over batch
            if self.trust_region is None:
                policy_loss = step_policy_loss
            else:
                policy_loss = self._add_kl_constraint(step_policy_loss, policies, avg_p)

            # Loss proportional to negative entropy
            entropy_loss = self.entropy_weight * (policies.log() * policies).sum(1).mean(0)

            # Value update
            q = q_values.gather(1, actions)
            value_loss = ((q_rets[mask] - q) ** 2 / 2).mean(0)  # Least squares loss

            # Update the retrace target
            truncated_rho = rhos.gather(1, actions).clamp(max=self.c)
            q_rets[mask] = truncated_rho * (q_rets[mask] - q.detach()) + values.detach()

            # Update total loss
            total_loss += policy_loss + value_loss + entropy_loss

            # Compute statistics
            mean_kl = (avg_p * (avg_p.log() - policies.log())).sum(1).mean()
            entropy = -(policies * policies.log()).sum(1).mean()
            kl_divs.append(mean_kl.item())
            entropy_losses.append(entropy_loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_values.append(entropy.item())

        # Transfer gradients to shared model and update
        self._update_networks(total_loss)
        return kl_divs, entropy_losses, entropy_values, policy_losses, value_losses

    def _add_kl_constraint(self, loss, policy, avg_policy):
        """Modify the given loss to enforce the KL constraint.

        Args:
            loss (torch.Tensor): Tensor containing the single step loss.
            policy (torch.Tensor): Tensor of shape (batch_size, K) containing the probabilities
                (from the local model) for each action. Must be connected to the loss graph.
            avg_policy (torch.Tensor): Tensor of shape (batch_size, K) containing probabilities
                (from the shared avg mode) for each action. Must be connected to the loss graph.

        Returns:
            Tensor containing the new step loss.
        """
        g = torch.autograd.grad(loss, policy)[0]  # Grad of loss w.r.t. policy
        kl = (avg_policy * (avg_policy.log() - policy.log())).sum()
        k = torch.autograd.grad(-kl, policy)[0]  # Grad of KL between policy and avg_policy
        k_dot_g = (k * g).sum(1)
        k_dot_k = (k * k).sum(1)
        k_factor = ((k_dot_g - self.trust_region) / k_dot_k).unsqueeze(1)
        k_factor = torch.where(k_factor > 0., k_factor, torch.zeros_like(k_factor))
        z = (g - k_factor * k).detach()
        constrained_loss = (policy * z).sum()
        return constrained_loss

    def _initial_q_ret(self, training_data):
        # Compute the initial retrace action-value.
        q_rets = torch.empty(len(training_data), 1)
        for i in range(len(training_data)):
            if training_data[i].last_state is None:
                q_rets[i, 0] = 0.
            else:
                with torch.no_grad():
                    policy, q_values = self.model(training_data[i].last_state)
                    q_rets[i, 0] = (policy * q_values).sum(dim=1)
        return q_rets

    def _get_training_data(self, trajectory: memory.Trajectory):
        # Create a TraceTrainingData object from the given trajectory.
        train_data = TraceTrainingData()
        if trajectory.episode_ended():
            states = torch.cat(tuple(state for state in trajectory.states))
            last_state = None
        else:  # Last state is passed separately
            states = torch.cat(tuple(state for state in trajectory.states[:-1]))
            last_state = trajectory.states[-1]
        policies, q_values = self.model(states)
        values = (policies * q_values).sum(dim=1, keepdim=True)
        with torch.no_grad():
            avg_policies, _ = self.shared_avg_model(states)
        policies = [policy.unsqueeze(0) for policy in policies]
        q_values = [q_val.unsqueeze(0) for q_val in q_values]
        values = [value.unsqueeze(0) for value in values]
        avg_policies = [avg_policy.unsqueeze(0) for avg_policy in avg_policies]
        train_data.init_from(actions=trajectory.actions, policies=policies, q_values=q_values,
                             values=values, rewards=trajectory.rewards, avg_policies=avg_policies,
                             old_policies=trajectory.p_actions, last_state=last_state)
        return train_data

    def _update_networks(self, loss):
        # Call backward() on the loss, clip the gradient and update
        # the network with or without locks.

        def update_avg_policy():
            # Update shared_average_model
            for param, avg_param in zip(self.shared_model.parameters(),
                                        self.shared_avg_model.parameters()):
                avg_param.data.copy_(self.tau * avg_param + (1 - self.tau) * param)
        loss.backward()
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        if self.shared_model_lock is not None:
            with self.shared_model_lock:
                self._transfer_grads_to_shared_model(overwrite_check=False)
                if self.use_lock_update:
                    self.shared_optimizer.step()
                    update_avg_policy()
            if not self.use_lock_update:  # Perform updates outside lock
                self.shared_optimizer.step()
                update_avg_policy()
        else:  # Do updates without lock
            self._transfer_grads_to_shared_model(overwrite_check=True)
            self.shared_optimizer.step()
            update_avg_policy()

    def _transfer_grads_to_shared_model(self, overwrite_check):
        # Transfers gradients from thread-specific model to shared model
        for param, shared_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if shared_param.grad is not None and overwrite_check:
                return
            shared_param.grad = param.grad

    def _log_to_queue(self, rewards, episode_lengths, p_loss, v_loss, on_kl_div, entropy_loss,
                      entropy_mean, off_kl_div):
        if self.summary_queue is not None:
            data_dict = {
                'rewards': (rewards, episode_lengths),
                'on-policy/losses/entropy-loss': (entropy_loss, None),
                'on-policy/losses/policy-loss': (p_loss, None),
                'on-policy/losses/value-loss': (v_loss, None),
                'on-policy/stats/mean-kl': (on_kl_div, None),
                'on-policy/stats/mean-entropy': (entropy_mean, None)
            }
            if off_kl_div is not None:
                data_dict['off-policy-kl'] = (off_kl_div, None)
            self.summary_queue.put(data_dict, block=False)
