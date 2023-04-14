import torch
import numpy as np
import itertools

from .base import Algorithm
from research.networks.base import ActorCriticValuePolicy
from research.utils import utils
from research.utils.evaluate import eval_policy
from functools import partial

from collections import deque, namedtuple

def gumbel_loss(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    loss = torch.exp(z) - z - 1
    return loss.mean()

def gumbel_rescale_loss(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    max_z = torch.max(z)
    max_z = torch.where(max_z < -1.0, torch.tensor(-1.0, dtype=torch.float, device=max_z.device), max_z)
    max_z = max_z.detach() # Detach the gradients
    loss = torch.exp(z - max_z) - z*torch.exp(-max_z) - torch.exp(-max_z)
    return loss.mean()

def expectile_loss(diff, expectile=0.8):
    """Loss function for iql expectile value difference."""
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def mse_loss(pred, label):
    return ((label - pred)**2).mean()

class XQL(Algorithm):

    def __init__(self, env, network_class, dataset_class, 
                       tau=0.005,
                       policy_noise=0.1,
                       init_temperature=0.1,
                       critic_freq=1,
                       value_freq=1,
                       actor_freq=2,
                       target_freq=2,
                       env_freq=1,
                       init_steps=1000,
                       reset_freq=1e4,
                       alpha=None,
                       exp_clip=10,
                       beta=1.0,
                       loss="gumbel",
                       value_action_noise=0.0,
                       bc_low=None,
                       rkl_low=None,
                       use_value_log_prob=False,
                       max_grad_norm=None,
                       max_grad_value=None,
                       **kwargs):
        
        # Save values needed for optim setup.
        self.init_temperature = init_temperature
        self._alpha = alpha
        super().__init__(env, network_class, dataset_class, **kwargs)
        assert isinstance(self.network, ActorCriticValuePolicy)

        # Save extra parameters
        self.tau = tau
        self.policy_noise = policy_noise
        self.critic_freq = critic_freq
        self.value_freq = value_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.env_freq = env_freq
        self.init_steps = init_steps
        self.reset_freq = reset_freq
        self.exp_clip = exp_clip
        self.beta = beta
        assert loss in {"gumbel", "gumbel_rescale", "mse", "expectile"}
        self.loss = loss
        self.value_action_noise = value_action_noise
        self.use_value_log_prob = use_value_log_prob
        self.action_range = (self.env.action_space.low, self.env.action_space.high)
        self.action_range_tensor = utils.to_device(utils.to_tensor(self.action_range), self.device)
        self.bc_low = bc_low
        self.rkl_low = rkl_low
        # Gradient clipping
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value
        
        # For extra validation
        self._memory = deque(maxlen=256)
        self._experience = namedtuple("_experience", field_names=["init_state", "init_action", "epi_reward"])

    @property
    def alpha(self):
        return self.log_alpha.exp()
        
    def setup_network(self, network_class, network_kwargs):
        self.network = network_class(self.env.observation_space, self.env.action_space, 
                                     **network_kwargs).to(self.device)
        self.target_network = network_class(self.env.observation_space, self.env.action_space, 
                                     **network_kwargs).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self, optim_class, optim_kwargs):
        # Default optimizer initialization
        self.optim['actor'] = optim_class(self.network.actor.parameters(), **optim_kwargs)
        # Update the encoder with the critic
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
        self.optim['critic'] = optim_class(critic_params, **optim_kwargs)
        self.optim['value'] = optim_class(self.network.value.parameters(), **optim_kwargs)

        self.target_entropy = -np.prod(self.env.action_space.low.shape)
        if self._alpha is None:
            # Setup the learned entropy coefficients. This has to be done first so its present in the setup_optim call.
            self.log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float).to(self.device)
            self.log_alpha.requires_grad = True
            self.optim['log_alpha'] = optim_class([self.log_alpha], **optim_kwargs)
        else:
            self.log_alpha = torch.tensor(np.log(self._alpha), dtype=torch.float).to(self.device)
            self.log_alpha.requires_grad = False
    
    def _step_env(self):
        # Step the environment and store the transition data.
        metrics = dict()
        if self._env_steps < self.init_steps:
            action = self.env.action_space.sample()
        else:
            self.eval_mode()
            with torch.no_grad():
                action = self.predict(self._current_obs, sample=True)
            self.train_mode()
        
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        next_obs, reward, done, info = self.env.step(action)
        self._episode_length += 1
        self._episode_reward += reward
        if self._init_action is None: self._init_action = action

        if 'discount' in info:
            discount = info['discount']
        elif hasattr(self.env, "_max_episode_steps") and self._episode_length == self.env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences.
        self.dataset.add(next_obs, action, reward, done, discount)

        if done:
            self._num_ep += 1
            # update metrics
            metrics['reward'] = self._episode_reward
            metrics['length'] = self._episode_length
            metrics['num_ep'] = self._num_ep
            # store to validation memory
            e = self._experience(self._init_obs, self._init_action, self._episode_reward)
            self._memory.append(e)
            # Reset the environment
            self._current_obs = self.env.reset()
            self.dataset.add(self._current_obs) # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
            # reset validation data
            self._init_obs = self._current_obs
            self._init_action = None
        else:
            self._current_obs = next_obs

        self._env_steps += 1
        metrics['env_steps'] = self._env_steps
        return metrics

    def _setup_train(self):
        self._current_obs = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._num_ep = 0
        self._env_steps = 0
        self.dataset.add(self._current_obs) # Store the initial reset observation!
        self._init_obs = self._current_obs
        self._init_action = None

    def _train_step(self, batch):
        all_metrics = {}

        if not self.offline and (self.steps % self.env_freq == 0 or self._env_steps < self.init_steps):
            # step the environment with freq env_freq or if we are before learning starts
            metrics = self._step_env()
            all_metrics.update(metrics)
            if self._env_steps < self.init_steps:
                return all_metrics # return here.
        
        if 'obs' not in batch:
            return all_metrics
        
        batch['obs'] = self.network.encoder(batch['obs'])
        with torch.no_grad():
            batch['next_obs'] = self.target_network.encoder(batch['next_obs'])

        # Update Q
        if self.steps % self.critic_freq == 0:
            # Q Loss:
            with torch.no_grad():
                target_v = self.target_network.value(batch['next_obs'])
                target_v = torch.min(target_v, dim=0)[0].squeeze(-1)
                target_q = batch['reward'] + batch['discount']*target_v

            qs = self.network.critic(batch['obs'], batch['action'])
            # Note: Could also just compute the mean over a broadcasted target. TO investigate later.
            q_loss = sum([mse_loss(qs[i], target_q) for i in range(qs.shape[0])])

            self.optim['critic'].zero_grad(set_to_none=True)
            q_loss.backward()
            self.optim['critic'].step()

            all_metrics['q_loss'] = q_loss.item()
            all_metrics['target_q'] = target_q.mean().item()
        
        action = batch['action']
        # action = self.network.predict(batch['obs'].detach(), sample=True)
        # Update Value
        if self.steps % self.value_freq == 0:
            if self.value_action_noise > 0.0:
                with torch.no_grad():
                    noise = (torch.randn_like(action) * self.value_action_noise).clamp(-0.5, 0.5)
                    action = (action.detach() + noise).clamp(*self.action_range_tensor)
            '''
            # Use policy noise!
            if self.value_action_noise > 0.0 and np.random.sample() < self.value_action_noise:
                with torch.no_grad():
                    # What about using noise from the policy? (helpful in the early stage of training?)
                    obs = batch['obs'].detach() # Detach the encoder so it isn't updated.
                    dist = self.network.actor(obs)
                    action = dist.rsample()
            '''
            qs = self.target_network.critic(batch['obs'], action)
            target_v_pi = (torch.min(qs, dim=0)[0]).detach()

            '''
            if self.use_value_log_prob: # What is this? Practical implementation?
                target_v_pi = (target_v_pi  - self.alpha.detach() * log_prob).detach()
            '''
            vs = self.network.value(batch['obs']).squeeze(-1)
            if self.loss == "gumbel":
                value_loss_fn = partial(gumbel_loss, beta=self.beta, clip=self.exp_clip) # (v_pred, target_v_pi, beta, self.exp_clip)
            elif self.loss == "gumbel_rescale":
                value_loss_fn = partial(gumbel_rescale_loss, beta=self.beta, clip=self.exp_clip)
            elif self.loss == "mse":
                value_loss_fn = mse_loss
            elif self.loss == "expectile":
                value_loss_fn = expectile_loss
            else:
                raise ValueError("Invalid loss specified.")
            # value_loss = value_loss_fn(v_pred, target_v_pi)
            # value_loss = value_loss.mean()
            # Note: Could also just compute the mean over a broadcasted target. TO investigate later.
            # value_loss_greedy = sum([value_loss_fn(vs[i], target_v_pi_greedy).mean() for i in range(vs.shape[0])])
            value_loss = sum([value_loss_fn(vs[i], target_v_pi) for i in range(vs.shape[0])])

            self.optim['value'].zero_grad(set_to_none=True)
            value_loss.backward()
            # Gradient clipping
            if self.max_grad_value is not None:
                torch.nn.utils.clip_grad_value_(self.network.value.parameters(), self.max_grad_value)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.network.value.parameters(), self.max_grad_norm)
            self.optim['value'].step()

            all_metrics['value_loss'] = value_loss.item()
            all_metrics['target_v'] = target_v_pi.mean().item()

        # Update Actor
        if self.steps % self.actor_freq == 0:
            # Forward KL (action sampled off-policy)
            obs = batch['obs'].detach() # Detach the encoder so it isn't updated.
            mu = batch['action'].detach()
            
            dist = self.network.actor(obs)
            log_prob_mu = dist.log_prob(mu).sum(dim=-1) # To avoid overfitting to behavior, designate the lower bound
            
            qs = self.network.critic(obs, mu)
            q = torch.min(qs, dim=0)[0]
            v = self.network.value(obs).squeeze(-1)
            exp_a = torch.exp((q - v) / self.beta)
            
            forward_kl = -(exp_a * log_prob_mu).mean()
            
            # Reverse KL (action sampled on-policy)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            qs = self.network.critic(obs, action)
            q = torch.mean(qs, dim=0)[0]
            
            if self._alpha is None:
                reverse_kl = self.alpha.detach() * log_prob - q
                
                # Update alpha
                self.optim['log_alpha'].zero_grad(set_to_none=True)
                alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.optim['log_alpha'].step()
            else:
                reverse_kl = self._alpha * log_prob - q
            
            if self.offline:
                if self.rkl_low is not None:
                    reverse_kl = torch.clamp(reverse_kl, min=-self.rkl_low)
                bc = -log_prob_mu # Maximize log_prob_mu
                if self.bc_low is not None:
                    bc = torch.clamp(bc, min=-self.bc_low) # Clamp bc with 
                actor_loss = reverse_kl.mean() 
                
                if self.steps < self.init_steps:
                    actor_loss = bc.mean()
            else:
                actor_loss = reverse_kl.mean()

            self.optim['actor'].zero_grad(set_to_none=True)
            actor_loss.backward()
            # Gradient clipping
            if self.max_grad_value is not None:
                torch.nn.utils.clip_grad_value_(self.network.actor.parameters(), self.max_grad_value)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.max_grad_norm)
            self.optim['actor'].step()
            
            all_metrics['reverse_kl'] = reverse_kl.mean().item()
            # all_metrics['behavior_loss'] = (-log_prob_mu.mean()).item()
            all_metrics['entropy'] = (-log_prob.mean()).item()
            all_metrics['alpha'] = self.alpha.detach().item()

        if self.steps % self.target_freq == 0:
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")

    def _validation_extras(self, *args):
        '''
        Record difference between MC estimates of Q^{\pi(s,a)} and Q^{\pi_\replay}(s,a)
        '''
        Qs_pi, Qs_behavior, Qs_behavior_estimate = [], [], []
        discount = self.dataset.discount
        if self.offline:
            for traj in self.dataset._trajs[:256]:
                obs, act, rew, _, _, _ = traj[0]
                obs = torch.from_numpy(obs).float().to(next(self.network.parameters()).device).unsqueeze(0)
                act = torch.from_numpy(act).float().to(next(self.network.parameters()).device).unsqueeze(0)
                Qs_behavior_estimate.append(self.network.critic(obs, act).detach().cpu().numpy())
                
                t, Q_behavior = 0,0
                for obs, act, rew, _, _, _ in traj:
                    Q_behavior += (discount ** t) * rew
                    t += 1
                Qs_behavior.append(Q_behavior)
                
                metrics = eval_policy(self.eval_env, self, 1, prefix=[obs, act], discount=discount)
                Qs_pi.append(metrics['reward_discount'])
        else:
            for s_0, a_0, Q_behavior in self._memory:
                Qs_behavior.append(Q_behavior)
                # Compute MC estimate of the current policy, starting from a specific state
                metrics = eval_policy(self.eval_env, self, 1, prefix=[s_0, a_0], discount=discount)
                Qs_pi.append(metrics['reward_discount'])
                s_0 = torch.from_numpy(s_0).float().to(next(self.network.parameters()).device).unsqueeze(0)
                a_0 = torch.from_numpy(a_0).float().to(next(self.network.parameters()).device).unsqueeze(0)
                Qs_behavior_estimate.append(self.network.critic(s_0, a_0).detach().cpu().numpy())
        
        val_metrics = dict(qs_pi=np.mean(Qs_pi), qs_behavior=np.mean(Qs_behavior), qs_behavior_estimated=np.mean(Qs_behavior_estimate))
        for k, v in val_metrics.items():
            val_metrics[k] = np.mean(v)
        return val_metrics
