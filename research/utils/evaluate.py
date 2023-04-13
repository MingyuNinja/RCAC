import torch
import numpy as np
import collections

MAX_METRICS = {'success', 'is_success'}
LAST_METRICS = {}
MEAN_METRICS = {}

def eval_policy(env, model, num_ep, prefix=None, discount=0.99):
    ep_rewards = []
    ep_rewards_dis = []
    ep_lengths = []
    ep_metrics = collections.defaultdict(list)

    for i in range(num_ep):
        # Reset Metrics
        done = False
        ep_reward = 0
        ep_reward_dis = 0
        ep_length = 0
        ep_metric = collections.defaultdict(list)
        
        obs = env.reset()
        # If given (s_0, a_0), perform MC estimate of the 
        if prefix is not None: 
            env.env.state = prefix[0]
            obs, reward, done, info = env.step(prefix[1])
            ep_reward += reward
            ep_reward_dis = reward + discount * ep_reward_dis
            ep_length += 1
        
        while not done:
            with torch.no_grad():
                action = model.predict(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_reward_dis = reward + discount * ep_reward_dis
            ep_length += 1
            for k, v in info.items():
                if isinstance(v, float) or np.isscalar(v):
                    ep_metric[k].append(v)
            
        try:
            ep_reward = env.get_normalized_score(ep_reward)
        except:
            pass

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        ep_rewards_dis.append(ep_reward_dis)
        for k, v in ep_metric.items():
            if k in MAX_METRICS:
                ep_metrics[k].append(np.max(v))
            elif k in LAST_METRICS: # Append the last value
                ep_metrics[k].append(v[-1])
            elif k in MEAN_METRICS:
                ep_metrics[k].append(np.mean(v))
            else:
                ep_metrics[k].append(np.sum(v))

    metrics = dict(reward=np.mean(ep_rewards), stddev=np.std(ep_rewards), length=np.mean(ep_lengths), reward_discount=np.mean(ep_reward_dis))
    for k, v in ep_metrics.items():
        metrics[k] = np.mean(v)
    return metrics