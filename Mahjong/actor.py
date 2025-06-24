from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import MyModel
import random

class Actor(Process):
    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        
    def run(self):
        torch.set_num_threads(1)
    
        # connect to model pool
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        
        # create network model
        model = MyModel()
        
        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        model.load_state_dict(state_dict)
        
        # collect data
        env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
        policies = {player : model for player in env.agent_names} # all four players use the latest model
        
        episode = 0
        total_try = 0
        total_rewards = {agent_name: 0 for agent_name in env.agent_names}
        # for episode in range(self.config['episodes_per_actor']):
        while episode < self.config['episodes_per_actor']:
            total_try += 1
            # update model
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                model.load_state_dict(state_dict)
                version = latest
            
            # run one episode and collect data
            obs = env.reset()
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'reward' : [],
                'value' : []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                # each player take action
                actions = {}
                values = {}
                assert len(obs) in [1,3]
                for agent_name in obs:
                    if agent_name != 'player_1': # 只训player1试试
                        arr = obs[agent_name]['action_mask']
                        indices = np.where(arr == 1)[0]
                        assert indices.size > 0
                        action = random.choice(indices).item()
                        actions[agent_name] = action
                        continue

                    agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(state['observation'])
                    agent_data['state']['action_mask'].append(state['action_mask'])
                    state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
                    state['action_mask'] = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0)
                    model.train(False) # Batch Norm inference mode
                    with torch.no_grad():
                        logits, value = model(state)
                        action_dist = torch.distributions.Categorical(logits = logits)
                        action = action_dist.sample().item()
                        # print(action_dist)
                        value = value.item()
                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(actions[agent_name])
                    agent_data['value'].append(values[agent_name])
                # interact with env
                next_obs, rewards, done = env.step(actions)

                if not done:
                    assert all(value == 0 for value in rewards.values())
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(rewards[agent_name])
                    total_rewards[agent_name] += rewards[agent_name]
                obs = next_obs
            
            # print('----------', rewards); exit(0)
            if all(value == 0 for value in rewards.values()):
                continue
            episode += 1
            print(self.name, 'Episode', episode, 'hu_rate', f'{episode/total_try:.2f}', 'Model', latest['id'], 'Reward', rewards, 'Total_rewards', total_rewards, flush=True) # ?? 为何 episode没到1000

            # postprocessing episode data for each agent
            for agent_name, agent_data in episode_data.items():
                if agent_name != 'player_1':
                    continue

                if agent_data['reward'][-1] <= 0: # 只训练胡牌的agent
                    continue

                if len(agent_data['action']) < len(agent_data['reward']):
                    agent_data['reward'].pop(0)
                obs = np.stack(agent_data['state']['observation'])
                mask = np.stack(agent_data['state']['action_mask'])
                actions = np.array(agent_data['action'], dtype = np.int64)
                rewards = np.array(agent_data['reward'], dtype = np.float32)
                values = np.array(agent_data['value'], dtype = np.float32)
                next_values = np.array(agent_data['value'][1:] + [0], dtype = np.float32)
                
                td_target = rewards + next_values * self.config['gamma']
                td_delta = td_target - values
                advs = []
                adv = 0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv) # GAE
                advs.reverse()
                advantages = np.array(advs, dtype = np.float32)
                
                # send samples to replay_buffer (per agent)
                self.replay_buffer.push({
                    'state': {
                        'observation': obs,
                        'action_mask': mask
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': td_target
                })

