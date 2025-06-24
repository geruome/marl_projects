from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import MyModel
import random


class Tester4(): # 4人, 指定预训练模型/纯随机。
    def __init__(self, config):
        self.config = config
        self.run()

    def run(self):
        policies = self.config['policies']
        assert len(policies) == 4
                
        env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
        models = {}
        for i in range(4):
            policy = policies[i]
            if policy == 'random':
                models[env.agent_names[i]] = policy
                pass
            else:   
                model = MyModel()
                state_dict = torch.load(policy)
                model.load_state_dict(state_dict)
                model.eval()
                models[env.agent_names[i]] = model
        
        episode = 0
        total_rewards = {}
        for agent_name in env.agent_names:
            total_rewards[agent_name] = 0

        for episode in range(self.config['episodes']):
        # while episode < self.config['episodes']:            
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
                    if models[agent_name] == 'random':
                        arr = obs[agent_name]['action_mask']
                        indices = np.where(arr == 1)[0]
                        assert indices.size > 0
                        action = random.choice(indices).item()
                        actions[agent_name] = action
                    else:
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
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(rewards[agent_name])
                    total_rewards[agent_name] += rewards[agent_name]

                obs = next_obs
            
            # print('----------', rewards); exit(0)
            if not all(value == 0 for value in rewards.values()):
                hu_episode += 1
            print('Episode', episode, 'Reward', rewards, 'Total_rewards', total_rewards, flush=True)

        print(total_rewards)


if __name__ == '__main__':
    config = {
        'episodes': 1000,
        'policies': ['expe/06242109/model_100.pt', 'random', 'random', 'random']
    }
    tester = Tester4(config)