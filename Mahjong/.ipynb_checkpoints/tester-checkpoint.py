from multiprocessing import Process
import numpy as np
import torch

from env import MahjongGBEnv
from feature import FeatureAgent
from model import MyModel
import random
from pdb import set_trace as stx
from utils import set_all_seeds
import time
from tqdm import tqdm


def obs2action(agent, model, obs):
    legal_actions = np.where(obs['action_mask'] == 1)[0]
    assert legal_actions.size

    if 1 in legal_actions:
        return 1
    
    next_states_obs_list = []
    choices_for_value_eval = []

    for action_idx in legal_actions:
        next_state_observation_np = agent.get_next_state(action_idx)
        next_states_obs_list.append(next_state_observation_np)
        choices_for_value_eval.append((action_idx, next_state_observation_np))

    batched_next_states_tensor = torch.from_numpy(np.stack(next_states_obs_list)).float().cuda()

    with torch.no_grad():
        values_tensor = model(batched_next_states_tensor)

    epsilon = 0.00

    if random.random() < epsilon:
        my_action = random.choice(legal_actions)
    else:
        actions_with_values = []
        for i, (action_idx, _) in enumerate(choices_for_value_eval):
            actions_with_values.append((action_idx, values_tensor[i].item()))
        
        random.shuffle(actions_with_values)
        max_value = -float('inf')
        best_action_idx = -1
        for action_idx, value in actions_with_values:
            if value > max_value:
                max_value = value
                best_action_idx = action_idx
        my_action = best_action_idx

    return my_action


class Tester(): # 指定预训练模型/纯随机。
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
                model = model.to(torch.device('cuda'))
                model.eval()
                models[env.agent_names[i]] = model
        
        episode = 0
        total_rewards = {}
        win_rate = {}
        lens = {}
        for agent_name in env.agent_names:
            total_rewards[agent_name] = 0
            win_rate[agent_name] = 0
            lens[agent_name] = 0

        for episode in tqdm(range(1, self.config['episodes'] + 1)):
            # run one episode and collect data
            obs = env.reset()
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'reward' : [], # 初始obs的reward
                'value' : []
            } for agent_name in env.agent_names}
            done = False
            episode_len = 0
            while not done:
                episode_len += 1
                # each player take action
                actions = {}
                values = {}
                expected_state = None
                assert len(obs) in [1,3]
                for agent_name in obs:
                    if models[agent_name] == 'random':
                        arr = obs[agent_name]['action_mask']
                        if arr[0] == 1: # Pass
                            actions[agent_name] = 0
                            continue
                        indices = np.where(arr == 1)[0]
                        assert indices.size > 0
                        action = random.choice(indices).item()
                        actions[agent_name] = action
                        continue

                    agent_data = episode_data[agent_name]

                    my_action = obs2action(env.agents[int(agent_name[-1])-1], models[agent_name], obs[agent_name])

                    actions[agent_name] = my_action
                    agent_data['action'].append(actions[agent_name])
                # interact with env
                next_obs, rewards, done = env.step(actions)
                obs = next_obs

            for agent_name in rewards:
                total_rewards[agent_name] += rewards[agent_name]
                if rewards[agent_name] > 0:
                    win_rate[agent_name] += 1
                    lens[agent_name] += episode_len

            if episode % 50 == 0:
                print('Episode', episode, end=' ')
                for k in total_rewards:
                    print(f"{total_rewards[k]/episode:.3f}, {win_rate[k]/episode:.3f}, {0 if win_rate[k]==0 else lens[k]/win_rate[k]:.3f}", end=' ')
                print(flush=True)


if __name__ == '__main__':

    # seed = int(time.time())
    seed = 3407
    set_all_seeds(seed)
    print(f"Seed: {seed}")

    path = 'expe/0628003319/models/model_3300000.pt'
    config = {
        'episodes': 10000,
        'policies': [path, 'random', 'random', 'random', ] # 'random',
    }
    tester = Tester(config)