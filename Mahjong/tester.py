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
    model.train(False)

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

    batched_next_states_tensor = torch.from_numpy(np.stack(next_states_obs_list)).float()

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
                model.eval()
                models[env.agent_names[i]] = model
        
        episode = 0
        total_rewards = {}
        for agent_name in env.agent_names:
            total_rewards[agent_name] = 0

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
            while not done:
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
                    # state = obs[agent_name]
                    # agent_data['state']['observation'].append(state['observation'])
                    # state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)

                    # legal_actions = np.where(state['action_mask'] == 1)[0]
                    # if 1 in legal_actions: # 能胡必胡
                    #     # 不纯e-greedy, 多 吃/碰/杠
                    #     actions[agent_name] = 1
                    #     continue

                    # with torch.no_grad():
                    #     choices = []
                    #     next_states_to_batch = []

                    #     for action in legal_actions:
                    #         next_state_obs = env.agents[int(agent_name[-1])-1].get_next_state(action)
                    #         next_states_to_batch.append(next_state_obs)

                    #     batched_next_states_np = np.stack(next_states_to_batch)
                            
                    #     tensor = torch.from_numpy(batched_next_states_np).float()
                    #     with torch.no_grad():
                    #         values_tensor = models[agent_name](tensor)

                    #     for i, action in enumerate(legal_actions):
                    #         value = values_tensor[i].item() 
                    #         choices.append((action, next_states_to_batch[i], value))


                    # epsilon = 0.00
                    # # e-greedy
                    # if random.random() < epsilon:
                    #     my_action, expected_state, value = random.choice(choices)
                    # else:
                    #     random.shuffle(choices)
                    #     max_value = -float('inf')
                    #     for action, next_state, value in choices:
                    #         if max_value < value:
                    #             my_action = action
                    #             expected_state = next_state
                    #             max_value = value

                    my_action = obs2action(env.agents[int(agent_name[-1])-1], models[agent_name], obs[agent_name])

                    actions[agent_name] = my_action
                    agent_data['action'].append(actions[agent_name])
                # interact with env
                next_obs, rewards, done = env.step(actions)
                for agent_name in rewards:
                    total_rewards[agent_name] += rewards[agent_name]
                obs = next_obs
            # print('----------', rewards); exit(0)
            # if not all(value == 0 for value in rewards.values()):
            #     hu_episode += 1
            print('Episode', episode, end=' ')
            for k in total_rewards:
                v = total_rewards[k] / episode
                print(f"{v:.2f}", end=' ')
            print(flush=True)

        # print(total_rewards)


if __name__ == '__main__':

    seed = int(time.time())
    set_all_seeds(seed)
    print(f"Seed: {seed}")

    path = 'pretrained_weights/0626202920_37000.pt'
    config = {
        'episodes': 1000,
        
        'policies': [path, 'random', 'random', 'random', ] # 'random',
    }
    tester = Tester(config)