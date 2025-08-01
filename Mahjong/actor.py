from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import MyModel
import random
from utils import set_all_seeds
import time


class Actor(Process):
    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config

        self.name = config.get('name', 'Actor-?')
        seed = config['seed'] # + int(self.name[-1]) ?? 为什么不同就训不动了
        set_all_seeds(seed)
        self.reward_buffer = []
        

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
        # policies = {player : model for player in env.agent_names} # all four players use the latest model
        
        episode = 0
        epsilon = self.config['max_epsilon']        

        while episode < self.config['episodes_per_actor']:
            # time.sleep(1)
            # update model
            retry_cnt = 0
            while True:
                try:
                    latest = model_pool.get_latest_model()
                    if latest['id'] > version['id']:
                        state_dict = model_pool.load_model(latest)
                        model.load_state_dict(state_dict)
                        version = latest
                    break
                except Exception as e:
                    print(f"Error during loading latest_model: {e}\nretrying...", flush=True)
                    time.sleep(0.1)
                    retry_cnt += 1
                    if retry_cnt >= 3:
                        assert 0


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
                    if agent_name != 'player_1': # 只训player1, 其他全pass
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
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(state['observation'])
                    state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
                    model.train(False) # Batch Norm inference mode

                    legal_actions = np.where(state['action_mask'] == 1)[0]
                    if 1 in legal_actions: # 能胡必胡
                        # 不纯e-greedy, 多 吃/碰/杠
                        actions[agent_name] = 1
                        agent_data['action'].append(actions[agent_name])
                        continue

                    with torch.no_grad():
                        choices = []
                        next_states_to_batch = []
                        # for action in legal_actions:
                        #     next_state = env.agents[int(agent_name[-1])-1].get_next_state(action)
                        #     next_state = torch.tensor(next_state)
                        #     value = model(next_state)
                        #     choices.append(action, next_state, value)

                        for action in legal_actions:
                            next_state_obs = env.agents[int(agent_name[-1])-1].get_next_state(action)
                            next_states_to_batch.append(next_state_obs)

                        batched_next_states_np = np.stack(next_states_to_batch)
                            
                        tensor = torch.from_numpy(batched_next_states_np).float()
                        with torch.no_grad():
                            values_tensor = model(tensor)

                        for i, action in enumerate(legal_actions):
                            value = values_tensor[i].item() 
                            choices.append((action, next_states_to_batch[i], value))

                    # e-greedy
                    if random.random() < epsilon:
                        my_action, expected_state, value = random.choice(choices)
                    else:
                        random.shuffle(choices)
                        max_value = -float('inf')
                        for action, next_state, value in choices:
                            if max_value < value:
                                my_action = action
                                expected_state = next_state
                                max_value = value

                    actions[agent_name] = my_action
                    agent_data['action'].append(actions[agent_name])
                # interact with env
                next_obs, rewards, done = env.step(actions)
                # if not done and 'player_1' in next_obs: # 胡了的state / 没操作成的state ??
                #     print(expected_state, next_obs['player_1']['observation'])
                #     assert (expected_state == next_obs['player_1']['observation']).all()

                if not done:
                    assert all(value == 0 for value in rewards.values())
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(rewards[agent_name]) # here
                    # rewards[agent_name] / 8, rewards[agent_name] > 0
                obs = next_obs

            _reward = max(rewards['player_1'], 0)
            # _reward = rewards['player_1']
            self.replay_buffer.push_reward(_reward)

            if rewards['player_1'] <= 0:
                continue
            episode += 1

            max_epsilon = self.config['max_epsilon']
            min_epsilon = self.config['min_epsilon']
            total_episode = self.config['episodes_per_actor']

            epsilon = max_epsilon + (episode / total_episode) * (min_epsilon - max_epsilon)
            # print(episode, epsilon)

            print(self.name, 'Episode', episode, 'Model', latest['id'], 'Reward', rewards['player_1'], flush=True)
            # print("Avg_reward: ")

            # postprocessing episode data for each agent
            for agent_name, agent_data in episode_data.items():
                if agent_name != 'player_1':
                    continue

                if agent_data['reward'][-1] <= 0: # 只训练胡牌的agent
                    continue

                obs = np.stack(agent_data['state']['observation'])
                rewards = np.array(agent_data['reward'], dtype = np.float32)
                next_value = 0
                td_targets = []

                with torch.no_grad():
                    values = model(torch.tensor(obs))
                values = values.numpy()
                
                # print(values.shape, rewards.shape)
                for value, reward in zip(values[::-1], rewards[::-1]):
                    td_targets.append(reward + next_value*self.config['gamma'])
                    next_value = float(value)
                
                td_targets.reverse()

                # send samples to replay_buffer (per agent)
                self.replay_buffer.push({
                    'states': obs,
                    # 'rewards': rewards,
                    'td_targets': td_targets,
                })
                # self.replay_buffer.push_reward(

                # )
                # self.reward_buffer = []

        print(f"End of {self.name} !!")