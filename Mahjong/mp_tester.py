from multiprocessing import Process, Queue
import numpy as np
import torch

from env import MahjongGBEnv
from feature import FeatureAgent
from model import MyModel
import random
from utils import set_all_seeds
import time
from tqdm import tqdm


class Tester(Process):
    def __init__(self, config, result_queue):
        super().__init__()
        self.config = config
        self.result_queue = result_queue

    def run(self):
        policies = self.config['policies']
        assert len(policies) == 4
                
        env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
        models = {}
        for i in range(4):
            policy = policies[i]
            if policy == 'random':
                models[env.agent_names[i]] = policy
            else:   
                model = MyModel()
                state_dict = torch.load(policy)
                model.load_state_dict(state_dict)
                model.eval()
                models[env.agent_names[i]] = model
        
        total_rewards = {}
        for agent_name in env.agent_names:
            total_rewards[agent_name] = 0

        for episode in tqdm(range(1, self.config['episodes'] + 1)):
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
                actions = {}
                values = {}
                expected_state = None
                assert len(obs) in [1,3]
                for agent_name in obs:
                    if models[agent_name] == 'random':
                        arr = obs[agent_name]['action_mask']
                        if arr[0] == 1:
                            actions[agent_name] = 0
                            continue
                        indices = np.where(arr == 1)[0]
                        assert indices.size > 0
                        action = random.choice(indices).item()
                        actions[agent_name] = action
                        continue

                    model = models[agent_name]
                    
                    agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(state['observation'])
                    state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
                    model.train(False)

                    legal_actions = np.where(state['action_mask'] == 1)[0]
                    if 1 in legal_actions:
                        actions[agent_name] = 1
                        continue

                    with torch.no_grad():
                        choices = []
                        next_states_to_batch = []

                        current_agent_obj = env.agents[int(agent_name.split('_')[-1])-1]
                        
                        for action in legal_actions:
                            next_state_obs = current_agent_obj.get_next_state(action)
                            next_states_to_batch.append(next_state_obs)

                        if not next_states_to_batch:
                            actions[agent_name] = 0
                            continue

                        batched_next_states_np = np.stack(next_states_to_batch)
                        
                        tensor = torch.from_numpy(batched_next_states_np).float()
                        with torch.no_grad():
                            values_tensor = model(tensor)

                        for i, action in enumerate(legal_actions):
                            value = values_tensor[i].item() 
                            choices.append((action, next_states_to_batch[i], value))


                    epsilon = 0.05
                    if random.random() < epsilon:
                        my_action, expected_state, value = random.choice(choices)
                    else:
                        random.shuffle(choices)
                        max_value = -float('inf')
                        for action, next_state, value_item in choices:
                            if max_value < value_item:
                                my_action = action
                                expected_state = next_state
                                max_value = value_item

                    actions[agent_name] = my_action
                    agent_data['action'].append(actions[agent_name])
                
                next_obs, rewards, done = env.step(actions)
                for agent_name in rewards:
                    total_rewards[agent_name] += rewards[agent_name]
                obs = next_obs
            
            print('Episode', episode, end=' ', flush=True)
            for k in total_rewards:
                v = total_rewards[k] / episode
                print(f"{v:.2f}", end=' ', flush=True)
            print(flush=True)

        self.result_queue.put(total_rewards)


if __name__ == '__main__':
    seed = int(time.time())
    set_all_seeds(seed)
    print(f"Seed: {seed}")

    num_testers = 4

    config = {
        'episodes': 250,
        'policies': ['expe/06261616/models/model_11000.pt', 'random', 'random', 'random']
    }
    
    results_queue = Queue()

    testers = []
    for i in range(num_testers):
        tester = Tester(config, results_queue)
        testers.append(tester)
        tester.start()

    for tester in testers:
        tester.join()

    dummy_env = MahjongGBEnv(config={'agent_clz': FeatureAgent})
    final_total_rewards = {agent_name: 0 for agent_name in dummy_env.agent_names}
    
    total_episodes_evaluated = num_testers * config['episodes']

    while not results_queue.empty():
        tester_rewards = results_queue.get()
        for agent_name, reward in tester_rewards.items():
            if agent_name in final_total_rewards:
                final_total_rewards[agent_name] += reward
            else:
                final_total_rewards[agent_name] = reward

    print("\n--- All Tester Processes Finished ---")
    print(f"Total episodes evaluated across all testers: {total_episodes_evaluated}")
    print("Aggregated Average Rewards:")
    for agent_name, total_reward_sum in final_total_rewards.items():
        overall_avg_reward = total_reward_sum / total_episodes_evaluated
        print(f"- {agent_name}: {overall_avg_reward:.2f}")