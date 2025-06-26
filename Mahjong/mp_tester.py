from multiprocessing import Process, Queue
import numpy as np
import torch
import random
import time
# from replay_buffer import ReplayBuffer # Not directly used in Tester for multi-process setup
# from model_pool import ModelPoolClient # Not directly used in Tester for multi-process setup
from env import MahjongGBEnv
from feature import FeatureAgent
from model import MyModel
from utils import set_all_seeds

class Tester(Process): # Make Tester inherit from Process
    def __init__(self, config, result_queue):
        super().__init__() # Call the constructor of the parent class (Process)
        self.config = config
        self.result_queue = result_queue # Queue to send results back to main process

    def run(self): # This method will be executed when the process starts
        # Set a unique seed for each process
        # Using process ID or a combination with time can ensure uniqueness
        process_seed = self.config.get('base_seed', int(time.time())) + self.pid # Use process ID for unique seed
        set_all_seeds(process_seed)
        print(f"Process {self.pid} started with seed: {process_seed}", flush=True)

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
                # Load model to CPU first, then transfer to device if needed (though Tester is usually CPU-bound)
                state_dict = torch.load(policy, map_location='cpu') 
                model.load_state_dict(state_dict)
                model.eval()
                models[env.agent_names[i]] = model
        
        total_rewards = {agent_name: 0 for agent_name in env.agent_names}

        # Each Tester process runs its own set of episodes
        # Divide total episodes among testers or let each tester run total_episodes
        # For simplicity, let each Tester run `episodes_per_tester`
        episodes_to_run = self.config['episodes_per_tester'] # Renamed for clarity

        for episode_idx in range(episodes_to_run):
            # run one episode and collect data
            obs = env.reset()
            done = False
            
            # Reset episode rewards for printout (total_rewards accumulates across episodes)
            episode_rewards = {agent_name: 0 for agent_name in env.agent_names}

            while not done:
                actions = {}
                # assert len(obs) in [1,3] # This assertion might fail if env returns obs for all players simultaneously
                                          # typically, obs only contains observable states for current acting player(s)
                
                # Determine which agent is acting based on 'obs' keys
                current_acting_agents = list(obs.keys())
                
                for agent_name in current_acting_agents: # Iterate only over agents that need to act
                    model = models.get(agent_name) # Get the specific model for this agent

                    if model == 'random': # If it's a random policy
                        arr = obs[agent_name]['action_mask']
                        if arr[0] == 1: # Pass
                            actions[agent_name] = 0
                            continue
                        indices = np.where(arr == 1)[0]
                        if indices.size == 0: # No legal actions, should not happen if environment is correct
                            actions[agent_name] = 0 # Default to Pass if no legal actions (fallback)
                        else:
                            actions[agent_name] = random.choice(indices).item()
                        continue
                    
                    # --- Logic for learned agent (player_1) ---
                    # Assuming only 'player_1' uses a learned model based on original code
                    # If other agents also use learned models, this structure needs adjustment
                    if agent_name == 'player_1':
                        state = obs[agent_name]
                        
                        # Convert observation to tensor
                        current_obs_tensor = torch.tensor(state['observation'], dtype=torch.float).unsqueeze(0)
                        
                        model.eval() # Set model to evaluation mode for inference

                        legal_actions = np.where(state['action_mask'] == 1)[0]
                        
                        if 1 in legal_actions: # 能胡必胡
                            actions[agent_name] = 1 # Hu action
                            continue

                        # Prepare for value prediction of next states
                        choices = []
                        next_states_to_batch = []

                        for action in legal_actions:
                            # get_next_state must be a method of the agent instance in the environment
                            # env.agents list is 0-indexed, player_1 is agent 0
                            agent_idx = int(agent_name.split('_')[-1]) - 1 
                            next_state_obs = env.agents[agent_idx].get_next_state(action)
                            next_states_to_batch.append(next_state_obs)

                        if len(next_states_to_batch) == 0: # Handle cases with no valid next states
                            actions[agent_name] = 0 # Default to Pass
                            # print(f"Warning: No valid next states for {agent_name}, defaulting to Pass.")
                            continue

                        batched_next_states_np = np.stack(next_states_to_batch)
                        tensor = torch.from_numpy(batched_next_states_np).float()

                        with torch.no_grad():
                            values_tensor = model(tensor)

                        for i, action in enumerate(legal_actions):
                            value = values_tensor[i].item() 
                            choices.append((action, value)) # Only need action and value for selection

                        epsilon = self.config['epsilon'] # Use epsilon from config
                        
                        # Epsilon-greedy action selection
                        if random.random() < epsilon:
                            my_action = random.choice([c[0] for c in choices]) # Randomly pick an action
                        else:
                            # Select action with max value (break ties randomly)
                            random.shuffle(choices) # Shuffle to break ties randomly
                            max_value = -float('inf')
                            my_action = None
                            for action, value in choices:
                                if value > max_value: # Strict greater than ensures max_value is updated
                                    my_action = action
                                    max_value = value
                        
                        actions[agent_name] = my_action
                    else:
                        # Fallback for other agents if they weren't explicitly 'random' or 'player_1' logic
                        # This should ideally be covered by the initial 'models' dict
                        print(f"Warning: Unhandled agent type for {agent_name}, defaulting to Pass.")
                        actions[agent_name] = 0 # Default to Pass
                
                next_obs, rewards, done = env.step(actions)
                
                for agent_name in rewards:
                    total_rewards[agent_name] += rewards[agent_name]
                    episode_rewards[agent_name] += rewards[agent_name] # Accumulate for current episode print
                
                obs = next_obs
            
            # Print episode summary
            win_rate = total_rewards['player_1'] / (episode_idx + 1)
            # Use self.pid for process-specific logging
            print(f'Process {self.pid} | Episode {episode_idx + 1} | Player 1 Reward: {episode_rewards["player_1"]:.2f} | Total Player 1 Reward: {total_rewards["player_1"]:.2f} | Win Rate: {win_rate:.2f}', flush=True)

        # After all episodes are done for this process, put results into the queue
        self.result_queue.put(total_rewards)
        print(f"Process {self.pid} finished and put results in queue.", flush=True)


if __name__ == '__main__':
    # Initial setup for main process
    base_seed = int(time.time())
    set_all_seeds(base_seed)
    print(f"Main process seed: {base_seed}")

    num_testers = 4 # Number of parallel Tester processes
    episodes_per_tester = 1000 # Each tester will run this many episodes
    
    main_config = {
        'episodes_per_tester': episodes_per_tester, # Each Tester runs this many episodes
        'policies': ['expe/06261520/models/model_48000.pt', 'random', 'random', 'random'],
        'epsilon': 0.05, # Epsilon for e-greedy in Tester
        'base_seed': base_seed # Pass base_seed for unique seed generation in subprocesses
    }
    
    results_queue = Queue() # Queue to collect results from Tester processes
    
    testers = []
    for i in range(num_testers):
        tester = Tester(main_config, results_queue)
        testers.append(tester)
        tester.start() # Start each Tester process

    # Wait for all Tester processes to complete
    for tester in testers:
        tester.join()

    # Collect results from the queue
    final_total_rewards = {
        'player_0': 0, 'player_1': 0, 'player_2': 0, 'player_3': 0 # Assuming these are env.agent_names
    } 
    # Initialize with actual agent names from env if possible, or use 'player_1' etc.
    # For simplicity, using generic player names. You might want to get them from env
    # Example: env = MahjongGBEnv(config={'agent_clz': FeatureAgent}); final_total_rewards = {name: 0 for name in env.agent_names}

    total_episodes_run = num_testers * episodes_per_tester

    while not results_queue.empty():
        tester_rewards = results_queue.get()
        for agent_name, reward in tester_rewards.items():
            # Adjusting agent_name for generic aggregation if necessary
            # Assuming agent_name is 'player_1', 'player_2' etc. already
            if agent_name in final_total_rewards:
                final_total_rewards[agent_name] += reward
            else: # If agent_name not pre-initialized, add it
                final_total_rewards[agent_name] = reward


    print("\n--- All Tester Processes Finished ---")
    print(f"Total Episodes Run Across All Testers: {total_episodes_run}")
    print("Aggregated Total Rewards:")
    for agent_name, reward in final_total_rewards.items():
        if 'player_1' in agent_name: # Assuming player_1 is the one you track win rate for
            aggregated_win_rate = reward / total_episodes_run
            print(f"- {agent_name}: {reward:.2f} (Aggregated Win Rate: {aggregated_win_rate:.2f})")
        else:
            print(f"- {agent_name}: {reward:.2f}")