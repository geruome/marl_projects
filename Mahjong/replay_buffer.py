from multiprocessing import Queue
from collections import deque
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, capacity, queue_size, max_sample_count=4):
        self.queue = Queue(queue_size)
        self.capacity = capacity
        self.buffer = [] # Stores (sample_data, current_sample_count, timestamp_sample_in)
        
        self.max_sample_count = max_sample_count
        self.max_age_samples = capacity

        self.stats = {
            'sample_in': 0, 
            'sample_out': 0, 
            'episode_in': 0, 
            'sample_removed_count_limit': 0, 
            'sample_removed_age_limit': 0
        }

        self.queue_reward = Queue(50000)
        self.reward_buffer = []

    def push(self, samples):
        self.queue.put(samples)
        
    def push_reward(self, reward):
        self.queue_reward.put(reward)

    def avg_reward(self):
        while not self.queue_reward.empty():
            reward = self.queue_reward.get()
            self.reward_buffer.append(reward)
        l = min(5000, len(self.reward_buffer)) # 
        if l == 0:
            return 0
        return sum(self.reward_buffer[-l:]) / l

    def _flush(self):
        while not self.queue_reward.empty():
            reward = self.queue_reward.get()
            self.reward_buffer.append(reward)

        # Filter out old or excessively sampled data
        samples_to_keep = []
        current_sample_count_in_buffer = self.stats['sample_in'] # Use this as the current "time" for age calculation

        for sample_data, current_count, timestamp in self.buffer:
            if current_count >= self.max_sample_count:
                self.stats['sample_removed_count_limit'] += 1
                continue
            
            if current_sample_count_in_buffer - timestamp > self.max_age_samples:
                self.stats['sample_removed_age_limit'] += 1
                continue

            samples_to_keep.append((sample_data, current_count, timestamp))
        
        self.buffer = samples_to_keep

        # Add new data from queue
        while not self.queue.empty():
            episode_data = self.queue.get()
            unpacked_data = self._unpack(episode_data)
            
            for sample_item in unpacked_data:
                self.buffer.append((sample_item, 0, self.stats['sample_in'])) 
                self.stats['sample_in'] += 1

            self.stats['episode_in'] += 1

    def sample(self, batch_size):
        self._flush()
        assert batch_size <= len(self.buffer)

        actual_batch_size = min(batch_size, len(self.buffer))
        sampled_indices = random.sample(range(len(self.buffer)), actual_batch_size)
        
        samples_to_return = []
        for idx in sampled_indices:
            sample_data, current_count, timestamp = self.buffer[idx]
            self.buffer[idx] = (sample_data, current_count + 1, timestamp)
            samples_to_return.append(sample_data)

        self.stats['sample_out'] += actual_batch_size
        return self._pack(samples_to_return)
    
    def size(self): # only called by learner
        self._flush()
        return len(self.buffer)
    
    def clear(self): # only called by learner
        self._flush()
        self.buffer.clear()
    
    def _unpack(self, data):
        # convert dict (of dict...) of list of (num/ndarray/list) to list of dict (of dict...)
        if type(data) == dict:
            res = []
            for key, value in data.items():
                values = self._unpack(value)
                if not res: res = [{} for i in range(len(values))]
                for i, v in enumerate(values):
                    res[i][key] = v
            return res
        else:
            return list(data)
            
    def _pack(self, data):
        # convert list of dict (of dict...) to dict (of dict...) of numpy array
        if type(data[0]) == dict:
            keys = data[0].keys()
            res = {}
            for key in keys:
                values = [x[key] for x in data]
                res[key] = self._pack(values)
            return res
        elif type(data[0]) == np.ndarray:
            return np.stack(data)
        else:
            return np.array(data)