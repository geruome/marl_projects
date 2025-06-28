from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
from utils import set_all_seeds
import time


if __name__ == '__main__':
    seed = int(time.time())
    # seed = 1750872601
    set_all_seeds(seed)

    config = {
        'replay_buffer_size': 5000, # 小点,只学习近期数据
        'replay_buffer_episode': 400, # Queue参数. Queue收集episode,unpack给buffer
        'max_sample_count': 8, 
        'model_pool_size': 2, # ? 1就够吧
        'model_pool_name': 'model-pool',
        'num_actors': 12,
        'episodes_per_actor': 1000000, # episodes_per_actor * 150? * count * num_actors / B。但实测是 episodes_per_actor * 6.5 = iters
        'gamma': 0.99,
        'lambda': 0.95,
        'min_sample': 256, # 
        'batch_size': 128,
        'ppo_epochs': 5,
        'clip': 0.2,
        'lr': 1e-3,
        'value_coeff': 1,
        'entropy_coeff': 0.01, # 还可以手动(e-greedy)鼓励探索。
        'device': 'cuda',
        'total_iters': 5000000,
        'ckpt_save_interval': 1000,
        'seed': seed,
        'pretrained_weights': 'expe/0627211756/models/model_647000.pt',
        'max_epsilon': 0.05, 
        'min_epsilon': 0.05,
        'note': 'reward(fan)=1, >=8',
    }
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'], config['max_sample_count'])

    learner = Learner(config, replay_buffer)

    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    
    # Process对象的start(): 创建一个新进程 + 调用run()
    for actor in actors: actor.start()
    learner.start()
    
    for actor in actors: actor.join() # join(): 让主进程等待子进程完成run()
    learner.terminate()

    # 子进程最后报错。如何正常结束