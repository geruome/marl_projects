from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner

if __name__ == '__main__':
    config = {
        'replay_buffer_size': 5000, # 小点,只学习近期数据
        'replay_buffer_episode': 400, # Queue参数. Queue收集episode,unpack给buffer
        'model_pool_size': 4, # ? 1就够吧
        'model_pool_name': 'model-pool',
        'num_actors': 8,
        'episodes_per_actor': 1000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 256, # ?
        'batch_size': 256,
        'epochs': 5,
        'clip': 0.2,
        'lr': 1e-3,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cuda',
        'total_iters': 10000,
        'ckpt_save_interval': 100,
    }
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    
    # Process对象的start(): 创建一个新进程 + 调用run()
    for actor in actors: actor.start()
    learner.start()
    
    for actor in actors: actor.join() # join(): 让主进程等待子进程完成run()
    learner.terminate() # 跑完再停