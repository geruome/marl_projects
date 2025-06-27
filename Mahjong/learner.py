from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F
import os
import json

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import MyModel
from utils import set_all_seeds
from torch.utils.tensorboard import SummaryWriter
import shutil


class Learner(Process): # 
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config

        set_all_seeds(config['seed'])
        self.expr_dir = os.path.join('expe', time.strftime('%m%d%H%M%S', time.localtime()))
        os.makedirs(self.expr_dir, exist_ok=True)
        with open(os.path.join(self.expr_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        shutil.copy('model.py', os.path.join(self.expr_dir, 'model.py'))
        os.makedirs(os.path.join(self.expr_dir, 'models'), exist_ok=True)

        
    def run(self):
        # create model pool. 负责模型的发布和管理，Actor 会从这里拉取最新模型
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        
        # initialize model params
        device = torch.device(self.config['device'])
        model = MyModel()

        self.logger = SummaryWriter(self.expr_dir)

        if self.config['pretrained_weights']:
            state_dict = torch.load(self.config['pretrained_weights'])
            model.load_state_dict(state_dict)

        # send to model pool
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        model = model.to(device)
        
        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr']) # ??
        
        iterations = 0
        while iterations < self.config['total_iters']:
            iterations += 1
            while self.replay_buffer.size() < self.config['min_sample']:
                time.sleep(1)
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            states = torch.tensor(batch['states']).to(device)
            td_targets = torch.tensor(batch['td_targets']).to(device)

            # print('Iteration %d, replay buffer in %d out %d' % (iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']), flush=True)
            avg_reward = self.replay_buffer.avg_reward()
            # print(f'Iteration {iterations}, avg_reward {avg_reward:.2f}', flush=True)
            self.logger.add_scalar('Avg_reward', avg_reward, iterations) # ???卡住

            model.train(True) # Batch Norm training mode
            values = model(states)
            td_targets = td_targets.view_as(values)

            loss = F.mse_loss(values, td_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.logger.add_scalar('Loss', loss.item(), iterations)

            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)
            
            # save checkpoints
            if iterations % self.config['ckpt_save_interval'] == 0:
                path = os.path.join(self.expr_dir, 'models', f'model_{iterations:06d}.pt')
                torch.save(model.state_dict(), path)

        print("End !!")
        # self.logger.close()
