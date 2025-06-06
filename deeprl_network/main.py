"""
Main function for training and evaluating MARL algorithms in NMARL envs
@author: Tianshu Chu
"""

import argparse
import configparser
import logging
import threading
from torch.utils.tensorboard.writer import SummaryWriter
from envs.cacc_env import CACCEnv
from envs.large_grid_env import LargeGridEnv
from envs.real_net_env import RealNetEnv
from agents.models import IA2C, IA2C_FP, MA2C_NC, IA2C_CU, MA2C_CNET, MA2C_DIAL, IC3Net
from utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag)
import os
import time
from pdb import set_trace as stx


def parse_args():
    default_config_dir = 'config/config_ma2c_nc_grid_hybrid.ini'
    # 'config_ia2c_grid.ini', 'config_ma2c_nc_grid.ini', 'config_ic3net_grid.ini' 'config_ma2c_nc_grid_hybrid.ini'
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    sp = subparsers.add_parser('train', help='train a single agent under base dir')
    sp.add_argument('--config-dir', type=str, required=False,
                    default=default_config_dir, help="experiment config path")
    sp = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp.add_argument('--evaluation-seeds', type=str, required=False,
                    default=','.join([str(i) for i in range(2000, 2500, 10)]),
                    help="random seeds for evaluation, split by ,")
    sp.add_argument('--demo', action='store_true', help="shows SUMO gui")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=0):
    scenario = config.get('scenario')
    if scenario.startswith('atsc'):
        if scenario.endswith('large_grid'):
            return LargeGridEnv(config, port=port)
        else:
            return RealNetEnv(config, port=port)
    else:
        return CACCEnv(config)


def init_agent(env, config, total_step, seed):
    if env.agent == 'ia2c':
        return IA2C(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                    total_step, config, seed=seed)
    elif env.agent == 'ia2c_fp':
        return IA2C_FP(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_nc':
        return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_cnet':
        # this is actually CommNet
        return MA2C_CNET(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    elif env.agent == 'ma2c_cu':
        return IA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_dial':
        return MA2C_DIAL(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    elif env.agent == 'ma2c_ic3net':
        return IC3Net(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    else:
        return None


def train(args):
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)
    
    base_dir = os.path.join('expe', time.strftime('%m%d_%H%M', time.localtime())) # +'_'+env.agent)
    dirs = init_dir(base_dir)
    copy_file(config_dir, dirs['data'])
    init_log(dirs['log'])

    # init env    
    env = init_env(config['ENV_CONFIG'])

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)
    
    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')
    model = init_agent(env, config['MODEL_CONFIG'], total_step, seed)
    # model.load(file_path='expe/0601_1529/model/checkpoint-1000080.pt')
    # model.load(dirs['model'], train_mode=True)
        
    # disable multi-threading for safe SUMO implementation
    summary_writer = SummaryWriter(dirs['log'],) # flush_secs=10000)
    trainer = Trainer(env, model, global_counter, summary_writer, expe_path=dirs) # output_path=dirs['data'])
    trainer.run()

    # save model
    final_step = global_counter.cur_step
    model.save(dirs['model'], final_step)
    summary_writer.close()


def evaluate_fn(expe_dir, output_dir, seeds, port, demo=False):
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # dirs = init_dir(expe_dir)
    os.makedirs(expe_dir, exist_ok=True)
    copy_file(config_dir, expe_dir)
    open(os.path.join(expe_dir, args.model_path.split('/')[1]), 'w').write('Hello, this is my file.\n')

    # init env
    env = init_env(config['ENV_CONFIG'], port=port)
    env.init_test_seeds(seeds)

    # load model for agent
    model = init_agent(env, config['MODEL_CONFIG'], 0, 0)
    model.load(args.model_path)
    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir, gui=demo)
    evaluator.run()


def evaluate(args):
    args.config_dir = 'config/config_ma2c_nc_grid_hybrid.ini'
    args.model_path = 'expe/0603_1720_hybrid/model/checkpoint-1000000.pt' 
    # 'expe/0530_2009_ppo/0602_1001/model/checkpoint-800000.pt'
    base_dir = os.path.join('expe', time.strftime('%m%d_%H%M', time.localtime()) + '_eval/')
    # enforce the same evaluation seeds across agents
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    seeds = [int(s) for s in seeds.split(',')]
    evaluate_fn(base_dir, base_dir, seeds, 8888, )


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
