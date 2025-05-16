import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from pdb import set_trace as stx
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


# 复制环境类
class Env:
    def __init__(self, num_firms, p, h, c, initial_inventory, poisson_lambda=10, max_steps=100):
        """
        初始化供应链管理仿真环境。
        
        :param num_firms: 企业数量
        :param p: 各企业的价格列表
        :param h: 库存持有成本
        :param c: 损失销售成本
        :param initial_inventory: 每个企业的初始库存
        :param poisson_lambda: 最下游企业需求的泊松分布均值
        :param max_steps: 每个episode的最大步数
        """
        self.num_firms = num_firms
        self.p = p  # 企业的价格列表
        self.h = h  # 库存持有成本
        self.c = c  # 损失销售成本
        self.poisson_lambda = poisson_lambda  # 泊松分布的均值
        self.max_steps = max_steps  # 每个episode的最大步数
        self.initial_inventory = initial_inventory  # 初始库存
        
        # 初始化库存
        self.inventory = np.full((num_firms, 1), initial_inventory)
        # 初始化订单量
        self.orders = np.zeros((num_firms, 1))
        # 初始化已满足的需求量
        self.satisfied_demand = np.zeros((num_firms, 1))
        # 记录当前步数
        self.current_step = 0
        # 标记episode是否结束
        self.done = False

    def reset(self):
        """
        重置环境状态。
        """
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory)
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((self.num_firms, 1))
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        获取每个企业的观察信息，包括订单量、满足的需求量和库存。
        每个企业的状态是独立的，包括自己观察的订单、需求和库存。
        """
        return np.concatenate((self.orders, self.satisfied_demand, self.inventory), axis=1)

    def _generate_demand(self):
        """
        根据规则生成每个企业的需求。
        最下游企业的需求遵循泊松分布，其他企业的需求等于下游企业的订单量。
        """
        demand = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            if i == 0:
                # 最下游企业的需求遵循泊松分布，均值为 poisson_lambda
                demand[i] = np.random.poisson(self.poisson_lambda)
            else:
                # 上游企业的需求等于下游企业的订单量
                demand[i] = self.orders[i - 1]  # d_{i+1,t} = q_{it}
        return demand

    def step(self, actions):
        """
        执行一个时间步的仿真，根据给定的行动 (每个企业的订单量) 更新环境状态。
        
        :param actions: 每个企业的订单量 (shape: (num_firms, 1))，即每个智能体的行动
        :return: next_state, reward, done
        """
        self.orders = actions  # 更新订单量
        
        # 生成各企业的需求
        self.demand = self._generate_demand()

        # 计算每个企业收到的订单量和满足的需求
        for i in range(self.num_firms):
            self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])
        
        # 更新库存
        for i in range(self.num_firms):
            self.inventory[i] = self.inventory[i] + self.orders[i] - self.satisfied_demand[i]
        
        # 计算每个企业的奖励: p_i * d_{it} - p_{i+1} * q_{it} - h * I_{it}
        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))  # 损失销售费用
        
        for i in range(self.num_firms):
            rewards[i] += self.p[i] * self.satisfied_demand[i] - (self.p[i+1] if i+1 < self.num_firms else 0) * self.orders[i] - self.h * self.inventory[i]
            
            # 损失销售计算
            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c
        
        rewards -= loss_sales  # 总奖励扣除损失销售成本
        
        # 增加步数
        self.current_step += 1
        
        # 判断是否结束（比如达到最大步数）
        if self.current_step >= self.max_steps:
            self.done = True
        
        return self._get_observation(), rewards, self.done


def plot(dp, act):
    plt.rcParams.update({
        "font.family": "serif",
        # "font.size": 13,
        "font.weight": "bold",       # 默认粗体
        "axes.labelweight": "bold",  # 坐标轴标签加粗
        "axes.titlesize": 14,        # 坐标轴标题大小
        # "xtick.labelsize": 12,       # 坐标轴刻度字号
        "ytick.labelsize": 12,
        "legend.fontsize": 12        # 图例字号
    })
    s = np.arange(1, 41)
    # V = dp[1:] - np.min(dp[1:])         # Normalize V
    V = dp[1:]
    pi_star = act[1:41]                 # Optimal policy

    fig, ax1 = plt.subplots(figsize=(6.5, 4))

    # Left y-axis: normalized V
    color = 'tab:blue'
    ax1.set_xlabel('Inventory level $s$')
    # ax1.set_ylabel('Normalized value $V_s - \min V$', color=color)
    ax1.set_ylabel('Value function $V_s$', color=color)
    ax1.plot(s, V, marker='o', ms=4, color=color, label='Value function $V_s$')
    ax1.tick_params(axis='y', labelcolor=color)

    # Right y-axis: policy
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Optimal order $\pi^*(s)$', color=color)
    ax2.plot(s, pi_star, marker='s', ms=4, color=color, label='Policy $\pi^*(s)$')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and layout
    # fig.suptitle('Converged Value Function and Optimal Policy')
    fig.tight_layout()
    plt.savefig('dp_pi.png')
    plt.grid(True)
    plt.show()


def value_iteration(h, c, p, gamma=0.99):
    """
    h: 库存持有成本
    c: 损失销售成本
    p: 差价
    return: 值函数、最优动作
    """
    np.random.seed(int(time.time()))
    np.set_printoptions(precision=3)

    dp = np.random.uniform(10, 100, size=41)
    act = [0]*41

    def pr():
        res = dp[1:]
        mi = res.min()
        print(res)
        # print(res-mi)
        print(act[1:41])

    for iter in range(3000):
        dp_copy = np.copy(dp)
        # dp_copy = dp
        # for i in reversed(range(1, 41)):
        for i in range(1, 41):
            ma = -1e9; id = 0
            up = max(min(40-i, 20), 1)
            for o in range(1, up+1): # i+o>40一定不最优
                sum = 0
                for d in range(1, 21):
                    if d<=i:
                        sum += gamma*dp_copy[i-d+o] + p*o - h*(i-d+o)
                    else:
                        sum += gamma*dp_copy[o] + p*o - h*o - c*(d-i)
                avg = sum/20
                if avg>ma:
                    ma = avg; id = o
            dp[i] = ma; act[i] = id
    
    pr()
    
    # plot(dp, act)
    return dp, act


def test(env: Env, target_firm_id, best_acts, num_episodes=10):
    """
    :param env: 环境
    :param num_episodes: 测试的episodes数量
    :return: 所有episode的奖励和详细信息
    """
    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []
    
    for i_episode in tqdm(range(1, num_episodes+1)):
        state = env.reset()
        score = 0
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []
        
        for t in range(env.max_steps):
            # 对特定企业采取动作，其他企业随机决策
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == target_firm_id:
                    # 使用智能体策略，不使用探索
                    firm_state = state[firm_id].reshape(1, -1)
                    # best_acts = [-1, 16, 16, 16, 16, 16, 15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 12, 11, 10, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    # best_acts = [-1, 16, 16, 16, 16, 15, 15, 14, 14, 14, 13, 13, 13, 13, 14, 12, 12, 11, 10, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # 正序
                    # best_acts = [-1, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 14, 12, 13, 13, 13, 12, 11, 10, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # 逆序
                    inventory = firm_state[0, 2]
                    assert inventory > 0
                    if inventory >= len(best_acts):
                        action = 1
                    else:
                        action = best_acts[int(inventory)]
                    actions[firm_id] = action
                else:
                    # 对其他企业采取随机策略
                    actions[firm_id] = np.random.randint(1, 21)
            
            # 执行动作
            next_state, rewards, done = env.step(actions)
            
            # 记录关键指标
            episode_inventory.append(env.inventory[target_firm_id][0])
            episode_orders.append(actions[target_firm_id][0])
            episode_demand.append(env.demand[target_firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[target_firm_id][0])
            
            # 该企业的奖励
            reward = rewards[target_firm_id][0]
            score += reward
            
            # 更新状态
            state = next_state
            
            if done:
                break
        
        # 记录分数和历史数据
        scores.append(score)
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)
        
        # print(f'Test Episode {i_episode}/{num_episodes} | Score: {score:.2f}')
    
    print(f"Average score: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}")
    return scores, inventory_history, orders_history, demand_history, satisfied_demand_history


def plot_test_results(scores, inventory_history, orders_history, demand_history, satisfied_demand_history):
    """
    绘制测试结果
    
    :param scores: 每个episode的奖励
    :param inventory_history: 每个episode的库存历史
    :param orders_history: 每个episode的订单历史
    :param demand_history: 每个episode的需求历史
    :param satisfied_demand_history: 每个episode的满足需求历史
    """
    # 计算平均值，用于绘图
    avg_inventory = np.mean(inventory_history, axis=0)
    avg_orders = np.mean(orders_history, axis=0)
    avg_demand = np.mean(demand_history, axis=0)
    avg_satisfied_demand = np.mean(satisfied_demand_history, axis=0)
    
    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 库存图表
    axs[0, 0].plot(avg_inventory)
    axs[0, 0].set_title('平均库存')
    axs[0, 0].set_xlabel('时间步')
    axs[0, 0].set_ylabel('库存量')
    
    # 订单图表
    axs[0, 1].plot(avg_orders)
    axs[0, 1].set_title('平均订单量')
    axs[0, 1].set_xlabel('时间步')
    axs[0, 1].set_ylabel('订单量')
    
    # 需求和满足需求图表
    axs[1, 0].plot(avg_demand, label='需求')
    axs[1, 0].plot(avg_satisfied_demand, label='满足的需求')
    axs[1, 0].set_title('平均需求 vs 满足的需求')
    axs[1, 0].set_xlabel('时间步')
    axs[1, 0].set_ylabel('数量')
    # axs[1, 0].set_ylim(1, 20)
    axs[1, 0].legend()
    
    print(f"satisfied demand rate: {np.mean(avg_satisfied_demand)}/{np.mean(avg_demand)} = {np.mean(avg_satisfied_demand)/np.mean(avg_demand)}")

    # axs[1, 1].bar(range(len(scores)), scores) # 柱状图
    hist, bins = np.histogram(scores, bins=50) # 直方图
    axs[1, 1].bar(bins[:-1], hist, width=np.diff(bins), align='edge')
    axs[1, 1].set_title('测试episode奖励')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('总奖励')
    
    plt.tight_layout()
    plt.savefig('figures/test.png') # here
    plt.close()


if __name__ == "__main__":
    # 创建保存模型和图表的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # 初始化环境参数
    num_firms = 3  # 假设有3个企业
    p = [10, 9, 8]  # 价格列表
    h = 0.5  # 库存持有成本
    c = 2  # 损失销售成本
    initial_inventory = 100  # 初始库存
    poisson_lambda = 10  # 泊松分布的均值
    max_steps = 100  # 每个episode的最大步数
    
    # 创建仿真环境
    env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)
    
    # 为第二个企业创建DQN智能体
    firm_id = 1  # 选择第二个企业进行训练
    state_size = 3  # 每个企业的状态维度：订单、满足的需求和库存
    action_size = 20  # 假设最大订单量为20
        
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 绘图显示负号
    
    values, best_acts = value_iteration(h, c, p[firm_id]-p[firm_id+1])

    # 测试训练好的智能体
    test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history = test(env, firm_id, best_acts, num_episodes=10000,)
    
    # 绘制测试结果
    plot_test_results(test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history)
