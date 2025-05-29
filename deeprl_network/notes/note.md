### PPT要求
ATSC Grid 和 ATSC Monaco环境代码参考：deeprl_network/envs/large_grid_env.py & real_net_env.py

1. Independent Advantage Actor Critic （IA2C）
https://ieeexplore.ieee.org/document/8667868
2. Individualized Controlled Continuous Communication Model（IC3Net）
https://github.com/IC3Net/IC3Net
https://arxiv.org/pdf/1812.09755.pdf

https://github.com/PKU-MARL/HARL

具体包括但不限于以下几部分：
一、算法设计，已给出两个基线多智能体算法IA2C和IC3Net，在两个环境中，分别以最小化Intersection delay和最小化Queue length为单个优化目标，并且：
- 1.学习、借鉴基线算法IC3Net，针对给定ATSC实验环境，**给出改进方案或设计新算法**（二选一）并分析改进原因或设计方案，不限于理论，网络结构，实现技巧等方面的改进和设计。对比训练曲线，保存不同算法的模型测试比较最终策略效果。结果图包括但不限于：学习曲线， 最终策略的Intersection delay和Queue length对比
- 2. 学习基线算法IA2C，参考由单智能体A2C算法拓展而来的多智能体IA2C算法，并学习针对ATSC中多智能体系统设计的MA2C算法，**扩展单智能体SAC或单智能体PPO**(二选一)算法为多智能体算法，与IA2C, MA2C对比训练曲线和最终策略效果。结果图要求同1。
二、多目标优化，ATSC的两个环境均有两个优化目标，分别为最小化Intersection delay和最小化Queue length，综合考虑同时优化两个目标，设计综合优化目标的奖励函数，并分析最终策略性能，绘制结果图，可通过可视化分析模拟面板数据优化设计方案。


### 环境配置
https://sumo.dlr.de/docs/Installing/Linux_Build.html 

<!-- sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:sumo/stable -->
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

export SUMO_HOME=/usr/share/sumo
export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH

cd envs/large_grid_data/
python build_file.py

可视化：
cd envs/large_grid_data
sumo-gui -c exp_0.sumocfg

### 
25个agent。
每个agent动作空间为5：phases = ['GGgrrrGGgrrr', 'rrrGrGrrrGrG', 'rrrGGrrrrGGr','rrrGGGrrrrrr', 'rrrrrrrrrGGG']. 每个12字符控制了该路口的3*4个灯。
状态主要是：node.wave_state. 车流量。
状态空间: 角,边,中 = 36,48,60

env.n_s_ls, env.n_a_ls: 状态空间长度,动作空间长度 
n_n: num_neighbor
env.distance_mask.shape: (25, 25). 各个agent距离.

LstmPolicy 包含了 actor_head 和 critic_head. 
LSTM,batch前后做了state传递. 

train时是直接按概率随机选动作: np.random.choice(np.arange(len(pi)), p=pi)
test时贪心。

action = self.model.forward(ob, done)
value = self.model.forward(ob, done, self.naction, 'v'). 
self._run_critic_head(h, np.array([naction])). critic_head的做法是把 h 和 one_hot(naction) concat. 这是合理的，因为策略梯度定理的baseline中，只要不包含当前agent action即可。


control_interval_sec：5, 黄灯：2 ??

IA2C优化目标：objective = queue. 怎么算的 ???  
Intersection delay 又怎么算 ???
好慢啊 ????

MA2C是整体一个大的actor、critic. 


### changes
actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1)). 错了 !!,不用log_softmax,内部会自己做。

spatial rewards: 对于当前agent, R += (alpha^dis) * reward. 不止考虑当前点的reward.

他的 adv = R-V. 用R估计Q ??? Reinforce ???

vs[:,1]
array([-6.8494906, -6.8052955, -6.863706 , -7.0475116, -7.0539894,
       -7.0510798, -7.0283413, -7.170772 , -7.015326 , -7.109921 ,
       -7.185642 , -7.2333493, -7.274774 , -7.419404 , -7.632034 ,
       -7.838477 , -7.7869964, -7.7530227, -7.766166 , -7.757078 ,
       -7.734191 , -7.757602 , -7.803926 , -7.894373 , -7.875008 ,
       -7.884476 , -7.8902626, -7.905335 , -7.9093404, -7.942168 ,
       -7.968404 , -8.015292 , -8.036086 , -8.07841  , -8.118147 ,
       -8.15472  , -8.161368 , -8.20045  , -8.277474 , -8.351079 ,
       -8.47661  , -8.61452  , -8.768036 , -8.903623 , -8.970426 ,
       -9.011455 , -9.0268545, -9.045116 , -9.054437 , -9.078088 ,
       -9.084437 , -9.102674 , -9.113695 , -9.129328 , -9.138702 ,
       -9.163825 , -9.164876 , -9.189061 , -9.179352 , -9.171317 ,
       -9.177709 , -9.23139  , -9.212136 , -9.235853 , -9.213273 ,
       -9.2761345, -9.298218 , -9.29707  , -9.312328 , -9.331329 ,
       -9.326385 , -9.33467  , -9.359961 , -9.351081 , -9.348458 ,
       -9.357836 , -9.375567 , -9.383488 , -9.39315  , -9.402997 ,
       -9.387936 , -9.413258 , -9.40646  , -9.430082 , -9.437663 ,
       -9.446443 , -9.464542 , -9.47196  , -9.475523 , -9.459885 ,
       -9.466429 , -9.512142 , -9.520471 , -9.496861 , -9.492448 ,
       -9.497475 , -9.513464 , -9.509709 , -9.530091 , -9.536367 ,
       -9.515377 , -9.532276 , -9.537311 , -9.529083 , -9.545961 ,
       -9.560585 , -9.55647  , -9.620927 , -9.60945  , -9.645255 ,
       -9.586708 , -9.596126 , -9.6511   , -9.685275 , -9.747642 ,
       -9.709349 , -9.6964245, -9.617096 , -9.651008 , -9.701138 ],
      dtype=float32)

np.array(self.rs) 单步reward
array([-0.001 , -0.0045, -0.0045, -0.0045, -0.012 , -0.013 , -0.0135,
       -0.013 , -0.019 , -0.0265, -0.0265, -0.0315, -0.036 , -0.038 ,
       -0.035 , -0.038 , -0.038 , -0.043 , -0.036 , -0.0335, -0.037 ,
       -0.0445, -0.041 , -0.047 , -0.053 , -0.0575, -0.06  , -0.0605,
       -0.058 , -0.0595, -0.062 , -0.067 , -0.069 , -0.064 , -0.058 ,
       -0.0545, -0.058 , -0.052 , -0.0515, -0.0555, -0.0575, -0.0595,
       -0.053 , -0.0585, -0.071 , -0.07  , -0.0755, -0.0775, -0.073 ,
       -0.066 , -0.0735, -0.082 , -0.068 , -0.061 , -0.0705, -0.07  ,
       -0.07  , -0.064 , -0.064 , -0.0585, -0.0525, -0.0585, -0.0705,
       -0.0515, -0.048 , -0.0525, -0.0535, -0.0615, -0.0705, -0.06  ,
       -0.057 , -0.048 , -0.052 , -0.0665, -0.064 , -0.0515, -0.0525,
       -0.057 , -0.049 , -0.055 , -0.062 , -0.0525, -0.0565, -0.0505,
       -0.057 , -0.0655, -0.0645, -0.067 , -0.064 , -0.0515, -0.063 ,
       -0.0755, -0.073 , -0.0685, -0.07  , -0.0765, -0.078 , -0.081 ,
       -0.084 , -0.0785, -0.071 , -0.0715, -0.067 , -0.062 , -0.072 ,
       -0.0675, -0.06  , -0.072 , -0.075 , -0.074 , -0.078 , -0.08  ,
       -0.083 , -0.073 , -0.082 , -0.0795, -0.074 , -0.0725, -0.076 ,
       -0.0665])

np.array(R) (last_V)
array([-10.995105,  -9.676867,  -9.885537,  -9.980827,  -9.852894,
        -9.844563,  -9.602427,  -9.443381,  -9.583128, -10.706152,
       -10.186242, -10.248742,  -9.58682 ,  -9.742153, -10.342079,
       -10.764072, -10.584856, -10.327369, -10.50308 , -10.135435,
       -11.42891 , -10.646751, -10.612808, -10.452703, -10.678249],
      dtype=float32)


### MA2C_NC
baseline:
![alt text](image.png)

去掉done,效果超好:
![alt text](image-1.png)

改成 gae+td0. 
朴素的用折扣回报同时作为V_target和Q却能训 ??

只改td0：

Adam不行，RMScrop行 ???
Adam 的自适应学习率问题 (尤其是在 RL 中):
历史梯度的累积问题： Adam 通过计算梯度的指数移动平均（一阶矩 m 和二阶矩 v）来自适应地调整学习率。在 RL 训练中，特别是对于策略梯度方法（如 REINFORCE, A2C, PPO），梯度通常具有高方差和非稳态的特性。
高方差意味着单个批次的梯度可能很不稳定。
非稳态意味着梯度分布可能随时间发生显著变化（例如，智能体探索新区域、学到新技能）。
可能导致学习率过早衰减或过大： Adam 可能会根据早期的大梯度（特别是训练初期的探索阶段）计算出较大的二阶矩 v，从而导致后续学习率（lr / sqrt(v)）过早地被抑制，变得过小。这使得智能体在后期难以探索或跳出局部最优。
RMSprop 的特性：
更简单的自适应： RMSprop 也使用梯度的平方的移动平均来调整学习率，但它不使用一阶矩的修正（不像 Adam）。它更专注于解决梯度在不同维度上的尺度差异。
对稀疏梯度的处理： RMSprop 在处理稀疏梯度时可能比 Adam 更稳定一些，因为它没有 Adam 那么复杂的动量累积机制。
学习率的激进性： RMSprop 可能会在某些情况下显得比 Adam 更“激进”，它的学习率可能波动更大，这在 RL 探索过程中有时是有益的。它不像 Adam 那样容易在早期就把学习率固定在某个小值。
对 RL 中基线方差的敏感性：
RL 的梯度通常有高方差，尤其是在早期探索阶段。价值函数（基线）的估计质量直接影响优势函数 Adv 的方差。
如果 Adam 对这种高方差梯度更敏感，可能导致其更新方向不稳定。而 RMSprop 可能会以某种方式（例如，对过去梯度的记忆更短）更好地处理这种不稳定性。


训练时：np.random.choice(np.arange(len(pi)), p=pi)。on-policy, 能否温度之类 ??

reward_norm = 2000
每个step: global_reward = np.mean(reward)(agent层面上). 递增
tensorboard记录的eposide reward只是对该值再平均。

value_loss基本到0了
loss/nc_policy_loss ??? 
期望的是adv大，但是logprob也大

不同机子上速度严重不同。


###
python3 main.py train --config-dir config/config_ia2c_grid.ini 
nohup python3 main.py train > output.log 2>&1 &
