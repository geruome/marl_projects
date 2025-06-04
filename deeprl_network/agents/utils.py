import numpy as np
import torch
import torch.nn as nn
from pdb import set_trace as stx
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

"""
initializers
"""
def init_layer(layer, layer_type):
    if layer_type == 'fc':
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
    elif layer_type == 'lstm':
        nn.init.orthogonal_(layer.weight_ih.data)
        nn.init.orthogonal_(layer.weight_hh.data)
        nn.init.constant_(layer.bias_ih.data, 0)
        nn.init.constant_(layer.bias_hh.data, 0)

"""
layer helpers
"""
def batch_to_seq(x):
    n_step = x.shape[0]
    return torch.chunk(x, n_step)
    # if len(x.shape) == 1:
    #     x = torch.unsqueeze(x, -1)


def run_rnn(layer, xs, dones, s):
    # shape: 64, 64, 64*2
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    # n_in = int(xs[0].shape[1])
    # n_out = int(s.shape[0]) // 2
    s = torch.unsqueeze(s, 0)
    h, c = torch.chunk(s, 2, dim=1)
    outputs = []
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        h, c = layer(x, (h, c))
        outputs.append(h)
    s = torch.cat([h, c], dim=1)
    return torch.cat(outputs), torch.squeeze(s)


def one_hot(x, oh_dim, dim=-1):
    oh_shape = list(x.shape)
    if dim == -1:
        oh_shape.append(oh_dim)
    else:
        oh_shape = oh_shape[:dim+1] + [oh_dim] + oh_shape[dim+1:]
    x_oh = torch.zeros(oh_shape)
    x = torch.unsqueeze(x, -1)
    if dim == -1:
        x_oh = x_oh.scatter(dim, x, 1)
    else:
        x_oh = x_oh.scatter(dim+1, x, 1)
    return x_oh


"""
buffers
"""
class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer): # here
    def __init__(self, gamma, alpha, distance_mask):
        self.gamma = gamma
        self.alpha = alpha
        if alpha > 0:
            self.distance_mask = distance_mask
            self.max_distance = np.max(distance_mask, axis=-1)
        self.reset()

    def reset(self, done=False):
        # the done before each step is required
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.adds = []
        self.dones = [done]

    def add_transition(self, ob, na, a, r, v, done):
        self.obs.append(ob)
        self.adds.append(na)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def sample_transition(self, R, dt=0):
        # R = self._get_value(ob, done, action). 不是reward ??
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        nas = np.array(self.adds, dtype=np.int32)
        acts = np.array(self.acts, dtype=np.int32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        # use pre-step dones here
        dones = np.array(self.dones[:-1], dtype=bool)
        self.reset(self.dones[-1])
        return obs, nas, acts, dones, Rs, Advs

    def _add_R_Adv(self, R):
        assert 0
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def _add_st_R_Adv(self, R, dt):
        assert 0
        Rs = []
        Advs = []
        # use post-step dones here
        tdiff = dt
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = self.gamma * R * (1.-done)
            if done:
                tdiff = 0
            # additional spatial rewards
            tmax = min(tdiff, self.max_distance)
            for t in range(tmax + 1):
                rt = np.sum(r[self.distance_mask == t])
                R += (self.gamma * self.alpha) ** t * rt
            Adv = R - v
            tdiff += 1
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    # def _add_s_R_Adv(self, R):
    #     Rs = []
    #     Advs = []
    #     # use post-step dones here
    #     for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
    #         R = self.gamma * R * (1.-done)
    #         # additional spatial rewards
    #         for t in range(self.max_distance + 1):
    #             rt = np.sum(r[self.distance_mask == t])
    #             R += (self.alpha ** t) * rt
    #         Adv = R - v
    #         Rs.append(R)
    #         Advs.append(Adv)
    #     Rs.reverse()
    #     Advs.reverse()
    #     self.Rs = Rs
    #     self.Advs = Advs

    def _add_s_R_Adv(self, last_v_estimate):
        # Rs 用于价值网络更新的目标（TD0目标），Advs 用于策略网络更新
        Rs = []
        Advs = []

        # GAE 的参数
        # self.gamma 已经是折扣因子 gamma
        # self.lambda_gae 是 GAE 中的 lambda 参数，通常在 0 到 1 之间
        if not hasattr(self, 'lambda_gae'):
            self.lambda_gae = 0.95
            # raise AttributeError("self.lambda_gae is not defined. Please initialize it (e.g., self.lambda_gae = 0.95).")
        
        gae_advantage = 0.0

        v_next = last_v_estimate
        for t in reversed(range(len(self.rs))):
            rts = self.rs[t]
            rt = 0
            for t in range(self.max_distance + 1):
                rt += (self.alpha ** t) * np.sum(rts[self.distance_mask == t])
            v_t = self.vs[t]
            done_t = self.dones[t] # 注意：这里 done_t 表示的是 S_t 到 S_{t+1} 是否终止

            delta = rt + self.gamma * v_next * (1. - done_t) - v_t
            # GAE(t) = delta_t + gamma * lambda * GAE(t+1) * (1 - done_t)
            gae_advantage = delta + self.gamma * self.lambda_gae * (1. - done_t) * gae_advantage
            
            td0_target = rt + self.gamma * v_next * (1. - done_t)

            Advs.append(gae_advantage)
            Rs.append(td0_target) # 这里 Rs 存储的是 TD(0) 目标

            # 更新 v_next 为当前时间步的价值，用于下一次循环 (t-1) 的计算
            v_next = v_t

        # 反转列表以恢复正序
        Rs.reverse()
        Advs.reverse()

        self.Rs = np.array(Rs) # 转换为 numpy 数组
        self.Advs = np.array(Advs) # 转换为 numpy 数组
        # stx()


class MultiAgentOnPolicyBuffer(OnPolicyBuffer): # here
    def __init__(self, gamma, alpha, distance_mask):
        # assert 0
        super().__init__(gamma, alpha, distance_mask)

    def sample_transition(self, R, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)
        obs = np.transpose(np.array(self.obs, dtype=np.float32), (1, 0, 2))
        policies = np.transpose(np.array(self.adds, dtype=np.float32), (1, 0, 2))
        acts = np.transpose(np.array(self.acts, dtype=np.int32))
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        dones = np.array(self.dones[:-1], dtype=bool)
        self.reset(self.dones[-1])
        return obs, policies, acts, dones, Rs, Advs

    def _add_R_Adv(self, R): # original
        Rs = []
        Advs = []
        vs = np.array(self.vs) # 120,25
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = r + self.gamma * cur_R # * (1.-done)
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs) # 25,120
        self.Advs = np.array(Advs) # 25,120

    # def _add_R_Adv(self, last_vs): # TD0
    #     Rs = []
    #     Advs = []
    #     vs = np.array(self.vs) # 120,25
    #     for i in range(vs.shape[1]):
    #         cur_Rs = []
    #         cur_Advs = []
    #         cur_R = last_vs[i]
    #         v_next = last_vs[i]
    #         for r, v, in zip(self.rs[::-1], vs[::-1,i]):
    #             cur_R = r + self.gamma * cur_R
    #             TD_target = r + self.gamma * v_next
    #             v_next = v                
    #             cur_Adv = cur_R - v
    #             # cur_Rs.append(cur_R)
    #             cur_Rs.append(TD_target)
    #             cur_Advs.append(cur_Adv)
    #         cur_Rs.reverse()
    #         cur_Advs.reverse()
    #         Rs.append(cur_Rs)
    #         Advs.append(cur_Advs)
    #     self.Rs = np.array(Rs) # 25,120
    #     self.Advs = np.array(Advs) # 25,120
    #     # print(((vs.T-self.Rs)**2).mean())

    # def _add_R_Adv(self, last_vs): # GAE + TD0
    #     if not hasattr(self, 'gamma'):
    #         raise AttributeError("self.gamma (discount factor) is not defined. Please initialize it.")
    #     if not hasattr(self, 'lambda_gae'):
    #         self.lambda_gae = 0.95

    #     num_timesteps = len(self.rs)
    #     num_agents = len(self.vs[0]) 

    #     rs_np = np.array(self.rs) 
    #     vs_np = np.array(self.vs)         

    #     all_Rs = np.zeros((num_agents, num_timesteps))
    #     all_Advs = np.zeros((num_agents, num_timesteps))

    #     for i in range(num_agents):
    #         gae_advantage_accum = 0.0
    #         v_next_agent = last_vs[i]
    #         for t in reversed(range(num_timesteps)):
    #             r_current_step = rs_np[t]
    #             v_current_step = vs_np[t, i] 
    #             delta = r_current_step + self.gamma * v_next_agent - v_current_step
    #             gae_advantage_accum = delta + self.gamma * self.lambda_gae * gae_advantage_accum
    #             td0_target = r_current_step + self.gamma * v_next_agent
    #             all_Advs[i, t] = gae_advantage_accum
    #             all_Rs[i, t] = td0_target
    #             v_next_agent = v_current_step

    #     self.Rs = all_Rs  # 形状为 (num_agents, num_timesteps)
    #     self.Advs = all_Advs # 形状为 (num_agents, num_timesteps)


"""
util functions
"""
class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val


def get_optimizer_cosine_warmup_scheduler(model, total_steps, warmup_rate=0.01, 
                                     learning_rate=1e-3, min_lr=1e-5, weight_decay=0.01):

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # betas=(0.9, 0.999), eps=1e-8,
    warmup_steps = int(warmup_rate * total_steps)
    
    # 2. 定义预热函数
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # 余弦衰减阶段：从1到min_lr/learning_rate的比例
            # 计算余弦衰减的“有效”步数和总步数
            cosine_total_steps = total_steps - warmup_steps
            cosine_current_step = current_step - warmup_steps
            
            # 避免除以零
            if cosine_total_steps == 0:
                return min_lr / learning_rate
            
            # 标准余弦衰减公式
            # lr_ratio = min_lr + 0.5 * (learning_rate - min_lr) * (1 + cos(pi * current_step / total_steps))
            # 我们这里返回的是学习率的比例，所以需要调整
            
            # cosine_annealing_ratio = 0.5 * (1 + math.cos(math.pi * cosine_current_step / cosine_total_steps))
            # 这里的余弦衰减是从 learning_rate 衰减到 min_lr
            # 衰减的范围是 (learning_rate - min_lr)
            # 所以比例是 (min_lr + (learning_rate - min_lr) * 0.5 * (1 + cos(...))) / learning_rate
            
            # 更简洁的表达方式是，将其视为从 learning_rate 衰减到 min_lr 的比例
            decay_ratio = (cosine_current_step / cosine_total_steps)
            
            # 使用标准的余弦衰减公式，但要注意 min_lr 是相对的
            # 余弦衰减是从 max_lr 到 min_lr，但我们的 LambdaLR 返回的是一个乘数 (0到1)
            # 所以需要计算当前学习率与最大学习率的比例
            
            # 学习率从 learning_rate 衰减到 min_lr 的余弦曲线
            # 最终学习率 = min_lr + (learning_rate - min_lr) * 0.5 * (1 + cos(pi * decay_ratio))
            # LambdaLR 返回的是：最终学习率 / learning_rate
            
            factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * decay_ratio)))
            return float(min_lr / learning_rate + (1 - min_lr / learning_rate) * factor)


    # 3. 定义学习率调度器
    # LambdaLR 允许你传入一个函数，该函数会计算当前学习率的乘数
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    return optimizer, scheduler