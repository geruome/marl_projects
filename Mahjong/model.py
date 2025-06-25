import torch
from torch import nn
import torch.nn.functional as F


class InitCNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten()
        )
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        hidden = self._tower(obs)
        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value = self._value_branch(hidden)
        return masked_logits, value


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # --- 序数牌处理模块 (共享权重) ---
        # 输入形状: (Batch_size, 2, 9) -- 2通道 (手牌/pack), 9列 (牌值1-9)
        # 我们将对万、条、饼分别传入这个模块
        self.num_suit_conv_block = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1, bias=False), # 2D输入会被处理为1D
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Flatten() # 展平输出，例如 32 * 9 = 288
        )
        
        # --- 字牌处理模块 (独立于序数牌，通常用MLP或不同Conv) ---
        # 输入形状: (Batch_size, 2, 9) -- 2通道 (手牌/pack), 9列 (字牌)
        # 考虑到字牌没有连续性，可以直接使用Conv1d，但也可以考虑MLP
        # 这里仍用Conv1d，但其捕获的是各字牌的独立状态组合
        self.honor_suit_conv_block = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=1, padding=0, bias=False), # kernel_size=1 相当于对每列独立处理
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Flatten() # 例如 32 * 9 = 288
        )

        # --- 特征融合与决策层 ---
        # 序数牌输出: 3 * (32 * 9) = 3 * 288 = 864
        # 字牌输出: 32 * 9 = 288
        # 总融合特征维度 = 864 + 288 = 1152
        
        self.fc_common = nn.Sequential(
            nn.Linear(3 * (32 * 9) + (32 * 9), 512), # 3个序数牌花色 + 1个字牌花色
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True)
        )

        # --- Actor 策略头 (输出 Logits) ---
        self._logits = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 235) # 动作空间大小
        )
        
        # --- Critic 价值头 (输出 Value) ---
        self._value_branch = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1) # 单个价值输出
        )
        
        # --- 权重初始化 ---
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        obs = input_dict["observation"].float() # obs 形状为 (Batch_size, 8, 9)

        # 分割 obs 为不同花色的输入
        # 序数牌输入 (万, 条, 饼)
        num_suit_inputs = [
            obs[:, 0:2, :], # 万 (行 0, 1)
            obs[:, 2:4, :], # 条 (行 2, 3)
            obs[:, 4:6, :]  # 饼 (行 4, 5)
        ]
        
        # 字牌输入 (风+箭)
        honor_suit_input = obs[:, 6:8, :] # 风+箭 (行 6, 7)

        # 处理序数牌 (共享权重)
        num_suit_features = []
        for i in range(3): # 遍历万、条、饼
            # Conv1d 期待 (Batch, Channels, Length)
            feature = self.num_suit_conv_block(num_suit_inputs[i])
            num_suit_features.append(feature)
        
        # 处理字牌
        honor_suit_feature = self.honor_suit_conv_block(honor_suit_input)

        # 拼接所有特征
        combined_features = torch.cat(num_suit_features + [honor_suit_feature], dim=1)
        
        # 融合特征并通过全连接层
        hidden = self.fc_common(combined_features)

        # 策略头和价值头
        logits = self._logits(hidden)
        value = self._value_branch(hidden)

        # 应用 action_mask
        mask = input_dict["action_mask"].float()
        # clamp log(0) to a very small number instead of -inf
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38) 
        masked_logits = logits + inf_mask

        return masked_logits, value
    

# class MyModel(nn.Module):
    