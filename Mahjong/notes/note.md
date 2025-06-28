##
只有 >=8 番才能胡。

要勾选长时运行 + 简单交互 !!

episodes_per_actor = 1000 完之后就结束了。一堆报错不用管。

235个动作。action_mask = pass1 + hu1 + discard34 + chi63(3*7*3) + peng34 + gang34 + angang34 + bugang34

MahjongFanCalculator 不胡牌就报错。

<!-- state添加牌池等信息。 但只考虑hand就差不多够了-->
告诉agent每个操作后的state。
model只要value model. action e-greedy.

reward：番数(不胡牌没番，怎么办) + 胡牌快慢
番数作为中间参考。

单独训一个人。
四人纯随机胡牌率：0.03

<!-- seed无效 -->
<!-- 加速cuda. -->

多进程tester更慢了..

Adam VS RMSprop. Adam是对的，而且稳步上升。

avg_reward：(reward=1)
expe/0626194320_great/models/model_34000.pt: 0.73
expe/0626200248/models/model_42000.pt: 0.75
expe/0626202920/models/model_24000.pt: 0.772
expe/0626205954_great/models/model_39000.pt: 0.72 
expe/0626205954_great/models/model_39000.pt: 0.73
expe/0627113246/models/model_045000.pt: 0.764
expe/0627113246/models/model_092000.pt: 0.768
expe/0627124638/models/model_900000.pt: 0.808
expe/0627124638/models/model_950000.pt: 0.796
expe/0627170944/models/model_323000.pt: 0.841
expe/0627170944/models/model_665000.pt: 0.848

fan >= 8:
expe/0627211756/models/model_486000.pt: 0.358
expe/0627211756/models/model_538000.pt: 0.377
expe/0628003319/models/model_1842000.pt: 0.393
expe/0628003319/models/model_3106000.pt: 0.401

<!-- 负奖励 ?? 自举 ?? -->
<!-- self.reward[player] = -30 为什么会触发这个 .. -->
<!-- (>=8 fan), 极限是 0.31 (eps=0.05) -->

从0开始训效果更好 ??

<!-- 调batch_size -->
<!-- 接近胡的时候操作的很对，之前的操作不太对 ??????
不会杠，乱吃，乱打 -->

试试policy_model
linear reward. 


##
pip install PyMahjongGB

nohup python train.py > output.log 2>&1 &

nohup python tester.py > output.log 2>&1 &

rm my_bot.zip  &&  zip my_bot.zip __main__.py model.py agent.py feature.py

weight.pt 之类的，要放在个人空间data, 不能携zip上传

