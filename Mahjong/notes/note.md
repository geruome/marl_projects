##
只有 >=8 番才能胡。
特征提取。单人尽快胡 + 不让对手胡（不点炮）。
凑番数 + 尽快胡。

自己创建房间比赛。
小组比赛。
本地不同agent ???

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

状态设计 ???!!!
四花色独自得出状态，再混合。
3花色 / 1-9 打乱。

中间reward如何添加 ?? 中间给小,最后给最大的。

<!-- seed无效 -->
<!-- 加速cuda. -->

多进程tester更慢了..

Adam VS RMSprop. Adam是对的，而且稳步上升。

avg_reward：(reward=1)
expe/0626194320_great/models/model_34000.pt: 0.73
expe/0626200248/models/model_42000.pt: 0.75
expe/0626202920/models/model_24000.pt: 0.76

expe/0626205954_great/models/model_39000.pt (>=8): 0.26

添加 loss / hu_rate 项。

改成看番数。

负奖励 ?? 自举 ??

0626223307 往后胡的越来越大 ??

self.reward[player] = -30 为什么会触发这个 ..

勾选长时运行 + 简单交互。

接近胡的时候操作的很对，之前的操作不太对 ??????


##
pip install PyMahjongGB

nohup python train.py > output.log 2>&1 &

nohup python tester.py > output.log 2>&1 &

rm my_bot.zip  &&  zip my_bot.zip __main__.py model.py agent.py feature.py

weight.pt 之类的，要放在个人空间data, 不能携zip上传

