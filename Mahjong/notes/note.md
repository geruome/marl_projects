##

只有 >=8 番才能胡。
特征提取。单人尽快胡 + 不让对手胡（不点炮）。
凑番数 + 尽快胡。

自己创建房间比赛。
小组比赛。
本地不同agent ???

episodes_per_actor = 1000 完之后就结束了。一堆报错不用管。

235个动作。action_mask = pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34

state: 34种牌0-4即可, 却要 6 * 4 * 9. 

MahjongFanCalculator 不胡牌就报错。


简化state. (好像也不是很需要简化)
<!-- state添加牌池等信息。 但只考虑hand就差不多够了-->
训一个只靠自己接牌的model，作为baseline。

告诉agent每个操作后的state。
model只要value model. action e-greedy.

reward：番数(不胡牌没番，怎么办) + 胡牌快慢
番数作为中间参考。

修改obs、model。
先验。

座位 / 顺序如何定的 ??
replay_buffer完善。
加速cuda.

单独训一个人。
四人纯随机胡牌率：0.03

File /model-pool exist ????
目前学到了什么

##
nohup python train.py > output.log 2>&1 &
nohup python tester.py > output.log 2>&1 &