##

只有 >=8 番才能胡。
特征提取。单人尽快胡 + 不让对手胡（不点炮）。
凑番数 + 尽快胡。


自己创建房间比赛。
小组比赛。
本地不同agent ???


episodes_per_actor = 1000 完之后就结束了。一堆报错不用管。

reward全0是不是没一点用 ??

235个动作。action_mask = pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34

state: 144张牌



##
nohup python train.py > output.log 2>&1 &