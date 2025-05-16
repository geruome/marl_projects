import numpy as np
import time

np.random.seed(int(time.time()))
np.set_printoptions(precision=3)

h = 0.5  # 库存持有成本
c = 2  # 损失销售成本
# p1 = 9 # 售价
# p2 = 8 # 进价
p = 1 # 差价
gamma = 0.99

dp = np.random.uniform(10, 100, size=41)
act = [0]*41

def pr():
    res = dp[1:]
    mi = res.min()
    print(res)
    print(res-mi)
    # print(np.diff(dp[1:41]))
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
    if iter > 0 and iter % 300 == 0:
        pr()

pr()