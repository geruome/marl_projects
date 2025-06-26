import matplotlib.pyplot as plt

file_path = 'output.log' # 'expe/0626194320_great/output.log' 

iterations = []
rewards = []

with open(file_path, 'r') as f:
    for line in f:
        if "Iteration" in line and "avg_reward" in line:
            parts = line.split(',')
            iteration_str = parts[0].split(' ')[1]
            reward_str = parts[1].split(' ')[-1]
            iterations.append(int(iteration_str))
            rewards.append(float(reward_str))

plt.figure(figsize=(10, 6))
plt.plot(iterations, rewards)
plt.xlabel("Iteration")
plt.ylabel("Average Reward")
plt.title("Average Reward Over Iterations")
plt.grid(True)
plt.savefig("reward.png")
plt.show()