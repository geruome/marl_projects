
项目结构：

```
.
├── DQN.py # Optimized DQN implementation 
├── DQN_new.py # DQN in modified environment
├── fullenv.py # DQN with full observation
├── fullenv_new.py # DQN in modified environment with full observation
├── value_iteration_method.py # Value iteration implementation
├── experiments
│ ├── vanilla_DQN # results with different eps
|       ├─ eps_0.xx
│ ├── DP # Value iteration evaluation results
│ └── new_env # Experiments under modified environment
│       ├─── full  # full observation
│       ├── single # local observation
```
每个实验子文件下包含 figures 和 权重文件.
report中没有展示完整效果，更多效果可在figures中查看。

直接python运行即可。
```
python DQN.py # Section 1
主要修改：参数调优 + early stop.

python value_iteration_method.py # Section 2
主要修改：value_iteration函数

python fullenv_new.py # Section 3
主要修改：Env.step, Env.get_action
```
默认加载预训练好的权重。
需要训练时，修改这段代码：
```py
    # scores = train_dqn(env, agent, num_episodes=10000, max_t=max_steps, eps_start=1.0, eps_end=0.05, eps_decay=0.995)
    # plot_training_results(scores)
    weight_path = xxx
    agent.load(weight_path)
```
