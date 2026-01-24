import numpy as np
import random
import time


# --- 1. 定义环境 (Environment) ---
class LineWorld:
    def __init__(self):
        self.state = 0  # 起点

    def step(self, action):
        # 行动逻辑
        if action == 1:
            self.state += 1  # Right
        elif action == 0:
            self.state -= 1  # Left
        self.state = np.clip(self.state, 0, 3)  # 限制边界

        # 奖励机制
        if self.state == 3:  # 到达终点
            return self.state, 10, True
        else:
            return self.state, -1, False  # 没到终点扣1分

    def reset(self):
        self.state = 0
        return self.state


# --- 参数设置 ---
alpha = 0.1  # 学习率 (步子迈多大)
epsilon = 0.1  # 探索率 (10%概率瞎走)
gamma = 0.9  # 折扣因子 (未来的钱打9折，我们暂未深入讲，设为0.9即可)
q_table = np.zeros((4, 2))  # 初始化 Q 表

# --- 2. 主循环 (The Loop) ---
env = LineWorld()

# 训练 100 个回合
for episode in range(100):
    state = env.reset()
    done = False

    while not done:
        # --- 步骤 A: 选择动作 (Epsilon-Greedy) ---
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1])  # 探索
        else:
            action = np.argmax(q_table[state])  # 利用

        # --- 步骤 B: 执行并观测 ---
        next_state, reward, done = env.step(action)

        # --- 步骤 C: 核心更新公式 (Q-Learning) ---
        # 这里的 max_next_q 就是 max Q(s', a')
        max_next_q = np.max(q_table[next_state])

        # 更新 Q(s, a)
        # 目标 = 奖励 + (折扣 * 未来最大价值)
        target = reward + gamma * max_next_q
        q_table[state, action] += alpha * (target - q_table[state, action])

        # 状态转移
        state = next_state

# --- 打印最终结果 ---
print("训练后的 Q 表:")
print(q_table)