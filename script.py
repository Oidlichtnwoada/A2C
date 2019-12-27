import gym
import matplotlib.pyplot as plt
import numpy as np

from A2C import A2CAgent

a2c_agent = A2CAgent(gym.make('CartPole-v0'))
print(f'episodic reward before training: {np.sum(a2c_agent.test(True)[3])}')
plt.plot(a2c_agent.train())
plt.xlabel('episode index')
plt.ylabel('episodic reward')
plt.show()
print(f'episodic reward after training: {np.sum(a2c_agent.test(True)[3])}')
