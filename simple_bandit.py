import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, bandit, exploration_rate):
        self.bandit = bandit
        self.exploration_rate = exploration_rate
        self.cur_estimates = self.first_estimates()
        self.all_estimates = [[self.cur_estimates[i]] for i in range(len(self.cur_estimates))]
        self.rewards = []
        self.avg_rewards = []

    def first_estimates(self):
        estimates = []
        for i in range(len(self.bandit)):
            estimates.append(np.random.normal(self.bandit[0], 1))
        return estimates

    def select_action(self):
        if np.random.random() < self.exploration_rate:
            return np.random.choice(len(self.cur_estimates))
        else:
            return np.argmax(self.cur_estimates)

    def get_reward(self, action):
        reward = np.random.normal(self.bandit[action], 1)
        self.rewards.append(reward)
        self.avg_rewards.append(np.mean(self.rewards))
        self.all_estimates[action].append(reward)
        self.cur_estimates[action] = np.mean(self.all_estimates[action])

def run(nb_plays):
    nb_arms = 10
    exploration_rates = [0, 0.01, 0.1, 0.2, 0.3, 1]
    bandit = np.random.normal(0, 1, nb_arms)
    agents_list = []
    for i in range(len(exploration_rates)):
        ex = exploration_rates[i]
        agents_list.append(Agent(bandit, ex))

    for i in range(nb_plays):
        for agent in agents_list:
            action = agent.select_action()
            agent.get_reward(action)

    time = [i for i in range(nb_plays)]
    for agent in agents_list:
        plt.plot(time, agent.avg_rewards, label="{e}".format(e=agent.exploration_rate))
    plt.legend(loc='best')
    plt.show()

run(400)
