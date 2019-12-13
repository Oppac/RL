import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, bandit, time_dep=False):
        self.nb_arms = bandit[0]
        self.nb_steps = bandit[1]
        self.reset()
        self.time_dep = time_dep

    def reset(self):
        self.time = 1
        self.all_rewards = np.zeros(self.nb_steps)
        self.hist_reward_arms = [[], [], [], []]
        self.q_estimates = np.zeros(self.nb_arms)
        self.hist_estimates = []

class AgentEpsilon(Agent):
    def __init__(self, bandit, epsilon=0.2, time_dep=False):
        Agent.__init__(self, bandit, time_dep)
        self.epsilon = epsilon

    def select_action(self):
        if self.time_dep:
            self.epsilon = 1 / np.sqrt(self.time)
        self.time += 1

        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_estimates)

class AgentSoftmax(Agent):
    def __init__(self, bandit, temperature=1, time_dep=False):
        Agent.__init__(self, bandit, time_dep)
        self.temperature = temperature

    def select_action(self):
        if self.time_dep:
            temperature = 4 * np.divide(1000 - self.time, 1000)
        self.time += 1

        e = np.exp(np.divide(self.q_estimates, self.temperature))
        softmax = np.divide(e, np.sum(e))
        return np.random.choice(len(self.q_estimates), 1, p=softmax)[0]


def plot_reward(names, agent_list, time):
    for n, agent in enumerate(agent_list):
        cum_score = np.cumsum(agent.all_rewards)
        avg_score = [cum_score[i] / i for i in range(1, len(agent.all_rewards))]
        plt.plot(list(range(time-1)), avg_score, label=names[n])
    plt.legend(loc="upper right")
    plt.title("Cumulative average reward over time")
    plt.xlabel("Time")
    plt.ylabel("Average reward")
    plt.show()

def plot_estimates(names, agent_list, arm, qstar, time):
    plt.plot(list(range(time)), [qstar for _ in range(time)], label='Q*')
    for n, agent in enumerate(agent_list):
        estimates = [i[arm] for i in agent.hist_estimates]
        plt.plot(list(range(time)), estimates, label=names[n])
    plt.legend(loc="best")
    plt.xlabel("Time")
    plt.ylabel("Estimated value")
    plt.title("Estimates over time")
    plt.show()

def plot_actions(name, agent, nb_arms):
    nb_actions = [len(i) for i in agent.hist_reward_arms]
    plt.bar(list(range(nb_arms)), nb_actions)
    plt.xticks(list(range(nb_arms)))
    plt.xlabel("Actions")
    plt.ylabel("Number of time selected")
    plt.title(f"Selected action of agent {name}")
    plt.show()


def main():
    time_steps = 1000
    bandit_qstar = np.array([[2.3, .9], [2.1, .6], [1.5, .4], [1.3, 2]])

    bandit_info = [len(bandit_qstar), time_steps]
    agent_list = [AgentEpsilon(bandit_info, 1),
                  AgentEpsilon(bandit_info, 0),
                  AgentEpsilon(bandit_info, 0.1),
                  AgentEpsilon(bandit_info, 0.2),
                  AgentSoftmax(bandit_info, 1),
                  AgentSoftmax(bandit_info, 0.1),
                  AgentEpsilon(bandit_info, time_dep=True),
                  AgentSoftmax(bandit_info, time_dep=True)
                 ]

    for agent in agent_list:
        for t in range(time_steps):
            agent_action = agent.select_action()
            reward = np.random.normal(bandit_qstar[agent_action][0],
                                      bandit_qstar[agent_action][1])

            agent.all_rewards[t] = reward
            agent.hist_reward_arms[agent_action].append(reward)
            agent.hist_estimates.append(agent.q_estimates.copy())
            agent.q_estimates[agent_action] = (
                            sum(agent.hist_reward_arms[agent_action]) /
                            len(agent.hist_reward_arms[agent_action])
                            )


    names = ['Random', 'Epsilon:0', 'Epsilon:0.1',
             'Epsilon:0.2', 'Softmax:1', 'Softmax:0.1',
             'EpsilonTime', 'SoftmaxTime']

    plot_reward(names, agent_list, time_steps)
    plot_estimates(names, agent_list, 0, 2.3, time_steps)
    plot_actions(names[-1], agent_list[-1], 4)


main()
