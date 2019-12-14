import time
import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, arms, time_dep=False):
        self.nb_arms = arms
        self.reset()
        self.time_dep = time_dep

    def reset(self):
        self.time = 1
        self.hist_reward_arms = [[], [], [], []]
        self.q_estimates = np.zeros(self.nb_arms)

class AgentEpsilon(Agent):
    def __init__(self, arms, epsilon=0.2, time_dep=False):
        Agent.__init__(self, arms, time_dep)
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
    def __init__(self, arms, temperature=1, time_dep=False):
        Agent.__init__(self, arms, time_dep)
        self.temperature = temperature

    def select_action(self):
        if self.time_dep:
            temperature = 4 * np.divide(1000 - self.time, 1000)
        self.time += 1

        e = np.exp(np.divide(self.q_estimates, self.temperature))
        softmax = np.divide(e, np.sum(e))
        return np.random.choice(len(self.q_estimates), 1, p=softmax)[0]


def plot_reward(names, scores, time):
    for i in range(len(scores)):
        plt.plot(np.arange(time), scores[i], label=names[i])
    plt.legend(loc="lower right")
    plt.title("Cumulative average reward over time")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.show()

def plot_estimates(names, estimates, arm, qstar, time):
    plt.plot(np.arange(time), np.full(time, qstar), label=f'Q*: {qstar}')
    for i in range(len(estimates)):
        plt.plot(np.arange(time), estimates[i], label=names[i])
    plt.legend(loc="best")
    plt.xlabel("Time")
    plt.ylabel("Estimated value")
    plt.title(f"Estimates over time for arm {arm}")
    plt.show()

def plot_actions(name, agent_actions, nb_arms):
    plt.bar(np.arange(nb_arms), agent_actions, label=name)
    plt.xticks(np.arange(nb_arms))
    plt.xlabel("Actions")
    plt.ylabel("Number of time selected")
    plt.title(f"Selected action of agent {name}")
    plt.show()


def main():
    trials = 30
    time_steps = 1000
    bandit_qstar = np.array([[2.3, .9], [2.1, .6], [1.5, .4], [1.3, 2]])

    agent_list = [AgentEpsilon(len(bandit_qstar), 1),
                  AgentEpsilon(len(bandit_qstar), 0),
                  AgentEpsilon(len(bandit_qstar), 0.1),
                  AgentEpsilon(len(bandit_qstar), 0.2),
                  AgentSoftmax(len(bandit_qstar), 1),
                  AgentSoftmax(len(bandit_qstar), 0.1),
                  AgentEpsilon(len(bandit_qstar), time_dep=True),
                  AgentSoftmax(len(bandit_qstar), time_dep=True)
                 ]

    all_scores = np.zeros((len(agent_list), time_steps))
    all_estimates = [np.zeros((len(agent_list), time_steps))
                     for _ in range(len(bandit_qstar))]
    all_actions = np.zeros((len(agent_list), len(bandit_qstar)))

    for _ in range(trials):
        for i, agent in enumerate(agent_list):
            for t in range(time_steps):
                agent_action = agent.select_action()
                all_actions[i][agent_action] += 1
                reward = np.random.normal(bandit_qstar[agent_action][0],
                                          bandit_qstar[agent_action][1])

                all_scores[i][t] += reward
                agent.hist_reward_arms[agent_action].append(reward)
                agent.q_estimates[agent_action] = (
                                sum(agent.hist_reward_arms[agent_action]) /
                                len(agent.hist_reward_arms[agent_action])
                                )
                for k in range(len(bandit_qstar)):
                    all_estimates[k][i][t] += agent.q_estimates[k]
            agent.reset()

    names = ['Random', 'Epsilon:0', 'Epsilon:0.1',
             'Epsilon:0.2', 'Softmax:1', 'Softmax:0.1',
             'EpsilonTime', 'SoftmaxTime']

    all_scores = np.divide(np.divide(np.cumsum(all_scores, axis=1), trials),
                           np.arange(1, time_steps+1))
    plot_reward(names, all_scores, time_steps)

    all_estimates = np.divide(all_estimates, trials)
    for i in range(len(bandit_qstar)):
        plot_estimates(names, all_estimates[i], i, bandit_qstar[i][0], time_steps)

    all_actions = np.divide(all_actions, trials)
    for i in range(len(agent_list)):
        plot_actions(names[i], all_actions[i], len(bandit_qstar))

#start_time = time.time()
main()
#print(f"--- %{time.time() - start_time} seconds ---")
