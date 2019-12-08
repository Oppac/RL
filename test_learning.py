import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon
        self.score = [0]
        self.hist_reward_arms = [[], [], [], []]
        self.estimation_arms = [0, 0, 0, 0]

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.estimation_arms)


def main():

    trials = 1000
    bandit_values = np.array([[2.3, .9], [2.1, .6], [1.5, .4], [1.3, 2]])
    np.random.shuffle(bandit_values)

    agent_list = [Agent(1), Agent(0), Agent(0.1), Agent(0.2)]

    for agent in agent_list:
        for i in range(trials):
            agent_action = agent.select_action()
            reward = np.random.normal(bandit_values[agent_action][0], bandit_values[agent_action][1])
            agent.hist_reward_arms[agent_action].append(reward)
            agent.score.append(agent.score[-1] + reward / len(agent.score))
            agent.estimation_arms[agent_action] = sum(agent.hist_reward_arms[agent_action]) / len(agent.hist_reward_arms[agent_action])


    print({agent.epsilon: agent.estimation_arms for agent in agent_list})

    for agent in agent_list:
        plt.plot(list(range(trials+1)), agent.score, label=r"$\epsilon$ = {e}".format(e=agent.epsilon))
    plt.legend(loc="lower right")
    #plt.show()


'''
    for i in range(len(agent_list)):
        fig, ax = plt.subplots()
        ax.bar(list(range(4)), agent_list[i].estimation_arms)
        ax.set_title(f"Agent epsilon {agent_list[i].epsilon}")
        plt.savefig(f'agent_{i}.png')
'''


main()
