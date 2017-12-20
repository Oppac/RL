import numpy as np
import matplotlib.pyplot as plt

class NArmedBandit:
    def __init__(self, arms_values):
        self.arms = [len(arms_values)]
        self.q_star = arms_values[:, 0]
        self.sigma = arms_values[:, 1]

    def get_reward(self, action):
        try:
            #Draw random samples from a normal Gaussian distribution
            return np.random.normal(self.q_star[action], self.sigma[action])
        except ValueError:
            print("get_reward : Invalid action")


class EpsilonPolicy:
   #epsilon = 1 : random / epsilon = 0 : greedy
    def __init__(self, epsilon=0):
       self.epsilon = epsilon

    def select_action(self, agent, plays=None):
        if plays != None:
            self.epsilon = np.divide(1, np.sqrt(plays))

        if np.random.random() < self.epsilon:
            #explore
            return np.random.choice(len(agent.values_estimates))
        else:
            #exploit
            action = np.argmax(agent.values_estimates)
            if_same_value = np.where(agent.values_estimates == action)[0]
            if len(if_same_value) == 0:
                return action
            else:
                return np.random.choice(if_same_value)

#Not sure it works correctly => to test further
class SoftMaxPolicy:
    def __init__(self, temperature=0):
        self.temp = temperature

    def select_action(self, agent, plays=None):
        if plays != None:
            self.temp = 4 * np.divide(1000-plays, 1000)
            if self.temp == 0: #prevent divide by 0
                self.temp = 0.004

        qt = agent.values_estimates
        e = np.exp(np.divide(qt, self.temp))
        cdf = np.divide(e, np.sum(e))
        pb = np.random.choice(len(agent.values_estimates), 1, p=cdf)
        action = pb.astype(np.int32)[0]
        return action

# The agent choose from  a set of actions at each steps. It used it's past
# actions to choose the next action.
class Agent:
    def __init__(self, bandit, policy, over_time=False):
        self.policy = policy
        self.arms = bandit.arms
        self._values_estimates = np.zeros(self.arms)
        self.previous_actions = np.zeros((4,2))
        self.last_action = None
        self.over_time = over_time

    def reset(self):
        self._values_estimates[:] = 0
        self.previous_actions[:] = [0, 0]
        self.last_action = None

    def select_action(self, plays=None):
        action = self.policy.select_action(self, plays)
        self.last_action = action
        return action

    def update_estimates(self, reward):
        self.previous_actions[self.last_action][0] += 1
        self.previous_actions[self.last_action][1] += reward
        self._values_estimates[self.last_action] = np.divide(self.previous_actions[self.last_action][1],
                                                    self.previous_actions[self.last_action][0])

    @property
    def values_estimates(self):
        return self._values_estimates


#The environment contains the bandit and the list of agents
class Environment:
    def __init__(self, bandit, agent_list):
        self.bandit = bandit
        self.agent_list = agent_list

    def reset(self):
        for agent in self.agent_list:
            agent.reset()

    def run_multiple_times(self, steps, trials):
        total_scores = np.zeros((steps+1, len(self.agent_list)))
        total_total = np.zeros((steps+1, len(self.agent_list)))

        #For the arms plotting
        action0 = np.zeros((steps, len(self.agent_list)))
        action1 = np.zeros((steps, len(self.agent_list)))
        action2 = np.zeros((steps, len(self.agent_list)))
        action3 = np.zeros((steps, len(self.agent_list)))

        #For the actions histograms
        action_chosen = np.zeros((len(self.agent_list), 4))
        action_j = 0


        for t in range(trials):
            current_scores = np.zeros((steps+1, len(self.agent_list)))
            current_total = np.zeros(len(self.agent_list))
            self.reset()

            for plays in range(steps):
                for i, agent in enumerate(self.agent_list):
                    if agent.over_time == False:
                        action = agent.select_action()
                    else:
                        action = agent.select_action(plays+1)

                    action_chosen[action_j, action] += 1
                    reward = self.bandit.get_reward(action)
                    agent.update_estimates(reward)
                    action0[plays, action_j] += agent.values_estimates[0]
                    action1[plays, action_j] += agent.values_estimates[1]
                    action2[plays, action_j] += agent.values_estimates[2]
                    action3[plays, action_j] += agent.values_estimates[3]

                    current_total[i] += reward
                    current_scores[plays+1, i] = np.divide(current_total[i], plays+1)
                    total_total[plays+1, i] += current_scores[plays+1, i]

                    if action_j == len(self.agent_list)-1:
                        action_j = 0
                    else:
                        action_j += 1
        for s in range(steps):
            for i, agent in enumerate(self.agent_list):
                total_scores[s+1, i] = np.divide(total_total[s+1, i], trials)

        actions_estimates = np.array([np.divide(action0,trials),
                                      np.divide(action1, trials),
                                      np.divide(action2, trials),
                                      np.divide(action3, trials)])
        return (total_scores, action_chosen/trials, actions_estimates)

    def plot_results_all(self, scores, names):
        plt.title("4 Armed Bandit")
        for i, name in enumerate(names):
            plt.plot(scores[:, i], label=name)
        plt.ylabel('Average reward')
        plt.xlabel('Plays')
        plt.legend()
        plt.show()

    def plot_histo(self, action_chosen, names):
        pos = np.arange(4)
        for i in range(len(action_chosen)):
            plt.subplot(3, 3, i+1)
            plt.bar(pos, action_chosen[i, :], 1, edgecolor='black')
            plt.xticks(pos)
            plt.xlabel(names[i])
        plt.show()

    def plot_arms(self, action_estimates, names):
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.plot(action_estimates[i])
            plt.xlabel("arm " + str(i))
        plt.legend(names, bbox_to_anchor=(1,1), loc='best', ncol=1)
        plt.show()











