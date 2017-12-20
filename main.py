import numpy as np
import bandit as bd

def run(steps, trials):
    # Table 1
    names = ['Random', 'Epsilon:0', "Epsilon:0.1",
             "Epsilon:0.2", "SoftMax:1", "SoftMax:0.1",
             "EpsilonTime", "SoftMaxTime"
             ]
    arms_values_ex1 = np.array([[2.3, 0.9], [2.1, 0.6], [1.5, 0.4], [1.3, 2.0]])
    arms_values_ex2 = np.array([[2.3, 1.8], [2.1, 1.2], [1.5, 0.8], [1.3, 4.0]])
    bandit_4_arms = bd.NArmedBandit(arms_values_ex1)
    agent_list = [bd.Agent(bandit_4_arms, bd.EpsilonPolicy(1)),
                  bd.Agent(bandit_4_arms, bd.EpsilonPolicy(0)),
                  bd.Agent(bandit_4_arms, bd.EpsilonPolicy(0.1)),
                  bd.Agent(bandit_4_arms, bd.EpsilonPolicy(0.2)),
                  bd.Agent(bandit_4_arms, bd.SoftMaxPolicy(1)),
                  bd.Agent(bandit_4_arms, bd.SoftMaxPolicy(0.1)),
                  bd.Agent(bandit_4_arms, bd.EpsilonPolicy(), True),
                  bd.Agent(bandit_4_arms, bd.SoftMaxPolicy(), True),
                  ]
    n1 = bd.Environment(bandit_4_arms, agent_list)

    scores, actions, estimates = n1.run_multiple_times(steps, trials)
    n1.plot_results_all(scores, names)
    n1.plot_arms(estimates, names)
    n1.plot_histo(actions, names)

run (1000, 2000)
