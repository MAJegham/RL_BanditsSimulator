 # epsilon_greedy.py
 # author: aziz jegham
 # Created on Tue July 30 2020
 # Copyright (C) 2020 aziz jegham
 # License: GNU General Public License version 3

from rlsimulator.simulator import Simulator
from rlsimulator.policies.policies import EpsilonGreedyPolicy
from rlsimulator.bandits.bandits import BernoulliBandit
from rlsimulator.plots.plots import plotRewards
from rlsimulator.plots.plots import plotEvals
from rlsimulator.plots.plots import plotAggregates

from matplotlib import pyplot as plt

policy_l = EpsilonGreedyPolicy(0.1)
simulator_l = Simulator(policy_l)

bandit_1 = BernoulliBandit(0.2)
bandit_2 = BernoulliBandit(0.5)
bandit_3 = BernoulliBandit(0.8)

#optimistic evaluations
simulator_l.addBandit(bandit_1, 1)
simulator_l.addBandit(bandit_2, 1)
simulator_l.addBandit(bandit_3, 1)

plotEvals(simulator_l, 500)

simulator_l.reinit()
plotRewards(simulator_l, 500)

simulator_l.reinit()
plotAggregates(simulator_l, 2500, 20, 50)

plt.show()