 # epsilon_greedy.py
 # author: aziz jegham
 # Created on Tue July 30 2020
 # Copyright (C) 2020 aziz jegham
 # License: GNU General Public License version 3

from rlsimulator.simulator import Simulator
from rlsimulator.policies.policies import EpsilonGreedyPolicy
from rlsimulator.policies.policies import UCBPolicy
from rlsimulator.bandits.bandits import _BaseBandit
from rlsimulator.bandits.bandits import NormalBandit
from rlsimulator.plots.plots import plotRewards
from rlsimulator.plots.plots import plotEvals
from rlsimulator.plots.plots import plotAggregates

from matplotlib import pyplot as plt
import numpy as np

class PeriodicNormalBandit(_BaseBandit):
    def __init__(self, means_p, std_p, subPeriod_p):
        self.means_ = means_p
        self.std_ = std_p
        self.subPeriod_ = subPeriod_p

    def getReward(self, step_p):
        period_l = self.subPeriod_ * len(self.means_)
        mean_l = self.means_[step_p % period_l // self.subPeriod_]
        return np.random.normal(mean_l, self.std_, 1)[0]


policy_l = UCBPolicy(50)
# policy_l = EpsilonGreedyPolicy(0.20)
simulator_l = Simulator(policy_l)

bandit_1 = PeriodicNormalBandit([20, 100, 10, 200], 5, 100)
bandit_2 = PeriodicNormalBandit([200, 10, 100, 20], 5, 100)
bandit_3 = NormalBandit(100, 30)

#optimistic evaluations
simulator_l.addBandit(bandit_1, 200)
simulator_l.addBandit(bandit_2, 200)
simulator_l.addBandit(bandit_3, 200)

plotEvals(simulator_l, 5000)

simulator_l.reinit()
plotRewards(simulator_l, 5000)

simulator_l.reinit()
plotAggregates(simulator_l, 2500, 20, 50)

plt.show()