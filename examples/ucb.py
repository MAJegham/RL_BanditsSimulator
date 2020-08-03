 # ucb.py
 # author: aziz jegham
 # Created on Tue July 30 2020
 # Copyright (C) 2020 aziz jegham
 # License: GNU General Public License version 3

from rlsimulator.simulator import Simulator
from rlsimulator.policies.policies import UCBPolicy
from rlsimulator.bandits.bandits import BernoulliBandit

policy_l = UCBPolicy(1)
simulator_l = Simulator(policy_l)

#over-optimistic evaluations with ucb
bandit_1 = BernoulliBandit(0.2)
bandit_2 = BernoulliBandit(0.5)
bandit_3 = BernoulliBandit(0.8)

simulator_l.addBandit(bandit_1, 2)
simulator_l.addBandit(bandit_2, 2)
simulator_l.addBandit(bandit_3, 2)

print(simulator_l.policy_.vectBanditsEvals_)

for i in range(10):
    print("step %d : " % (i+1) )
    print("\t %s  ==> %s" % (simulator_l.nextStep(), simulator_l.policy_.vectBanditsEvals_) )
