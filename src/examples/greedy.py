from ..simulator import Simulator
from ..policies.policies import GreedyPolicy
from ..bandits.bandits import BernoulliBandit

policy_l = GreedyPolicy()
simulator_l = Simulator(policy_l)

#optimistic evaluations
bandit_1 = BernoulliBandit(1,0.2)
bandit_2 = BernoulliBandit(1,0.5)
bandit_3 = BernoulliBandit(1,0.8)

simulator_l.addBandit(bandit_1)
simulator_l.addBandit(bandit_2)
simulator_l.addBandit(bandit_3)

print(simulator_l.policy_.vectBanditsEvals_)

for i in range(10):
    print("step %d : " % (i+1) )
    print("\t %s  ==> %s" % (simulator_l.nextStep(), simulator_l.policy_.vectBanditsEvals_) )
