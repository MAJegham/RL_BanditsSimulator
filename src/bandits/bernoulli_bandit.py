from bandits.abstract_bandit import Bandit

import numpy as np

class BernoulliBandit(Bandit):
    def __init__(self, initialEval_p, proba_p):
        self.initialEval_ = initialEval_p
        self.proba_ = proba_p

    def getReward(self, step_p):
        return np.random.binomial(1, self.proba_)