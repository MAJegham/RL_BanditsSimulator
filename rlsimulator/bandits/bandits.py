import numpy as np

class _BaseBandit:
    def __init__(self, initialEval_p):
        self.initialEval_ = initialEval_p

    def getReward(self, step_p):
        pass

class BernoulliBandit(_BaseBandit):
    def __init__(self, initialEval_p, proba_p):
        self.initialEval_ = initialEval_p
        self.proba_ = proba_p

    def getReward(self, step_p):
        return np.random.binomial(1, self.proba_)

class NormalBandit(_BaseBandit):
    def __init__(self, initialEval_p, mean_p, std_p):
        self.initialEval_ = initialEval_p
        self.mean_ = mean_p
        self.std_ = std_p

    def getReward(self, step_p):
        return np.random.normal(self.mean_, self.std_, 1)[0]