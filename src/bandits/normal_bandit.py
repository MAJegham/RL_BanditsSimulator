import numpy as np

class normalBandit(Bandit):
    def __init__(self, initialEval_p, mean_p, std_p):
        self.initialEval_ = initialEval_p
        self.mean_ = mean_p
        self.std_ = std_p

    def getReward(self, step_p):
        return np.random.normal(self.mean_, self.std_, 1)[0]