 # bandits.py
 # author: aziz jegham
 # Created on Tue July 30 2020
 # Copyright (C) 2020 aziz jegham
 # License: GNU General Public License version 3

import numpy as np

class _BaseBandit:
    """
    Abstract base class for bandits.
    
    Attributes
    ----------
    initialEval_ : initial score associated to the bandit

    Methods
    -------
    getReward : abstract. returns the reward won upon choosing the bandit.


    """
    def __init__(self, initialEval_p):
        self.initialEval_ = initialEval_p

    def getReward(self, step_p):
        pass

class BernoulliBandit(_BaseBandit):
    """
    Implements a stationary bernoulli bandit.

    The bandit has a probability proba_ to issue a reward.

    Parameters 
    ----------
    initialEval_p : double indicating the initial score assigned to the bandit.

    proba_p : the probability of producing a reward


    Attributes
    ----------
    initialEval_ : initial score associated to the bandit

    proba_ : probability that the bandit will produce a reward when chosen

    Methods
    -------
    getReward : returns the reward won upon choosing the bandit.
    """
    
    def __init__(self, initialEval_p, proba_p):
        self.initialEval_ = initialEval_p
        self.proba_ = proba_p

    def getReward(self, step_p):
        """
            returns the reward won upon choosing the bandit.
            Produces a reward of 1 with proBandits have an initial evaluation representing the score of the bandit used when choosing
        among multiple ones.
    Bandits implement a getReward method that returns the reward won upon choosing the bandit.
bability proba_ otherwise 0

            step_p : the time step of pulling. Unused here but useful for non-stationary bandits.
        """
        return np.random.binomial(1, self.proba_)

class NormalBandit(_BaseBandit):
    """
    Implements a stationary normal bandit.

    The bandit produces a reward that has a normal distribution.

    Parameters 
    ----------
    initialEval_p : double indicating the initial score assigned to the bandit.

    mean_p : mean of the reward's distribution
    
    std_p : standard deviation of the reward's distribution


    Attributes
    ----------
    initialEval_ : initial score associated to the bandit

    mean_ : mean of the reward's distribution
    
    std_ : standard deviation of the reward's distribution

    Methods
    -------
    getReward : returns the reward won upon choosing the bandit.
    """
    def __init__(self, initialEval_p, mean_p, std_p):
        self.initialEval_ = initialEval_p
        self.mean_ = mean_p
        self.std_ = std_p

    def getReward(self, step_p):
        """
            returns the reward won upon choosing the bandit.
            Produces a reward pulled from a normal distribution N(mean_, std_)

            step_p : the time step of pulling. Unused here but useful for non-stationary bandits.
        """
        return np.random.normal(self.mean_, self.std_, 1)[0]