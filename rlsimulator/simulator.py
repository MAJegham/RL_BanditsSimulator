 # simulator.py
 # author: aziz jegham
 # Created on Tue July 30 2020
 # Copyright (C) 2020 aziz jegham
 # License: GNU General Public License version 3

class Simulator:
    """
    Class to simulate a bandits problem

    Parameters
    ----------
    policy_p : the policy to use. 

    Attributes
    ----------
    step_ : timestep ie number of times actions were performed.
    
    nbBandits_ : number of the bandits used in the simulator.

    policy_ : the policy to use to choose the action to perform.

    banditsList_ : List of the bandits available in the simulator.

    actionsList_ : List of the actions performed ie. indices of the bandits pulled at each step.

    rewardsList_ : reward obtained at each timestep.

    Methods
    -------
    addBandit : adds a bandit to the available bandits in the simulator.
    
    nextStep : returns the chosen action and its reward.
    """

    def __init__(self, policy_p):
        self.step_ = 0
        self.nbBandits_ = 0
        
        self.policy_ = policy_p
        
        self.banditsList_ = []
        self.actionsList_ = []
        self.rewardsList_= []


    def addBandit(self, bandit_p, initialEval_p):
        """
        adds a bandit to the available bandits in the simulator.

        Parameters
        ----------        
        bandit_p : the bandit to add.
        
        initialEval_p : initial score associated to the bandit
        """
        self.banditsList_.append(bandit_p)
        self.policy_.vectBanditsEvals_.append(initialEval_p)
        self.policy_.vectBanditsParamEstimates_.append(initialEval_p)
        self.policy_.vectCountBanditsPulls_.append(0)
        self.nbBandits_ += 1


    def nextStep(self):
        """
        returns the chosen action and its reward.
        """
        self.step_ += 1

        action_l = self.policy_.getNexAction()
        self.actionsList_.append(action_l)
        
        reward_l = self.banditsList_[action_l].getReward(self.step_) 
        self.rewardsList_.append(reward_l)

        self.policy_.update(action_l, reward_l)

        return action_l, reward_l


    