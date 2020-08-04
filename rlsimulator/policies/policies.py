 # policies.py
 # author: aziz jegham
 # Created on Tue July 30 2020
 # Copyright (C) 2020 aziz jegham
 # License: GNU General Public License version 3

import numpy as np

class _BasePolicy:
    """
    Abstract base class for drawing policies.
    Policies describe the way we score bandits and how do we choose among them.

    Attributes
    ----------
    vectCountBanditsPulls_ : list describing the number of times we pulled the corresponding bandit.
    vectBanditsEvals_ : list describing the scores associated to the corresponding bandit.
    vectBanditsParamEstimates_ : list describing the estimate associated to the key parameter of
     the bandit's probability distribution.
    step_ : timestep ie number of times actions were performed.

    Methods
    -------
    getNexAction : abstract. returns the best action to be performed according to the policy.
    
    update : abstract. updates the policy's attributes after an action is performed and a reward is
     drawn. Needs to be called each time after the reward of getNextAction is revealed.  

    exploitActionsList : returns the list of greedy actions ie. those having the highest score. 

    exploreActionsList : returns the list of exploration actions ie. those not having the highest score.
    """

    def __init__(self):
        self.vectCountBanditsPulls_ = []
        self.vectBanditsEvals_ = []
        self.vectBanditsParamEstimates_ = []
        self.step_ = 0

    def reinit(self, initialEvals_p):
        self.vectCountBanditsPulls_ = [0] * len(initialEvals_p)
        self.vectBanditsEvals_ = initialEvals_p.copy()
        self.vectBanditsParamEstimates_ = initialEvals_p.copy()
        self.step_ = 0

    def addBandit(self, initialEval_p):
        self.vectBanditsEvals_.append(initialEval_p)
        self.vectBanditsParamEstimates_.append(initialEval_p)
        self.vectCountBanditsPulls_.append(0)

    def getNexAction(self):
        pass

    def update(action_p, reward_p):
        """
        updates the policy's attributes.
        
        parameters
        ----------
        action_p : the action performed

        reward_p : the reward won upon performing the action
        """
        pass

    def exploitActionsList(self):
        greedyEvaluation_l = np.max(self.vectBanditsEvals_)
        return np.flatnonzero(self.vectBanditsEvals_ == greedyEvaluation_l)

    def exploreActionsList(self):
        greedyEvaluation_l = np.max(self.vectBanditsEvals_)
        return np.flatnonzero(self.vectBanditsEvals_ < greedyEvaluation_l)

class GreedyPolicy(_BasePolicy):
    """
    Class implementing a greedy policy.
    The policy always returns the action having the highest evaluation.
    Bandits' evaluations are updated using a sample-average method 

    Attributes
    ----------
    vectCountBanditsPulls_ : list describing the number of times we pulled the corresponding bandit.
    vectBanditsEvals_ : list describing the scores associated to the corresponding bandit.
    vectBanditsParamEstimates_ : list describing the estimate associated to the key parameter of
     the bandit's probability distribution.
    step_ : timestep ie number of times actions were performed.

    Methods
    -------
    getNexAction : returns the best action to be performed according to the policy.
    
    update : updates the policy's attributes after an action is performed and a reward is
     drawn. Needs to be called each time after the reward of getNextAction is revealed.  
    """

    def getNexAction(self):
        self.step_ += 1
        return np.random.choice(self.exploitActionsList())

    def update(self, action_p, reward_p):
        self.vectCountBanditsPulls_[action_p] += 1
        self.vectBanditsEvals_[action_p] += (1/(1+self.vectCountBanditsPulls_[action_p]))*(reward_p - self.vectBanditsEvals_[action_p])
        self.vectBanditsParamEstimates_[action_p] += (1/(1+self.vectCountBanditsPulls_[action_p]))*(reward_p - self.vectBanditsParamEstimates_[action_p])

class EpsilonGreedyPolicy(_BasePolicy):
    """
    Class implementing an epsilon-greedy policy.
    The policy returns a greedy action with a given probability otherwise it performs an exploration
     action.
    Bandits' evaluations are updated using a sample-average method 

    Attributes
    ----------
    vectCountBanditsPulls_ : list describing the number of times we pulled the corresponding bandit.
    
    vectBanditsEvals_ : list describing the scores associated to the corresponding bandit.
    
    vectBanditsParamEstimates_ : list describing the estimate associated to the key parameter of
     the bandit's probability distribution.
    
    step_ : timestep ie number of times actions were performed.
    
    epsilon_ : probability of performing a greedy action.

    Methods
    -------
    getNexAction : returns the best action to be performed according to the policy.
    
    update : updates the policy's attributes after an action is performed and a reward is
     drawn. Needs to be called each time after the reward of getNextAction is revealed.  
    """

    def __init__(self, epsilon_p):
        super().__init__()
        self.epsilon_ = epsilon_p

    def getNexAction(self):
        self.step_ += 1
        
        doExploreAction_l = ( np.random.binomial(1, self.epsilon_) == 1 )
        greedyEvaluation_l = np.max(self.vectBanditsEvals_)

        if doExploreAction_l:
            exploreActionsList_l = self.exploreActionsList()
            if len(exploreActionsList_l) != 0 :
                return np.random.choice(exploreActionsList_l)
            else:
                return np.random.choice(np.arange(len(self.vectBanditsEvals_)))
        else:
            greedyActions_l = self.exploitActionsList()
            return np.random.choice(greedyActions_l)

    def update(self, action_p, reward_p):
        self.vectCountBanditsPulls_[action_p] += 1
        observationsCount_l = 1+self.vectCountBanditsPulls_[action_p]
        self.vectBanditsEvals_[action_p] += (1/(observationsCount_l))*(reward_p - self.vectBanditsEvals_[action_p])
        self.vectBanditsParamEstimates_[action_p] += (1/(observationsCount_l))*(reward_p - self.vectBanditsParamEstimates_[action_p])



class UCBPolicy(_BasePolicy):
    """
    Class implementing an Upper Confidence Bound (UCB) policy.
    The policy evaluates the bandit using a sample-average method and an uncertainty component.
    The uncertainty component encourages choosing actions that we are unconfident about their
     evaluations (they haven't been selected for a while or they have been selected only a few
     number of times) 
    
    Eval(Action, step) = SampleAverage(Action, step) + exploreParam_ * sqrt( ln(step) / PriorCount(Action, step) )

    Attributes
    ----------
    vectCountBanditsPulls_ : list describing the number of times we pulled the corresponding bandit.
    
    vectBanditsEvals_ : list describing the scores associated to the corresponding bandit.
    
    vectBanditsParamEstimates_ : list describing the estimate associated to the key parameter of
     the bandit's probability distribution.
    
    step_ : timestep ie number of times actions were performed.
    
    exploreParam_ : exploration coefficient.

    Methods
    -------
    getNexAction : returns the best action to be performed according to the policy.
    
    update : updates the policy's attributes after an action is performed and a reward is
     drawn. Needs to be called each time after the reward of getNextAction is revealed.  
    """

    def __init__(self, exploreParam_p):
        super().__init__()
        self.exploreParam_ = exploreParam_p

    def getNexAction(self):
        self.step_ += 1
        return np.random.choice(self.exploitActionsList())

    def update(self, action_p, reward_p):
        self.vectCountBanditsPulls_[action_p] += 1
        #actual estimate : ignores the initial estimate
        self.vectBanditsParamEstimates_[action_p] += (1/self.vectCountBanditsPulls_[action_p])*(reward_p - self.vectBanditsParamEstimates_[action_p])        

        uncertainties_l = []
        for action_cntr in range(len(self.vectBanditsEvals_)):
            uncertainty_l = np.sqrt(np.log(self.step_+1) / (1+self.vectCountBanditsPulls_[action_cntr]) )
            self.vectBanditsEvals_[action_cntr] = self.vectBanditsParamEstimates_[action_cntr] + self.exploreParam_ * uncertainty_l
