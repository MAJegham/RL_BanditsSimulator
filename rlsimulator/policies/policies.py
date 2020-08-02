import numpy as np

class _BasePolicy:
    def __init__(self):
        self.vectCountBanditsPulls_ = []
        self.vectBanditsEvals_ = []
        self.vectBanditsParamEstimates_ = []
        self.step_ = 0

    def getNexAction(self):
        pass

    def update(action_p, reward_p):
        pass

    def exploitActionsList(self):
        greedyEvaluation_l = np.max(self.vectBanditsEvals_)
        return np.flatnonzero(self.vectBanditsEvals_ == greedyEvaluation_l)

    def exploreActionsList(self):
        greedyEvaluation_l = np.max(self.vectBanditsEvals_)
        return np.flatnonzero(self.vectBanditsEvals_ < greedyEvaluation_l)

class GreedyPolicy(_BasePolicy):
    def getNexAction(self):
        self.step_ += 1
        return np.random.choice(self.exploitActionsList())

    def update(self, action_p, reward_p):
        self.vectCountBanditsPulls_[action_p] += 1
        self.vectBanditsEvals_[action_p] += (1/(1+self.vectCountBanditsPulls_[action_p]))*(reward_p - self.vectBanditsEvals_[action_p])
        self.vectBanditsParamEstimates_[action_p] += (1/(1+self.vectCountBanditsPulls_[action_p]))*(reward_p - self.vectBanditsParamEstimates_[action_p])

class EpsilonGreedyPolicy(_BasePolicy):
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
