class Simulator:
    def __init__(self, policy_p):
        self.step_ = 0
        self.nbBandits_ = 0
        
        self.policy_ = policy_p
        
        self.banditsList_ = []
        self.actionsList_ = []
        self.rewardsList_= []


    def addBandit(self, bandit_p):
        self.banditsList_.append(bandit_p)
        self.policy_.vectBanditsEvals_.append(bandit_p.initialEval_)
        self.policy_.vectCountBanditsPulls_.append(0)
        self.nbBandits_ += 1


    def nextStep(self):
        self.step_ += 1

        action_l = self.policy_.getNexAction()
        self.actionsList_.append(action_l)
        
        reward_l = self.banditsList_[action_l].getReward(self.step_) 
        self.rewardsList_.append(reward_l)

        self.policy_.update(action_l, reward_l)

        return action_l, reward_l


    