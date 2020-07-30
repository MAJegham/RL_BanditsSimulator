from policies.abstract_policy import Policy

import numpy as np

class GreedyPolicy(Policy):

    def getNexAction(self):
        self.step_ += 1
        return np.argmax(self.vectBanditsEvals_)

    def update(self, action_p, reward_p):
        self.vectCountBanditsPulls_[action_p] += 1
        self.vectBanditsEvals_[action_p] += (1/(self.vectCountBanditsPulls_[action_p]))*(reward_p - self.vectBanditsEvals_[action_p])
        