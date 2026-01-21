import numpy as np

from typing import Callable



class SampleAverageEstimator:
    """
    Keeps and updates estimates Q(a) for all arms a of a k-armed bandit.
    """

    def __init__(self, k_arms=10, initial_q=0):
        self.k = k_arms
        self.n = np.zeros(k_arms)           # number of times each action was taken
        self.q = np.full(k_arms, initial_q) # values estimates for each action


    def update(self, a, r):
        self.n[a] += 1
        step_size = 1.0 / self.n[a]
        self.q[a] = self.q[a] + step_size*(r - self.q[a])



class ConstantStepSizeEstimator:
    """
    Keeps and updates estimates Q(a) for all arms a of a k-armed bandit.
    """

    def __init__(self, k_arms, initial_q=0, step_size = 0.1):
        self.k = k_arms
        self.n = np.zeros(k_arms)           # number of times each action was taken
        self.q = np.full(k_arms, initial_q) # values estimates for each action
        self.step_size = step_size          # step size for the update after n steps


    def update(self, a, r):
        self.n[a] += 1
        self.q[a] = self.q[a] + self.step_size*(r - self.q[a])

