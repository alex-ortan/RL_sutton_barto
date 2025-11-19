import numpy as np

from typing import Callable



class SampleAverageEstimator:
    """
    Keeps and updates estimates Q(a) for all arms a of a k-armed bandit.
    """

    def __init__(self, k):
        self.k = k
        self.q = np.zeros(k)        # values estimates for each action
        self.n = np.zeros(k)        # number of times each action was taken


    def update(self, a, r):
        self.n[a] += 1
        self.q[a] += (1.0/self.n[a])*(r - self.q[a])



class ConstantStepSizeEstimator:
    """
    Keeps and updates estimates Q(a) for all arms a of a k-armed bandit.
    """

    def __init__(self, k, step_size = 0.1):
        self.k = k
        self.q = np.zeros(k)        # values estimates for each action
        self.n = np.zeros(k)        # number of times each action was taken
        self.step_size = step_size  # step size for the update after n steps


    def update(self, a, r):
        self.n[a] += 1
        self.q[a] += self.step_size*(r - self.q[a])

