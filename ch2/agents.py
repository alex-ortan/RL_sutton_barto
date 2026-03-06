import numpy as np

from estimators import Estimator


class EpsilonGreedyAgent:
    """
    Implements the epsilon-greedy algorithm for k-armed bandits.
    """

    def __init__(self, k_arms=10, epsilon=0.1, estimator: Estimator = None):
        """
        Initializes the agent.

        Args:
            k_arms: number of arms in the bandit
            epsilon: Probability of exploration (0 <= epsilon <= 1).
            estimator: Initialized estimator class used to estimate values (e.g., SampleAverageEstimator).
        """

        self.k = k_arms
        self.epsilon = epsilon
        self.estimator = estimator

    def select_action(self):
        """
        Chooses an action based on the epsilon-greedy strategy.
        It uses the Q values stored in the estimator.
        """

        if np.random.rand() < self.epsilon:
            # Explore: Choose a random action
            action = np.random.randint(self.k)
        else:
            # Exploit: Choose a greedy action based on current q values, breaking ties randomly
            actions = np.flatnonzero(self.estimator.q == self.estimator.q.max())
            action = np.random.choice(actions).item()

        return action

    def update(self, action, reward):
        """
        Passes the update instruction to the underlying estimator.
        """
        self.estimator.update(action, reward)
