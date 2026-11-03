import numpy as np

from scipy.special import softmax

from estimators import Estimator


class EpsilonGreedyAgent:
    """
    Implements the epsilon-greedy algorithm for k-armed bandits.
    """

    def __init__(self, k_arms=10, *, epsilon=0.1, estimator: Estimator = None, seed=42):
        """
        Initializes the agent.

        Args:
            k_arms: number of arms in the bandit
            epsilon: Probability of exploration (0 <= epsilon <= 1).
            estimator: Initialized estimator class used to estimate values (e.g., SampleAverageEstimator).
            seed: seed for random number generator
        """

        self.k = k_arms
        self.epsilon = epsilon
        self.estimator = estimator
        self.rng = np.random.default_rng(seed)

    def select_action(self):
        """
        Chooses an action based on the epsilon-greedy strategy.
        It uses the Q values stored in the estimator.
        """

        if self.rng.random() < self.epsilon:
            # Explore: Choose a random action
            action = self.rng.integers(self.k).item()
        else:
            # Exploit: Choose a greedy action based on current q values, breaking ties randomly
            actions = np.flatnonzero(self.estimator.q == self.estimator.q.max())
            action = self.rng.choice(actions).item()

        return action

    def update(self, action, reward):
        """
        Passes the update instruction to the underlying estimator.
        """
        self.estimator. update(action, reward)


class UCBAgent:
    """
    Implements Upper-Confidence-Bound algorithm for k-armed bandits.
    """

    def __init__(self, k_arms=10, *, c=1, estimator: Estimator = None, seed=42):
        """
        Initializes the agent.

        Args:
            k_arms: number of arms in the bandit
            c: degree of exploration
            estimator: Initialized estimator class used to estimate values (e.g., SampleAverageEstimator).
            t: current time step
            seed: seed for random number generator
        """

        self.k = k_arms
        self.c = c
        self.estimator = estimator
        self.t = 1
        self.rng = np.random.default_rng(seed)

    def select_action(self):
        """
        Chooses an action using the upper-confidence-bound strategy.
        It uses the Q values stored in the estimator.
        """

        unexplored_actions = np.flatnonzero(self.estimator.n == 0)
        if len(unexplored_actions) > 0:
            # Explore: If some actions have not been chosen yet, choose one of them at random
            action = self.rng.choice(unexplored_actions).item()
        else:
            # Exploit: Choose a greedy action based on current q values and uncertainty

            # Define preference function as value estimate plus an uncertainty term
            pref = self.estimator.q + self.c*np.sqrt(np.log(self.t)/self.estimator.n)

            # Choose action that maximizes preference function, breaking ties randomly
            actions = np.flatnonzero(pref == pref.max())
            action = self.rng.choice(actions).item()

        return action

    def update(self, action, reward):
        """
        Passes the update instruction to the underlying estimator.
        """
        self.estimator.update(action, reward)

        # Track time steps taken
        self.t += 1


class GradientBanditAgent:
    """
    Implements Gradient Bandit algorithm for k-armed bandits.
    """

    def __init__(self, k_arms=10, *, alpha=0.1, seed=42):
        """
        Initializes the agent.

        Args:
            k_arms: number of arms in the bandit
            alpha: step size parameter
            h: preference of each action
            p: action probability distribution
            r: average reward
            t: current time step
            seed: seed for random number generator
        """

        self.k = k_arms
        self.alpha = alpha
        self.h = np.zeros(k_arms)          # initially all actions have equal preference
        self.p = softmax(self.h)
        self.r = 0.0
        self.t = 1
        self.rng = np.random.default_rng(seed)

    def select_action(self):
        """
        Chooses an action using based on the current preference function
        """
        # Sample action from probability distribution
        action = self.rng.choice(len(self.p), p=self.p)

        return action

    def update(self, action, reward):
        """
        Update preference function.
        """
        # Update average reward
        self.r = self.r + 1.0 / self.t * (reward - self.r)

        for a in range(0, self.k):
            if a == action:
                self.h[a] += self.alpha * (reward - self.r) * (1 - self.p[a])
            else:
                self.h[a] += - self.alpha * (reward - self.r) * self.p[a]

        # Update action probabilities based on preference
        self.p = softmax(self.h)

        # Track time steps taken
        self.t += 1
