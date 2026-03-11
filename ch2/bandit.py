import numpy as np


class Bandit:
    """
    Defines the bandit environment
    """

    def __init__(self, k, *, q_true_mean=0, q_true_var=1, q_var=1, drift_var=0.0, seed=42):
        """
        k-armed bandit, where the rewards from each arm are normally distributed, with random means and variance 1.
        The mean rewards for each arm themselves are drawn from a normal distribution with mean 0 and variance 1.

        Args:
            k: number of arms in the bandit
            q_true_mean: mean of the q_true values for each of the k arms
            q_true_var: variance of the q_true values each of the k arms
            q_var: variance of the sampled rewards; their mean is q_true
            drift_var: if the bandit is non stationary, the q_true values take a random walk with variance drift_var and mean 0 at every step
            seed: seed for random number generator
        """

        self.k = k
        self._drift_var = drift_var
        self.rng = np.random.default_rng(seed)


        self._q_true = self.rng.normal(q_true_mean, q_true_var, k)
        self._q_var = [q_var] * k
        self._best_action = np.argmax(self._q_true)

    def sample(self, action, n=1):
        """
        Provides a way to sample multiple reward values from the bandit distribution at a given point in time.
        This is useful in non-stationary scenarios where calling self.reward() also changes the distribution.
        """

        return self.rng.normal(self._q_true[action], self._q_var[action], n)

    def reward(self, action):
        """
        Takes the given action and returns the reward value.
        Updates the q_true values if self._drift_var is non-zero.
        """

        if not (0 <= action < self.k):
            raise ValueError("Invalid action selected.")

        r = self.sample(action).item()

        # Add non-stationarity through a random walk of the true q values
        self._q_true += self.rng.normal(0, self._drift_var, self.k)

        return r
