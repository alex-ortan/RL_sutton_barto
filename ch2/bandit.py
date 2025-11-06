import numpy as np

# Random number generator initialized with fixed seed
# This is at the top so every time a Bandit is initialized it can generate different rewards
Rng = np.random.default_rng(seed=42)


class Bandit:
    """
    k-armed bandit, where the rewards from each arm are normally distributed, with random means and variance 1.
    The mean rewards for each arm themselves are drawn from a normal distribution with mean 0 and variance 1.

    """
    def __init__(self, 
                 k, 
                 q_true_mean = 0, 
                 q_true_var = 1, 
                 q_var = 1,
                 drift_var = 0.0):

        self._k = k
        self._drift_var = drift_var

        self._q_true = Rng.normal(q_true_mean, q_true_var, k)
        self._q_var = [q_var]*k
        self._best_action = np.argmax(self._q_true)


    def sample(self, action, n = 1):
        
        return Rng.normal(self._q_true[action], self._q_var[action], n)


    def reward(self, action):

        r = self.sample(action).item()

        # Add non-stationarity through a random walk of the true q values
        self._q_true += Rng.normal(0, self._drift_var, self._k)

        return r
