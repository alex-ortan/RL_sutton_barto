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
                 action_values_mean = 0, 
                 action_values_var = 1, 
                 reward_var = 1):

        self._vals = Rng.normal(action_values_mean, action_values_var, k)
        self._vars = [reward_var]*k
        self._best_action = np.argmax(self._vals)


    def reward(self, action):
        
        return Rng.normal(self._vals[action], self._vars[action])

