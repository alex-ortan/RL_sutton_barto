import numpy as np

from estimators import Estimator, SampleAverageEstimator, ConstantStepSizeEstimator


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


    def update_estimates(self, action, reward):
        """
        Passes the update instruction to the underlying estimator.
        """
        self.estimator.update(action, reward)


def bandit_agent_run(bandit, agent, n_steps):
    """ 
    Run an agent on a k-armed bandit for n_steps time steps.

    Args:
        bandit: bandit environment with a reward() method that returns a real number for every possible action a.
        agent: a class with a select_action method that decides on the next move
        n_steps: number of time steps played
    """

    actions = []                 # sequence of actions taken (n_steps,)
    rewards = []                 # sequence of rewards received (n_steps,)
    
    for t in range(0,n_steps):
        # Determine action to take
        a = agent.select_action()
        actions.append(a)
        
        # Determine reward for the chosen action
        r = bandit.reward(a)
        rewards.append(r)
        
        # Update value estimates
        agent.update_estimates(a, r)

    return rewards, actions
