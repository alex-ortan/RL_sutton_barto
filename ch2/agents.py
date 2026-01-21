import numpy as np

from estimators import SampleAverageEstimator, ConstantStepSizeEstimator


class EpsilonGreedyAgent:
    """
    Implements the epsilon-greedy algorithm for k-armed bandits.
    """

    def __init__(self, k_arms=10, epsilon=0.1, estimator=None):
        """
        Initializes the agent.
        
        Args:
            k_arms: number of arms in the bandit
            epsilon: Probability of exploration (0 <= epsilon <= 1).
            estimator: Initialized estimator class used to estimate values (e.g., SampleAverageEstimator).
        """

        self.k = k_arms
        self.epsilon = epsilon

        # Initialize the estimator object
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
            action = np.random.choice(actions)
            
        return action


def bandit_epsilon_greedy_run(bandit, k, T, epsilon, estimator_class, step_size = None):
    """ 
    Run the epsilon-greedy algorithm on a particular k-armed bandit for T time steps.

    Args:
        bandit: bandit reward function; bandit(a) returns a real number for every possible action a.
        k: number of arms in the bandit
        T: number of time steps played
        epsilon: probability with which an exploratory action is taken
        estimator_name: name of the method used for estimating q values. "SampleAverage" or "ConstantStepSize"
        step_size: the size of the step for the ConstantStepSize estimator
    """

    # Initialize estimator and agent
    if step_size:
        estimator = estimator_class(k, step_size)
    else:
        estimator = estimator_class(k)
    agent = EpsilonGreedyAgent(k, epsilon, estimator)

    actions = []                 # sequence of actions taken (T,)
    rewards = []                 # sequence of rewards received (T,)
    
    for t in range(0,T):
        # Determine action to take
        a = agent.select_action()
        actions.append(a)
        
        # Determine reward for the chosen action
        r = bandit(a)
        rewards.append(r)
        
        # Update value estimates
        estimator.update(a, r)

    return rewards, actions
