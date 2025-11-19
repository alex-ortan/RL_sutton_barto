import numpy as np

from estimators import SampleAverageEstimator, ConstantStepSizeEstimator


def bandit_epsilon_greedy_run(bandit, k, T, epsilon, estimator_name, step_size = None):
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

    if estimator_name == "SampleAverage":
        estimator = SampleAverageEstimator(k)
    elif estimator_name == "ConstantStepSize":
        if step_size:
            estimator = ConstantStepSizeEstimator(k, step_size)
        else:
            estimator = ConstantStepSizeEstimator(k)
    else:
       raise ValueError('Only estimators "SampleAverage" or ConstantStepSize" are supported.')

    actions = []                 # sequence of actions taken (T,)
    rewards = []                 # sequence of rewards received (T,)
    
    for t in range(0,T):
        # Determine greedy action, randomly breaking ties randomly
        greedy_actions = np.flatnonzero(estimator.q == estimator.q.max())
        greedy_action = np.random.choice(greedy_actions)
        
        # Determine exploratory action randomly
        exploratory_action = np.random.choice(range(0,10))
        exploratory_action
        
        # Choose exploratory action with epsilon probability, and the greedy action otherwise
        a = np.random.choice([greedy_action, exploratory_action], p=[1-epsilon, epsilon])
        actions.append(a)
        
        # Determine reward for the chosen action
        r = bandit(a)
        rewards.append(r)
        
        # Update value estimates
        estimator.update(a, r)

    return rewards, actions
