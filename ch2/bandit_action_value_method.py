import numpy as np

from typing import Callable


def bandit_av_method_run(bandit, k, T, epsilon, step_size: Callable[[int], float] = lambda x: 1/x):
     """
     Run the epsilon-greedy algorithm on a particular k-armed bandit for T time steps.

     Args:
        bandit: bandit reward function; bandit(a) returns a real number for every possible action a.
        k: number of arms in the bandit
        T: number of time steps played
        epsilon: probability with which an exploratory action is taken
        step_size: function of n that determines the size of the update step (alpha in equation 2.5 on page 32)
                   by default the step size used is 1/n, leading to a sample average method
    """

    n = np.zeros(k)              # number of times each action was taken (k,)
    actions = []                 # sequence of actions taken (T,)
    rewards = []                 # sequence of rewards received (T,)
    q_estimates = np.zeros(k)    # values estimates for each action at every step (k,)
    
    for t in range(0,T):
        # Determine greedy action, randomly breaking ties randomly
        greedy_actions = np.flatnonzero(q_estimates == q_estimates.max())
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
        n[a] += 1
        q_estimates[a] += step_size(n[a])*(r - q_estimates[a])
    
    return rewards, actions
