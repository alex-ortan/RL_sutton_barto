import numpy as np

from bandit import Bandit
from estimators import SampleAverageEstimator, ConstantStepSizeEstimator
from agents import EpsilonGreedyAgent, bandit_agent_run


def run_simulation(R, epsilon, T, k, estimator_class=SampleAverageEstimator, **kwargs):
    """
    Run R runs of the epsilon-greedy algorithm on R different k-armed bandits for T time steps, and average the rewards from all runs for each time step.

    Args:
        R: number of runs
        epsilon: probability with which an exploratory action is taken
        T: number of timesteps played
        k: number of arms in the bandit
        estimator_class: SampleAverageEstimator or ConstantStepSizeEstimator

    OptionalArgs:
        q_true_mean: mean of the q_true values for each of the k arms
        q_true_var: variance of the q_true values each of the k arms
        drift_var: if the bandit is non stationary, the q_true values take a random walk with variance drift_var and mean 0 at every step
        
    """

    # Get optional bandit and algorithm args from kwargs
    bandit_args = {k:kwargs[k] for k in kwargs.keys() & ["q_true_mean", "q_true_var", "drift_var"]}

    average_rewards = np.zeros(T)
    number_optimal_actions = np.zeros(T)

    #TODO: parallelize to save on time instead of memory
    for r in range(0,R):
        # Initialize environment
        bandit = Bandit(k, **bandit_args)
        bandit_best_action = bandit._best_action
        
        estimator = estimator_class(k)

        # Initialize agent
        agent = EpsilonGreedyAgent(k, epsilon, estimator)

        # Perform one run of the agent on the environment
        rewards, actions = bandit_agent_run(bandit, agent, T)

        # Update average rewards and number of optimal actions with results from current run
        average_rewards = average_rewards + 1/(r+1)*(rewards - average_rewards)
        number_optimal_actions += (actions == bandit_best_action)

    # Calculate percent optimal actions across all runs
    percent_optimal_actions = number_optimal_actions*100/R

    return average_rewards, percent_optimal_actions
