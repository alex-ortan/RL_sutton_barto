import numpy as np

from concurrent.futures import ProcessPoolExecutor

from agents import EpsilonGreedyAgent, UCBAgent, GradientBanditAgent
from bandit import Bandit
from estimators import SampleAverageEstimator


def bandit_agent_run(bandit, agent, n_steps):
    """ 
    Run an agent on a k-armed bandit for n_steps time steps.

    Args:
        bandit: bandit environment with a reward() method that returns a real number for every possible action a.
        agent: a class with a select_action method that decides on the next move
        n_steps: number of time steps played
    """

    optimal_actions = []    # sequence of optimal action indicators (n_steps,)
    rewards = []            # sequence of rewards received (n_steps,)

    for t in range(0, n_steps):
        # Determine action to take
        a = agent.select_action()

        # Record whether action was optimal for benchmarking purposes
        optimal_actions.append(a == bandit._best_action)

        # Determine reward for the chosen action
        r = bandit.reward(a)
        rewards.append(r)

        # Update value estimates
        agent.update(a, r)

    return rewards, optimal_actions


def run_simulation(T, k, estimator_class, agent_class, seed, **kwargs):
    """
    Run a single run of the epsilon-greedy algorithm on a k-armed bandit for T time steps.

    Args:
        epsilon: probability with which an exploratory action is taken
        T: number of timesteps played
        k: number of arms in the bandit
        bandit_args: arguments needed to initialize the bandit
        estimator_class: SampleAverageEstimator or ConstantStepSizeEstimator
        seed: seed for random number generators in both bandit and agent
    """

    # Get optional bandit and algorithm args from kwargs
    bandit_args = {k: kwargs[k] for k in kwargs.keys() & ["q_true_mean", "q_true_var", "q_var", "drift_var"]}

    # Get optional estimator args from kwargs
    estimator_args = {k: kwargs[k] for k in kwargs.keys() & ["initial_q", "step_size"]}

    # Select appropriate agent arguments depending on the agent class, if known:
    if agent_class == EpsilonGreedyAgent:
        agent_args = {k: kwargs[k] for k in kwargs.keys() & ["epsilon"]}
    elif agent_class == UCBAgent:
        agent_args = {k: kwargs[k] for k in kwargs.keys() & ["c"]}
    elif agent_class == GradientBanditAgent:
        agent_args = {k: kwargs[k] for k in kwargs.keys() & ["alpha", "baseline"]}

    # Initialize environment
    bandit_args["seed"] = seed
    bandit = Bandit(k, **bandit_args)

    # Initialize estimator if a class is provided
    if estimator_class:
        estimator = estimator_class(k, **estimator_args)
        agent_args["estimator"] = estimator

    # Initialize agent
    agent = agent_class(k, seed=seed, **agent_args)

    # Perform one run of the agent on the environment
    rewards, optimal_actions = bandit_agent_run(bandit, agent, T)

    return rewards, optimal_actions


def run_experiment(R, T, k, estimator_class=SampleAverageEstimator, agent_class=EpsilonGreedyAgent, **kwargs):
    """
    Run R runs of the epsilon-greedy algorithm on R different k-armed bandits for T time steps, and average the rewards from all runs for each time step.

    Args:
        R: number of runs
        T: number of timesteps played
        k: number of arms in the bandit
        estimator_class: SampleAverageEstimator or ConstantStepSizeEstimator
        agent_class: EpsilonGreedyAgent or UCBAgent

    OptionalArgs:
        q_true_mean: mean of the q_true values for each of the k arms
        q_true_var: variance of the q_true values each of the k arms
        drift_var: if the bandit is non stationary, the q_true values take a random walk with variance drift_var and mean 0 at every step
        initial_q: initial estimate for the estimators
        epsilon: probability with which an exploratory action is taken
        c: exploration constant for UCB

    """
    all_rewards = np.zeros((R, T))
    all_optimal_actions = np.zeros((R, T))

    for r in range(R):
        # Perform one run of the agent on the environment
        rewards, optimal_actions = run_simulation(T, k, estimator_class, agent_class, seed=r, **kwargs)

        all_rewards[r, :] = rewards
        all_optimal_actions[r, :] = optimal_actions
    
    # Calculate average rewards and number of optimal actions with results from current run
    average_rewards = np.average(all_rewards, axis=0)
    number_optimal_actions = all_optimal_actions.sum(axis=0)

    # Calculate percent optimal actions across all runs
    percent_optimal_actions = number_optimal_actions * 100 / R

    return {"average_rewards": average_rewards,
            "percent_optimal_actions": percent_optimal_actions}


def run_experiment_parallel(R, T, k, estimator_class=SampleAverageEstimator, agent_class=EpsilonGreedyAgent, **kwargs):
    """
    Run R runs of the epsilon-greedy algorithm on R different k-armed bandits for T time steps, and average the rewards from all runs for each time step.

    Args:
        R: number of runs
        T: number of timesteps played
        k: number of arms in the bandit
        estimator_class: SampleAverageEstimator or ConstantStepSizeEstimator
        agent_class: EpsilonGreedyAgent or UCBAgent

    OptionalArgs:
        q_true_mean: mean of the q_true values for each of the k arms
        q_true_var: variance of the q_true values each of the k arms
        drift_var: if the bandit is non stationary, the q_true values take a random walk with variance drift_var and mean 0 at every step
        initial_q: initial estimate for the estimators
        epsilon: probability with which an exploratory action is taken
        c: exploration constant for UCB

    """

    all_rewards = np.zeros((R, T))
    all_optimal_actions = np.zeros((R, T))

    with ProcessPoolExecutor() as executor:
        # Pass round number r as seed in run_simulation to ensure experiment is repeatable but each run is still different
        rounds = [executor.submit(run_simulation, T, k, estimator_class, agent_class, seed=r, **kwargs) for r in range(R)]

    for r in range(R):
        # Output returned in the order assigned
        rewards, optimal_actions = rounds[r].result()

        all_rewards[r, :] = rewards
        all_optimal_actions[r, :] = optimal_actions
    
    # Calculate average rewards and number of optimal actions with results from current run
    average_rewards = np.average(all_rewards, axis=0)
    number_optimal_actions = all_optimal_actions.sum(axis=0)

    # Calculate percent optimal actions across all runs
    percent_optimal_actions = number_optimal_actions * 100 / R

    return {"average_rewards": average_rewards,
            "percent_optimal_actions": percent_optimal_actions}
