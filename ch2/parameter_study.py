"""Exercise 2.11: epsilon-greedy, UCB, optimistic, and gradient-bandit sweeps.

Run: .venv/bin/python ch2/parameter_study.py --runs 200
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit
from numpy.random import Generator
from numpy.typing import NDArray
from seaborn.axisgrid import FacetGrid


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
RunScores = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]
UCB_C_VALUES = 2.0 ** np.arange(-4, 10)
Q_INIT_VALUES = 2.0 ** np.arange(-7, 10)
GRADIENT_ALPHAS = 2.0 ** np.arange(-7, 3)


@njit(cache=True)
def select_action(q: FloatArray, epsilon: float, rng: Generator) -> int:
    """Explore randomly, or choose uniformly among the best estimated arms."""
    if rng.random() < epsilon:
        return rng.integers(0, len(q))

    best_actions = np.flatnonzero(q == q.max())
    return best_actions[rng.integers(0, len(best_actions))]


@njit(cache=True)
def select_ucb_action(
    q: FloatArray, counts: IntArray, time: int, c: float, rng: Generator,
) -> int:
    """Choose an untried arm first; otherwise maximize the UCB score."""
    untried = np.flatnonzero(counts == 0)
    if len(untried):
        return untried[rng.integers(0, len(untried))]

    scores = q + c * np.sqrt(np.log(time) / counts)
    best_actions = np.flatnonzero(scores == scores.max())
    return best_actions[rng.integers(0, len(best_actions))]


@njit(cache=True)
def simulate_agent(
    parameter: float,
    sample_average: bool,
    drift: FloatArray,
    reward_noise: FloatArray,
    score_steps: int,
    rng: Generator,
    q_init: float = 0.0,
    ucb: bool = False,
) -> float:
    """Run one agent; parameter is epsilon, or c when ucb=True."""
    steps, arms = drift.shape
    q = np.full(arms, q_init, dtype=np.float64)
    true_q = np.zeros(arms)
    counts = np.zeros(arms, dtype=np.int64)
    reward_sum = 0.0

    for t in range(steps):
        if ucb:
            action = select_ucb_action(q, counts, t + 1, parameter, rng)
        else:
            action = select_action(q, parameter, rng)
        reward = true_q[action] + reward_noise[t]

        counts[action] += 1
        if sample_average:
            step_size = 1.0 / counts[action]
        else:
            step_size = 0.1
        q[action] += step_size * (reward - q[action])

        if t >= steps - score_steps:
            reward_sum += reward

        # Every arm drifts, including those the agent didn't select.
        true_q += drift[t]

    return reward_sum / score_steps


@njit(cache=True)
def softmax(preferences: FloatArray) -> FloatArray:
    """Use the standard numerical stability trick"""
    weights = np.exp(preferences - preferences.max())
    return weights / weights.sum()


@njit(cache=True)
def update_preferences(
    preferences: FloatArray,
    probabilities: FloatArray,
    action: int,
    advantage: float,
    alpha: float,
) -> None:
    """Apply the gradient update using the probabilities that chose the action."""
    for arm in range(len(preferences)):
        if arm == action:
            preferences[arm] += alpha * advantage * (1.0 - probabilities[arm])
        else:
            preferences[arm] -= alpha * advantage * probabilities[arm]


@njit(cache=True)
def simulate_gradient_agent(
    alpha: float,
    drift: FloatArray,
    reward_noise: FloatArray,
    score_steps: int,
    rng: Generator,
    baseline_step: float = 0.0,
) -> float:
    """Use a mean baseline by default, or a moving baseline experimentally.

    baseline_step=0 preserves the original sample average. A positive step
    uses exponential averaging. Both baselines include the current reward.
    """
    if not 0.0 <= baseline_step <= 1.0:
        raise ValueError("baseline_step must be between zero and one")
    steps, arms = drift.shape
    preferences = np.zeros(arms)
    true_q = np.zeros(arms)
    average_reward = 0.0
    reward_sum = 0.0

    for t in range(steps):
        probabilities = softmax(preferences)
        # Sample a categorical distribution using cumulative probabilities.
        draw = rng.random()
        cumulative = 0.0
        action = arms - 1
        for arm in range(arms):
            cumulative += probabilities[arm]
            if draw < cumulative:
                action = arm
                break

        reward = true_q[action] + reward_noise[t]
        if baseline_step == 0.0:
            average_reward += (reward - average_reward) / (t + 1)
        else:
            average_reward += baseline_step * (reward - average_reward)
        advantage = reward - average_reward
        update_preferences(preferences, probabilities, action, advantage, alpha)

        if t >= steps - score_steps:
            reward_sum += reward
        true_q += drift[t]

    return reward_sum / score_steps


def run_study(
    runs: int = 200,
    steps: int = 200000,
    score_steps: int = 100000,
    seed: int = 42,
) -> tuple[FloatArray, RunScores]:
    """Return epsilon values and five score arrays shaped (setting, run).

    Each run has a shared environment across settings, with independent agent
    decisions. Numba compiles the per-agent loop; Python organizes the study.
    Methods are epsilon-greedy with sample averages, epsilon-greedy with
    alpha=0.1, UCB with sample averages, and optimistic greedy with alpha=0.1.
    UCB uses UCB_C_VALUES; optimistic greedy uses Q_INIT_VALUES and epsilon=0.
    Gradient bandits sweep GRADIENT_ALPHAS with a sample-average reward baseline.
    """
    if runs < 2 or not 0 < score_steps <= steps:
        raise ValueError("Need at least two runs and 0 < score_steps <= steps")
    epsilons = 2.0 ** np.arange(-7, 0)
    # UCB can have more parameter settings than epsilon-greedy.
    run_scores = (
        np.empty((len(epsilons), runs)),
        np.empty((len(epsilons), runs)),
        np.empty((len(UCB_C_VALUES), runs)),
        np.empty((len(Q_INIT_VALUES), runs)),
        np.empty((len(GRADIENT_ALPHAS), runs)),
    )
    run_seeds = np.random.SeedSequence(seed).spawn(runs)
    for run, run_seed in enumerate(run_seeds):
        environment_seed, action_seed = run_seed.spawn(2)
        environment_rng = np.random.default_rng(environment_seed)
        # Generate one environment per run and reuse it across settings.
        drift = environment_rng.normal(0, 0.01, size=(steps, 10))
        reward_noise = environment_rng.normal(size=steps)
        for method, sample_average in enumerate((True, False)):
            for i, epsilon in enumerate(epsilons):
                rng = np.random.default_rng(action_seed.spawn(1)[0])
                run_scores[method][i, run] = simulate_agent(
                    epsilon, sample_average, drift, reward_noise, score_steps, rng,
                )

        for i, c in enumerate(UCB_C_VALUES):
            rng = np.random.default_rng(action_seed.spawn(1)[0])
            run_scores[2][i, run] = simulate_agent(
                c, True, drift, reward_noise, score_steps, rng, ucb=True,
            )

        # Greedy action selection, alpha=0.1, varying initial estimates.
        for i, q_init in enumerate(Q_INIT_VALUES):
            rng = np.random.default_rng(action_seed.spawn(1)[0])
            run_scores[3][i, run] = simulate_agent(
                0.0, False, drift, reward_noise, score_steps, rng, q_init=q_init,
            )

        for i, alpha in enumerate(GRADIENT_ALPHAS):
            rng = np.random.default_rng(action_seed.spawn(1)[0])
            run_scores[4][i, run] = simulate_gradient_agent(
                alpha, drift, reward_noise, score_steps, rng,
            )

        if (run + 1) % 10 == 0 or run + 1 == runs:
            print(f"Completed {run + 1:,}/{runs:,} runs for all settings", flush=True)

    return epsilons, run_scores


def study_frame(epsilons: FloatArray, run_scores: RunScores) -> pd.DataFrame:
    """One row per independent run's evaluation-window average reward."""
    methods = (
        ("ε-greedy: sample average", epsilons),
        ("ε-greedy: α = 0.1", epsilons),
        ("UCB: sample average", UCB_C_VALUES),
        ("Optimistic greedy: α = 0.1", Q_INIT_VALUES),
        ("Gradient bandit: mean baseline", GRADIENT_ALPHAS),
    )
    if len(run_scores) != len(methods) or any(
        scores.shape != (len(parameters), run_scores[0].shape[-1])
        for (_, parameters), scores in zip(methods, run_scores)
    ):
        raise ValueError("Expected all five methods; regenerate results with run_study()")
    return pd.DataFrame([
        {"method": name, "parameter": parameter, "run": run, "reward": reward}
        for (name, parameters), method_scores in zip(methods, run_scores)
        for parameter, scores in zip(parameters, method_scores)
        for run, reward in enumerate(scores)
    ])


def plot_study(
    epsilons: FloatArray,
    run_scores: RunScores,
    steps: int = 200000,
    score_steps: int = 100000,
) -> FacetGrid:
    """Plot means with approximate 95% intervals across runs using Seaborn."""
    sns.set_theme(style="whitegrid", context="notebook")
    graph = sns.relplot(
        data=study_frame(epsilons, run_scores),
        x="parameter", y="reward", hue="method", style="method",
        kind="line", markers=True, dashes=False,
        errorbar=("se", 1.96), height=5, aspect=1.5,
    )
    graph.set(xscale="log")
    ticks = np.unique(np.concatenate((epsilons, UCB_C_VALUES, Q_INIT_VALUES, GRADIENT_ALPHAS)))
    graph.set(xticks=ticks, xticklabels=[f"1/{round(1 / p)}" if p < 1 else f"{p:g}" for p in ticks])
    graph.set_axis_labels("Parameter value: ε, c, $Q_0$, or α for the corresponding method",
                          f"Average reward over last {score_steps:,} steps")
    graph.legend.set_title("Method")
    graph.figure.suptitle(
        f"Exercise 2.11: bandit parameter study\n"
        f"Nonstationary · {steps:,} steps · {run_scores[0].shape[-1]:,} runs · drift SD = 0.01",
        y=1.04,
    )
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    epsilons, run_scores = run_study(runs=args.runs, seed=args.seed)
    output = Path(__file__).resolve().parents[1] / "images"
    output.mkdir(exist_ok=True)
    np.savez(output / "parameter_study.npz", epsilons=epsilons,
             c_values=UCB_C_VALUES,
             q_init_values=Q_INIT_VALUES, optimistic_scores=run_scores[3],
             gradient_alphas=GRADIENT_ALPHAS, gradient_scores=run_scores[4],
             gradient_baseline="sample_average_including_current_reward",
             epsilon_scores=np.stack(run_scores[:2]), ucb_scores=run_scores[2],
             runs=args.runs, steps=200000,
             score_steps=100000, seed=args.seed, alpha=0.1, drift_sd=0.01)
    study_frame(epsilons, run_scores).to_csv(output / "parameter_study.csv", index=False)
    graph = plot_study(epsilons, run_scores)
    path = output / "parameter_study.png"
    graph.savefig(path, dpi=160)
    for name, parameters, scores in zip(
        ("ε-greedy sample average", "ε-greedy constant step size", "UCB sample average",
         "Optimistic greedy", "Gradient bandit"),
        (epsilons, epsilons, UCB_C_VALUES, Q_INIT_VALUES, GRADIENT_ALPHAS),
        (s.mean(axis=-1) for s in run_scores),
    ):
        best = scores.argmax()
        print(f"{name}: best tested parameter = {parameters[best]:g}, reward = {scores[best]:.4f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
