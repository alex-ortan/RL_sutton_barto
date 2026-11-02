"""Policy iteration for Sutton & Barto, Example 4.2: Jack's Car Rental.

Run: .venv/bin/python ch4/jacks_car_rental.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson


def jacks_car_rental(
    max_cars: int = 20,
    max_move: int = 5,
    gamma: float = 0.9,
    theta: float = 1e-8,
    max_sweeps: int = 10000,
    max_iterations: int = 100,
) -> tuple[
    NDArray[np.int_],
    NDArray[np.float64],
    list[NDArray[np.int_]],
    list[int],
]:
    """Return (policy, values, policies, evaluation_sweeps).

    State (i, j) is the end-of-day inventory at locations 1 and 2. Positive
    actions move cars from 1 to 2 overnight; negative actions move them back.
    Transfers must leave both inventories between 0 and max_cars. Rentals
    happen the next day, followed by returns; surplus returns are removed.

    Rentals earn $10 and moving costs $2 per car, with the first car moved
    from location 1 to 2 free. Each location charges $4 for parking more than
    10 cars after moving. Independent Poisson rental rates are (3, 4), and
    return rates are (3, 2), with Exercise 4.7's modified reward rules.

    Policy evaluation stops when the largest value change is below theta.
    Policy improvement repeats until no action changes. The returned policies
    include the initial policy and each improvement; evaluation_sweeps records
    the number of Bellman updates used to evaluate each policy.
    """
    if not isinstance(max_cars, int) or max_cars < 1:
        raise ValueError("max_cars must be a positive integer")
    if not isinstance(max_move, int) or max_move < 0:
        raise ValueError("max_move must be a nonnegative integer")
    if not 0 <= gamma < 1:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be positive and finite")
    for name, limit in (("max_sweeps", max_sweeps), ("max_iterations", max_iterations)):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{name} must be a positive integer")

    size = max_cars + 1

    def transition_rewards(demand_rate, return_rate):
        # transition[a, b] = P(finish w b cars | start with a cars)
        transition = np.zeros((size, size))
        rewards = np.zeros(size)
        for available in range(size):
            rented = np.arange(available + 1)
            rental_prob = poisson.pmf(rented, demand_rate)
            # probs need to sum to 1
            rental_prob[-1] = poisson.sf(available - 1, demand_rate)
            # $10/rental
            rewards[available] = 10 * (rental_prob @ rented)
            # Overnight parking is based on inventory after moving, before rentals.
            if available > 10:
                rewards[available] -= 4
            for count, probability in enumerate(rental_prob):
                remaining = available - count
                space = max_cars - remaining
                transition[available, remaining:max_cars] += (
                    probability * poisson.pmf(np.arange(space), return_rate)
                )
                transition[available, max_cars] += (
                    probability * poisson.sf(space - 1, return_rate)
                )
        return transition, rewards

    first_transition, first_reward = transition_rewards(3, 3)
    second_transition, second_reward = transition_rewards(4, 2)

    actions = np.arange(-min(max_move, max_cars), min(max_move, max_cars) + 1)
    first = np.zeros((size, size), dtype=int)
    second = np.zeros((size, size), dtype=int)
    rental_income = np.zeros((size, size))
    feasible = np.zeros((len(actions), size, size), dtype=bool)
    safe_first = np.zeros((len(actions), size, size), dtype=int)
    safe_second = np.zeros((len(actions), size, size), dtype=int)
    moving_cost = np.zeros((len(actions), size, size))

    for i in range(size):
        for j in range(size):
            first[i, j] = i
            second[i, j] = j
            rental_income[i, j] = first_reward[i] + second_reward[j]

            for k, action in enumerate(actions):
                after_first = i - action
                after_second = j + action
                feasible[k, i, j] = (0 <= after_first <= max_cars
                                     and 0 <= after_second <= max_cars)
                # Keep indices in bounds; infeasible actions are excluded later.
                safe_first[k, i, j] = min(max(after_first, 0), max_cars)
                safe_second[k, i, j] = min(max(after_second, 0), max_cars)
                # $2
                if action > 0:
                    moving_cost[k, i, j] = 2 * (action - 1)
                else:
                    moving_cost[k, i, j] = 2 * -action

    policy = np.zeros((size, size), dtype=int)
    values = np.zeros((size, size))
    policies = [policy.copy()]
    evaluation_sweeps = []
    for _ in range(max_iterations):
        # 2. Evaluate the current policy with repeated Bellman updates.
        # Independence gives E[V(next)] = P1 @ V @ P2.T, indexed by the
        # inventories AFTER moving cars. Reuse V from the preceding policy.
        morning_first, morning_second = first - policy, second + policy
        for sweep in range(1, max_sweeps + 1):
            morning_values = (rental_income
                              + gamma * (first_transition @ values @ second_transition.T))
            policy_cost = 2 * np.abs(policy) - 2 * (policy > 0)
            updated = morning_values[morning_first, morning_second] - policy_cost
            delta = np.max(np.abs(updated - values))
            values = updated
            if delta < theta:
                break
        else:
            raise RuntimeError("policy evaluation did not converge within max_sweeps")
        evaluation_sweeps.append(sweep)

        # 3. Improve the policy: greedily choose the best feasible transfer.
        morning_values = (rental_income
                          + gamma * (first_transition @ values @ second_transition.T))
        q = morning_values[safe_first, safe_second] - moving_cost
        q = np.where(feasible, q, -np.inf)
        improved = actions[np.argmax(q, axis=0)]
        old_q = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                k = np.searchsorted(actions, policy[i, j])
                old_q[i, j] = q[k, i, j]
        tied = np.isclose(old_q, q.max(axis=0), rtol=0, atol=1e-10)
        improved[tied] = policy[tied]  # Retain the current action on numerical ties.
        if np.array_equal(improved, policy): # policy_stable
            break # great output look
        policy = improved
        policies.append(policy.copy())
    else:
        raise RuntimeError("policy iteration did not stabilize within max_iterations")

    return policy, values, policies, evaluation_sweeps


if __name__ == "__main__":
    plt.switch_backend("Agg")
    policy, values, policies, evaluation_sweeps = jacks_car_rental()

    output = Path(__file__).resolve().parents[1] / "images"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "jacks_car_rental.npz", policy=policy, values=values,
             policies=np.stack(policies), evaluation_sweeps=evaluation_sweeps)
    panels = len(policies) + 1
    rows = (panels + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows), squeeze=False)
    limit = max(1, int(np.max(np.abs(policies))))
    for index, snapshot in enumerate(policies):
        ax = axes.flat[index]
        image = ax.imshow(snapshot, origin="lower", vmin=-limit, vmax=limit,
                          cmap="RdBu_r", interpolation="nearest")
        ax.set_title(f"Policy π{index}" + (" (stable)" if index == panels - 2 else ""))
        fig.colorbar(image, ax=ax, label="Cars moved: + means 1 → 2")
    ax = axes.flat[panels - 1]
    image = ax.imshow(values, origin="lower", cmap="viridis")
    ax.set_title("Final value function V")
    fig.colorbar(image, ax=ax, label="Expected discounted net income ($)")
    for ax in list(axes.flat)[:panels]:
        ax.set_xlabel("Cars at location 2")
        ax.set_ylabel("Cars at location 1")
        ax.set_xticks(np.arange(0, policy.shape[0], 5))
        ax.set_yticks(np.arange(0, policy.shape[0], 5))
    for ax in list(axes.flat)[panels:]:
        ax.set_visible(False)
    fig.suptitle("Jack's Car Rental — Example 4.2")
    fig.tight_layout()
    fig.savefig(output / "jacks_car_rental.png", dpi=160)
    plt.close(fig)
    for index, sweeps in enumerate(evaluation_sweeps):
        print(f"Policy {index}: evaluated in {sweeps} sweeps")
    print("Policy stable")
    print(f"Saved results and policy plot to {output}")
