from __future__ import annotations

from typing import Callable
from typing import List
from typing import Protocol

import numpy as np
from dynpric import GreedyFirm
from dynpric import History
from dynpric import OLSFirm
from dynpric import RandomFirm
from dynpric.priors import Belief
from dynpric.priors import BetaPrior
from dynpric.priors import GammaPrior
from scipy.optimize import linprog
from scipy.optimize.optimize import OptimizeResult


def find_optimal_price(prices, demand, c) -> OptimizeResult:
    assert len(prices) == len(demand)
    # The reason for the minus sign is that scipy only does minimizations
    objective = [-(p * d) for p, d in zip(prices, demand)]

    # --- Constraints ---
    # 1. Demand is smaller equal than available inventory
    c1 = [demand, c]

    # Sum of probabilities smaller equal one
    c2 = [(1, 1, 1, 1), 1]

    # 3. Probability of picking a price must be or equal to greater than zero
    c3 = [
        [(-1, 0, 0, 0), 0],
        [(0, -1, 0, 0), 0],
        [(0, 0, -1, 0), 0],
        [(0, 0, 0, -1), 0],
    ]

    constraints = [c1, c2, *c3]

    lhs_ineq = []
    rhs_ineq = []

    for lhs, rhs in constraints:
        lhs_ineq.append(lhs)
        rhs_ineq.append(rhs)

    opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, method='revised simplex')

    print(opt.x, demand)
    return opt


Strategy = Callable[[Belief], float]


def thompson(b: Belief) -> float:
    return b.prior.sample()  # type: ignore


def greedy(b: Belief) -> float:
    return b.prior.expected_value  # type: ignore


def estimate_demand(strategy: Strategy, beliefs: List[Belief]) -> List[int]:
    """
    For each price level return an estimated quantity
    """
    parameters = [strategy(belief) for belief in beliefs]
    belief_type = type(beliefs[0].prior)
    if belief_type == BetaPrior:
        return parameters
    elif belief_type == GammaPrior:
        demand = [np.random.poisson(p) for p in parameters]
        # demand = [np.random.poisson(10) for p in parameters]
        return demand
    else:
        raise NotImplementedError


class TSFixedFirm:
    def __init__(
        self,
        name: str,
        beliefs: List[Belief],
        strategy: Strategy,
        inventory: int,
        n_periods: int,
    ) -> None:
        self.name = name
        self.beliefs = beliefs
        self.strategy = strategy
        self.inventory = inventory
        self.c = inventory / n_periods

    @property
    def price(self) -> float:
        demand = estimate_demand(self.strategy, self.beliefs)
        prices = [belief.price for belief in self.beliefs]

        # Given estimated demands for each price level and the inventory
        # constraint, optimize for best price to set
        optimization_result = find_optimal_price(prices, demand, self.c)

        def sample_price(probs, prices) -> float:
            assert len(probs) == len(prices)

            # Ensure probs are always positive
            rounded_probs = np.round(probs, decimals=3)
            if any(p < 0 for p in rounded_probs):
                raise ValueError(rounded_probs)

            # Normalize probs to add up to one
            normalized_probs = np.divide(rounded_probs, np.sum(rounded_probs))
            # normalized_probs = [p / np.sum(rounded_probs) for p in rounded_probs]
            sampled_price = np.random.choice(prices, size=1, p=normalized_probs)
            return float(sampled_price)

        chosen_price = sample_price(optimization_result.x, prices)
        return float(chosen_price)

    def observe_market(self, history: History) -> None:
        last_period = history[-1]
        price_set = last_period.prices[self]
        demand = last_period.demand[self]
        # Update belief
        (belief,) = [belief for belief in self.beliefs if belief.price == price_set]
        belief.prior.update(demand)

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"
