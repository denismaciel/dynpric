from __future__ import annotations

import functools
from typing import Callable
from typing import Sequence

import numpy as np
from dynpric import History
from dynpric.priors import Belief
from dynpric.priors import BetaPrior
from dynpric.priors import GammaPrior
from dynpric.priors import Prior
from scipy.optimize import linprog
from scipy.optimize.optimize import OptimizeResult


LeftHandSide = Sequence[int]
Inequality = tuple[LeftHandSide, int]


def constraint_price_prob_is_positive(n_prices: int) -> Sequence[Inequality]:

    """
    Example
    -------
    For four possible price, results in the following constraint matrix:

       [
            ((-1, 0, 0, 0), 0),
            ((0, -1, 0, 0), 0),
            ((0, 0, -1, 0), 0),
            ((0, 0, 0, -1), 0),
       ]
    """
    range_prices = tuple(range(n_prices))

    def _left_side(j: int) -> LeftHandSide:
        return tuple([-1 * int(j == i) for i in range_prices])

    def _line(j: int) -> Inequality:
        return (_left_side(j), 0)

    return [_line(j) for j in range_prices]


def find_optimal_price(
    prices: list[float], demand: list[int], c: float
) -> OptimizeResult:
    assert len(prices) == len(demand)
    n_prices = len(prices)

    # The reason for the minus sign is that scipy only does minimizations
    objective = [-(p * d) for p, d in zip(prices, demand)]

    # --- Constraints ---
    # 1. Demand is smaller equal than available inventory
    c1 = [demand, c]

    # Sum of probabilities must be <= 1
    c2 = [tuple(1 for _ in range(n_prices)), 1]

    # 3. Probability of picking a price must be or equal to greater than zero
    c3 = constraint_price_prob_is_positive(len(prices))

    constraints = [c1, c2, *c3]

    lhs_ineq = []
    rhs_ineq = []

    for lhs, rhs in constraints:
        lhs_ineq.append(lhs)
        rhs_ineq.append(rhs)

    opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, method='revised simplex')

    print(opt.x, demand)
    return opt


# How to select the value of a prior
SamplingStrategy = Callable[[Belief], float]


class SamplingStrategies:
    @staticmethod
    def thompson(prior: Prior) -> int:
        return int(prior.sample())

    @staticmethod
    def greedy(prior: Prior) -> int:
        return int(prior.expected_value)


@functools.singledispatch
def sample_demand(param: Prior, strategy: SamplingStrategy) -> int:
    raise NotImplementedError


@sample_demand.register(BetaPrior)
def _(belief, strategy):  # type: ignore
    return strategy(belief)


@sample_demand.register(GammaPrior)  # type: ignore
def _(belief, strategy):
    return np.random.poisson(strategy(belief))


def sample_price(probs: Sequence[float], prices: Sequence[float]) -> float:
    """
    The optimization result is a distribution over the possible prices.
    Using such optimal distribution, sample a price.
    """
    assert len(probs) == len(prices)

    # Ensure probs are always positive
    rounded_probs = np.round(probs, decimals=3)

    if any(rounded_probs < 0):
        raise ValueError(rounded_probs)

    # Normalize probs to add up to one
    normalized_probs = np.divide(rounded_probs, np.sum(rounded_probs))
    price_set = np.random.choice(prices, size=1, p=normalized_probs)
    return float(price_set)


class TSFixedFirm:
    def __init__(
        self,
        name: str,
        beliefs: list[Belief],
        strategy: SamplingStrategy,
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
        demand = [sample_demand(belief.prior, self.strategy) for belief in self.beliefs]
        prices = [belief.price for belief in self.beliefs]

        # Given estimated demands for each price level and the inventory
        # constraint, optimize for best price to set
        optimization_result = find_optimal_price(prices, demand, self.c)
        chosen_price = sample_price(optimization_result.x, prices)
        return float(chosen_price)

    def observe_market(self, history: History) -> None:
        last_period = history[-1]
        price_set = last_period.prices[self]
        demand = last_period.demand[self]
        # Update belief
        (belief,) = [belief for belief in self.beliefs if belief.price == price_set]
        belief.prior.update(int(demand))

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name!r})'
