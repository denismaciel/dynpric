from __future__ import annotations

import functools
from typing import Callable
from typing import Sequence

import numpy as np
from dynpric import History
from dynpric.priors import BetaPrior
from dynpric.priors import GammaPrior
from dynpric.types import Belief
from dynpric.types import Prior
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
    prices: list[float], demand: list[int] | list[float], c: float | None = None
) -> OptimizeResult:
    assert len(prices) == len(demand)
    n_prices = len(prices)

    # The reason for the minus sign is that scipy only does minimizations
    objective = [-(p * d) for p, d in zip(prices, demand)]

    # --- Constraints ---
    # 1. Demand is smaller equal than available inventory
    if c is not None:
        c1 = [demand, c]
    else:
        c1 = []

    # Sum of probabilities must be <= 1
    c2 = [tuple(1 for _ in range(n_prices)), 1]

    # 3. Probability of picking a price must be or equal to greater than zero
    c3 = constraint_price_prob_is_positive(len(prices))

    constraints = [c1, c2, *c3]

    constraints = [constr for constr in constraints if constr is not None]

    lhs_ineq = []
    rhs_ineq = []

    for lhs, rhs in constraints:  # type: ignore
        lhs_ineq.append(lhs)  # type: ignore
        rhs_ineq.append(rhs)  # type: ignore

    opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, method='revised simplex')

    return opt


# How to select the value of a prior
SamplingStrategy = Callable[[Prior], float]


class SamplingStrategies:
    @staticmethod
    def thompson(prior: Prior) -> int | float:
        return prior.sample()

    @staticmethod
    def greedy(prior: Prior) -> int | float:
        return prior.expected_value


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
    try:
        price_set = np.random.choice(prices, size=1, p=normalized_probs)
    except ValueError:
        # When inventory is very low (1 or 2), normalized_probs might contain NaNs.
        # Default to picking the lowest price.
        print('NaNs in probs')
        price_set = prices[0]
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


class TSUpdateFirm:
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
        self.n_periods = n_periods
        self.current_period = 0

    @property
    def c(self) -> float:
        return self.inventory / (self.n_periods - self.current_period)

    def observe_market(self, history: History) -> None:
        last_period = history[-1]
        price_set = last_period.prices[self]
        demand = last_period.demand[self]

        if price_set == np.inf:
            assert demand <= 0
            return

        # Update belief
        (belief,) = [belief for belief in self.beliefs if belief.price == price_set]
        belief.prior.update(int(demand))

        # Update inventory
        self.inventory -= demand

        # Udpate current period
        self.current_period += 1

    @property
    def price(self) -> float:

        if self.inventory <= 0:
            return np.inf

        demand = [sample_demand(belief.prior, self.strategy) for belief in self.beliefs]
        prices = [belief.price for belief in self.beliefs]

        # Given estimated demands for each price level and the inventory
        # constraint, optimize for best price to set
        optimization_result = find_optimal_price(prices, demand, self.c)
        chosen_price = sample_price(optimization_result.x, prices)
        return float(chosen_price)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name!r})'


class TSIngoreInventoryFirm:
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
        self.n_periods = n_periods
        self.current_period = 0

    @property
    def price(self) -> float:

        if self.inventory <= 0:
            return np.inf

        demand = [sample_demand(belief.prior, self.strategy) for belief in self.beliefs]
        prices = [belief.price for belief in self.beliefs]
        optimization_result = find_optimal_price(prices, demand)
        chosen_price = sample_price(optimization_result.x, prices)
        return float(chosen_price)

    def observe_market(self, history: History) -> None:
        last_period = history[-1]
        price_set = last_period.prices[self]
        demand = last_period.demand[self]

        if price_set == np.inf:
            assert demand <= 0
            return

        # Update belief
        (belief,) = [belief for belief in self.beliefs if belief.price == price_set]
        belief.prior.update(int(demand))

        # Update inventory
        self.inventory -= demand

        # Udpate current period
        self.current_period += 1

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name!r})'
