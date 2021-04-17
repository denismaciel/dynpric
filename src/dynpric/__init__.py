from __future__ import annotations

import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Protocol
from typing import Collection
from typing import Sequence
from typing import Tuple

import numpy as np
import sklearn.linear_model
from numpy.random import exponential
from numpy.random import multinomial
from numpy.random import poisson
from numpy.random import uniform

Price = float
Quantity = float
Customer = float


class Firm(Protocol):
    name: str
    @property
    def price(self) -> float:
        ...
    def observe_market(self, history: History) -> None:
        ...

DemandRealized = Mapping[Firm, Quantity]
PricesSet = Mapping[Firm, Price]


class Period(NamedTuple):
    prices: PricesSet
    demand: DemandRealized


class Demand(Protocol):
    def allocate(self, prices_set: PricesSet) -> dict[Firm, int]:
        ...


def sample_shares(n: int) -> List[float]:
    samples: List[float] = [uniform() for _ in range(n)]
    total = sum(samples)
    shares = [x / total for x in samples]
    assert round(sum(shares), 5) == 1, sum(shares)
    return shares


History = List[Period]

Allocation = Dict[Firm, List[Customer]]


class PoissonDemand:
    """
    Poisson demand for a single firm.
    """

    def __init__(self) -> None:
        ...
    @staticmethod
    def allocate(prices_set: PricesSet) -> dict[Firm, int]:
        ((firm, price),) = prices_set.items()

        def λ(p: float) -> float:
            return max(100 - 2*p, 0)
        q: int = np.random.poisson(λ(price))
        return {firm: q}


def assign_randomly(customers: Collection[Customer], firms: Sequence[Firm]) -> Allocation:
    allocation: Dict[Firm, list[Customer]] = {firm: [] for firm in firms}
    for c in customers:
        selected = random.choice(firms)
        allocation[selected].append(c)
    return allocation


class InformsDemand:
    def __init__(self) -> None:

        # Sample parameters
        self.λ = round(uniform(50, 150))
        self.θ_sho, self.θ_loy = sample_shares(2)

        self.β_sho: int = round(uniform(5, 15))
        self.β_loy: int = round(uniform(1.5, 2) * self.β_sho)

    def _sample_customers(self) -> Tuple[np.ndarray, ...]:
        n = poisson(self.λ)
        n_sho, n_loy, = multinomial(
            n,
            (
                self.θ_sho,
                self.θ_loy,
            ),
        )
        assert n == n_sho + n_loy

        n_sho, n_loy,

        self.shoppers: np.ndarray = exponential(self.β_sho, n_sho)
        self.loyals: np.ndarray = exponential(self.β_loy, n_loy)

        return self.shoppers, self.loyals

    def allocate(self, prices_set: PricesSet) -> Dict[Firm, int]:
        self._sample_customers()

        firms = list(prices_set.keys())
        assigned = assign_randomly(self.loyals, firms)

        return {
            firm: sum(c > firm.price for c in customers)
            for firm, customers in assigned.items()
        }


class RandomFirm:
    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.min = min
        self.max = max
        self._set_price()

    def observe_market(self, history: History) -> None:
        prices_set, demand = history[-1]
        assert self.price == prices_set[self]
        self._set_price()

    def _set_price(self) -> None:
        self._price = round(uniform(self.min, self.max))

    @property
    def price(self) -> float:
        return self._price

    def __repr__(self) -> str:
        return f'RandomFirm(name={self.name})'


def train_linear_regression(
    prices: List[float], quantities: List[float]
) -> sklearn.linear_model.LinearRegression:
    model = sklearn.linear_model.LinearRegression()
    model.fit(np.array(prices).reshape(-1, 1), quantities)
    return model


def predict_quantity(
    model: sklearn.linear_model.LinearRegression, prices: List[int]
) -> List[float]:
    price_array = np.array(prices).reshape(-1, 1)
    quantity_array = model.predict(price_array)
    return list(quantity_array)


class OLSFirm:
    """
    - Other competitor's actions are random. Modeled as random noise.
    - 40 periods: exploration
    - 960 periods: 94% of the time: regular exploitation; 5% exploration;
    1% competitive disruption
    - Exploration
        * Price randomly sampled from (0, 100)
    - Exploitation
        * Four models
            * q ~ p
            * q ~ log p
            * log q ~ p
            * log q ~ log p
        * Chooses model with highest R2
        * Optimize through line search
        * Add small perturbation for further exploration
    - Competitive disruption
        * Set price to zero in order to confuse other competitors.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._price = self._explore()

    def _set_price(self, period: int) -> Price:
        if period <= 40:
            return self._explore()
        else:
            rnd = random.uniform(0, 1)
            if rnd < 0.01:
                return self._disrupt_competitors()
            elif rnd < 0.05:
                return self._explore()
            else:
                return self._exploit()

    @staticmethod
    def _explore() -> Price:
        return random.uniform(0, 100)

    def _exploit(self) -> Price:

        model = train_linear_regression(self.prices, self.quantities)

        prices = list(range(0, 100))
        predicted_q = predict_quantity(model, prices)
        revenues = [q * p for q, p in zip(prices, predicted_q)]
        idx_max = revenues.index(max(revenues))
        return prices[idx_max]

    def _disrupt_competitors(self) -> float:
        return 0.0

    def observe_market(self, history: History) -> None:
        self.prices = [prices[self] for prices, _ in history]
        self.quantities = [demand[self] for _, demand in history]
        period = len(history)
        self._price = self._set_price(period)

    @property
    def price(self) -> Price:
        return self._price

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.name})'


class GreedyFirm:
    """
    TODO
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._price = random.uniform(0, 100)

    @staticmethod
    def _set_prices(
        all_prices: List[Price], last_period_prices: List[Price]
    ) -> Price:

        min_price = min(last_period_prices) if last_period_prices else 0
        lower_10 = np.percentile(all_prices, 10) if all_prices else 0
        price = max(lower_10, 5) if min_price < lower_10 else min_price
        return price

    def observe_market(self, history: History) -> None:
        all_prices = [p for prices, _ in history[-30:] for _, p in prices.items()]

        prices, _ = history[-1]
        last_period_prices = [p for _, p in prices.items()]

        self._price = self._set_prices(all_prices, last_period_prices)

    @property
    def price(self) -> Price:
        return self._price

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.name})'


def simulate_market(n_periods: int, firms: List[Firm], demand: Demand) -> History:

    history: History = []
    for _ in range(n_periods):
        prices_set: PricesSet = {firm: firm.price for firm in firms}
        realized_demand: DemandRealized = demand.allocate(prices_set)

        history.append(Period(prices_set, realized_demand))

        for firm in firms:
            firm.observe_market(history)

    return history


if __name__ == '__main__':
    history = simulate_market(
        n_periods=100,
        firms=[
            OLSFirm('ols'),
            GreedyFirm('greedy2'),
        ],
        demand=InformsDemand(),
    )
