from __future__ import annotations

import collections
import random
from typing import Callable
from typing import Tuple
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Protocol
from typing import Sequence

from numpy.random import exponential
from numpy.random import multinomial
from numpy.random import poisson
from numpy.random import uniform
import numpy as np

Price = float
Quantity = float
Customer = float

class Firm(Protocol):
    name: str
    price: float
    observe_market: Callable[[Quantity, PricesSet], None]


DemandRealized = Dict[Firm, Quantity]
PricesSet = Dict[Firm, Price]


def sample_shares(n):
    samples = [uniform() for _ in range(n)]
    total = sum(samples)
    shares = [x / total for x in samples]
    assert round(sum(shares), 5) == 1, sum(shares)
    return shares


class Period(NamedTuple):
    prices: PricesSet
    demand: DemandRealized


class Demand(Protocol):
    allocate: Callable[[List[Firm]], DemandRealized]


History = List[Period]


def assign_randomly(customers: Sequence[Customer], firms: Sequence[Firm]):
    # allocation = collections.defaultdict(list)

    allocation = {firm: [] for firm in firms}
    for c in customers:
        allocation[random.choice(firms)].append(c)
    return allocation


def compute_demand(firm: Firm, customers: Sequence[Customer]):
    return sum(c >= firm.price for c in customers)


class InformsDemand:
    def __init__(self) -> None:

        # Sample parameters
        self.λ = round(uniform(50, 150))
        self.θ_sho, self.θ_loy = sample_shares(2)

        self.β_sho = round(uniform(5, 15))
        self.β_loy = uniform(1.5, 2) * self.β_sho

    def _sample_customers(self) -> Tuple[np.ndarray]:
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

        self.shoppers = exponential(self.β_sho, n_sho)
        self.loyals = exponential(self.β_loy, n_loy)

        return self.shoppers, self.loyals

    def allocate(self, prices_set: PricesSet):
        self._sample_customers()

        firms = list(prices_set.keys())
        assigned = assign_randomly(self.loyals, firms)

        return {
            firm: sum(c > firm.price for c in customers)
            for firm, customers in assigned.items()
        }


class RandomFirm:
    name: str
    price: Price

    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.min = min
        self.max = max
        self._p = self._set_price()


    def observe_market(self, quantity: Quantity, prices_set: PricesSet) -> None: 
        assert self.price == prices_set[self]
        self._set_price()

    def _set_price(self):
        self._price = round(uniform(self.min, self.max))

    @property
    def price(self):
        return self._price

    def __repr__(self):
        return f"RandomFirm(name={self.name})"




class OLSFirm:
    ...


class GreedyFirm:
    ...


def simulate_market(n_periods: int, firms: List[Firm], demand: Demand) -> History:

    history: History = []
    for i in range(n_periods):
        prices_set: PricesSet = {firm: firm.price for firm in firms}
        realized_demand: DemandRealized = demand.allocate(prices_set)

        # Inform to each firm the results of the market
        # The firm can observe the prices of all other firms
        # but can only see its realized demand
        for firm in firms:
            firm.observe_market(realized_demand[firm], prices_set)
        history.append(Period(prices_set, realized_demand))

    return history


if __name__ == '__main__':
    history = simulate_market(
        n_periods=500,
        firms=[
            RandomFirm('a', 1, 10),
            RandomFirm('b', 1, 10),
            RandomFirm('c', 1, 10),
            RandomFirm('5', 1, 10),
            RandomFirm('slkjd', 1, 10),
        ],
        demand=InformsDemand(),
    )

    print(history)
