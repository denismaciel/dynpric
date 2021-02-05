import collections
import random
from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Protocol
from typing import Sequence

import numpy as np
from numpy.random import exponential
from numpy.random import multinomial
from numpy.random import poisson
from numpy.random import uniform
from scipy.special import lambertw

Price = float
Prices = Sequence[Price]


class Firm(Protocol):
    name: str
    price: float
    observe_market: Callable[[int, Prices], None]


Customer = float
MarketAllocation = Dict[Firm, Sequence[Customer]]


def logit_demand(p, prices, α, β):
    assert p in prices
    nominator = np.exp(α + β * p)
    denominator = 1 + sum(np.exp(α + β * p) for p in prices)

    return nominator / denominator


def sample_shares(n):
    samples = [uniform() for _ in range(n)]
    total = sum(samples)
    shares = [x / total for x in samples]
    assert round(sum(shares), 5) == 1, sum(shares)
    return shares


class RandomFirm:
    name: str
    price: Price

    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.min = min
        self.max = max
        self._p = self._set_price()

    def observe_market(self, demand: int, disclosed_prices: Sequence[Price]) -> Price:
        # print(self.name, "sees demands of", demand)
        assert self.price == disclosed_prices[self]
        self._set_price()

    def _set_price(self):
        self._price = round(uniform(self.min, self.max))

    @property
    def price(self):
        return self._price


class InformsDemand:
    def __init__(self) -> None:

        # Sample parameters
        self.λ = round(uniform(50, 150))
        self.θ_sho, self.θ_loy = sample_shares(2)

        # self.θ_sci: float = 0.25
        # self.v_phd: float = 0.7
        # self.v_prof: float = 0.3

        self.β_sho = round(uniform(5, 15))
        self.β_loy = uniform(1.5, 2) * self.β_sho

        # Scientists

    def _sample_customers(self):
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

    def demand(self, firms: Sequence[Firm]):
        self._sample_customers()
        loyals = assign_randomly(self.loyals, firms)
        return loyals


def assign_randomly(
    customers: Sequence[Customer], firms: Sequence[Firm]
) -> MarketAllocation:
    allocation = collections.defaultdict(list)
    for c in customers:
        allocation[random.choice(firms)].append(c)
    return allocation


def compute_demand(firm: Firm, customers: Sequence[Customer]):
    return sum(c >= firm.price for c in customers)


if __name__ == '__main__':

    firms = (RandomFirm('a', 5, 10), RandomFirm('b', 13, 17), RandomFirm('c', 1, 4))
    market = InformsDemand()

    revenue_stream: Dict[Firm, Sequence[float]] = collections.defaultdict(list)

    for _ in range(1000):
        allocation = market.demand(firms)

        for firm, customers in allocation.items():
            disclosed_prices: Dict[Firm, Price] = {firm: firm.price for firm in firms}
            demand = compute_demand(firm, customers)
            revenue_stream[firm].append(firm.price * demand)
            firm.observe_market(demand, disclosed_prices)

    for firm, revenue in sorted(revenue_stream.items(), key=lambda x: x[0].name):
        print(firm.name, sum(revenue))
