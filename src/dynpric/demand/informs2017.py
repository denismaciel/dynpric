"""
INFORMS Demand from (cite paper)

Brief description of demand
---------------------------

Simulation parameters
    * Arrival process and demand mechanism are time independent
        * Arrival rate λ = uniform(50, 150)
        * Customer segment shares
            * θ_shoppers
            * θ_loyals
            * θ_scientists
                * γ_phd
                * γ_prof
    * Three customer segments
        * Loyal
        * Shoppers
        * Scientists
            * PhD
            * Professors
"""
import random
from typing import Collection
from typing import NamedTuple
from typing import Sequence

import numpy as np
from dynpric.types import Allocation
from dynpric.types import Customer
from dynpric.types import Firm
from dynpric.types import PricesSet
from numpy.random import exponential
from numpy.random import multinomial
from numpy.random import poisson
from numpy.random import uniform


def sample_shares(n: int) -> list[float]:
    samples: list[float] = [uniform() for _ in range(n)]
    total = sum(samples)
    shares = [x / total for x in samples]
    assert round(sum(shares), 5) == 1, sum(shares)
    return shares


def assign_randomly(
    customers: Collection[Customer], firms: Sequence[Firm]
) -> Allocation:
    allocation: dict[Firm, list[Customer]] = {firm: [] for firm in firms}
    for c in customers:
        selected = random.choice(firms)
        allocation[selected].append(c)
    return allocation


def assign_shoppers(
    shoppers: Collection[Customer], prices_set: PricesSet
) -> Allocation:
    lowest_price = min(prices_set.values())
    cheapest_firms = [
        firm for firm, price in prices_set.items() if price == lowest_price
    ]
    return assign_randomly(shoppers, cheapest_firms)


def merge_allocations(allocations: list[Allocation]) -> Allocation:
    # List of all firms across allocations
    firms: set[Firm] = set()
    for alloc in allocations:
        firms.update(alloc.keys())

    # Initialize empty allocation with all firms
    final_allocation: dict[Firm, list[Customer]] = {firm: [] for firm in firms}

    # Gather customers across all allocations for each firm
    for firm in final_allocation:
        for alloc in allocations:
            if firm in alloc:
                final_allocation[firm].extend(alloc[firm])
    return final_allocation


class TrialDemand(NamedTuple):
    λ: int
    # Share of customers
    θ_sho: float
    θ_loy: float
    # WIP: exponentially distributed with mean β
    β_sho: float
    β_loy: float

    def allocate(self, prices_set: PricesSet) -> dict[Firm, int]:
        firms = list(prices_set.keys())
        shoppers, loyals = customers_in_period_t(self)
        loy_allocation = assign_randomly(loyals, firms)
        sho_allocation = assign_shoppers(shoppers, prices_set)

        final = merge_allocations([loy_allocation, sho_allocation])

        return {
            firm: sum(c > prices_set[firm] for c in customers)
            for firm, customers in final.items()
        }


def customers_in_period_t(demand: TrialDemand) -> tuple[Collection[Customer], ...]:
    n = poisson(demand.λ)
    n_sho, n_loy = multinomial(n, (demand.θ_sho, demand.θ_loy))
    assert n == n_sho + n_loy

    shoppers: np.ndarray = exponential(demand.β_sho, n_sho)
    loyals: np.ndarray = exponential(demand.β_loy, n_loy)

    return shoppers, loyals


class InformsDemand:
    def __init__(self) -> None:
        # Arrival rate
        λ = round(uniform(50, 150))
        # Customer shares
        θ_sho, θ_loy = sample_shares(2)
        # Mean WIP
        β_sho: int = round(uniform(5, 15))
        β_loy: int = round(uniform(1.5, 2) * β_sho)
        self.trial_demand = TrialDemand(λ, θ_sho, θ_loy, β_sho, β_loy)

    def allocate(self, prices_set: PricesSet) -> dict[Firm, int]:
        return self.trial_demand.allocate(prices_set)
