from __future__ import annotations

from typing import Mapping
from typing import NamedTuple
from typing import Protocol

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


History = list[Period]
Allocation = dict[Firm, list[Customer]]
