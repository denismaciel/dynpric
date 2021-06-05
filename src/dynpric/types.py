from __future__ import annotations

import abc
from typing import Any
from typing import Mapping
from typing import NamedTuple
from typing import Protocol
from typing import TypedDict


class Belief(NamedTuple):
    price: Price
    prior: Prior


class Prior(abc.ABC):
    """
    Abstract base class for priors
    """

    @property
    @abc.abstractmethod
    def params(self) -> dict[str, float]:
        ...

    @property
    @abc.abstractmethod
    def expected_value(self) -> float:
        ...

    @abc.abstractmethod
    def sample(self) -> float:
        """
        Samples n values from distribution
        """

    @abc.abstractmethod
    def update(self, result: int) -> None:
        """
        Updates parameters with new information (result)
        """


class PriceLevel(NamedTuple):
    """
    Container of the information that charcterizes the state of a price level
    """

    price: float
    true_prob: float  # probability a customer makes a purchase at the price level


Price = float
Quantity = int
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


class Log(TypedDict):
    Firm: dict[str, Any]


class Demand(Protocol):
    def allocate(self, prices_set: PricesSet) -> DemandRealized:
        ...


History = list[Period]
TrialResults = tuple[list[Log], History]
Allocation = dict[Firm, list[Customer]]
