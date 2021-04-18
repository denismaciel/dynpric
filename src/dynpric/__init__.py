from __future__ import annotations

from dynpric.types import Demand
from dynpric.types import DemandRealized
from dynpric.types import Firm
from dynpric.types import History
from dynpric.types import Period
from dynpric.types import PricesSet


def simulate_market(n_periods: int, firms: list[Firm], demand: Demand) -> History:

    history: History = []
    for _ in range(n_periods):
        prices_set: PricesSet = {firm: firm.price for firm in firms}
        realized_demand: DemandRealized = demand.allocate(prices_set)

        history.append(Period(prices_set, realized_demand))

        for firm in firms:
            firm.observe_market(history)

    return history
