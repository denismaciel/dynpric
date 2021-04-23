from __future__ import annotations

from typing import Callable

from dynpric.types import Demand
from dynpric.types import DemandRealized
from dynpric.types import Firm
from dynpric.types import History
from dynpric.types import Log
from dynpric.types import Period
from dynpric.types import PricesSet


def simulate_market(
    n_periods: int,
    firms: list[Firm],
    demand: Demand,
    logger: Callable[[Firm, PricesSet, DemandRealized], dict[str, str | float] | None],
) -> tuple[list[Log], History]:

    history: History = []
    logs: list[Log] = []
    for _ in range(n_periods):
        prices_set: PricesSet = {firm: firm.price for firm in firms}
        realized_demand: DemandRealized = demand.allocate(prices_set)

        history.append(
            Period(
                prices_set,
                realized_demand,
            )
        )

        for firm in firms:
            firm.observe_market(history)

        log = {
            firm: logged_info
            for firm in firms
            if (logged_info := logger(firm, prices_set, realized_demand)) is not None
        }
        logs.append(log)  # type: ignore

    return logs, history
