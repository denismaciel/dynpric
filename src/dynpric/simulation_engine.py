# type: ignore
import multiprocessing
import os
import time
from functools import partial
from itertools import product
from typing import Callable

import pandas as pd
from dynpric.market import Market
from dynpric.market import Price
from dynpric.market import Quantity
from dynpric.seller import Seller


def run_period(t, market, seller, record_state):
    p = seller.choose_price()
    q = market.realize_demand(p)
    seller.observe_demand(q, p)
    return record_state(t, market, seller, p, q)


def run_trial(
    s,
    T,
    reporting_frequency,
    initializer: Callable[[], tuple[Market, Seller]],
    state_recorder: Callable[[int, Market, Seller, Price, Quantity], dict],
) -> dict:
    market, seller = initializer()
    periods = [run_period(t, market, seller, state_recorder) for t in range(T)]
    if s % reporting_frequency == 0:
        print(f'Finished trial number {s}')
    return {'s': s, 'periods': periods}


def trial_factory(
    initializer: Callable[[], tuple[Market, Seller]],
    state_recorder: Callable[[int, Market, Seller, Price, Quantity], dict],
) -> Callable[[int, int], dict]:
    return partial(run_trial, initializer=initializer, state_recorder=state_recorder)


def simulate(S: int, T: int, trial_runner: Callable, execution_mode='parallel'):
    """
    S: number of trials for the simulation
    T: number of time periods each trial has
    """
    print('Starting simulation...')
    start = time.perf_counter()

    REPORTING_FREQUENCY = S // 10

    if execution_mode == 'parallel':
        pool = multiprocessing.Pool(processes=os.cpu_count())
        result = pool.starmap(
            trial_runner, product(range(S), [T], [REPORTING_FREQUENCY])
        )
    elif execution_mode == 'sequential':
        result = [trial_runner(s, T, REPORTING_FREQUENCY) for s in range(S)]

    end = time.perf_counter()
    print(f'Simulation completed in {end - start} seconds')
    return result


def flatten_results(simulation: list[dict]) -> pd.DataFrame:
    """
    Creates a dataframe out of the results of a simulation.

    The expected structure of the simulation object:
    [
      {
         "s": 0,
         "periods": [
             {'t': 0, 'price': 29.99},
             {'t': 1, 'price': 34.99},
             ...
         ]
      },
      ...
    ]
    """
    flat = []
    for trial in simulation:
        for period in trial['periods']:
            flat.append({'s': trial['s'], **period})

    return pd.DataFrame(flat)
