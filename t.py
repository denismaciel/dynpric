from dynpric.priors import Belief
from dynpric.priors import GammaPrior

from dynpric.firms.thompson import TSFixedFirm, thompson

from dynpric import simulate_market
from dynpric import GreedyFirm
from dynpric import InformsDemand
from dynpric import PoissonDemand


def competitive_market():
    N_PERIODS = 100
    history = simulate_market(
        n_periods=N_PERIODS,
        firms=[
            GreedyFirm('greedy2'),
            TSFixedFirm('ts-fixed', beliefs, thompson, 300, N_PERIODS)
        ],
        demand=InformsDemand(),
    )

def poisson():

    beliefs = [
        Belief(29.9, GammaPrior(50, 1)),
        Belief(34.9, GammaPrior(40, 1)),
        Belief(39.9, GammaPrior(20, 1)),
        Belief(44.9, GammaPrior(10, 1)),
        Belief(50, GammaPrior(5, 1)),
    ]

    for b in beliefs: print(b.prior)

    N_PERIODS = 1000
    INVENTORY = 20 * N_PERIODS
    history = simulate_market(
        n_periods=N_PERIODS,
        firms=[
            TSFixedFirm('ts-fixed', beliefs, thompson, INVENTORY, N_PERIODS)
        ],
        demand=PoissonDemand(),
    )

    for b in beliefs:
        print()


if __name__ == '__main__':
    poisson()
