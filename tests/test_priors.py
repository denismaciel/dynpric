import pytest
from dynpric.priors import BetaPrior


def test_beta_params_property():
    b = BetaPrior(1, 1)
    assert b.params == {
        'alpha': 1,
        'beta': 1,
    }

    b = BetaPrior(1987, 0)
    assert b.params == {
        'alpha': 1987,
        'beta': 0,
    }


def test_beta_expected_value():
    b = BetaPrior(1, 1)
    assert b.expected_value == 0.5

    b = BetaPrior(1, 4)
    assert b.expected_value == 0.2


def test_beta_update():
    b = BetaPrior(1, 1)
    b.update(1)
    assert b.params == {
        'alpha': 2,
        'beta': 1,
    }

    b = BetaPrior(1, 1)
    b.update(0)
    assert b.params == {
        'alpha': 1,
        'beta': 2,
    }

    with pytest.raises(ValueError):
        b.update(-8)


def test_beta_sample():
    b = BetaPrior(1, 1)

    x = b.sample()
    assert len(x) == 1
    assert 0 <= x <= 1

    y = b.sample(100)
    assert len(y) == 100
    assert all(0 <= j <= 1 for j in y)
