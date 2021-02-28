import pytest
from dynpric.priors import BetaPrior


def test_beta_params_property():
    b = BetaPrior(1, 1)
    assert b.params == {
        'α': 1,
        'β': 1,
    }

    b = BetaPrior(1987, 0)
    assert b.params == {
        'α': 1987,
        'β': 0,
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
        'α': 2,
        'β': 1,
    }

    b = BetaPrior(1, 1)
    b.update(0)
    assert b.params == {
        'α': 1,
        'β': 2,
    }

    with pytest.raises(ValueError):
        b.update(-8)


def test_beta_sample():
    b = BetaPrior(1, 1)

    x = b.sample()
    assert 0 <= x <= 1
