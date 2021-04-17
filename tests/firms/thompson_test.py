from dynpric.firms.thompson import constraint_price_prob_is_positive


def test_constraint_price_prob_is_positive():
    assert constraint_price_prob_is_positive(1) == [
        ((-1,), 0),
    ]
    assert constraint_price_prob_is_positive(2) == [
        ((-1, 0), 0),
        ((0, -1), 0),
    ]
    assert constraint_price_prob_is_positive(4) == [
        ((-1, 0, 0, 0), 0),
        ((0, -1, 0, 0), 0),
        ((0, 0, -1, 0), 0),
        ((0, 0, 0, -1), 0),
    ]
