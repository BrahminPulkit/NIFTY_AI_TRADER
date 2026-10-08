from tools.run_label_objective_audit import classify_sign


def test_directional_sign_label_does_not_invent_ambiguous_order():
    assert classify_sign(1) == "WIN"
    assert classify_sign(-1) == "LOSS"
    assert classify_sign(0) == "AMBIGUOUS"
