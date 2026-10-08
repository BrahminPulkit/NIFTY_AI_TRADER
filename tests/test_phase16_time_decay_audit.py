from tools.run_phase16_time_decay_audit import intrinsic_value


def test_intrinsic_value_is_side_correct():
    assert intrinsic_value(22000, 21900, "CE") == 100
    assert intrinsic_value(22000, 22100, "CE") == 0
    assert intrinsic_value(22000, 22100, "PE") == 100
    assert intrinsic_value(22000, 21900, "PE") == 0
