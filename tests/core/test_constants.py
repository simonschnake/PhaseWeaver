from phase_weaver.core import constants


def test_frequency_band_constants_are_ordered():
    assert constants.CRISP_MIN_THZ < constants.CIRSP_MAX_THZ
    assert constants.CIRSP_MAX_THZ < constants.IR_MIN_THZ
    assert constants.IR_MIN_THZ < constants.IR_MAX_THZ
    assert constants.CIRSP_MAX_THZ < constants.COMBINE_THZ < constants.IR_MIN_THZ
