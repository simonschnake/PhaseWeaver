import numpy as np
import pytest

from phase_weaver.model import profile_model as mod

def test_default_background_values():
    p = mod._default_background()

    assert isinstance(p, mod.AsymSuperGaussParams)
    assert p.center == 0.0
    assert p.width == 2.191914604893585e-14
    assert p.skew == 0.0
    assert p.order == 1.0
    assert p.amplitude == 5819.0


def test_default_peak_values():
    p = mod._default_peak()

    assert isinstance(p, mod.AsymSuperGaussParams)
    assert p.center == -1e-14
    assert p.width == 7.21e-16
    assert p.skew == 0.0
    assert p.order == 0.5
    assert p.amplitude == 13000.0


def test_default_peak2_values():
    p = mod._default_peak2()

    assert isinstance(p, mod.AsymSuperGaussParams)
    assert p.center == 1e-14
    assert p.width == 7.21e-16
    assert p.skew == 0.0
    assert p.order == 0.5
    assert p.amplitude == 13000.0


def test_profile_model_state_defaults_and_slots():
    state = mod.ProfileModelState()

    assert state.dt == 1e-16
    assert state.t_max == 1e-13
    assert state.charge is None
    assert isinstance(state.background, mod.AsymSuperGaussParams)
    assert isinstance(state.peak, mod.AsymSuperGaussParams)
    assert state.peak2_enabled is False
    assert isinstance(state.peak2, mod.AsymSuperGaussParams)

    with pytest.raises(AttributeError):
        state.extra = 123


def test_profile_model_init_uses_default_state_and_builds_grid(monkeypatch):
    fake_grid = object()
    calls = []

    class FakeGrid:
        @staticmethod
        def from_dt_tmax(*, dt, t_max, snap_pow2, min_N):
            calls.append(
                {
                    "dt": dt,
                    "t_max": t_max,
                    "snap_pow2": snap_pow2,
                    "min_N": min_N,
                }
            )
            return fake_grid

    monkeypatch.setattr(mod, "Grid", FakeGrid)

    model = mod.ProfileModel()

    assert isinstance(model.state, mod.ProfileModelState)
    assert model.grid is fake_grid
    assert calls == [
        {
            "dt": 1e-16,
            "t_max": 1e-13,
            "snap_pow2": True,
            "min_N": 64,
        }
    ]


def test_profile_model_init_uses_provided_state(monkeypatch):
    fake_grid = object()
    calls = []

    class FakeGrid:
        @staticmethod
        def from_dt_tmax(*, dt, t_max, snap_pow2, min_N):
            calls.append(
                {
                    "dt": dt,
                    "t_max": t_max,
                    "snap_pow2": snap_pow2,
                    "min_N": min_N,
                }
            )
            return fake_grid

    monkeypatch.setattr(mod, "Grid", FakeGrid)

    state = mod.ProfileModelState(dt=2e-16, t_max=3e-13, charge=4.2)
    model = mod.ProfileModel(state)

    assert model.state is state
    assert model.grid is fake_grid
    assert calls == [
        {
            "dt": 2e-16,
            "t_max": 3e-13,
            "snap_pow2": True,
            "min_N": 64,
        }
    ]


def test_compute_profile_without_peak2(monkeypatch):
    t = np.array([-1.0, 0.0, 1.0])
    fake_grid = type("FakeGridInstance", (), {"t": t})()

    profile_calls = []
    asg_calls = []

    class FakeGrid:
        @staticmethod
        def from_dt_tmax(*, dt, t_max, snap_pow2, min_N):
            return fake_grid

    def fake_asymmetric_super_gaussian(
        t_in, *, center, width, skew, order, amplitude
    ):
        asg_calls.append(
            {
                "t": t_in.copy(),
                "center": center,
                "width": width,
                "skew": skew,
                "order": order,
                "amplitude": amplitude,
            }
        )

        if amplitude == 10.0:
            return np.array([1.0, 2.0, 3.0])
        if amplitude == 20.0:
            return np.array([4.0, 5.0, 6.0])

        raise AssertionError("unexpected amplitude")

    def fake_current_profile(*, grid, values, charge):
        profile_calls.append(
            {
                "grid": grid,
                "values": values.copy(),
                "charge": charge,
            }
        )
        return {
            "grid": grid,
            "values": values,
            "charge": charge,
        }

    monkeypatch.setattr(mod, "Grid", FakeGrid)
    monkeypatch.setattr(mod, "asymmetric_super_gaussian", fake_asymmetric_super_gaussian)
    monkeypatch.setattr(mod, "CurrentProfile", fake_current_profile)

    state = mod.ProfileModelState(
        charge=7.5,
        background=mod.AsymSuperGaussParams(
            center=123.0,  # should be ignored by compute_profile for background
            width=1.1,
            skew=0.1,
            order=1.2,
            amplitude=10.0,
        ),
        peak=mod.AsymSuperGaussParams(
            center=-0.25,
            width=2.2,
            skew=-0.2,
            order=0.8,
            amplitude=20.0,
        ),
        peak2_enabled=False,
        peak2=mod.AsymSuperGaussParams(
            center=0.5,
            width=3.3,
            skew=0.3,
            order=1.5,
            amplitude=30.0,
        ),
    )

    model = mod.ProfileModel(state)
    result = model.compute_profile()

    assert len(asg_calls) == 2

    # Background call: center is forced to 0.0
    assert asg_calls[0]["center"] == 0.0
    assert asg_calls[0]["width"] == 1.1
    assert asg_calls[0]["skew"] == 0.1
    assert asg_calls[0]["order"] == 1.2
    assert asg_calls[0]["amplitude"] == 10.0
    np.testing.assert_array_equal(asg_calls[0]["t"], t)

    # Peak call: uses peak.center as-is
    assert asg_calls[1]["center"] == -0.25
    assert asg_calls[1]["width"] == 2.2
    assert asg_calls[1]["skew"] == -0.2
    assert asg_calls[1]["order"] == 0.8
    assert asg_calls[1]["amplitude"] == 20.0
    np.testing.assert_array_equal(asg_calls[1]["t"], t)

    assert len(profile_calls) == 1
    assert profile_calls[0]["grid"] is fake_grid
    assert profile_calls[0]["charge"] == 7.5
    np.testing.assert_array_equal(profile_calls[0]["values"], np.array([5.0, 7.0, 9.0]))

    assert result["grid"] is fake_grid
    assert result["charge"] == 7.5
    np.testing.assert_array_equal(result["values"], np.array([5.0, 7.0, 9.0]))


def test_compute_profile_with_peak2(monkeypatch):
    t = np.array([0.0, 1.0])
    fake_grid = type("FakeGridInstance", (), {"t": t})()

    profile_calls = []
    asg_calls = []

    class FakeGrid:
        @staticmethod
        def from_dt_tmax(*, dt, t_max, snap_pow2, min_N):
            return fake_grid

    returned = {
        11.0: np.array([1.0, 10.0]),
        22.0: np.array([2.0, 20.0]),
        33.0: np.array([3.0, 30.0]),
    }

    def fake_asymmetric_super_gaussian(
        t_in, *, center, width, skew, order, amplitude
    ):
        asg_calls.append(
            {
                "t": t_in.copy(),
                "center": center,
                "width": width,
                "skew": skew,
                "order": order,
                "amplitude": amplitude,
            }
        )
        return returned[amplitude]

    def fake_current_profile(*, grid, values, charge):
        profile_calls.append(
            {
                "grid": grid,
                "values": values.copy(),
                "charge": charge,
            }
        )
        return ("profile", grid, values, charge)

    monkeypatch.setattr(mod, "Grid", FakeGrid)
    monkeypatch.setattr(mod, "asymmetric_super_gaussian", fake_asymmetric_super_gaussian)
    monkeypatch.setattr(mod, "CurrentProfile", fake_current_profile)

    state = mod.ProfileModelState(
        charge=None,
        background=mod.AsymSuperGaussParams(
            center=99.0,
            width=1.0,
            skew=0.0,
            order=1.0,
            amplitude=11.0,
        ),
        peak=mod.AsymSuperGaussParams(
            center=-1.0,
            width=2.0,
            skew=0.1,
            order=1.5,
            amplitude=22.0,
        ),
        peak2_enabled=True,
        peak2=mod.AsymSuperGaussParams(
            center=3.0,
            width=4.0,
            skew=-0.3,
            order=0.7,
            amplitude=33.0,
        ),
    )

    model = mod.ProfileModel(state)
    result = model.compute_profile()

    assert len(asg_calls) == 3

    assert asg_calls[0]["center"] == 0.0
    assert asg_calls[0]["amplitude"] == 11.0

    assert asg_calls[1]["center"] == -1.0
    assert asg_calls[1]["amplitude"] == 22.0

    assert asg_calls[2]["center"] == 3.0
    assert asg_calls[2]["width"] == 4.0
    assert asg_calls[2]["skew"] == -0.3
    assert asg_calls[2]["order"] == 0.7
    assert asg_calls[2]["amplitude"] == 33.0

    expected = np.array([6.0, 60.0])

    assert len(profile_calls) == 1
    assert profile_calls[0]["grid"] is fake_grid
    assert profile_calls[0]["charge"] is None
    np.testing.assert_array_equal(profile_calls[0]["values"], expected)

    assert result[0] == "profile"
    assert result[1] is fake_grid
    np.testing.assert_array_equal(result[2], expected)
    assert result[3] is None