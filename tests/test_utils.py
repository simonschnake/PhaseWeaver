import numpy as np

from phase_weaver.core.utils import safe_real  # adjust import if needed


def test_safe_real_returns_float_for_real_input():
    x = np.array([1, 2, 3], dtype=np.int32)
    y = safe_real(x)

    assert np.array_equal(y, np.array([1.0, 2.0, 3.0]))
    assert y.dtype == float
    assert not np.iscomplexobj(y)


def test_safe_real_extracts_real_part_for_complex_input():
    x = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    y = safe_real(x)

    assert np.array_equal(y, np.array([1.0, 3.0]))
    assert y.dtype == float
    assert not np.iscomplexobj(y)


def test_safe_real_handles_complex_leakage_tiny_imag():
    x = np.array([1.5 + 1e-12j, -2.0 - 1e-15j], dtype=np.complex128)
    y = safe_real(x)

    assert np.allclose(y, np.array([1.5, -2.0]))
    assert y.dtype == float


def test_safe_real_converts_python_list_to_float_ndarray():
    x = [1, 2.25, 3]
    y = safe_real(x)

    assert isinstance(y, np.ndarray)
    assert np.array_equal(y, np.array([1.0, 2.25, 3.0]))
    assert y.dtype == float


def test_safe_real_preserves_shape():
    x = np.array([[1 + 0j, 2 + 3j], [4 - 5j, 6 + 0j]])
    y = safe_real(x)

    assert y.shape == x.shape
    assert np.array_equal(y, np.array([[1.0, 2.0], [4.0, 6.0]]))


def test_safe_real_works_for_scalar_inputs():
    y1 = safe_real(3)
    y2 = safe_real(3 + 4j)

    assert np.array_equal(y1, np.array(3.0))
    assert np.array_equal(y2, np.array(3.0))
    assert y1.dtype == float
    assert y2.dtype == float