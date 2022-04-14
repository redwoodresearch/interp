import pytest
import jax
import jax.numpy as jnp

from interp.tools.positive_definite import check_valid_positive_semi_def, make_sym, nearest_positive_definite


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("size", range(10, 100, 20))
def test_converges_to_positive_definite(size, seed):
    A = make_sym(jax.random.normal(jax.random.PRNGKey(seed), (size, size)))
    B = nearest_positive_definite(A)

    check_valid_positive_semi_def(B)


def test_no_iters_results_in_nan():
    A = jnp.array([[-1.0, 0], [0, -1]])
    B = nearest_positive_definite(A, iter_limit=0)

    assert jnp.isnan(B).all()


def test_nan_exits():
    A = jnp.array([[1.0, 0], [float("nan"), 1]])
    B = nearest_positive_definite(A)

    assert jnp.isnan(B).all()
