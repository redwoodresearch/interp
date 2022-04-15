from functools import partial
from typing import Literal, Tuple

import pytest
import jax.numpy as jnp
import jax

from interp.tools.custom_jvp import integrated_gradients_custom_jvp, ablation_custom_jvp, different_function_custom_jvp


def check_linear_is_same(get_wrapper, n: int, m: int, key: jax.random.KeyArray):

    mat = jax.random.uniform(key, (m, n), minval=-1.0, maxval=1.0)
    inp = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    direc = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)

    def f(x):
        return mat @ x

    p_actual, t_actual = jax.jvp(f, (inp,), (direc,))
    p_wrap, t_wrap = jax.jvp(get_wrapper(f), (inp,), (direc,))

    assert jnp.allclose(p_actual, p_wrap)
    assert jnp.allclose(t_actual, t_wrap)


@pytest.mark.parametrize("ig_min_max", [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 1.74)])
@pytest.mark.parametrize("ig_iters", [1, 10])
@pytest.mark.parametrize("n", [1, 6])
@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("seed", [2238])
def test_ig_linear_is_same(ig_min_max: Tuple[float, float], ig_iters: int, n: int, m: int, seed: int):
    check_linear_is_same(
        partial(integrated_gradients_custom_jvp, min_mul=ig_min_max[0], max_mul=ig_min_max[1], n=ig_iters),
        n,
        m,
        jax.random.PRNGKey(seed),
    )


@pytest.mark.parametrize("n", [1, 12])
@pytest.mark.parametrize("m", [1, 7])
@pytest.mark.parametrize("seed", [2239])
def test_ablation_linear_is_same(n: int, m: int, seed: int):
    check_linear_is_same(ablation_custom_jvp, n, m, jax.random.PRNGKey(seed))


@pytest.mark.parametrize("n", [1, 7])
@pytest.mark.parametrize("m", [1, 9])
@pytest.mark.parametrize("seed", [2240])
def test_same_different_func_linear_is_same(n: int, m: int, seed: int):
    check_linear_is_same(lambda f: different_function_custom_jvp(f, f), n, m, jax.random.PRNGKey(seed))


Activation = Literal["gelu", "relu", "softmax"]


def check_non_linear_is_same(get_wrapper, n: int, m: int, o: int, key: jax.random.KeyArray, activation: Activation):

    mat_mid = jax.random.uniform(key, (m, n), minval=-1.0, maxval=1.0)
    mat_out = jax.random.uniform(key, (o, m), minval=-1.0, maxval=1.0)
    inp = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    direc = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)

    activation_func = {"gelu": jax.nn.gelu, "relu": jax.nn.relu, "softmax": jax.nn.softmax}[activation]

    def f(x):
        return mat_out @ activation_func(mat_mid @ x)

    p_actual, t_actual = jax.jvp(f, (inp,), (direc,))
    p_wrap, t_wrap = jax.jvp(get_wrapper(f), (inp,), (direc,))

    assert jnp.allclose(p_actual, p_wrap)
    assert jnp.allclose(t_actual, t_wrap, atol=1e-5)


@pytest.mark.parametrize("ig_iters", [1, 9])
@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize("m", [1, 7])
@pytest.mark.parametrize("o", [1, 9])
@pytest.mark.parametrize("seed", [2238])
@pytest.mark.parametrize(
    "activation",
    [
        "gelu",
        "relu",
        "softmax",
    ],
)
def test_ig_at_1_non_linear_is_same(ig_iters: int, n: int, m: int, o: int, seed: int, activation: Activation):
    check_non_linear_is_same(
        partial(integrated_gradients_custom_jvp, min_mul=1.0, max_mul=1.0, n=ig_iters),
        n,
        m,
        o,
        jax.random.PRNGKey(seed),
        activation,
    )


@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize("m", [1, 7])
@pytest.mark.parametrize("o", [1, 9])
@pytest.mark.parametrize("seed", [2238])
@pytest.mark.parametrize(
    "activation",
    [
        "gelu",
        "relu",
        "softmax",
    ],
)
def test_same_different_func_non_linear_is_same(n: int, m: int, o: int, seed: int, activation: Activation):
    check_non_linear_is_same(
        lambda f: different_function_custom_jvp(f, f),
        n,
        m,
        o,
        jax.random.PRNGKey(seed),
        activation,
    )
