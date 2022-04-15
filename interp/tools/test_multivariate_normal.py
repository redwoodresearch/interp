from typing import Iterable, Literal, Tuple, Union
import math
from functools import partial
import einops

import pytest
import jax
import jax.numpy as jnp
from attrs import define, evolve

from interp.tools.multivariate_normal import FlatValueConfig, MultivariateNormal
from interp.tools.interpretability_tools import check_close_weak
from interp.tools.jax_tree_util import AttrsPartiallyStaticDefaultNonStatic
from interp.tools.immutable_dict import get_f


def check_close_weak_normal(l: MultivariateNormal, r: MultivariateNormal, atol=1e-2, norm_div_tol=1e-1):
    check_close_weak(l.mean, r.mean, atol=atol, norm_div_tol=norm_div_tol)
    check_close_weak(l.covariance, r.covariance, atol=atol, norm_div_tol=norm_div_tol)


def random_normal(
    key: jax.random.KeyArray,
    n: int,
    is_zero_cov: Union[Literal[False, True], float] = False,
    always_check_valid: bool = True,
    sampling_eps: float = 1e-3,
    val_multiplier: float = 1.0,
):
    min_v = -1.0 * val_multiplier
    max_v = 1.0 * val_multiplier
    key, subkey = jax.random.split(key)
    init_random_mat = jax.random.uniform(subkey, (n, n), minval=min_v, maxval=max_v)

    if not isinstance(is_zero_cov, float) and is_zero_cov:
        initial_cov = jnp.zeros((n, n))
    else:
        initial_cov = jnp.eye(n)

    key, subkey = jax.random.split(key)
    dist = MultivariateNormal(
        jax.random.uniform(subkey, (n,), minval=min_v, maxval=max_v),
        initial_cov,
        sampling_eps=sampling_eps,
        always_check_valid=always_check_valid,
    ).lin_op(lambda x: init_random_mat @ x)

    key, subkey = jax.random.split(key)
    if isinstance(is_zero_cov, float):
        zero = jax.random.bernoulli(subkey, p=jnp.array(is_zero_cov), shape=(n,))
        dist = dist.normal_like(dist.mean, dist.covariance.at[zero].set(0.0).at[:, zero].set(0.0))

    return dist


jitted_random_normal = jax.jit(random_normal, static_argnames=["n", "is_zero_cov", "always_check_valid"])


@jax.tree_util.register_pytree_node_class
@define
class ExtraAndOperated(AttrsPartiallyStaticDefaultNonStatic):
    extra: jnp.ndarray
    operated: jnp.ndarray

    def apply(self, f):
        return evolve(self, operated=f(self.operated))

    @staticmethod
    def wrap(f):
        return lambda x: x.apply(f)

    @staticmethod
    def get_o(x):
        return x.operated

    @staticmethod
    def get_e(x):
        return x.extra

    @staticmethod
    def combine(new, orig):
        return ExtraAndOperated(orig.extra, new)


def extra_wrap(x: MultivariateNormal, extra: int, operated_shape=(-1,)):
    config, _ = FlatValueConfig.from_tree(
        ExtraAndOperated(jnp.zeros(extra), jnp.zeros(x.size - extra).reshape(*operated_shape))
    )
    return evolve(x, flat_value_config=config)


def test_linear_operator_scale():
    means = jnp.array([1.0, 8.4, 8.0, -2.0])

    dist = extra_wrap(MultivariateNormal(means, jnp.eye(4, 4), always_check_valid=True), 1)
    linear_operator_arr = lambda x: x * 3
    linear_operator = ExtraAndOperated.wrap(linear_operator_arr)
    iters = 100_000

    expected_dist = MultivariateNormal(
        means.at[1:].mul(3.0), jnp.eye(4, 4).at[1:, 1:].mul(3.0 ** 2), always_check_valid=True
    )

    check_close_weak_normal(dist.lin_op(linear_operator), expected_dist)
    check_close_weak_normal(
        dist.monte_carlo_non_linearity(jax.random.PRNGKey(0), lambda x, _: linear_operator(x), iters=iters),
        expected_dist,
    )
    check_close_weak_normal(
        dist.monte_carlo_non_linearity(
            jax.random.PRNGKey(0),
            lambda x, _: linear_operator_arr(x),
            iters=iters,
            sample_selector=ExtraAndOperated.get_o,
            combine=ExtraAndOperated.combine,
        ),
        expected_dist,
    )


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("m", [1, 5])
@pytest.mark.parametrize("extra", [2])
@pytest.mark.parametrize("is_zero_cov", [False, True, 0.5])
@pytest.mark.parametrize("seed", [3])
def test_linear_operators_random(n: int, m: int, extra: int, is_zero_cov, seed: int):
    key = jax.random.PRNGKey(seed)
    total = n + extra

    key, subkey = jax.random.split(key)
    dist = extra_wrap(random_normal(subkey, total, is_zero_cov), extra)

    key, subkey = jax.random.split(key)
    operator_mat = jax.random.uniform(subkey, (m, n), minval=-1.0, maxval=1.0)

    linear_operator = ExtraAndOperated.wrap(lambda x: operator_mat @ x)
    iters = 100_000

    linear = dist.lin_op(linear_operator)
    monte_carlo = dist.monte_carlo_non_linearity(key, lambda x, _: linear_operator(x), iters=iters)
    monte_carlo_select = dist.monte_carlo_non_linearity(
        jax.random.PRNGKey(0),
        lambda x, _: operator_mat @ x,
        iters=iters,
        sample_selector=ExtraAndOperated.get_o,
        combine=ExtraAndOperated.combine,
    )

    check_close_weak_normal(linear, monte_carlo)
    check_close_weak_normal(linear, monte_carlo_select)

    if not isinstance(is_zero_cov, float) and is_zero_cov:
        assert (jnp.abs(linear.covariance) == 0.0).all()


@pytest.mark.parametrize("n", [1, 15])
@pytest.mark.parametrize("extra", [0, 2])
@pytest.mark.parametrize("seed", [4])
def test_add_random(n: int, extra: int, seed: int):
    key = jax.random.PRNGKey(seed)
    total = n + extra

    key, subkey = jax.random.split(key)
    dist = extra_wrap(random_normal(subkey, total), extra)

    key, subkey = jax.random.split(key)
    addr = jax.random.uniform(subkey, (n,), minval=-1.0, maxval=1.0)

    operator = ExtraAndOperated.wrap(lambda x: x + addr)
    iters = 100_000

    check_close_weak_normal(
        dist.add(operator), dist.monte_carlo_non_linearity(key, lambda x, _: operator(x), iters=iters)
    )


@pytest.mark.parametrize("n", [1, 15])
@pytest.mark.parametrize("extra", [0, 2])
@pytest.mark.parametrize("seed", [5])
def test_set_random(n: int, extra: int, seed: int):
    key = jax.random.PRNGKey(seed)
    total = n + extra

    key, subkey = jax.random.split(key)
    dist = extra_wrap(random_normal(subkey, total), extra)

    key, subkey = jax.random.split(key)
    should_set = jax.random.bernoulli(subkey, shape=(n,))

    key, subkey = jax.random.split(key)
    set_to = jax.random.uniform(subkey, shape=(n,))

    def set_items(val, set_to):
        return val.apply(lambda x: jnp.where(should_set, set_to, x))

    iters = 100_000

    check_close_weak_normal(
        dist.set(set_to, set_items), dist.monte_carlo_non_linearity(key, lambda val, _: set_items(val, set_to), iters)
    )


def test_one_sample():
    n = 6
    m = 7
    extra = 5
    seed = 8

    key = jax.random.PRNGKey(seed)
    total = n + extra

    key, subkey = jax.random.split(key)
    dist = extra_wrap(random_normal(subkey, total, True), extra)

    key, subkey = jax.random.split(key)
    operator_mat = jax.random.uniform(subkey, (m, n), minval=-1.0, maxval=1.0)

    linear_operator = ExtraAndOperated.wrap(lambda x: operator_mat @ x)
    iters = 1

    linear = dist.lin_op(linear_operator)
    monte_carlo = dist.monte_carlo_non_linearity(key, lambda x, _: linear_operator(x), iters)
    check_close_weak(linear.mean, monte_carlo.mean)
    assert (monte_carlo.covariance == 0.0).all()


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("m", [1, 5])
@pytest.mark.parametrize("extra", [2])
@pytest.mark.parametrize("seed", [6])
@pytest.mark.parametrize("few_iters", [False, True])
@pytest.mark.parametrize("tweak_sampling", [False, True])
def test_softmax(n, m, extra, seed, few_iters, tweak_sampling):
    if few_iters and (n == 1 or m == 1):
        return

    key = jax.random.PRNGKey(seed)
    total = n * m + extra

    iters = 2 if few_iters else 100_000

    key, subkey = jax.random.split(key)
    dist = extra_wrap(random_normal(subkey, total, False), extra, operated_shape=(n, m))

    def f(x: jnp.ndarray):
        return jax.nn.softmax(x).sum(axis=0)

    key, subkey = jax.random.split(key)
    samples = dist.sample(subkey, (iters,))
    out = jnp.concatenate(
        [samples[:, :extra], jax.vmap(f)(samples[:, extra:].reshape(-1, n, m)).reshape(-1, m)], axis=-1
    )

    mean = out.mean(axis=0)
    cov = jnp.broadcast_to(jnp.cov(out, rowvar=False), (m + extra, m + extra))

    key, subkey = jax.random.split(key)
    wrap_f = lambda x, _: f(x)
    if tweak_sampling:
        new_dist = dist.monte_carlo_non_linearity_operate_gaussian(
            subkey,
            wrap_f,
            lambda x: x.lin_op(lambda x: x * 1.5),
            iters=iters,
            sample_selector=ExtraAndOperated.get_o,
            combine=ExtraAndOperated.combine,
        )
    else:
        new_dist = dist.monte_carlo_non_linearity(
            subkey,
            wrap_f,
            iters=iters,
            sample_selector=ExtraAndOperated.get_o,
            combine=ExtraAndOperated.combine,
        )
    new_dist.check_valid()

    if not few_iters:
        check_close_weak_normal(MultivariateNormal(mean, cov), new_dist, atol=1e-3, norm_div_tol=3e-2)


l_mul_selector = lambda x: x["vals"]["l"][0][0]
r_mul_selector = lambda x: x["vals"]["r"]
mul_combine = lambda new, orig: {"new": new, "orig": orig}


@partial(jax.jit, static_argnames=["l_axes_names", "r_axes_names", "out_axes_names"])
def jit_mul_select(
    dist: MultivariateNormal,
    l_axes_names: Tuple[Union[int, str]],
    r_axes_names: Tuple[Union[int, str]],
    out_axes_names: Tuple[Union[int, str]],
) -> MultivariateNormal:
    return dist.mul_select(
        l_mul_selector, list(l_axes_names), r_mul_selector, list(r_axes_names), list(out_axes_names), mul_combine
    )


def run_test_mul(
    extra: int,
    l_shape: Tuple[int, ...],
    l_einsum: Iterable[int],
    r_shape: Tuple[int, ...],
    r_einsum: Iterable[int],
    out_einsum: Iterable[int],
    seed: int,
    is_zero_cov: Union[Literal[False, True], float] = False,
    val_multiplier: float = 1.0,
    few_iters: bool = False,
):
    key = jax.random.PRNGKey(seed)

    l_size = math.prod(l_shape)
    r_size = math.prod(r_shape)

    key, subkey = jax.random.split(key)
    dist = random_normal(
        subkey, extra + l_size + r_size, is_zero_cov=is_zero_cov, val_multiplier=val_multiplier
    ).lin_op(
        # obfuscate pytree some for testing
        lambda x: {
            "extra2": x[:extra],
            "vals": {
                "l": [(x[extra : l_size + extra].reshape(l_shape),)],
                "extra": x[:extra],
                "extra_": x[:extra],
                "r": x[l_size + extra : r_size + l_size + extra].reshape(r_shape),
            },
            "extra": x[:extra],
        }
    )

    l_einsum_tup = tuple(l_einsum)
    r_einsum_tup = tuple(r_einsum)
    out_einsum_tup = tuple(out_einsum)

    exact_dist = jit_mul_select(evolve(dist, always_check_valid=False), l_einsum_tup, r_einsum_tup, out_einsum_tup)
    exact_dist.check_valid()

    key, subkey = jax.random.split(key)

    monte_dist = dist.monte_carlo_non_linearity(
        subkey,
        lambda vals, _: jnp.einsum(vals[0], l_einsum_tup, vals[1], r_einsum_tup, out_einsum_tup),
        iters=2 if few_iters else 100_000,
        sample_selector=lambda x: (l_mul_selector(x), r_mul_selector(x)),
        combine=mul_combine,
    )

    assert exact_dist.flat_value_config == monte_dist.flat_value_config
    assert (FlatValueConfig.from_tree(exact_dist.mean_as()["orig"])[1] == dist.mean).all()
    assert (FlatValueConfig.from_tree(monte_dist.mean_as()["orig"])[1] == dist.mean).all()

    if not few_iters:
        check_close_weak_normal(exact_dist, monte_dist)
        check_close_weak_normal(exact_dist.lin_op(get_f("new")), monte_dist.lin_op(get_f("new")))

    assert (exact_dist.covariance[monte_dist.covariance == 0.0] == 0.0).all()


def test_mul_minimal():
    run_test_mul(extra=0, l_shape=(1,), l_einsum=[0], r_shape=(1,), r_einsum=[1], out_einsum=[0, 1], seed=4)


def test_mul_minimal_zero():
    run_test_mul(
        extra=0, l_shape=(1,), l_einsum=[0], r_shape=(1,), r_einsum=[1], out_einsum=[0, 1], seed=4, is_zero_cov=True
    )


def test_mul_few_iters():
    for seed in range(8):
        run_test_mul(
            extra=1,
            l_shape=(1,),
            l_einsum=[0],
            r_shape=(1,),
            r_einsum=[1],
            out_einsum=[0, 1],
            seed=seed,
            few_iters=True,
        )


def test_mul_tiny():
    run_test_mul(extra=0, l_shape=(3,), l_einsum=[0], r_shape=(2,), r_einsum=[1], out_einsum=[0, 1], seed=5)


def test_mul_tiny_extra():
    run_test_mul(extra=2, l_shape=(4,), l_einsum=[0], r_shape=(3,), r_einsum=[1], out_einsum=[1, 0], seed=12)


def test_mul_tiny_extra_zeros():
    run_test_mul(
        extra=2, l_shape=(4,), l_einsum=[0], r_shape=(3,), r_einsum=[1], out_einsum=[1, 0], seed=15, is_zero_cov=0.6
    )


def test_mul_tiny_full_reduce():
    run_test_mul(extra=2, l_shape=(4,), l_einsum=[0], r_shape=(3,), r_einsum=[1], out_einsum=[], seed=12)


def test_mul_small():
    run_test_mul(
        extra=2,
        l_shape=(2, 4, 3),
        l_einsum=[2, 1, 0],
        r_shape=(4, 2, 1),
        r_einsum=[1, 2, 3],
        out_einsum=[3, 1],
        seed=6,
    )


def test_mul_small_zeros():
    run_test_mul(
        extra=2,
        l_shape=(2, 4, 3),
        l_einsum=[2, 1, 0],
        r_shape=(4, 2, 1),
        r_einsum=[1, 2, 3],
        out_einsum=[3, 1],
        seed=9,
        is_zero_cov=0.6,
    )


def test_mul_single_l():
    run_test_mul(
        extra=3,
        l_shape=(3,),
        l_einsum=[0],
        r_shape=(5, 3, 4),
        r_einsum=[1, 2, 3],
        out_einsum=[2, 3, 0, 1],
        seed=23,
    )


def test_mul_single_r():
    run_test_mul(
        extra=2,
        l_shape=(3, 5, 4),
        l_einsum=[0, 1, 2],
        r_shape=(3,),
        r_einsum=[3],
        out_einsum=[2, 3, 0, 1],
        seed=11,
    )


def test_mul_permute():
    run_test_mul(
        extra=7,
        l_shape=(1, 2, 3),
        l_einsum=[0, 1, 2],
        r_shape=(2, 3, 1),
        r_einsum=[3, 4, 5],
        out_einsum=[5, 4, 3, 0, 1, 2],
        seed=17,
    )


def test_static_equal():
    key = jax.random.PRNGKey(7)
    key, subkey = jax.random.split(key)
    dist_l = extra_wrap(random_normal(subkey, 8), 1)
    _, tree_def_l = jax.tree_util.tree_flatten(dist_l)
    key, subkey = jax.random.split(key)
    dist_r = extra_wrap(random_normal(subkey, 8), 1)
    _, tree_def_r = jax.tree_util.tree_flatten(dist_r)

    assert tree_def_l == tree_def_r
    assert hash(tree_def_l) == hash(tree_def_r)


@pytest.mark.parametrize("n", [1, 3, 34])
@pytest.mark.parametrize("use_setter", [False, True])
@pytest.mark.parametrize("seed", [8])
def test_condition(n, use_setter, seed):
    key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    dist = random_normal(subkey, n)

    selector = lambda x: x[n // 2 :]

    key, subkey = jax.random.split(key)
    new_val = dist.lin_op(selector).sample(subkey, ())

    cond_dist = dist.condition(selector, new_val, (lambda x, val: x.at[n // 2 :].set(val)) if use_setter else None)

    if use_setter:
        assert (cond_dist.mean[n // 2 :] == new_val).all()
        assert (cond_dist.covariance[n // 2 :] == 0.0).all()
        assert (cond_dist.covariance[:, n // 2 :] == 0.0).all()
    else:
        assert jnp.allclose(cond_dist.mean[n // 2 :], new_val, atol=1e-6)
        assert jnp.allclose(cond_dist.covariance[n // 2 :], 0.0, atol=1e-5)
        assert jnp.allclose(cond_dist.covariance[:, n // 2 :], 0.0, atol=1e-5)

    cov_1_1 = dist.covariance[: n // 2, : n // 2]
    cov_1_2 = dist.covariance[: n // 2, n // 2 :]
    cov_2_2 = dist.covariance[n // 2 :, n // 2 :]

    new_mean_1 = dist.mean[: n // 2] + jnp.einsum(
        "o t, t T, T", cov_1_2, jnp.linalg.pinv(cov_2_2), new_val - dist.mean[n // 2 :]
    )

    new_cov_1_1 = cov_1_1 - jnp.einsum("o t, t T, O T -> o O", cov_1_2, jnp.linalg.pinv(cov_2_2), cov_1_2)

    assert jnp.allclose(cond_dist.mean[: n // 2], new_mean_1)
    assert jnp.allclose(cond_dist.covariance[: n // 2, : n // 2], new_cov_1_1, atol=1e-6)


@pytest.mark.parametrize("n", [3, 34])
@pytest.mark.parametrize("n_to_mix", [2, 8])
@pytest.mark.parametrize("use_setter", [False, True])
@pytest.mark.parametrize("seed", [2, 9])
def test_condition_mixture(n, n_to_mix, use_setter, seed):
    if n_to_mix >= n:
        return

    key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    dist = random_normal(subkey, n)

    selector = lambda x: x[n // 2 :]
    setter = (lambda x, val: x.at[n // 2 :].set(val)) if use_setter else None

    key, subkey = jax.random.split(key)
    new_vals = dist.lin_op(selector).sample(subkey, (n_to_mix,))
    key, subkey = jax.random.split(key)
    logits = jax.random.uniform(subkey, (n_to_mix,))
    weights = jax.nn.softmax(logits)

    cond_dist = dist.condition_mixture(selector, new_vals, weights, setter=setter)

    key, *subkeys = jax.random.split(key, n_to_mix + 1)

    iters_per = 100_000
    vals_per = jnp.concatenate(
        [
            dist.condition(selector, new_val, setter=setter).sample(subkey, (iters_per,))
            for new_val, subkey in zip(new_vals, subkeys)
        ]
    )
    rep_weights = einops.repeat(weights, "n -> n iters", iters=iters_per).flatten()

    mean = jnp.average(vals_per, axis=0, weights=rep_weights)
    cov = jnp.broadcast_to(jnp.cov(vals_per, rowvar=False, bias=True, aweights=rep_weights), cond_dist.covariance.shape)

    check_close_weak(cond_dist.mean, mean, atol=1e-3, norm_div_tol=1e-2)
    check_close_weak(cond_dist.covariance, cov, atol=1e-3, norm_div_tol=1e-2)


@pytest.mark.parametrize("n", [1, 15])
@pytest.mark.parametrize("extra", [2])
@pytest.mark.parametrize("is_zero_cov", [False, True, 0.5])
@pytest.mark.parametrize("seed", [5])
def test_exp_log_same(n, extra, is_zero_cov, seed):
    key = jax.random.PRNGKey(seed)
    total = n + extra

    key, subkey = jax.random.split(key)
    dist = extra_wrap(random_normal(subkey, total, is_zero_cov), extra)

    dist_after = dist.exp(ExtraAndOperated.get_o, ExtraAndOperated.combine).log(
        ExtraAndOperated.get_o, ExtraAndOperated.combine
    )

    check_close_weak_normal(dist, dist_after, atol=1e-5, norm_div_tol=0.0)
    assert (dist_after.covariance[dist.covariance == 0.0] == 0.0).all()


@pytest.mark.parametrize("n", [1, 15])
@pytest.mark.parametrize("extra", [2])
@pytest.mark.parametrize("is_zero_cov", [False, True, 0.5])
@pytest.mark.parametrize("seed", [5])
def test_exp_monte(n, extra, is_zero_cov, seed):
    key = jax.random.PRNGKey(seed)
    total = n + extra

    key, subkey = jax.random.split(key)
    dist = extra_wrap(random_normal(subkey, total, is_zero_cov).lin_op(lambda x: x / 2), extra)

    exp_dist = dist.exp(ExtraAndOperated.get_o, ExtraAndOperated.combine)

    exp_monte_carlo = dist.monte_carlo_non_linearity(
        jax.random.PRNGKey(2),
        lambda x, _: jnp.exp(x),
        iters=100_000,
        sample_selector=ExtraAndOperated.get_o,
        combine=ExtraAndOperated.combine,
    )

    check_close_weak(
        exp_dist.covariance_as().operated.extra, exp_monte_carlo.covariance_as().operated.extra, norm_div_tol=5e-2
    )

    check_close_weak_normal(exp_dist.lin_op(ExtraAndOperated.get_o), exp_monte_carlo.lin_op(ExtraAndOperated.get_o))
    check_close_weak_normal(exp_dist.lin_op(ExtraAndOperated.get_e), exp_monte_carlo.lin_op(ExtraAndOperated.get_e))

    assert (exp_dist.covariance[exp_monte_carlo.covariance == 0.0] == 0.0).all()
