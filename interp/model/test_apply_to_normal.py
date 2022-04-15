from typing import Literal, Union
import math

import pytest
import jax
import jax.numpy as jnp
from attrs import evolve

from interp.model.gpt_model import (
    Gpt,
    gpt_new_dist_from_final_embed,
    input_dist_zero_cov,
    gpt_apply_to_normal,
    input_dist_zero_cov_toks,
    gpt_gen_new_dist_to_input,
)
from interp.model.apply_to_normal_utils import pad_dist, un_pad_dist
from interp.model.gpt_modules import AttnApplyToNormalConfig, UnidirectionalAttn
from interp.model.model_fixtures import ModelParams, tiny_tiny_random_model
from interp.tools.multivariate_normal import MultivariateNormal
from interp.tools.immutable_dict import get_f, gets_f, remove
from interp.tools.interpretability_tools import check_close_weak
from interp.tools.test_multivariate_normal import check_close_weak_normal, random_normal

_ = tiny_tiny_random_model


attn_scores_jit = jax.jit(UnidirectionalAttn.attn_scores_apply_to_normal, static_argnames="mask")


def close_by_key(l: MultivariateNormal, r: MultivariateNormal, atol=1e-2, norm_div_tol=1e-1):
    keys = l.mean_as().keys()
    check_close_weak_normal(l, r, atol=atol, norm_div_tol=norm_div_tol)
    for k in keys:
        check_close_weak_normal(l.lin_op(get_f(k)), r.lin_op(get_f(k)), atol=atol, norm_div_tol=norm_div_tol)


def run_test_attn_scores(
    extra: int,
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_size: int,
    mask: bool,
    seed: int,
    is_zero_cov: Union[Literal[False, True], float] = False,
):
    key = jax.random.PRNGKey(seed)

    def get_shape_size(seq):
        shape = (batch_size, num_heads, seq, head_size)
        return shape, math.prod(shape)

    q_shape, q_size = get_shape_size(seq_len_q)
    k_shape, k_size = get_shape_size(seq_len_k)

    key, subkey = jax.random.split(key)
    dist = random_normal(
        subkey, extra + q_size + k_size, is_zero_cov=is_zero_cov, val_multiplier=1.0 / head_size
    ).lin_op(
        lambda x: {
            "extra": x[:extra],
            "q": x[extra : q_size + extra].reshape(q_shape),
            "k": x[q_size + extra : k_size + q_size + extra].reshape(k_shape),
        }
    )

    exact_dist = attn_scores_jit(evolve(dist, always_check_valid=False), mask=mask)
    exact_dist.check_valid()

    key, subkey = jax.random.split(key)

    def attn_scores(x, _):
        return UnidirectionalAttn.attn_scores_static(x["q"], x["k"], head_size, mask=mask)

    monte_dist = dist.monte_carlo_non_linearity(
        subkey,
        attn_scores,
        iters=20_000,
        sample_selector=gets_f(["q", "k"]),
        combine=lambda new, orig: {"attn_scores": new, **remove(orig, ["q", "k"])},
    )

    assert exact_dist.flat_value_config == monte_dist.flat_value_config
    assert set(exact_dist.mean_as().keys()) == {"extra", "attn_scores"}

    close_by_key(exact_dist, monte_dist)

    assert (exact_dist.covariance[monte_dist.covariance == 0.0] == 0.0).all()


def test_attn_scores_minimal():
    for mask in [False, True]:
        run_test_attn_scores(
            extra=0, batch_size=1, num_heads=1, seq_len_q=1, seq_len_k=1, head_size=1, mask=mask, seed=1
        )


def test_attn_scores_minimal_zero():
    for mask in [False, True]:
        run_test_attn_scores(
            extra=0,
            batch_size=1,
            num_heads=1,
            seq_len_q=1,
            seq_len_k=1,
            head_size=1,
            mask=mask,
            seed=5,
            is_zero_cov=True,
        )


def test_attn_scores_extra():
    for mask in [False, True]:
        run_test_attn_scores(
            extra=3, batch_size=1, num_heads=1, seq_len_q=1, seq_len_k=1, head_size=1, mask=mask, seed=182
        )


def test_attn_scores_head_size():
    run_test_attn_scores(
        extra=2, batch_size=1, num_heads=1, seq_len_q=2, seq_len_k=2, head_size=7, mask=True, seed=12980
    )


def test_attn_scores_many():
    run_test_attn_scores(
        extra=7, batch_size=2, num_heads=3, seq_len_q=5, seq_len_k=4, head_size=3, mask=False, seed=12983
    )


def test_attn_scores_many_square():
    run_test_attn_scores(
        extra=9, batch_size=2, num_heads=3, seq_len_q=3, seq_len_k=3, head_size=3, mask=True, seed=12971
    )


def test_attn_scores_many_square_zeros():
    run_test_attn_scores(
        extra=9,
        batch_size=2,
        num_heads=3,
        seq_len_q=3,
        seq_len_k=3,
        head_size=3,
        mask=True,
        seed=12971,
        is_zero_cov=0.8,
    )


jitted_mul_v_probs = jax.jit(UnidirectionalAttn.mul_probs_v_apply_to_normal)


def run_test_mul_v_probs(
    extra: int,
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_size: int,
    seed: int,
    is_zero_cov: Union[Literal[False, True], float] = False,
):
    key = jax.random.PRNGKey(seed)

    v_shape = (batch_size, num_heads, seq_len_k, head_size)
    v_size = math.prod(v_shape)
    probs_shape = (batch_size, num_heads, seq_len_q, seq_len_k)
    probs_size = math.prod(probs_shape)

    key, subkey = jax.random.split(key)
    dist = random_normal(
        subkey, extra + v_size + probs_size, is_zero_cov=is_zero_cov, val_multiplier=1.0 / seq_len_k
    ).lin_op(
        lambda x: {
            "extra": x[:extra],
            "v": x[extra : v_size + extra].reshape(v_shape),
            "attn_probs": x[v_size + extra : probs_size + v_size + extra].reshape(probs_shape),
        }
    )

    exact_dist = jitted_mul_v_probs(evolve(dist, always_check_valid=False))
    exact_dist.check_valid()

    key, subkey = jax.random.split(key)

    def mul_v_probs(x, _):
        return UnidirectionalAttn.mul_probs_v(x["attn_probs"], x["v"])

    monte_dist = dist.monte_carlo_non_linearity(
        subkey,
        mul_v_probs,
        iters=50_000,
        sample_selector=gets_f(["attn_probs", "v"]),
        combine=lambda new, orig: {"combined_v": new, **remove(orig, ["v", "attn_probs"])},
    )

    assert exact_dist.flat_value_config == monte_dist.flat_value_config
    assert set(exact_dist.mean_as().keys()) == {"extra", "combined_v"}

    close_by_key(exact_dist, monte_dist)

    assert (exact_dist.covariance[monte_dist.covariance == 0.0] == 0.0).all()


def test_mul_v_probs_minimal():
    run_test_mul_v_probs(extra=0, batch_size=1, num_heads=1, seq_len_q=1, seq_len_k=1, head_size=1, seed=1)


def test_mul_v_probs_minimal_zero():
    run_test_mul_v_probs(
        extra=0, batch_size=1, num_heads=1, seq_len_q=1, seq_len_k=1, head_size=1, seed=1, is_zero_cov=True
    )


def test_mul_v_probs_extra():
    run_test_mul_v_probs(extra=1, batch_size=1, num_heads=1, seq_len_q=1, seq_len_k=1, head_size=1, seed=12)


def test_mul_v_probs_seq_k():
    run_test_mul_v_probs(extra=1, batch_size=1, num_heads=1, seq_len_q=1, seq_len_k=6, head_size=1, seed=15)


def test_mul_v_probs_extra_zero():
    run_test_mul_v_probs(
        extra=1, batch_size=1, num_heads=1, seq_len_q=1, seq_len_k=1, head_size=1, seed=12, is_zero_cov=True
    )


def test_mul_v_probs_many():
    run_test_mul_v_probs(extra=5, batch_size=2, num_heads=3, seq_len_q=3, seq_len_k=2, head_size=4, seed=2838)


def test_mul_v_probs_many_zeros():
    run_test_mul_v_probs(
        extra=5, batch_size=2, num_heads=3, seq_len_q=3, seq_len_k=2, head_size=4, seed=2838, is_zero_cov=0.7
    )


@pytest.mark.parametrize("seq_len", [1, 3])
@pytest.mark.parametrize("is_zero_cov", [False, True, 0.7])
def test_new_from_final_embed(
    tiny_tiny_random_model: ModelParams, seq_len: int, is_zero_cov: Union[Literal[False, True], float]
):
    seed = 3
    extra = 237
    is_zero_cov = False

    model, params = tiny_tiny_random_model

    model_b = model.bind(params)

    seq_idx = min(1, seq_len - 1)

    key, subkey = jax.random.split(jax.random.PRNGKey(seed))
    dist = random_normal(
        subkey, extra + model.hidden_size * seq_len, is_zero_cov=is_zero_cov, always_check_valid=False
    ).lin_op(
        lambda x: {
            "extra": x[:extra],
            "final_embeds": x[extra : seq_len * model.hidden_size + extra].reshape(1, seq_len, model.hidden_size),
        }
    )

    precise_dist, mean_probs, mean_include_prob = gpt_new_dist_from_final_embed(
        model, params, key, dist, iters=10_000, final_embed_seq_idx=seq_idx
    )
    assert mean_include_prob == 1.0

    def sample_tok_embed(x, key):
        return model_b.embedding.token_embedding.embedding[jax.random.categorical(key, model_b.embedding.unembed(x))]

    selector = lambda x: x["final_embeds"][0, seq_idx]

    key, subkey = jax.random.split(jax.random.PRNGKey(seed))
    monte_dist = dist.monte_carlo_non_linearity(
        subkey,
        sample_tok_embed,
        iters=100_000,
        sample_selector=selector,
        combine=lambda new, orig: {**orig, "new_input_embed_dist": new},
    )

    assert precise_dist.flat_value_config == monte_dist.flat_value_config
    assert set(precise_dist.mean_as().keys()) == {"extra", "final_embeds", "new_input_embed_dist"}

    close_by_key(precise_dist, monte_dist, atol=1e-3, norm_div_tol=3e-2)

    assert (precise_dist.covariance[monte_dist.covariance == 0.0] == 0.0).all()

    key, subkey = jax.random.split(jax.random.PRNGKey(seed))

    monte_mean_probs = jax.nn.softmax(model_b.embedding.unembed(dist.lin_op(selector).sample(subkey, (10_000,)))).mean(
        axis=0
    )

    check_close_weak(mean_probs, monte_mean_probs, norm_div_tol=0.1)


@pytest.mark.parametrize("seq_len", [1, 3, 13])
def test_same_behaviour_on_fixed(tiny_tiny_random_model: ModelParams, seq_len: int):
    seed = 3

    model, params = tiny_tiny_random_model

    model_b = model.bind(params)

    key, subkey = jax.random.split(jax.random.PRNGKey(seed))
    toks = jax.random.randint(
        subkey,
        (
            1,
            seq_len,
        ),
        minval=0,
        maxval=model.vocab_size,
    )

    embeds = model_b.embedding(toks)["tok"]

    dist, _ = gpt_apply_to_normal(
        model,
        params,
        key,
        input_dist_zero_cov(embeds),
        config=Gpt.ApplyToNormalConfig(attn_config=AttnApplyToNormalConfig(1)),
    )
    out = model_b(toks)
    assert (jnp.abs(dist.covariance) == 0.0).all()

    means = dist.mean_as()
    assert set(means.keys()) == {"final_embeds", "input_embeds"}

    assert (means["input_embeds"] == embeds).all()
    actual = model_b.embedding.unembed(means["final_embeds"])

    assert jnp.allclose(actual, out, atol=1e-6)


def test_padded_equiv(tiny_tiny_random_model: ModelParams):
    seed = 3

    model, params = tiny_tiny_random_model

    key, subkey = jax.random.split(jax.random.PRNGKey(seed))
    dist, _ = gpt_apply_to_normal(
        model,
        params,
        subkey,
        input_dist_zero_cov_toks(model, params, jnp.array([[0]])),
        config=Gpt.ApplyToNormalConfig(attn_config=AttnApplyToNormalConfig(1)),
    )
    dist.check_valid()

    config = Gpt.ApplyToNormalConfig(attn_config=AttnApplyToNormalConfig(10_000))

    start_toks = 1
    additional = 2

    inp_dist = None

    for i in range(additional):
        key, subkey = jax.random.split(key)
        inp_dist = gpt_gen_new_dist_to_input(model, params, subkey, dist, iters=1_000)[0].lin_op(
            lambda x: x["input_embeds"]
        )

        key, subkey = jax.random.split(key)
        dist_padded, _ = gpt_apply_to_normal(
            model,
            params,
            subkey,
            pad_dist(inp_dist, pad_to=start_toks + additional),
            config=config,
        )
        dist_padded.check_valid()
        dist_padded_same = un_pad_dist(dist_padded, i + start_toks + 1)
        key, subkey = jax.random.split(key)
        dist_non_padded, _ = gpt_apply_to_normal(model, params, subkey, inp_dist, config=config)
        dist_non_padded.check_valid()

        assert dist_padded_same.flat_value_config == dist_non_padded.flat_value_config
        close_by_key(dist_padded_same, dist_non_padded)

        dist = dist_non_padded

    assert inp_dist is not None

    with pytest.raises(AssertionError):
        key, subkey = jax.random.split(key)
        inp_dist = gpt_gen_new_dist_to_input(model, params, subkey, dist, iters=1_000)[0]
        pad_dist(inp_dist, pad_to=start_toks + additional - 1)
