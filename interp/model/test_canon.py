from typing import Optional

import pytest
import jax
import jax.numpy as jnp

from flax.core.scope import FrozenVariableDict
from interp.model.canon import canonicalize
from interp.model.gpt_model import Gpt, gpt_call_no_log
from interp.model.model_fixtures import (
    loaded_model,
    tiny_random_model,
    loaded_model_random_params,
    random_model_tiny_mlps,
    random_model_tiny_relu_layer_norm,
    example_seq,
    example_seq_no_begin,
    ModelParams,
)

_ = (
    loaded_model,
    tiny_random_model,
    loaded_model_random_params,
    random_model_tiny_mlps,
    example_seq,
    example_seq_no_begin,
)


def run_for_params(
    model: Gpt,
    params: FrozenVariableDict,
    inp: jnp.ndarray,
    allow_relative_diff: bool = False,
    layers=None,
    heads=None,
    matrices=["qk", "ov"],
    emb: Optional[str] = "t_emb",
    relu: bool = False,
    norm: bool = False,
):
    """The output of the model should be invariant under canonicalization."""

    out_orig = gpt_call_no_log(model, params, inp)

    params_canon = canonicalize(model, params, layers, heads, matrices, emb, relu=relu, norm=norm)

    out_canon = gpt_call_no_log(model, params_canon, inp)

    orig_probs = jax.nn.softmax(out_orig, axis=-1)
    canon_probs = jax.nn.softmax(out_canon, axis=-1)

    diff = jnp.abs(orig_probs - canon_probs)

    # Seeing some differences like one having logit 1e-4 and the other 1e-5;
    # these are ok, softmax to focus on the important bits
    assert jnp.allclose(orig_probs, canon_probs, atol=1e-3) or (allow_relative_diff and (diff / orig_probs < 0.1).all())


def base_test_output_invariant(
    loaded_model: ModelParams,
    example_seq: jnp.ndarray,
    *,
    other_params: Optional[FrozenVariableDict] = None,
    allow_relative_diff: bool = False,
    layers=[0, 1],
    heads=[1, 2, 4],
    matrices=["qk", "ov"],
    emb="t_emb",
):
    model, params = loaded_model
    if other_params is not None:
        params = other_params

    run_for_params(
        model,
        params,
        example_seq,
        allow_relative_diff=allow_relative_diff,
        layers=layers,
        heads=heads,
        matrices=matrices,
        emb=emb,
    )


@pytest.mark.parametrize("layers", [1, [0, 1], None])
def test_output_invariant_layers(tiny_random_model: ModelParams, example_seq: jnp.ndarray, layers):
    return base_test_output_invariant(tiny_random_model, example_seq, layers=layers)


@pytest.mark.parametrize("heads", [3, None, [1, 2, 3]])
def test_output_invariant_heads(tiny_random_model: ModelParams, example_seq: jnp.ndarray, heads):
    return base_test_output_invariant(tiny_random_model, example_seq, heads=heads)


@pytest.mark.parametrize("matrices", [["qk"], [], ["qk", "ov"]])
def test_output_invariant_matrices(tiny_random_model: ModelParams, example_seq: jnp.ndarray, matrices):
    return base_test_output_invariant(tiny_random_model, example_seq, matrices=matrices)


@pytest.mark.parametrize("emb", ["t_emb", "p_emb", None])
def test_output_invariant_emb(tiny_random_model: ModelParams, example_seq: jnp.ndarray, emb):
    return base_test_output_invariant(tiny_random_model, example_seq, emb=emb)


@pytest.mark.parametrize("emb", ["t_emb"])
def test_output_invariant_with_mlps(random_model_tiny_mlps: ModelParams, example_seq: jnp.ndarray, emb):
    model, params = random_model_tiny_mlps
    run_for_params(model, params, example_seq, emb=emb)


@pytest.mark.parametrize("norm", [True])
def test_output_invariant_with_relu_layer_norm(
    random_model_tiny_relu_layer_norm: ModelParams, example_seq: jnp.ndarray, norm
):
    model, params = random_model_tiny_relu_layer_norm
    run_for_params(model, params, example_seq, norm=norm, relu=True, emb=None)


@pytest.mark.skip(reason="too slow in CI. somewhat good to run this locally after change")
@pytest.mark.parametrize("use_loaded_model", [False, True])
def test_output_invariant_loaded(
    loaded_model: ModelParams,
    loaded_model_random_params: FrozenVariableDict,
    example_seq: jnp.ndarray,
    use_loaded_model,
):
    return base_test_output_invariant(
        loaded_model,
        example_seq,
        other_params=None if use_loaded_model else loaded_model_random_params,
        allow_relative_diff=use_loaded_model,
    )
