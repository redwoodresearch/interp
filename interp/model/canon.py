from typing import Optional, Union, Iterable
from interp.tools.assert_never import assert_never

import numpy as np
import jax.numpy as jnp
from einops import rearrange
from flax.core.scope import FrozenVariableDict
import flax.core.frozen_dict as fd
from interp.tools.svd import canonicalize_matrix_product
from interp.model.gpt_model import Gpt


def canonicalize(
    model: Gpt,
    params: FrozenVariableDict,
    layers: Union[Iterable[int], int, None] = None,
    heads: Union[Iterable[int], int, None] = None,
    matrices: Iterable[str] = ["qk", "ov"],
    emb: Optional[str] = "t_emb",  # TODO doesn't make sense with layer norms
    relu: bool = True,  # split magnitude between input and output vectors on Relu MLPs if present
    norm: bool = False,  # bake as much of the norm as possible into the surrounding layers. For batch norm, this completely obsoletes the layer, whereas for layer norm, it obsoletes weight, bias, and mean subtraction, but not variance effects.
) -> FrozenVariableDict:
    """
    Return a model that is the canonicalized version of the input.

    We can canonicalize any pair of matrices that is always multiplied together, like the Q and K in a head, since their shared space can be rotated arbitrarily.

    We can also canonicalize the embedding matrix by rotating the entire model's embedding space. This however does not make sense with layer norms, and these are simply not implemented here.

    :param orig: model to canonicalize
    :param layers: which layers to canonicalize QK and OV matrices in. If None, will do all.
    :param heads: which heads to canonicalize QK and OV matrices in. If None, will do all.
    :param matrices: which of QK and OV to canonicalize.
    :param emb: whether to canonicalize embedding matrix. If t_emb, will canonicalize the token embedding matrix and apply the appropriate transformation to all the other matrices in the model, and similarly for p_emb.
    """
    assert not set(matrices) - set(["qk", "ov"])

    if layers is None:
        layers = model.num_layers
    if isinstance(layers, int):
        layers = range(layers)

    if heads is None:
        heads = model.num_heads
    if isinstance(heads, int):
        heads = range(heads)

    layers = list(layers)
    heads = list(heads)
    matrices = list(matrices)

    for l in layers:
        for h in heads:
            for mats in matrices:
                A, B, _ = canonicalize_matrix_product(model, params, l, h, mats)
                # deep learnining style big-O right here...
                # note that rebinding each time is (in general) required as we want to use the new params
                params = model.bind(params).set_attn_weights(mats[0], A, l, h)
                params = model.bind(params).set_attn_weights(mats[1], B, l, h)
        if relu and model.use_mlp and model.mlp_act_type == "relu":
            # The Relu function is invariant to scaling around 0, so the input to relu and output from relu
            # can be scaled relative to each other. This splits the scale between in and out matrices.
            i_kernel = jnp.transpose(params["params"][f"blocks_{l}"]["linear1"]["kernel"])
            i_bias = params["params"][f"blocks_{l}"]["linear1"]["bias"]
            o_kernel = params["params"][f"blocks_{l}"]["linear2"]["kernel"]
            magnitudes_half = jnp.sqrt(jnp.linalg.norm(i_kernel, axis=1) * jnp.linalg.norm(o_kernel, axis=1))
            print(i_kernel.shape, i_bias.shape, o_kernel.shape, magnitudes_half.shape)
            params_mut = params.unfreeze()
            params_mut["params"][f"blocks_{l}"]["linear1"]["bias"] = (
                i_bias / jnp.linalg.norm(i_kernel, axis=1) * magnitudes_half
            )
            params_mut["params"][f"blocks_{l}"]["linear1"]["kernel"] = (
                jnp.transpose(i_kernel) / jnp.linalg.norm(i_kernel, axis=1) * magnitudes_half
            )
            params_mut["params"][f"blocks_{l}"]["linear2"]["kernel"] = (
                o_kernel
                / jnp.expand_dims(jnp.linalg.norm(o_kernel, axis=1), axis=1)
                * jnp.expand_dims(magnitudes_half, axis=1)
            )
            params = fd.freeze(params_mut)
        if norm:
            if model.norm_type == "layer_norm":
                assert model.attn_bias, "can only canonicalize norm with attn bias"
                params_mut = params.unfreeze()
                ln1_bias = params["params"][f"blocks_{l}"]["norm1"]["bias"]
                ln1_scale = params["params"][f"blocks_{l}"]["norm1"]["scale"]
                ln2_bias = params["params"][f"blocks_{l}"]["norm2"]["bias"]
                ln2_scale = params["params"][f"blocks_{l}"]["norm2"]["scale"]

                # move bias/scale into the following attn/mlp
                params_mut["params"][f"blocks_{l}"]["norm1"]["scale"] = jnp.ones_like(ln1_scale)
                params_mut["params"][f"blocks_{l}"]["norm1"]["bias"] = jnp.zeros_like(ln1_bias)
                params_mut["params"][f"blocks_{l}"]["norm2"]["scale"] = jnp.ones_like(ln2_scale)
                params_mut["params"][f"blocks_{l}"]["norm2"]["bias"] = jnp.zeros_like(ln2_bias)

                attn_kernel = params["params"][f"blocks_{l}"]["attention"]["attn_weights"]["kernel"]
                print(attn_kernel.shape)
                attn_kernel = jnp.einsum("hw,h->hw", attn_kernel, ln1_scale)
                print(
                    attn_kernel.shape,
                    model.hidden_size,
                    jnp.diag(np.full((model.hidden_size,), 1 / model.hidden_size)).shape,
                )
                # Bake "subtract mean" from layer norm into the following matrix. The mean of a vector is
                # that vector dotted with the all (1/N) vector. We can turn "subtract mean" into a matmul
                # by making a square matrix of all -(1/N) with 1 added on the diagonal.
                attn_kernel -= jnp.einsum(
                    "vh,hw->vw", np.full((model.hidden_size, model.hidden_size), 1 / model.hidden_size), attn_kernel
                )
                params_mut["params"][f"blocks_{l}"]["attention"]["attn_weights"]["kernel"] = attn_kernel

                attn_bias = params["params"][f"blocks_{l}"]["attention"]["attn_weights"]["bias"]
                attn_bias += jnp.einsum("hw,h->w", attn_kernel, ln1_bias)
                params_mut["params"][f"blocks_{l}"]["attention"]["attn_weights"]["bias"] = attn_bias

                mlp_in_kernel = params["params"][f"blocks_{l}"]["linear1"]["kernel"]
                mlp_in_kernel = jnp.einsum("hw,h->hw", mlp_in_kernel, ln2_scale)
                mlp_in_kernel -= jnp.einsum(
                    "vh,hw->vw", np.full((model.hidden_size, model.hidden_size), 1 / model.hidden_size), mlp_in_kernel
                )
                params_mut["params"][f"blocks_{l}"]["linear1"]["kernel"] = mlp_in_kernel

                mlp_in_bias = params["params"][f"blocks_{l}"]["linear1"]["bias"]
                mlp_in_bias += jnp.einsum("hw,h->w", mlp_in_kernel, ln2_bias)
                params_mut["params"][f"blocks_{l}"]["linear1"]["bias"] = mlp_in_bias

                params = fd.freeze(params_mut)

            elif model.norm_type == "none":
                pass
            else:
                raise Exception(f"canon not implemented for {model.norm_type}")

    if emb is not None:
        assert emb in {"t_emb", "p_emb"}
        assert model.norm_type == "none"

        wt = model.bind(params).embedding.token_embedding.embedding
        wp = model.bind(params).embedding.position_embedding.embedding

        x = jnp.transpose(jnp.linalg.svd(wt if emb == "t_emb" else wp, full_matrices=False)[2])
        wp_canon = jnp.einsum("ef, pe -> pf", x, wp)
        wt_canon = jnp.einsum("ef, te -> tf", x, wt)
        params = model.bind(params).set_emb_weights("t_emb", wt_canon)
        params = model.bind(params).set_emb_weights("p_emb", wp_canon)
        for layer in range(model.num_layers):
            [qs, ks, vs] = rearrange(
                jnp.einsum("k h d e, e f -> k h d f", model.bind(params).blocks[layer].attention.get_qkv_mats(), x),
                "k h d f -> k (h d) f",
            )
            o_s = rearrange(
                jnp.einsum("h e d, e f -> h f d", model.bind(params).blocks[layer].attention.get_o_mats(), x),
                "h f d -> f (h d)",
            )
            params = model.bind(params).set_attn_weights("q", qs, l=layer, h_low=0, h_high=model.num_heads)
            params = model.bind(params).set_attn_weights("k", ks, l=layer, h_low=0, h_high=model.num_heads)
            params = model.bind(params).set_attn_weights("v", vs, l=layer, h_low=0, h_high=model.num_heads)
            params = model.bind(params).set_attn_weights("o", o_s, l=layer, h_low=0, h_high=model.num_heads)

            if model.use_mlp:
                linear1 = jnp.einsum("d e, e f -> d f", model.bind(params).blocks[layer].linear1.get_weights(), x)
                linear2 = jnp.einsum("e d, e f -> f d", model.bind(params).blocks[layer].linear2.get_weights(), x)
                params = model.bind(params).set_linear_weights("l1", linear1, l=layer)
                params = model.bind(params).set_linear_weights("l2", linear2, l=layer)

    return params


def auto_canonicalize(model, params):
    """Tao guesses what canonicalizations make sense here and applies them"""
    return canonicalize(
        model,
        params,
        emb="t_emb" if model.norm_type == "none" else None,
        relu=model.use_mlp and model.mlp_act_type == "relu",
        norm=model.norm_type == "layer_norm",
    )
