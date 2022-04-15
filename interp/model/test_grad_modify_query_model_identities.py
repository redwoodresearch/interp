from functools import partial

import pytest
import jax
import jax.numpy as jnp
from flax.core.scope import FrozenVariableDict
from interp.model.gpt_model import Gpt, partial_gpt_call_just_logger

from interp.model.grad_modify_output import Attn, AttnLayer, skip_connect
from interp.model.model_loading import load_model
from interp.tools.attribution_backend_utils import stop_qkv_except
from interp.tools.grad_modify_query import run_queries, Query, ModifierCollectionTreeNode, TargetConf
from interp.tools.grad_modify_query_items import MulConf, ItemConf
from interp.tools.grad_modify_query_utils import compose_trees as ct
from interp.tools.interpretability_tools import check_close_weak
from interp.tools.log import Idxs, KeyIdxs
import interp.tools.optional as op
from interp.model.model_fixtures import (
    loaded_model,
    loaded_model_random_params,
    example_seq,
    example_seq_no_begin,
    ModelParams,
)

_ = loaded_model, loaded_model_random_params, example_seq, example_seq_no_begin


@partial(jax.jit, static_argnames=["model"])
def run_test_queries(model: Gpt, params: FrozenVariableDict, seq: jnp.ndarray):
    # TODO: add some more of these
    return run_queries(
        partial_gpt_call_just_logger(model, params, seq),
        {
            "no_derivs": Query(
                targets=[
                    TargetConf(KeyIdxs.single("blocks.attention.out_by_head", 0)),
                    TargetConf(KeyIdxs("embedding.tok")),
                    TargetConf(KeyIdxs.single("blocks.attention.inp", 1)),
                    TargetConf(KeyIdxs.single("blocks.attention.v", 1)),
                ]
            ),
            "deriv_on_inp": Query(
                targets=[
                    TargetConf(KeyIdxs.single("blocks.attention.v", 1)),
                ],
                modifier_collection_tree=ModifierCollectionTreeNode(
                    MulConf(ItemConf(KeyIdxs.single("blocks.attention.inp", 1)))
                ),
            ),
            "deriv": Query(
                targets=[
                    TargetConf(KeyIdxs.single("blocks.attention.inp", 1)),
                    TargetConf(KeyIdxs.single("blocks.attention.v", 1)),
                ],
                modifier_collection_tree=ct(
                    ModifierCollectionTreeNode(MulConf(ItemConf(KeyIdxs("embedding.tok")))),
                    *stop_qkv_except(Idxs.single(0), "v")
                ),
            ),
            "deriv_just_embed": Query(
                targets=[
                    TargetConf(KeyIdxs.single("blocks.attention.inp", 1)),
                    TargetConf(KeyIdxs.single("blocks.attention.v", 1)),
                ],
                modifier_collection_tree=ct(
                    ModifierCollectionTreeNode(MulConf(ItemConf(KeyIdxs("embedding.tok")))),
                    *stop_qkv_except(Idxs.single(0), "v"),
                    op.unwrap(skip_connect(model, end=AttnLayer(1)))
                ),
            ),
            "deriv_just_out": Query(
                targets=[
                    TargetConf(KeyIdxs.single("blocks.attention.inp", 1)),
                    TargetConf(KeyIdxs.single("blocks.attention.v", 1)),
                ],
                modifier_collection_tree=ModifierCollectionTreeNode(
                    MulConf(ItemConf(KeyIdxs.single("blocks.attention.out_by_head", 0)))
                ),
            ),
        },
    )


@pytest.mark.parametrize("use_loaded_model", [False, True])
def test_check_identities(
    loaded_model: ModelParams,
    loaded_model_random_params: FrozenVariableDict,
    example_seq: jnp.ndarray,
    use_loaded_model: bool,
):
    model, params = loaded_model
    if not use_loaded_model:
        params = loaded_model_random_params

    out = run_test_queries(model, params, example_seq)

    check_close_weak(
        out["deriv_just_out"]["blocks.attention.inp"],
        out["no_derivs"]["blocks.attention.out_by_head"].sum(axis=1),
    )
    assert jnp.allclose(out["deriv_just_embed"]["blocks.attention.inp"], out["no_derivs"]["embedding.tok"])
    check_close_weak(
        out["deriv_just_out"]["blocks.attention.inp"] + out["deriv_just_embed"]["blocks.attention.inp"],
        out["no_derivs"]["blocks.attention.inp"],
    )
    check_close_weak(out["deriv"]["blocks.attention.inp"], out["no_derivs"]["blocks.attention.inp"])
    check_close_weak(out["deriv"]["blocks.attention.v"], out["no_derivs"]["blocks.attention.v"])
    check_close_weak(out["deriv_on_inp"]["blocks.attention.v"], out["no_derivs"]["blocks.attention.v"])
