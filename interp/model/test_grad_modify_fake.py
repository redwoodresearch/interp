import pytest
import jax
import jax.numpy as jnp

from interp.model.gpt_model import Gpt, partial_gpt_call
from interp.model.grad_modify_fake import ablation_softmax_probs, integrated_gradients_softmax_probs
from interp.tools.log import Idxs, KeyIdxs
from interp.tools.grad_modify_query import ModifierCollectionTreeNode, Query, run_queries, TargetConf
from interp.model.model_fixtures import tiny_random_model, example_seq, example_seq_no_begin, ModelParams
from interp.tools.interpretability_tools import (
    cross_entropy_ablation_log_probs,
    get_cross_entropy_integrated_gradients_log_probs,
    losses_runner_log,
)

_ = tiny_random_model, example_seq, example_seq_no_begin


def check_equal_with_tree(
    model_params: ModelParams, example_seq: jnp.ndarray, root: ModifierCollectionTreeNode, fwd: bool = True
):
    model, params = model_params

    targets = [
        TargetConf(KeyIdxs("blocks.attention.out_by_head", Idxs.all())),
        TargetConf(KeyIdxs("cross_entropy.log_probs")),
    ]

    vals = run_queries(
        losses_runner_log(
            partial_gpt_call(model, params, config=Gpt.CallConfig(log_finish=False)), example_seq, example_seq
        ),
        {
            "with_tree": Query(targets=targets, modifier_collection_tree=root),
            "no_tree": Query(targets=targets),
        },
        use_fwd=fwd,
    )

    def assert_eq(x, y):
        assert (x == y).all()

    jax.tree_map(assert_eq, *vals.values())


@pytest.mark.parametrize("fwd", [False, True])
def test_ig_attn_probs(tiny_random_model: ModelParams, example_seq: jnp.ndarray, fwd):
    check_equal_with_tree(tiny_random_model, example_seq, integrated_gradients_softmax_probs(), fwd)


def test_ablation_attn_probs(tiny_random_model: ModelParams, example_seq: jnp.ndarray):
    check_equal_with_tree(tiny_random_model, example_seq, ablation_softmax_probs())


@pytest.mark.parametrize("fwd", [False, True])
def test_ig_log_softmax(tiny_random_model: ModelParams, example_seq: jnp.ndarray, fwd):
    check_equal_with_tree(tiny_random_model, example_seq, get_cross_entropy_integrated_gradients_log_probs(), fwd)


def test_ablation_log_softmax(tiny_random_model: ModelParams, example_seq: jnp.ndarray):
    check_equal_with_tree(tiny_random_model, example_seq, cross_entropy_ablation_log_probs)
