import os
from typing import Dict, List
from functools import partial

import jax.numpy as jnp
import jax

from interp.model.grad_modify_output import Attn, AttnLayer, Embeds, OutputConf, output_tree
from interp.model.gpt_model import gpt_call
from interp.model.model_fixtures import ModelParams, loaded_model
from interp.tools.attribution_backend_utils import fake_to_attn_probs, stop_qkv_except
from interp.tools.grad_modify_query_items import ItemConf, ItemIdx, StopGradConf
from interp.tools.grad_modify_query_utils import compose_trees as ct, compose_trees_maybe_empty
from interp.tools.grad_modify_query import (
    ModifierCollectionTreeNodeStack,
    ModifierCollectionTreeNode,
    Query,
    TargetConf,
    run_query,
)
from interp.tools.indexer import I
from interp.tools.interpretability_tools import (
    batch_tokenize,
    check_close_weak,
    get_interp_tokenizer,
    single_tokenize,
)
from interp.tools.log import Idxs, KeyIdxs
import interp.tools.optional as op
from interp.ui.attribution_backend import AttributionBackend, Edge, Fake
from interp.ui.attribution_backend_comp import LayerType, get_run_model_and_output

_ = loaded_model  # TODO test MLPs


def get_output_log(tree: ModifierCollectionTreeNodeStack, get_outputs=False):
    @partial(jax.jit, static_argnames=["model", "fake_log_probs"])
    def to_output_log(model, params, seqs, seq_idx, target, fake_log_probs: Fake) -> Dict[str, jnp.ndarray]:
        targets = [
            TargetConf(KeyIdxs("loss.loss"), display_name="loss"),
        ]

        if get_outputs:
            targets.extend(
                [
                    TargetConf(KeyIdxs("final_out.logits"), display_name="outputs", idx=(0, seq_idx)),
                    TargetConf(KeyIdxs("log_probs"), display_name="log_probs", idx=(0, seq_idx)),
                ]
            )
        return run_query(
            get_run_model_and_output(
                model,
                params,
                seqs,
                fake_log_probs=fake_log_probs,
                target=jnp.array([seq_idx, target]),
                logit_logprob_or_prob="logprob",
            ),
            Query(
                targets=targets,
                modifier_collection_tree=tree,
                use_fwd=get_outputs,
            ),
        )

    return to_output_log


def attribution_backend_equiv(
    loaded_model: ModelParams, fake_attn_probs: Fake, fake_log_probs: Fake, for_embeds: bool, outputs: bool
):
    print(f"{fake_attn_probs=}, {fake_log_probs=}, {for_embeds=}, {outputs=}")

    model, params = loaded_model

    s = "[BEGIN] \"I love Mrs. Dursley. I don't sleep right,\" Harry said. He waved his hands helplessly. \"Mrs. Dursley's sleep cycle is twenty-six hours long, I always go to sleep two hours later, every day. I can't fall asleep any earlier than that, and then the next day I go to sleep two hours later than that. 10PM, 12AM, 2AM, 4AM, until it goes around the clock. Mr. Dursley stops me from sleeping two hours everytime. Mr. Dursley dislikes his wife Mrs. Dursley."
    seq = batch_tokenize([s])
    seq_len = seq.shape[1]
    target = "ley"

    head_layer_0 = 0
    head_layer_1 = 4
    first_seq_idx = (30 if for_embeds else 31) if outputs else I[:]
    seq_idx = 101
    attn_0_head_0_to_attn_1_head_4: List[ModifierCollectionTreeNodeStack] = [
        output_tree(
            model,
            OutputConf(
                Embeds() if for_embeds else AttnLayer(0, by_head=True),
                item_idx=ItemIdx(I[0, first_seq_idx] if for_embeds else I[0, head_layer_0, first_seq_idx]),
                shape=() if outputs else (seq_len, 1),
                endpoint=AttnLayer(1),
            ),
        ),
        ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs.single("blocks.inp_res", 1)))),
        ModifierCollectionTreeNode(
            StopGradConf(
                ItemConf(
                    KeyIdxs.single("blocks.attention.out_by_head", 1),
                    ItemIdx(I[:, head_layer_1, seq_idx], except_idx=True),
                )
            )
        ),
        [[ct(*stop_qkv_except(Idxs.single(1), qkv))] for qkv in range(3)],
    ]
    attn_0_head_0_to_attn_1_head_4.extend(op.it(fake_to_attn_probs(fake_attn_probs)))
    out = get_output_log(compose_trees_maybe_empty(*attn_0_head_0_to_attn_1_head_4), get_outputs=outputs)(
        model, params, seq, seq_idx, single_tokenize(target), fake_log_probs
    )

    backend = AttributionBackend(model, params, get_interp_tokenizer(), s)
    backend.startTree(
        {
            "kind": "logprob",
            "data": {
                "seqIdx": seq_idx,
                "tokString": target,
                "comparisonTokString": None,
            },
        },
        fake_log_probs == "ig",
        False,
    )

    if outputs:
        assert isinstance(first_seq_idx, int)
        if for_embeds:
            path_item = {"layerWithIO": 0, "headOrNeuron": 0, "token": first_seq_idx, "isMlp": False}
        else:
            path_item = {"layerWithIO": 1, "headOrNeuron": head_layer_0, "token": first_seq_idx, "isMlp": False}

        for qkv in range(3):
            print(f"{qkv=}")
            tree_path = [
                {"layerWithIO": 2, "headOrNeuron": head_layer_1, "token": seq_idx, "qkv": qkv, "isMlp": False},
                path_item,
            ]
            logits_out = backend.logitsForSpecificPath(
                tree_path,
                fake_log_probs=fake_log_probs,
                fake_attn=fake_attn_probs,
                compute_stepwise=False,
            )
            logits_out_stepwise = backend.logitsForSpecificPath(
                tree_path,
                fake_log_probs=fake_log_probs,
                fake_attn=fake_attn_probs,
                compute_stepwise=True,
            )

            check_close_weak(logits_out["log_probs"], logits_out_stepwise["log_probs"], atol=1e-4, norm_div_tol=2e-3)

            actual = logits_out["log_probs"]
            expected = out["log_probs"][..., qkv]

            check_close_weak(actual, expected, atol=1e-4, norm_div_tol=2e-3)

    else:
        losses = out["loss"].squeeze(-1)
        print("losses shape", losses.shape)
        expanded_vnt = backend.expandTreeNode(
            [{"layerWithIO": 2, "headOrNeuron": head_layer_1, "token": seq_idx, "isMlp": False}],
            fake_attn_probs == "ig",
        )
        assert isinstance(expanded_vnt, dict)

        if for_embeds:
            expanded = expanded_vnt["embeds"].tensor
        else:
            expanded = expanded_vnt["heads"][:, 0, head_layer_0].tensor
        # numerical error from this is pretty wild
        check_close_weak(expanded, losses)


def add_wiggle_room_nonneg(min_v):  # for numerical error
    return max(0, min(0.999 * min_v, min_v - 0.0001))


def attribution_backend_equiv_mask(
    loaded_model: ModelParams,
    fake_attn_probs: Fake,
    for_embeds: bool,
):
    use_neg = False  # TODO write test for True case
    print(f"{fake_attn_probs=}, {for_embeds=}, {use_neg=}")

    model, params = loaded_model

    s = "[BEGIN] \"I love Mrs. Dursley. I don't sleep right,\" Harry said. He waved his hands helplessly. \"Mrs. Dursley's sleep cycle is twenty-six hours long, I always go to sleep two hours later, every day. I can't fall asleep any earlier than that, and then the next day I go to sleep two hours later than that. 10PM, 12AM, 2AM, 4AM, until it goes around the clock. Mr. Dursley stops me from sleeping two hours everytime. Mr. Dursley dislikes his wife Mrs. Dursley."
    target = "ley"

    head_layer_0 = 0
    head_layer_1 = 4
    first_seq_idx = 101 if for_embeds else 31
    seq_idx = 101
    if for_embeds:
        layer = -1
        layer_type = LayerType.embeds
    else:
        layer = 0
        layer_type = LayerType.heads

    backend = AttributionBackend(model, params, get_interp_tokenizer(), s)
    start_vnt = backend.startTree(
        {
            "kind": "logprob",
            "data": {
                "seqIdx": seq_idx,
                "tokString": target,
                "comparisonTokString": None,
            },
        },
    )
    start_vnt_heads = start_vnt["heads"][0].tensor
    start_vnt_layer_0 = start_vnt["embeds"][0].tensor if for_embeds else start_vnt_heads[0, head_layer_0]

    # Compute all things that will end up in our mask using the expand-tree-node approach
    threshold_head_layer_1 = start_vnt_heads[1, head_layer_1, seq_idx]
    assert threshold_head_layer_1 > 0, "test run is testing empty things, quitting"

    def get_threshold_head_layer_0(threshold_expand):
        heads_direct_over_threshold = jnp.where(start_vnt_heads >= threshold_expand)
        expanded_vnts = [
            backend.expandTreeNode(
                [{"layerWithIO": l + 1, "headOrNeuron": head, "token": seq, "isMlp": False}],
                fake_attn_probs == "ig",
            )
            for [l, head, seq] in jnp.stack(heads_direct_over_threshold, axis=1)
        ]
        total_attrib_layer_0 = (
            sum(
                [
                    jnp.sum((vnt["embeds"] if for_embeds else vnt["heads"][:, 0, head_layer_0]).tensor, axis=0)
                    for vnt in expanded_vnts
                ]
            )
            + start_vnt_layer_0
        )
        threshold_head_layer_0 = total_attrib_layer_0[first_seq_idx]
        return threshold_head_layer_0

    threshold_head_layer_0 = get_threshold_head_layer_0(threshold_head_layer_1)
    print("threshold", threshold_head_layer_0)
    assert threshold_head_layer_0 > 0, "test run is testing empty things, quitting"

    # Pick threshold s.t. mask will contain both chosen heads (this is actually not very reliable,
    # new threshold may pick up more L1 heads which may cause attribution to target L0 head to decrease
    # below threshold; but it works in this case)
    node_threshold = add_wiggle_room_nonneg(min(threshold_head_layer_1, threshold_head_layer_0))
    print("node threshold", node_threshold)
    new_threshold_head_layer_0 = get_threshold_head_layer_0(node_threshold)
    print("new_threshold", new_threshold_head_layer_0)
    assert new_threshold_head_layer_0 >= threshold_head_layer_0, "mask threshold changed, quitting"
    nodes, node_attribs, mask = backend.searchAttributionsFromStart(node_threshold, use_neg, fake_attn_probs == "ig")

    assert seq_idx in nodes[LayerType.output][model.num_layers]
    # check L1 head attribs match output from start tree
    head_layer_1_idx_in_nodes = jnp.where(
        jnp.equal(nodes[LayerType.heads][1], jnp.array([head_layer_1, seq_idx])).all(1)
    )
    assert jnp.allclose(node_attribs[LayerType.heads][1][head_layer_1_idx_in_nodes], threshold_head_layer_1)
    # check L0 head attribs match output from expand tree node
    nodes_layer_0 = nodes[layer_type][layer]
    head_layer_0_idx_in_nodes = (
        jnp.where(nodes_layer_0 == first_seq_idx)[0]
        if for_embeds
        else jnp.where(jnp.equal(nodes_layer_0, jnp.array([head_layer_0, first_seq_idx])).all(1))
    )
    check_close_weak(
        node_attribs[layer_type][layer][head_layer_0_idx_in_nodes],
        jnp.expand_dims(threshold_head_layer_0, axis=0),
        atol=1e-3,
    )

    # Compute edge attribs using the expand-tree-node approach
    expanded_vnt_head_layer_1 = backend.expandTreeNode(
        [{"layerWithIO": 2, "headOrNeuron": head_layer_1, "token": seq_idx, "isMlp": False}],
        fake_attn_probs == "ig",
    )
    qkv_edges = (
        expanded_vnt_head_layer_1["embeds"][:, first_seq_idx]
        if for_embeds
        else expanded_vnt_head_layer_1["heads"][:, 0, head_layer_0, first_seq_idx]
    )

    # Pick threshold s.t. we'll get all nonneg-attrib edges from head_layer_0 to head_layer_1
    edge_threshold = add_wiggle_room_nonneg(min(qkv_edges.tensor[qkv_edges.tensor >= 0]))
    edge_attribs = backend.getAttributionInMask(edge_threshold, use_neg=True)

    # check edge attribs match
    for qkv in range(3):
        if qkv_edges[qkv].tensor < edge_threshold:
            continue
        edge = Edge(LayerType.heads, 1, head_layer_1, seq_idx, qkv, layer_type, layer, head_layer_0, first_seq_idx)
        assert jnp.allclose(edge_attribs[edge], qkv_edges[qkv].tensor)


def test_attribution_backend_equiv_embeds(loaded_model):
    attribution_backend_equiv(loaded_model, "none", "none", True, True)


def test_attribution_backend_equiv_default(loaded_model):
    attribution_backend_equiv(loaded_model, "none", "none", False, False)
    attribution_backend_equiv(loaded_model, "none", "none", False, True)


def test_attribution_backend_equiv_ablation(loaded_model):
    attribution_backend_equiv(loaded_model, "ablation", "none", False, True)
    attribution_backend_equiv(loaded_model, "none", "ablation", False, True)


def test_attribution_backend_equiv_ig(loaded_model):
    attribution_backend_equiv(loaded_model, "ig", "none", False, True)
    attribution_backend_equiv(loaded_model, "none", "ig", False, True)
    attribution_backend_equiv(loaded_model, "ig", "none", False, False)
    attribution_backend_equiv(loaded_model, "none", "ig", False, False)


def test_attribution_backend_equiv_mask(loaded_model):
    attribution_backend_equiv_mask(loaded_model, "ig", False)
    attribution_backend_equiv_mask(loaded_model, "none", False)


def test_attribution_backend_equiv_mask_embeds(loaded_model):
    attribution_backend_equiv_mask(loaded_model, "none", True)
