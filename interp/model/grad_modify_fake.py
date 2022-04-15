from typing import Any
from interp.model.blocks import gelu

import jax

from interp.tools.grad_modify_query import ModifierCollectionTreeNode
from interp.tools.grad_modify_query_items import ItemIdx, ReplaceFuncConf, ItemConf
from interp.tools.custom_jvp import ablation_custom_jvp, integrated_gradients_custom_jvp, mix_with_linear_custom_jvp
from interp.tools.log import Idxs, KeyIdxs


def replace_softmax_probs(replacement, layer_idxs: Idxs = Idxs.all(), item_idx: ItemIdx = ItemIdx()):
    return ModifierCollectionTreeNode(
        ReplaceFuncConf(
            conf=ItemConf(KeyIdxs("blocks.attention.attn_probs", layer_idxs), item_idx),
            from_key_idxs=KeyIdxs("blocks.attention.attn_scores", layer_idxs),
            replacement=replacement,
        )
    )


def wrap_softmax_probs(wrapper, layer_idxs: Idxs = Idxs.all(), item_idx: ItemIdx = ItemIdx()):
    return replace_softmax_probs(wrapper(jax.nn.softmax), layer_idxs, item_idx)


def integrated_gradients_softmax_probs(layer_idxs: Idxs = Idxs.all(), item_idx: ItemIdx = ItemIdx()):
    return wrap_softmax_probs(integrated_gradients_custom_jvp, layer_idxs, item_idx)


def ablation_softmax_probs(layer_idxs: Idxs = Idxs.all(), item_idx: ItemIdx = ItemIdx()):
    return wrap_softmax_probs(ablation_custom_jvp, layer_idxs, item_idx)


def half_linear_gelu(fraction=0.5, layer_idxs: Idxs = Idxs.all(), item_idx: ItemIdx = ItemIdx()):
    return ModifierCollectionTreeNode(
        ReplaceFuncConf(
            conf=ItemConf(KeyIdxs("blocks.mlp.gelu", layer_idxs), item_idx),
            from_key_idxs=KeyIdxs("blocks.mlp.linear1", layer_idxs),
            replacement=mix_with_linear_custom_jvp(gelu, fraction),
        )
    )
