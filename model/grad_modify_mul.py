from typing import Any, Tuple, Union

import jax.numpy as jnp

from interp.model.gpt_model import Gpt
from interp.tools.grad_modify_query import ModifierCollectionTreeNodeStack
from interp.tools.grad_modify_query_items import ItemIdx
from interp.tools.indexer import I
from interp.model.gpt_modules import UnidirectionalAttn
from interp.tools.grad_modify_query_utils import MulBuilder
from interp.tools.log import Idxs, KeyIdxs


def tuple_len_check_get(idx, l: int) -> Tuple[Any, ...]:
    if isinstance(idx, tuple):
        assert len(idx) <= l
    else:
        idx = (idx,)

    return idx + (I[:],) * (len(idx) - l)


def attn_score_mul_builder(
    loggable,
    model: Gpt,
    tree_q: ModifierCollectionTreeNodeStack,
    tree_k: ModifierCollectionTreeNodeStack,
    layer: Union[int, jnp.ndarray],
    idx=I[:],  # batch, head
    seq_q=I[:],
    seq_k=I[:],
    shape=(),
    use_fwd_q: bool = True,
    use_fwd_k: bool = True,
    allow_non_first_deriv_q: bool = False,
    allow_non_first_deriv_k: bool = False,
) -> MulBuilder:
    base_idx = tuple_len_check_get(idx, 2)
    seq_q_idx = tuple_len_check_get(seq_q, 1)
    seq_k_idx = tuple_len_check_get(seq_k, 1)

    mulled_idx = base_idx + seq_q_idx + seq_k_idx
    q_idx = base_idx + seq_q_idx
    k_idx = base_idx + seq_k_idx

    def run(m: Gpt, q, k):
        attn = m.blocks[0].attention
        return attn.mask_out_upper_triangle(attn.attn_scores(q, k, mask=False), d=0.0)

    return MulBuilder(
        loggable,
        tree_q,
        tree_k,
        KeyIdxs.single("blocks.attention.final_q", layer),
        KeyIdxs.single("blocks.attention.final_k", layer),
        lambda q, k: model.apply({}, q, k, method=run),
        key_idxs=KeyIdxs.single("blocks.attention.attn_scores", layer),
        mulled_item_idx=ItemIdx(mulled_idx),
        shape=shape,
        idx_l=q_idx,
        idx_r=k_idx,
        use_fwd_l=use_fwd_q,
        use_fwd_r=use_fwd_k,
        allow_non_first_deriv_l=allow_non_first_deriv_q,
        allow_non_first_deriv_r=allow_non_first_deriv_k,
    )


def probs_v_mul_builder(
    loggable,
    tree_probs: ModifierCollectionTreeNodeStack,
    tree_v: ModifierCollectionTreeNodeStack,
    layer: int,
    idx=I[:],  # batch, head
    seq_q=I[:],
    seq_k=I[:],
    shape=(),
    use_fwd_probs: bool = True,
    use_fwd_v: bool = True,
    allow_non_first_deriv_probs: bool = False,
    allow_non_first_deriv_v: bool = False,
) -> MulBuilder:
    base_idx = tuple_len_check_get(idx, 2)
    seq_q_idx = tuple_len_check_get(seq_q, 1)
    seq_k_idx = tuple_len_check_get(seq_k, 1)

    probs_idx = base_idx + seq_q_idx + seq_k_idx
    v_idx = base_idx + seq_k_idx
    mulled_idx = base_idx + seq_q_idx

    return MulBuilder(
        loggable,
        tree_probs,
        tree_v,
        KeyIdxs.single("blocks.attention.attn_probs_dropout", layer),
        KeyIdxs.single("blocks.attention.v", layer),
        UnidirectionalAttn.mul_probs_v,
        key_idxs=KeyIdxs.single("blocks.attention.combined_v", layer),
        mulled_item_idx=ItemIdx(mulled_idx),
        shape=shape,
        idx_l=probs_idx,
        idx_r=v_idx,
        use_fwd_l=use_fwd_probs,
        use_fwd_r=use_fwd_v,
        allow_non_first_deriv_l=allow_non_first_deriv_probs,
        allow_non_first_deriv_r=allow_non_first_deriv_v,
    )
