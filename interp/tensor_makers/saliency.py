from functools import partial
import jax
from interp.tools.grad_modify_query import ModifierCollectionTreeNode, Query, TargetConf, run_query
from interp.tools.grad_modify_query_items import ItemConf, MulConf
from interp.tools.log import KeyIdxs
from interp.ui.very_named_tensor import LazyVeryNamedTensor
from interp.model.gpt_model import loss_by_token_loggable


def saliency_unjit(model, params, token_ids):
    result = run_query(
        loss_by_token_loggable(model, params, token_ids),
        query=Query(
            targets=[TargetConf(KeyIdxs("loss.loss_by_token"), display_name="loss")],
            modifier_collection_tree=ModifierCollectionTreeNode(
                MulConf(ItemConf(KeyIdxs(f"embedding.tok")), shape=[1, token_ids.shape[1], 1])
            ),
            use_fwd=True,
        ),
    )["loss"]
    print(result.shape)
    result = -result[0, :, 0, :, 0]
    return result


saliency = jax.jit(saliency_unjit, static_argnames=["model"])


def get_lvnt(model, params, tokenizer, string):
    token_ids = tokenizer(string, padding=False, return_tensors="jax")["input_ids"]
    token_strs = [tokenizer.decode(token_id) for token_id in token_ids[0]]

    vnt_attn = LazyVeryNamedTensor(
        partial(saliency, model, params, token_ids),
        dim_names=[
            "to",
            "from",
        ],
        dim_types=[
            "seq",
            "seq",
        ],
        dim_idx_names=[
            token_strs,
            token_strs,
        ],
        units="saliency",
        title="Saliency",
    )
    return vnt_attn
