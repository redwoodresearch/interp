from functools import partial
import jax
import jax.numpy as jnp
from interp.model.gpt_model import Gpt, gpt_call, gpt_call_no_log
from interp.tools.grad_modify_query import ModifierCollectionTreeNode, Query, TargetConf, run_query
from interp.tools.grad_modify_query_items import ItemConf, MulConf
from interp.tools.log import KeyIdxs, LogInfo, Logger
from interp.ui.very_named_tensor import LazyVeryNamedTensor
from interp.model.gpt_model import loss_by_token_loggable


def kl_saliency_unjit(model, params, token_ids):
    def kl_loggable(logger: Logger):
        log_info = LogInfo(logger)
        out, cache = gpt_call(model, params, token_ids, log_info=log_info, config=Gpt.CallConfig(log_finish=False))
        out = jax.nn.log_softmax(out, axis=-1)
        independent_out = jax.nn.log_softmax(gpt_call_no_log(model, params, token_ids)[0], axis=-1)
        kl_by_logit = jnp.exp(independent_out) * (independent_out - out)
        kl_by_token = jnp.sum(kl_by_logit, axis=-1)
        _, cache = log_info.sub("loss").log_and_modify(kl_by_token, "kl_by_token", cache)
        logger.check_finish_cache(cache)
        return cache

    result = run_query(
        kl_loggable,
        query=Query(
            targets=[TargetConf(KeyIdxs("loss.kl_by_token"), display_name="loss")],
            modifier_collection_tree=ModifierCollectionTreeNode(
                MulConf(ItemConf(KeyIdxs(f"embedding.tok")), shape=[1, token_ids.shape[1], 1])
            ),
            use_fwd=True,
        ),
    )["loss"]
    print(result.shape)
    result = result[0, :, 0, :, 0]
    return result


kl_saliency = jax.jit(kl_saliency_unjit, static_argnames=["model"])


def get_lvnt(model, params, tokenizer, string):
    token_ids = tokenizer(string, padding=False, return_tensors="jax")["input_ids"]
    token_strs = [tokenizer.decode(token_id) for token_id in token_ids[0]]

    vnt_attn = LazyVeryNamedTensor(
        partial(kl_saliency, model, params, token_ids),
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
        title="KL Saliency",
    )
    return vnt_attn
