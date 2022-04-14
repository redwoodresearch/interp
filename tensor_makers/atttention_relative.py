import jax
from interp.tools.log import Idxs, KeyIdxs, LoggerCache
from interp.ui.very_named_tensor import LazyVeryNamedTensor, VeryNamedTensor
import numpy as np
import jax.numpy as jnp
from interp.model.model_loading import load_model
from interp.model.gpt_model import inference


def compute_weighted_probs(model, params, token_ids):
    log = LoggerCache()

    log = inference(
        model,
        params,
        token_ids,
        to_log_idxed=[
            KeyIdxs("blocks.attention.attn_probs", Idxs.all()),
            KeyIdxs("blocks.attention.inp", Idxs.all()),
        ],
    )
    # %%
    # Let's visualize the attention scores by layer and head
    attn = jnp.einsum(
        "lbhqk,lbhk->lhqk",
        log["blocks.attention.attn_probs"],
        jnp.linalg.norm(
            jnp.einsum(
                "lbsv,lhov->lbhso",
                log["blocks.attention.inp"],
                model.bind(params).get_ov_combined_mats_all_layers(),
            ),
            axis=-1,
        )
        / jnp.linalg.norm(log["blocks.attention.inp"], axis=-1),
    )  # shape = (layer, head, seq, seq)
    return attn


compute_weighted_probs = jax.jit(compute_weighted_probs, static_argnames=["model"])


def get_lvnt(model, params, tokenizer, string):
    token_ids = tokenizer(string, padding=False, return_tensors="jax")["input_ids"]
    token_strs = [tokenizer.decode(token_id) for token_id in token_ids[0]]

    def fn():
        return compute_weighted_probs(model, params, token_ids)

    return LazyVeryNamedTensor(
        fn,
        dim_names="layer head Q K".split(),
        dim_types="layer head seq seq".split(),
        dim_idx_names=[
            [str(i) for i in range(model.num_layers)],
            [str(i) for i in range(model.num_heads)],
            token_strs,
            token_strs,
        ],
        units="prob",
        title="Relative Weighted Attn Probs",
    )
