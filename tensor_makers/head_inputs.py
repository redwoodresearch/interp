from interp.tools.log import Idxs, KeyIdxs
from interp.ui.very_named_tensor import LazyVeryNamedTensor, VeryNamedTensor
import numpy as np
import jax.numpy as jnp
from interp.model.model_loading import load_model
from interp.model.gpt_model import inference
import jax
from functools import partial


def fn_unjit(model, params, token_ids):
    log = inference(model, params, token_ids, to_log_idxed=[KeyIdxs(f"blocks.attention.inp", Idxs.all())])
    inputs = jnp.squeeze(log[f"blocks.attention.inp"], axis=1)  # layer seq hidden
    weights = model.bind(params).get_qkv_mats_all_layers()
    print(inputs.shape, weights.shape)
    qkvs = jnp.einsum("lsh,wlnih->lnwsi", inputs, weights)
    back = jnp.einsum("lnwsi,wlnih->lnwsh", qkvs, weights)
    return back


fn = jax.jit(fn_unjit, static_argnames=["model"])


def get_lvnt(model, params, tokenizer, string):
    token_ids = tokenizer(string, padding=False, return_tensors="jax")["input_ids"]
    token_strs = [tokenizer.decode(token_id) for token_id in token_ids[0]]

    vnt_attn = LazyVeryNamedTensor(
        partial(fn, model, params, token_ids),
        dim_names=[
            "layer",
            "head",
            "qkv",
            "seq",
            "hidden",
        ],
        dim_types=[
            "layer",
            "head",
            "qkv",
            "seq",
            "hidden",
        ],
        dim_idx_names=[
            [str(i) for i in range(model.num_layers)],
            [str(i) for i in range(model.num_heads)],
            ["q", "k", "v"],
            token_strs,
            [str(i) for i in range(model.hidden_size)],
        ],
        units="activation",
        title="Head Inputs",
    )
    return vnt_attn
