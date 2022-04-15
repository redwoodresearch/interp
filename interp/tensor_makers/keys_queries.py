from interp.tools.log import Idxs, KeyIdxs
from interp.ui.very_named_tensor import LazyVeryNamedTensor, VeryNamedTensor
import numpy as np
import jax.numpy as jnp
from interp.model.model_loading import load_model
from interp.model.gpt_model import inference


def get_lvnt(model, params, tokenizer, string):
    token_ids = tokenizer(string, padding=False, return_tensors="jax")["input_ids"]
    token_strs = [tokenizer.decode(token_id) for token_id in token_ids[0]]

    def fn():
        log = inference(
            model,
            params,
            token_ids,
            to_log_idxed=[KeyIdxs(f"blocks.attention.q", Idxs.all()), KeyIdxs(f"blocks.attention.k", Idxs.all())],
        )
        print("k shape", log[f"blocks.attention.k"].shape)
        all = jnp.squeeze(
            jnp.stack([log[f"blocks.attention.q"], log[f"blocks.attention.k"]], axis=3), axis=1
        )  # shape = (layer, head, qk, seq, head_size)
        return all

    vnt_attn = LazyVeryNamedTensor(
        fn,
        dim_names=[
            "layer",
            "head",
            "qk",
            "seq",
            "head_size",
        ],
        dim_types=[
            "layer",
            "head",
            "qk",
            "seq",
            "head_size",
        ],
        dim_idx_names=[
            [str(i) for i in range(model.num_layers)],
            [str(i) for i in range(model.num_heads)],
            ["q", "k"],
            token_strs,
            [str(i) for i in range(model.hidden_size // model.num_heads)],
        ],
        units="activation",
        title="Keys and Queries",
    )
    return vnt_attn
