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
        log = inference(model, params, token_ids, to_log_idxed=[KeyIdxs(f"blocks.attention.out_by_head", Idxs.all())])
        all = jnp.squeeze(log[f"blocks.attention.out_by_head"], axis=1)  # shape = (layer, [ai,ao,mi,mo], seq,hidden)
        return all

    vnt_attn = LazyVeryNamedTensor(
        fn,
        dim_names=[
            "layer",
            "head",
            "seq",
            "hidden",
        ],
        dim_types=[
            "layer",
            "head",
            "seq",
            "hidden",
        ],
        dim_idx_names=[
            [str(i) for i in range(model.num_layers)],
            [str(i) for i in range(model.num_heads)],
            token_strs,
            [str(i) for i in range(model.hidden_size)],
        ],
        units="activation",
        title="Head Outputs",
    )
    return vnt_attn
