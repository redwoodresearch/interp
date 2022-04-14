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
            to_log_idxed=[KeyIdxs(f"blocks.attention.inp", Idxs.all()), KeyIdxs(f"blocks.attention.out", Idxs.all())]
            + (
                [
                    KeyIdxs(f"blocks.mlp.inp", Idxs.all()),
                    KeyIdxs(f"blocks.mlp.out", Idxs.all()),
                ]
                if model.use_mlp
                else []
            ),
        )
        # Let's visualize the attention scores by layer and head
        all = jnp.squeeze(
            jnp.stack(
                [
                    log[f"blocks.attention.inp"],
                    log[f"blocks.attention.out"],
                ]
                + ([log[f"blocks.mlp.inp"], log[f"blocks.mlp.out"]] if model.use_mlp else []),
                axis=2,
            ),
            axis=1,
        )  # shape = (layer, [ai,ao,mi,mo], seq, hidden)
        return all

    vnt_attn = LazyVeryNamedTensor(
        fn,
        dim_names=[
            "layer",
            "amio",
            "seq",
            "hidden",
        ],
        dim_types=[
            "layer",
            "amio",
            "seq",
            "hidden",
        ],
        dim_idx_names=[
            [str(i) for i in range(model.num_layers)],
            ["Attention_Input", "Attention_Output"] + (["MLP_Input", "MLP_Output"] if model.use_mlp else []),
            token_strs,
            [str(i) for i in range(model.hidden_size)],
        ],
        units="activation",
        title="All Activations",
    )
    return vnt_attn
