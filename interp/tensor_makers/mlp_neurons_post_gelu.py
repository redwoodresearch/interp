from interp.ui.very_named_tensor import LazyVeryNamedTensor, VeryNamedTensor
import numpy as np
import jax.numpy as jnp
from interp.model.model_loading import load_model
from interp.model.gpt_model import inference
from interp.tools.log import Idxs, KeyIdxs


def get_lvnt(model, params, tokenizer, string):
    token_ids = tokenizer(string, padding=False, return_tensors="jax")["input_ids"]
    token_strs = [tokenizer.decode(token_id) for token_id in token_ids[0]]

    def fn():
        log = inference(model, params, token_ids, to_log_idxed=[KeyIdxs(f"blocks.mlp.gelu", Idxs.all())])
        # %%
        # Let's visualize the attention scores by layer and head
        neurons = jnp.squeeze(log[f"blocks.mlp.gelu"], axis=1)  # shape = (layer, head, seq, seq)
        return neurons

    vnt_attn = LazyVeryNamedTensor(
        fn,
        dim_names=[
            "layer",
            "seq",
            "neuron",
        ],
        dim_types=[
            "layer",
            "seq",
            "neuron",
        ],
        dim_idx_names=[
            [str(i) for i in range(model.num_layers)],
            token_strs,
            [str(i) for i in range(model.hidden_size * 4)],
        ],
        units="activation",
        title="MLP Neurons",
    )
    return vnt_attn


required_model_info_subtree = {"model_config": {"use_mlp": True}}
