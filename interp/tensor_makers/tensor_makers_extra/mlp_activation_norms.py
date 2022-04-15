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
        log = inference(model, params, token_ids, to_log_idxed=[KeyIdxs(f"blocks.mlp.out", Idxs.all())])
        # %%
        # Let's visualize the attention scores by layer and head
        outs = jnp.squeeze(log[f"blocks.mlp.out"], axis=1)  # shape = (layer, head, seq)
        return np.linalg.norm(outs, axis=-1)

    vnt_attn = LazyVeryNamedTensor(
        fn,
        dim_names=[
            "layer",
            "seq",
        ],
        dim_types=[
            "layer",
            "seq",
        ],
        dim_idx_names=[
            [str(i) for i in range(model.num_layers)],
            token_strs,
        ],
        units="activation_norm",
        title="MLP Block Output Norms",
    )
    return vnt_attn


required_model_info_subtree = {"model_config": {"use_mlp": True}}
