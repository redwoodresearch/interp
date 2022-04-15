from interp.tools.log import Idxs, KeyIdxs, LoggerCache, LoggerCacheAll, MutLogCache, LogCache
from interp.ui.very_named_tensor import LazyVeryNamedTensor, VeryNamedTensor
import numpy as np
import jax.numpy as jnp
from interp.model.model_loading import load_model
from interp.model.gpt_model import inference


def get_lvnt(model, params, tokenizer, string):
    token_ids = tokenizer(string, padding=False, return_tensors="jax")["input_ids"]
    token_strs = [tokenizer.decode(token_id) for token_id in token_ids[0]]

    def fn():
        cache = inference(model, params, token_ids, to_log_idxed=[KeyIdxs("blocks.attention.out_by_head", Idxs.all())])
        # Let's visualize the attention scores by layer and head
        return np.squeeze(np.linalg.norm(cache[f"blocks.attention.out_by_head"], axis=-1), axis=1)

    vnt_attn = LazyVeryNamedTensor(
        fn,
        dim_names=[
            "layer",
            "head",
            "seq",
        ],
        dim_types=[
            "layer",
            "head",
            "seq",
        ],
        dim_idx_names=[
            [str(i) for i in range(model.num_layers)],
            [str(i) for i in range(model.num_heads)],
            token_strs,
        ],
        units="activation_norm",
        title="Attention Head Output Norms",
    )
    return vnt_attn
