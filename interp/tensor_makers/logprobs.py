from interp.tools.data_loading import np_log_softmax
from interp.ui.very_named_tensor import LazyVeryNamedTensor, VeryNamedTensor
import numpy as np
import jax.numpy as jnp
from interp.model.model_loading import load_model
from interp.tools.log import LoggerCacheAll
from interp.model.gpt_model import gpt_call_no_log, inference
import jax


def fn_unjit(model, params, token_ids):
    out = inference(model, params, token_ids)["final_out.logits"]
    logprobs = jax.nn.log_softmax(out[0])
    print("OUT 0 SHAPE", logprobs.shape)
    top_token_ids = (-logprobs).argsort(axis=1)
    return top_token_ids, logprobs


fn = jax.jit(fn_unjit, static_argnames=["model"])


def get_lvnt(model, params, tokenizer, string):
    k = 20
    token_ids = tokenizer(string, padding=False, return_tensors="jax")["input_ids"]
    token_strs = [tokenizer.decode(token_id) for token_id in token_ids[0]]

    top_token_ids, logprobs = fn(model, params, token_ids)
    top_token_ids = np.array(top_token_ids[:, :k])
    shown_token_ids = np.unique(top_token_ids.flatten())
    logprobs = logprobs[:, shown_token_ids]

    vnt_preds = LazyVeryNamedTensor(
        lambda: logprobs,
        dim_names="seq vocab".split(),
        dim_types="seq vocab".split(),
        dim_idx_names=[token_strs, [tokenizer.decode([token_id]) for token_id in shown_token_ids]],
        units="logprob",
        title="Model Predictions",
    )
    return vnt_preds
