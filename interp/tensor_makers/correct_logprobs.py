from interp.tools.data_loading import np_log_softmax
from interp.ui.very_named_tensor import LazyVeryNamedTensor, VeryNamedTensor
import numpy as np
import jax.numpy as jnp
from interp.model.model_loading import load_model
from interp.tools.log import LoggerCacheAll
from interp.model.gpt_model import inference
import jax
from functools import partial


def fn_unjit(model, params, token_ids):
    out = inference(model, params, token_ids)["final_out.logits"]
    logprobs = jax.nn.log_softmax(out[0])
    correct_logprobs = jnp.take_along_axis(
        logprobs, jnp.expand_dims(jnp.concatenate([token_ids[0, 1:], np.array([0], dtype=jnp.int32)]), -1), 1
    )[:, 0]
    return correct_logprobs


fn_jit = jax.jit(fn_unjit, static_argnames=["model"])


def get_lvnt(model, params, tokenizer, string):
    token_ids = tokenizer(string, padding=False, return_tensors="jax")["input_ids"]
    token_strs = [tokenizer.decode(token_id) for token_id in token_ids[0]]

    vnt_preds = LazyVeryNamedTensor(
        partial(fn_jit, model, params, token_ids),
        dim_names=["seq"],
        dim_types=["seq"],
        dim_idx_names=[token_strs],
        units="logprob",
        title="Logprobs on Correct",
    )
    return vnt_preds
