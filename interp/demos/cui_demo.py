# %%

import jax

from interp.ui.very_named_tensor import VeryNamedTensor
from interp.model.model_loading import load_model
from interp.model.gpt_model import gpt_call
from interp.tools.log import LogInfo, LoggerCacheAll, KeyIdxs, Idxs, LogCache

import interp.cui as cui

await cui.init(port=6789)  # type: ignore

# %%

model, params, tok = load_model("jan5_attn_only_two_layers")
text = "[BEGIN] Mr and Mrs Dursley were proud to say that they were perfectly normal. Mr Dursley made drills."

logger = LoggerCacheAll()
token_ids = tok(text, padding=False, return_tensors="jax")["input_ids"]
token_strs = [tok.decode(token_id) for token_id in token_ids[0]]
out, cache_out = gpt_call(model, params, token_ids, log_info=LogInfo(logger))
cache = LogCache.unwrap(cache_out)


# %%

# Let's visualize the attention scores by layer and head
attn = cache.get(KeyIdxs("blocks.attention.attn_probs", Idxs.all())).squeeze(1)  # shape = (layer, head, seq, seq)

vnt_attn = VeryNamedTensor(
    attn,
    dim_names="layer head Q K".split(),
    dim_types="layer head seq seq".split(),
    dim_idx_names=[
        [str(i) for i in range(model.num_layers)],
        [str(i) for i in range(model.num_heads)],
        token_strs,
        token_strs,
    ],
    units="prob",
    title="Attention Probabilities",
)

# Also visualize the model's predictions
logprobs = jax.nn.log_softmax(out[0])
# Truncate the vocab_size dimension to reduce amount of data sent - this isn't the smartest way but works for demo
top_token_ids = (-logprobs.max(0)).argsort()[:50]
vnt_preds = VeryNamedTensor(
    logprobs[:, top_token_ids],
    dim_names="seq vocab".split(),
    dim_types="seq vocab".split(),
    dim_idx_names=[token_strs, [tok.decode(token_id) for token_id in top_token_ids]],
    units="logprob",
    title="Model Predictions",
)


# %%
# Run this and then click the link that appears.
await cui.show_tensors(  # type: ignore
    vnt_preds,
    vnt_attn,
)
