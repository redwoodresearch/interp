# %%

import itertools
import os
from typing import Optional

import jax.numpy as jnp
import jax

from interp.tools.jax_util import stack_tree
from interp.model.gpt_model import Gpt, partial_gpt_call, gpt_call
from interp.tools.interpretability_tools import (
    batch_tokenize,
    begin_token,
    losses_runner_no_log,
    run_on_tokens,
    toks_to_string_list,
)
from interp.model.model_loading import load_model
from interp.tools.log import LogInfo, LoggerCacheAll, LogCache
from interp.tools.data_loading import DATA_DIR


# %%

models_dir_local = os.path.expanduser("~/interp_models_jax")
model, params, tok = load_model(
    "jan5_attn_only_two_layers/",  # models_dir=models_dir_local
)
model: Gpt = model
params

# %%

gpt2, gpt2_params, _ = load_model(
    "gpt2/",
    # models_dir=models_dir_local,
)
gpt2_params


# %%

text = " \"I don't sleep right,\" Harry said. He waved his hands helplessly. \"My sleep cycle is twenty-six hours long, I always go to sleep two hours later, every day. I can't fall asleep any earlier than that, and then the next day I go to sleep two hours later than that. 10PM, 12AM, 2AM, 4AM, until it goes around the clock. Even if I try to wake up early, it makes no difference and I'm a wreck that whole day. That's why I haven't been going to a normal school up until now.\""
# text = " My name is Buck. My name is Buck."
example_seq_no_begin = batch_tokenize([text])
example_seq = jnp.concatenate([jnp.expand_dims(begin_token(), (0, 1)), example_seq_no_begin], axis=1)
example_seq_string_list = toks_to_string_list(example_seq[0])
example_seq_names = [f"{n} ({i})" for i, n in enumerate(example_seq_string_list)]
example_seq.shape

# %%

_, cache = gpt_call(model, params, example_seq, log_info=LogInfo(LoggerCacheAll(exclude={"blocks.mlp.out_by_neuron"})))
cache: Optional[LogCache] = cache
assert cache is not None
cache.cache.keys(), cache.idxed_cache.keys()

# %%

_, cache = gpt_call(
    gpt2,
    gpt2_params,
    example_seq_no_begin,
    log_info=LogInfo(LoggerCacheAll(exclude={"blocks.mlp.out_by_neuron"})),
)
cache: Optional[LogCache] = cache
assert cache is not None
cache.cache.keys(), cache.idxed_cache.keys()

# %%

print(cache.idxed_cache["blocks.attention.attn_probs"].get(10, check=True).shape)
print(cache.idxed_cache["blocks.attention.attn_probs"].values.shape)


# %%

import torch

fnames = os.listdir(f"{DATA_DIR}/owt_tokens_int16_val")[1:2]
all_tokens = [torch.load(f"{DATA_DIR}/owt_tokens_int16/{f}") for f in fnames]
data_pt = list(itertools.chain(*[torch.split(x["tokens"], x["lens"].tolist()) for x in all_tokens]))

# %%

max_size = 511

# %%

data = torch.stack(
    [data_pt_val[:max_size].to(torch.int64) + 32768 for data_pt_val in data_pt if data_pt_val.size(0) >= max_size],
    dim=0,
).numpy()
data.shape

# %%

data_subset = data[:300]
batch_size = 4

# %%

stack_tree(
    run_on_tokens(
        jax.jit(losses_runner_no_log(partial_gpt_call(model, params))),
        data_subset,
        batch_size,
    )
).mean()

# %%

stack_tree(
    run_on_tokens(
        jax.jit(losses_runner_no_log(partial_gpt_call(gpt2, gpt2_params))),
        data_subset,
        1,
        prepend_begin=False,
    )
).mean()
