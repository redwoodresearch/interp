# %%

"""
Exercises and examples for new jax code and the grad_modify_query code.

In general, it's quite useful to be familiar with how jax.jit works, how jax
deals with memory, and jax's rough edges in general. There are various jax
tutorials which should be helpful here, but I won't go through this here.

(Code for this partially taken from `interp/demos/jax_inference.py` and `interp/tools/test_grad_modify_query.py`)
"""

# %%

from functools import partial
import itertools
import os
from typing import Dict, Literal

import jax.numpy as jnp
import jax
import numpy as np
from flax.core.scope import FrozenVariableDict

from interp.model.gpt_model import Gpt, partial_gpt_call, gpt_call
from interp.tools.assert_never import assert_never
from interp.tools.grad_modify_query import (
    ModifierCollectionTreeNode,
    ModifierCollectionTreeNodeStack,
    Query,
    run_query,
    TargetConf,
)
from interp.tools.indexer import I
from interp.tools.grad_modify_query_items import AddConf, ItemIdx, MulConf, StopGradConf, ItemConf
from interp.tools.grad_modify_query_utils import MulBuilder, as_op, compose_trees
from interp.tools.interpretability_tools import (
    batch_tokenize,
    begin_token,
    losses_runner_no_log,
    run_on_tokens,
    single_tokenize,
    toks_to_string_list,
)
from interp.tools.jax_util import stack_tree
from interp.tools.custom_jvp import ablation_custom_jvp, integrated_gradients_custom_jvp
from interp.model.model_loading import load_model
from interp.tools.data_loading import DATA_DIR
from interp.tools.log import Idxs, KeyIdxs, LogInfo, LoggerCache, LoggerCacheAll, Logger, MutLogCache, LogCache

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
gpt2: Gpt = gpt2
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

"""
We're using flax which is a jax neural network/module library.

Jax doesn't allow for mutation of values, so it's important that we can treat
parameters as inputs and outputs. For this reason, flax has the parameters as a
separate value (`params` above) which can be 'bound' to the model or passed via
model.apply. See the flax docs for more detail.
"""

# %%

outputs = model.apply(params, example_seq)  # this calls the __call__ method of the model and sets up params
print(outputs.shape)
print(model.apply(params, method=model.get_k_mats_all_layers).shape)  # we can also use a different method for apply

# %%

"""
It's sometimes nicer to use bind.

_(The bind method for Gpt is wrapped so that that it returns the right type)_
"""

# %%

bound = model.bind(params)
outputs = bound(example_seq)
print(outputs.shape)
print(bound.get_k_mats_all_layers().shape)

# %%

"""
The model takes a log as input and updates the referenced cache in the log
"""

# %%

log = MutLogCache.new(logger=LoggerCacheAll())  # MutLogCache is a convenience helper for `Logger`

model.apply(params, example_seq, log=log)
cache: LogCache = LogCache.unwrap(log.cache)
cache.cache.keys(), cache.idxed_cache.keys()

# %%

"""
There's a jitted function which should be used by default. To avoid mutating
around the jit boundary, this function returns a cache.
"""

# %%

_, cache_out = gpt_call(model, params, example_seq, log_info=LogInfo(LoggerCacheAll()))
cache = LogCache.unwrap(cache_out)
cache.cache.keys(), cache.idxed_cache.keys()

# %%

_, cache_out = gpt_call(
    gpt2,
    gpt2_params,
    example_seq_no_begin,
    log_info=LogInfo(LoggerCacheAll(exclude={"blocks.mlp.out_by_neuron"})),
)
cache = LogCache.unwrap(cache_out)
cache.cache.keys(), cache.idxed_cache.keys()

# %%

"""
The cache has 2 parts, an idxed part and a non-idxed part. For this model, the
indexed part is just per block.

Some of this interface will be better motivated when select particular things
to cache (later).
"""

# %%

print(cache.cache["final_out.logits"].shape)
print(type(cache.idxed_cache["blocks.attention.attn_probs"]))
print(cache.idxed_cache["blocks.attention.attn_probs"].get(7, check=True).shape)
try:
    print(cache.idxed_cache["blocks.attention.attn_probs"].get(100, check=True).shape)
except AssertionError:
    ...


print(cache.idxed_cache["blocks.attention.attn_probs"].get_idxs(None).shape)  # None for all
print(cache.idxed_cache["blocks.attention.attn_probs"].get_idxs(jnp.array([1, 2, 7])).shape)

# %%

"""
We can also query the cache directly.
"""

# %%

print(cache.get(KeyIdxs("final_out.logits"), check=True).shape)
print(cache.get(KeyIdxs("blocks.attention.attn_probs", idxs=Idxs.all()), check=True).shape)
print(cache.get(KeyIdxs("blocks.attention.attn_probs", idxs=Idxs(idxs=jnp.array([1, 2, 7]))), check=True).shape)

# %%

"""
Take a look at the `Log` interface in `interp/tools/log.py`.

Now look at `interp/model/gpt_model.py` and `interp/model/gpt_modules.py` (and possibly `interp/model/blocks.py`).
Particularly note the logging calls.
"""

# %%

"""
The log interface should be straight forward to subclass, but there are
existing logging implementations which can do various tasks you might be
interested in.
"""

# %%

# more selective caching for memory or whatever (shouldn't matter for (runtime)
# perf if you jit and only return part of the log from the jitted function)
logger = LoggerCache(
    to_cache={"final_out.logits", "final_out.inp"},
)

logger.add(KeyIdxs("blocks.attention.attn_probs", Idxs(idxs=jnp.array([1, 2, 7]))))
logger.add(KeyIdxs("blocks.mlp.linear2", Idxs.all()))
logger.add(KeyIdxs("blocks.mlp.norm2.bias", Idxs.single(3)))
logger.add(KeyIdxs.single("blocks.mlp.norm2.bias", 3))  # equivalent to above


_, cache_out = gpt_call(gpt2, gpt2_params, example_seq_no_begin, log_info=LogInfo(logger))
cache = LogCache.unwrap(cache_out)
print(cache.cache.keys())
print(cache.idxed_cache.keys())

print(cache.get(KeyIdxs("final_out.logits"), check=True).shape)
print(cache.get(KeyIdxs("blocks.attention.attn_probs", idxs=Idxs.all()), check=True).shape)
print(cache.get(KeyIdxs("blocks.mlp.linear2", idxs=Idxs.all()), check=True).shape)
print(cache.get(KeyIdxs("blocks.mlp.linear2", idxs=Idxs.single(4)), check=True).shape)
print(cache.get(KeyIdxs("blocks.mlp.norm2.bias", idxs=Idxs.single(3)), check=True).shape)

# idxing is absolute
print(cache.get(KeyIdxs("blocks.attention.attn_probs", idxs=Idxs(idxs=jnp.array([1, 2, 7]))), check=True).shape)
print(cache.get(KeyIdxs("blocks.attention.attn_probs", idxs=Idxs(idxs=jnp.array([2]))), check=True).shape)

# %%

"""
Refering to the model and blocks as needed, use logging (probably LoggerCache) on gpt2 to get:
- The outputs of head 9 in layer 2 (hint: `out_by_head`)
- The normalizing multiplier for the attn layer norm in layer 4.
- The k value for head 4 in layer 6
- The neuron outputs for the mlp in layer 7 (after activation, before reducing dimensionality)

Note that LoggerCache doesn't support indexing, but you can just index the values afterward.
"""

# %%

# solution
logger = LoggerCache()
logger.add(KeyIdxs("blocks.attention.out_by_head", Idxs.single(2)))
logger.add(KeyIdxs("blocks.attention.norm1.overall_mul", Idxs.single(4)))
logger.add(KeyIdxs("blocks.attention.k", Idxs.single(6)))
logger.add(KeyIdxs("blocks.mlp.gelu", Idxs.single(7)))
_, cache_out = gpt_call(gpt2, gpt2_params, example_seq_no_begin, log_info=LogInfo(logger))
cache = LogCache.unwrap(cache_out)
print(cache.idxed_cache.keys())

# %%

"""
We'll get to the grad modification log and associated querying code later, but
first let's load some data and get the loss.

(just to show how this can be done with jax)
"""

# %%

import torch  # sometime loading torch before you've run any jax causes great OOMage

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

# %%

"""
Now let's look at the derivative querying code.

To start, we'll work with a simple linear example with scalars. (from `interp/tools/test_grad_modify_query.py`)

After that we'll run some queries on the actual model.
"""

# %%


def scalar_example(logger: Logger, finish=True):
    log = MutLogCache.new(logger)

    x_0 = (
        log.log_and_modify(jnp.array(10.0), "x_0_10")
        + log.log_and_modify(jnp.array(7.0), "x_0_7")
        + log.log_and_modify(jnp.array(5.0), "x_0_5")
    )
    x_0 = log.log_and_modify(x_0, "x_0")
    x_1 = (
        4.0 * log.log_and_modify(x_0, "x_0_x_1_4")
        + 5.0 * log.log_and_modify(x_0, "x_0_x_1_5")
        + log.log_and_modify(jnp.array(121.0), "x_1_121")
    )
    x_1 = log.log_and_modify(x_1, "x_1")
    x_2 = (
        23.0 * log.log_and_modify(x_0, "x_0_x_2_23")
        + 41.0 * log.log_and_modify(x_0, "x_0_x_2_41")
        + 13.0 * log.log_and_modify(x_1, "x_1_x_2_13")
        + 3.0 * log.log_and_modify(x_1, "x_1_x_2_3")
    )
    x_2 = log.log_and_modify(x_2, "x_2")

    x_3 = log.log_and_modify(x_1, "x_1_x_3_mul_x_2") * log.log_and_modify(
        x_2, "x_2_x_3_mul_x_1"
    ) + 4 * log.log_and_modify(x_0, "x_0_x_3_4")
    x_3 = log.log_and_modify(x_3, "x_3")

    if finish:
        log.check_finish()

    return log.cache


# %%

"""
This query doesn't take a derivative, it just gets the actual outputs.
"""

# %%

results = run_query(
    scalar_example,
    Query(
        targets=[
            TargetConf(KeyIdxs("x_0")),
            TargetConf(KeyIdxs("x_1"), display_name="x_1_value"),
            TargetConf(KeyIdxs("x_2")),
            TargetConf(KeyIdxs("x_3")),
        ]
    ),
)
print(results["x_0"])
print(results["x_1_value"])

x_0, x_1, x_2, x_3 = results["x_0"], results["x_1_value"], results["x_2"], results["x_3"]

# %%

"""
We can look at how x_1 changes as an 'input' to x_0 changes by multiplying by an alpha=1.
"""

# %%

results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_1"))],
        modifier_collection_tree=ModifierCollectionTreeNode(MulConf(ItemConf(KeyIdxs("x_0_10")))),
    ),
)
print(results["x_1"])

# %%

"""
We can also look at how x_1 changes as x_0 changes overall by adding a beta=0.
"""

# %%

results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_1"))],
        modifier_collection_tree=ModifierCollectionTreeNode(AddConf(ItemConf(KeyIdxs("x_0")))),
    ),
)
print(results["x_1"])

# %%

"""
It can also be useful to sum over multiple different gradient modifications.
This can be done by passing a list to ModifierCollectionTreeNode.
"""

# %%

results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_1"))],
        modifier_collection_tree=ModifierCollectionTreeNode(
            [MulConf(ItemConf(KeyIdxs("x_0_10"))).forget_type(), MulConf(ItemConf(KeyIdxs("x_0_7")))]
        ),
    ),
)
print(results["x_1"])

# %%

"""
A list of lists will stack over the outer dimension and sum over the inner
dimension.
"""

# %%

results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_1"))],
        modifier_collection_tree=ModifierCollectionTreeNode(
            [
                [MulConf(ItemConf(KeyIdxs("x_0_10"))).forget_type(), MulConf(ItemConf(KeyIdxs("x_0_7")))],
                [MulConf(ItemConf(KeyIdxs("x_0_5"))).forget_type()],
            ]
        ),
    ),
)
print(results["x_1"])


# %%

"""
Stopping gradients can be used to select a specific path through the computation.
"""

# %%

results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_1"))],
        modifier_collection_tree=ModifierCollectionTreeNode(
            MulConf(ItemConf(KeyIdxs("x_0_10"))),
            next_item=ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_1_4")))),
        ),
    ),
)
print(results["x_1"])

# %%

"""
It can be annoying to chain with nested 'next_item' all the time, so if we're
just joining a bunch of things end to end, we can use the `compose_trees`
utility function.
"""

# %%

results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_1"))],
        modifier_collection_tree=compose_trees(
            ModifierCollectionTreeNode(MulConf(ItemConf(KeyIdxs("x_0_10")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_1_4")))),
        ),
    ),
)
print(results["x_1"])  # same as above

# %%

"""
And more stop grads.
"""

# %%

results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_2"))],
        # order doesn't matter for nodes (unless you are taking multiple derivs
        # and the function is REAL gnarly
        modifier_collection_tree=compose_trees(
            ModifierCollectionTreeNode(MulConf(ItemConf(KeyIdxs("x_0_10")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_1_4")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_2_23")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_2_41")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_1_x_2_13")))),
        ),
    ),
)
print(results["x_2"])

# %%

"""
Exercise: use stop grads to select the exactly computation path from x_0 7 -> x_2 23 (and target x_2).
Next select x_0 7 -> x_1 4 -> x_2 13 (and target x_2).
"""

# %%

# solution
results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_2"))],
        modifier_collection_tree=compose_trees(
            ModifierCollectionTreeNode(MulConf(ItemConf(KeyIdxs("x_0_7")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_1")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_2_41")))),
        ),
    ),
)
print(results["x_2"])

results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_2"))],
        modifier_collection_tree=compose_trees(
            ModifierCollectionTreeNode(MulConf(ItemConf(KeyIdxs("x_0_7")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_1_5")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_2_23")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_2_41")))),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_1_x_2_3")))),
        ),
    ),
)
print(results["x_2"])

# %%

"""
It's also possible to sum/concat over nodes identically to modifications.

We can use this to branch on next_item and sum over nodes.
"""


# %%

common_stop_grads = compose_trees(
    ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_2_23")))),
    ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_2_41")))),
)


results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_2"))],
        modifier_collection_tree=compose_trees(
            common_stop_grads,
            ModifierCollectionTreeNode(
                MulConf(ItemConf(KeyIdxs("x_0_10"))),
                next_item=as_op(
                    [
                        compose_trees(
                            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_1_4")))),
                            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_1_x_2_13")))),
                        ),
                        compose_trees(
                            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_0_x_1_5")))),
                            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_1_x_2_3")))),
                        ),
                    ]
                ),
            ),
        ),
    ),
)
print(results["x_2"])  # 10 * 5 * 3 + 10 * 4 * 13

# %%

"""
I think it's often a bit nicer to shorten things as:
```
MCTN = ModifierCollectionTreeNode
SGC = StopGradConf
IC = ItemConf
cpf = combine_paths_first
```
But we'll use the full names here.
"""

# %%

"""
While this backend supports taking multiple derivatives, this is disabled by
default. (And can be enabled via a field of Query).
"""

# %%

try:
    results = run_query(
        scalar_example,
        Query(
            targets=[TargetConf(KeyIdxs("x_1"))],
            modifier_collection_tree=compose_trees(
                ModifierCollectionTreeNode(MulConf(ItemConf(KeyIdxs("x_0_10")))),
                ModifierCollectionTreeNode(MulConf(ItemConf(KeyIdxs("x_0_x_1_4")))),
            ),
        ),
    )
except AssertionError as e:
    print("error:", e)

# %%

"""
One task of interest is determing attribution (or similar) when you have a
product of two values. Specifically, suppose we have f(x(a) * y(b)).

We might be interested in the quantities:
- normal product rule derivative: `f'(x(a) * y(b)) (x'(a) y(b) +  x(a) y'(b))`
- just interaction between a and b derivative: `f'(x(a) * y(b)) x'(a) y'(b)`
- 'double counting' removed: `f'(x(a) * y(b)) (x'(a) y(b) +  x(a) y'(b)) - f'(x(a) * y(b)) x'(a) y'(b)`

These can be computed with the query api with the aid of a helper.
"""

# %%

"""
Let's look at the normal product rule derivative.
"""

# %%

results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_3"))],
        modifier_collection_tree=ModifierCollectionTreeNode(AddConf(ItemConf(KeyIdxs("x_1")))),
    ),
)
print(results["x_3"])
print(1 * x_2 + x_1 * (13 + 3))

# %%

results = run_query(
    scalar_example,
    Query(
        targets=[TargetConf(KeyIdxs("x_3"))],
        modifier_collection_tree=as_op(
            [
                compose_trees(
                    ModifierCollectionTreeNode(AddConf(ItemConf(KeyIdxs("x_1")))),
                    ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_1_x_2_13")))),
                    ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_1_x_2_3")))),
                ),
                compose_trees(
                    ModifierCollectionTreeNode(AddConf(ItemConf(KeyIdxs("x_0_x_2_23")))),
                ),
            ]
        ),
    ),
)
print(results["x_3"])
print(1 * x_2 + 23 * x_1)

# %%

"""
Now lets look at just the interaction.
"""

# %%

# This is a bit verbose, but there are shorthands we'll see in a sec for attn
# scores q k and attn probs v.
builder = MulBuilder(
    scalar_example,
    compose_trees(
        ModifierCollectionTreeNode(AddConf(ItemConf(KeyIdxs("x_1")))),
        ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_1_x_2_13")))),
        ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs("x_1_x_2_3")))),
    ),
    compose_trees(
        ModifierCollectionTreeNode(AddConf(ItemConf(KeyIdxs("x_0_x_2_23")))),
    ),
    KeyIdxs("x_1"),
    KeyIdxs("x_2"),
    lambda x, y: x * y,
    KeyIdxs("x_3"),
)

# %%


# NOTE: calling `.conjunctive()` actually runs some computation! (it isn't fully declarative)
# So if you want that part to be jitted, make sure to include it inside of the jit.
results = run_query(
    scalar_example, Query(targets=[TargetConf(KeyIdxs("x_3"))], modifier_collection_tree=builder.conjunctive())
)
results["x_3"]  # 1 * 23

# %%

# remove_conjunctive_from_sum also runs computation
results = run_query(
    scalar_example,
    Query(targets=[TargetConf(KeyIdxs("x_3"))], modifier_collection_tree=as_op(builder.remove_conjunctive_from_sum())),
)
results["x_3"]
print(1 * x_2 + 23 * x_1 - 1 * 23)

# %%

"""
Exercises:
Pick a path to x_1 and a path to x_2 starting from x_0 and compute all 3 of
these types of attribution for x_3.

Consider also playing with this example some more generally.
"""

# %%

"""
The run_queries helper might also be helpful for running multiple queries.
"""

# %%

"""
Now let's move on to attribution in a transformer.

We'll copy paste in some (very slightly pruned) code from the attribution
backend for easier learning
"""

# %%

last_am_seq_idx = (np.array(example_seq_string_list) == "AM").nonzero()[0][-1]
last_am = single_tokenize("AM")
last_am_seq_idx, last_am


# %%


# ig = integrated_gradients
Fake = Literal["none", "ablation", "ig"]


def fake_to_wrapper(fake: Fake):
    if fake == "ig":
        return integrated_gradients_custom_jvp
    elif fake == "ablation":
        return ablation_custom_jvp
    elif fake == "none":
        return lambda x: x
    else:
        assert_never(fake)


def get_run_model_and_output(
    model: Gpt,
    params: FrozenVariableDict,
    seqs: jnp.ndarray,
    fake_log_probs: Fake,
):
    def run(logger: Logger):
        log_info = LogInfo(logger)
        logits, cache = gpt_call(model, params, seqs, log_info=log_info, config=Gpt.CallConfig(log_finish=False))

        log_softmax = fake_to_wrapper(fake_log_probs)(jax.nn.log_softmax)
        _, cache = log_info.log_and_modify(log_softmax(logits), "log_probs", cache)

        logger.check_finish_cache(cache)

        return cache

    return run


# %%


def get_loss_query(root: ModifierCollectionTreeNodeStack):
    @partial(jax.jit, static_argnames=["fake_log_probs"])
    def loss_query(
        seqs,
        seq_idx,
        target,
        fake_log_probs: Fake = "none",
    ) -> Dict[str, jnp.ndarray]:
        # backward is better here because we have a single loss value.
        return run_query(
            get_run_model_and_output(model, params, seqs, fake_log_probs),
            Query(
                targets=[TargetConf(KeyIdxs("log_probs"), display_name="loss", idx=(0, seq_idx, target))],
                modifier_collection_tree=root,
                use_fwd=False,
            ),
        )

    return loss_query


# %%

# shape is just 'stacked' into the output (dimension naming might be more a thing later)
get_loss_query(
    ModifierCollectionTreeNode(
        MulConf(ItemConf(KeyIdxs.single("blocks.attention.out_by_head", 1), ItemIdx(0)), shape=(model.num_heads, 1, 1))
    ),
)(example_seq, last_am_seq_idx - 1, last_am)["loss"].squeeze((-2, -1))

# %%

"""
We do see head 1.4 having a good time (as expected). Let's see which of the seq idxs was most
important for that head.
"""

# %%

contribution = get_loss_query(
    compose_trees(
        ModifierCollectionTreeNode(
            MulConf(ItemConf(KeyIdxs("embedding.overall"), ItemIdx(0)), shape=(example_seq.shape[1], 1))
        ),
        ModifierCollectionTreeNode(
            StopGradConf(ItemConf(KeyIdxs.single("blocks.attention.out_by_head", 1), ItemIdx(I[0, 4], except_idx=True)))
        ),
    )
)(example_seq, last_am_seq_idx - 1, last_am)["loss"].squeeze((-1))
print(contribution)

top_k_vals, top_k_idxs = jax.lax.top_k(contribution, 5)
print(top_k_vals)
print(top_k_idxs)
print(np.array(example_seq_names)[np.array(top_k_idxs)])

# %%

"""
Exercise: Try to nail down that path through which these sequence idxs contribute.

You may find attn_score_mul_builder and probs_v_mul_builder in interp/model/grad_modify_mul.py helpful.

After working on this for some amount of time, try the same task in the attribution ui.
"""

# %%

"""
It's possible to modify how derivatives are computed. For instance, we could use integrated gradients.

_(This only changes non-linear things)_

Another potentially useful thing is doing ablations.

See interp/experiments/induction_head_investigation.py for some examples of this.

Note that ablations *aren't* linear: summing over the attribution for various
ablations doesn't yield the value for the overall ablation. Unfortunately, this
means that some of the computations you might be interested in which ablate
multiple things at the same time aren't easily possible. I'll
add a wrapper for computing this sort of thing at some point, but this
is a bit complex/annoying for a few reasons.
"""
