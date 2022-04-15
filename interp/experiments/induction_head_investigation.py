# %%

from functools import partial
import os
import itertools
from copy import copy, deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from jax.interpreters.ad import JVPTracer

from tabulate import tabulate
import msgpack
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import plotly.express as px
from attrs import evolve, define, Factory, frozen
import einops

from interp.model.gpt_model import Gpt, partial_gpt_call
from interp.model.grad_modify_fake import ablation_softmax_probs, integrated_gradients_softmax_probs
from interp.model.grad_modify_mul import attn_score_mul_builder, probs_v_mul_builder
from interp.tools.attribution_backend_utils import stop_qkv_except
from interp.tools.grad_modify_query_items import ItemConf, ItemIdx, MulConf, NoneConf, StopGradConf, ModifyInBasisConf
from interp.tools.grad_modify_query_utils import MulBuilder, compose_trees as ct
from interp.tools.interpretability_tools import (
    batch_tokenize,
    losses_runner,
    losses_runner_log,
    print_max_min_by_tok_k,
    single_tokenize,
    LossesRunnerTreeConfig,
    begin_token,
    losses_runner_tree,
    toks_to_string_list,
    run_on_tokens,
    cross_entropy_ablation_log_probs,
    get_cross_entropy_integrated_gradients_log_probs,
)
from interp.tools.jax_util import stack_tree
from interp.model.model_loading import load_model
from interp.tools.grad_modify_query import (
    GradModifierConf,
    ModifierCollectionTreeNodeStack,
    ModifierCollectionTreeNode,
    Query,
    TargetConf,
    run_queries,
    run_query,
)
from interp.tools.indexer import I
from interp.model.grad_modify_output import (
    Attn,
    AttnLayer,
    Embeds,
    FinalOutput,
    OutputConf,
    losses_runner_outputs,
    output_tree,
)
from interp.tools.log import Idxs, KeyIdxs
from interp.ui.attribution_backend import AttributionBackend, AttributionRoot
from interp.tools.data_loading import DATA_DIR
import interp
import interp.tools.optional as op

# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# %%

"""
This notebook does some quick ablations on the induction circuit to see what's
actually important.
"""

# %%

# interp.init()  # if you want interp to work, run in jupyter notebook classic

# %%

models_dir_local = os.path.expanduser("~/interp_models_jax")
model, params, tok = load_model("jan5_attn_only_two_layers/", models_dir=models_dir_local)
model: Gpt = model
params

# %%

text = " \"I don't sleep right,\" Harry said. He waved his hands helplessly. \"My sleep cycle is twenty-six hours long, I always go to sleep two hours later, every day. I can't fall asleep any earlier than that, and then the next day I go to sleep two hours later than that. 10PM, 12AM, 2AM, 4AM, until it goes around the clock. Even if I try to wake up early, it makes no difference and I'm a wreck that whole day. That's why I haven't been going to a normal school up until now.\""
text = " \"I love Mrs. Dursley. I don't sleep right,\" Harry said. He waved his hands helplessly. \"Mrs. Dursley's sleep cycle is twenty-six hours long, I always go to sleep two hours later, every day. I can't fall asleep any earlier than that, and then the next day I go to sleep two hours later than that. 10PM, 12AM, 2AM, 4AM, until it goes around the clock. Mr. Dursley stops me from sleeping two hours everytime. Mr. Dursley dislikes his wife Mrs. Dursley."
# text = " My name is Buck. My name is Buck."
example_seq_no_begin = batch_tokenize([text])
example_seq = jnp.concatenate([jnp.expand_dims(begin_token(), (0, 1)), example_seq_no_begin], axis=1)
example_seq_string_list = toks_to_string_list(example_seq[0])
example_seq_names = [f"{n} ({i})" for i, n in enumerate(example_seq_string_list)]
example_seq.shape

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

"""
We'll want to ablate 'through' the log softmax instead of differentiating.

We'll also see what the derivative and integrated derivative look like.
"""

# %%

out_tree = output_tree(model, OutputConf(AttnLayer(1, by_head=True), endpoint=FinalOutput(), item_idx=ItemIdx(I[:, 4])))

extra_tree: List[List[Optional[ModifierCollectionTreeNode]]] = [
    [ct(evolve(out_tree, use_fwd=True), cross_entropy_ablation_log_probs)],
    [ct(evolve(out_tree, use_fwd=True), get_cross_entropy_integrated_gradients_log_probs())],
    [evolve(out_tree, use_fwd=False)],
]

run_split = jax.jit(
    losses_runner_outputs(
        model,
        params,
        config=LossesRunnerTreeConfig(get_loss_no_deriv=True),
        get_extra_tree=lambda runner, seq_len: extra_tree,
    )
)

# %%

out_removed = stack_tree(
    run_on_tokens(
        run_split,
        data_subset,
        batch_size,
        disable_progress=False,
    )
)
out_removed

# %%

loss_removed = out_removed["deriv"]["loss"].mean(axis=0)
loss_orig = out_removed["no_deriv"]["loss"].mean()

ablation_loss = loss_removed[0]
print("ablation:", ablation_loss)
print("integrated gradient", loss_removed[1])
print("normal deriv", loss_removed[2])
print("original loss", loss_orig)

# %%

"""
Now let's start excluding some heads and see how the head does.
"""

# %%

tok_to_layer_1_attn_tree = output_tree(model, OutputConf(Embeds(), endpoint=AttnLayer(1)))
tok_to_layer_1_attn_tree

# %%


def all_except_run(
    q_heads,
    k_heads,
    v_heads,
    exclude_tok_to_k=False,
    ablation_over_ig_attn: Optional[bool] = True,
    ablation_over_ig_log_probs: Optional[bool] = True,
    config: LossesRunnerTreeConfig = LossesRunnerTreeConfig(avg_loss=False),
):
    def get_trees(runner, seq_len) -> ModifierCollectionTreeNodeStack:
        q_tree: ModifierCollectionTreeNodeStack = ModifierCollectionTreeNode(
            MulConf(ItemConf(KeyIdxs.single("blocks.attention.out_by_head", 0), ItemIdx(I[:, q_heads])))
        )
        k_tree: List[Optional[ModifierCollectionTreeNode]] = [
            ModifierCollectionTreeNode(
                MulConf(ItemConf(KeyIdxs.single("blocks.attention.out_by_head", 0), ItemIdx(I[:, k_heads])))
            ),
        ]
        if exclude_tok_to_k:
            k_tree.append(tok_to_layer_1_attn_tree)

        v_tree: ModifierCollectionTreeNodeStack = ModifierCollectionTreeNode(
            MulConf(ItemConf(KeyIdxs.single("blocks.attention.out_by_head", 0), ItemIdx(I[:, v_heads])))
        )

        base_tree = ct(
            ModifierCollectionTreeNode(
                StopGradConf(
                    ItemConf(KeyIdxs.single("blocks.attention.out_by_head", 1), ItemIdx(I[:, 4], except_idx=True))
                )
            ),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs.single("blocks.inp_res", 1)))),
            op.map(
                ablation_over_ig_attn,
                lambda x: ablation_softmax_probs(Idxs.single(1))
                if x
                else integrated_gradients_softmax_probs(Idxs.single(1)),
            ),
        )

        attn_mul: List[Optional[ModifierCollectionTreeNode]] = list(
            attn_score_mul_builder(
                runner,
                model,
                ct(base_tree, q_tree),
                ct(base_tree, k_tree),
                1,
            ).remove_conjunctive_from_sum()
        )

        all_mulled: List[Optional[ModifierCollectionTreeNode]] = list(
            probs_v_mul_builder(
                runner,
                attn_mul,
                ct(base_tree, v_tree),
                1,
            ).remove_conjunctive_from_sum()
        )

        return ct(
            base_tree,
            all_mulled,
            op.map(
                ablation_over_ig_log_probs,
                lambda x: cross_entropy_ablation_log_probs if x else get_cross_entropy_integrated_gradients_log_probs(),
            ),
        )

    return jax.jit(
        losses_runner_tree(
            partial_gpt_call(model, params, config=Gpt.CallConfig(log_finish=False)),
            get_trees,
            config=evolve(config, use_fwd=True),
        )
    )


# %%

# selected based on some experimentation
v_heads = list(range(1, 8))
base_q_heads = list(range(8))
base_k_heads = list(range(1, 8))

# %%

"""
Let's see how this does.
"""

# %%

run = all_except_run(base_q_heads, base_k_heads, v_heads)

# %%

out_except = stack_tree(run_on_tokens(run, data_subset, 2, disable_progress=False), stack=False)
mean_base = out_except["deriv"]["loss"].squeeze().mean(axis=(0, 1))

print(mean_base)
print("score", (ablation_loss - mean_base) / ablation_loss)

# %%

"""
Now let's try looping over including a single head on both k and q.
"""

# %%

for head_to_include in range(8):
    new_q_heads = copy(base_q_heads)
    new_q_heads.remove(head_to_include)
    new_k_heads = copy(base_k_heads)
    if head_to_include in new_k_heads:
        new_k_heads.remove(head_to_include)

    run = all_except_run(new_q_heads, new_k_heads, v_heads)

    # compile times on this are brutal...
    out_except = stack_tree(run_on_tokens(run, data[:100], 1, disable_progress=False), stack=False)
    out_except

    print("included:", head_to_include)
    mean_l = out_except["deriv"]["loss"].squeeze().mean(axis=(0, 1))
    print(mean_l)
    print("score", (ablation_loss - mean_l) / ablation_loss)

# %%

"""
Note that above we included layer 0 head 0 in v. It's somewhat suprising that
this head makes a reasonable contribution to v (but I claim it does).

Let's figure out exactly what this contribution is.

(This part of the notebook is undocumented and unexplained atm, but will
hopefully be better in the future after I write things up).
"""

# %%


def effect_of_layer_0_0_v(
    ablation_over_ig_log_probs: Optional[bool] = True,
    shape_on_toks=True,
    avg_loss=False,
    log_probs=True,
    fwd=True,
    toks_subset=I[:],
    extra_tree: ModifierCollectionTreeNodeStack = None,
):
    @jax.jit
    def f(toks, targets):
        runner = losses_runner_log(
            partial_gpt_call(model, params, config=Gpt.CallConfig(log_finish=False)), toks, targets
        )

        head_layer_0 = 0
        head_layer_1 = 4

        if not isinstance(toks_subset, slice):
            assert toks.shape[0] == 1

        shape = (
            (toks.shape[0], toks.shape[1] if isinstance(toks_subset, slice) else toks_subset.shape[0], 1)
            if shape_on_toks
            else ()
        )
        tree = ct(
            ModifierCollectionTreeNode(
                MulConf(ItemConf(KeyIdxs.single("blocks.attention.inp", 0), ItemIdx(I[:, toks_subset])), shape=shape)
            ),
            ModifierCollectionTreeNode(
                StopGradConf(
                    ItemConf(
                        KeyIdxs("blocks.attention.out_by_head", Idxs.single(0)),
                        item_idx=ItemIdx(I[:, head_layer_0], except_idx=True),
                    ),
                )
            ),
            ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs.single("blocks.inp_res", 1)))),
            *stop_qkv_except(Idxs.single(1), "v"),
            ModifierCollectionTreeNode(
                StopGradConf(
                    ItemConf(
                        KeyIdxs.single("blocks.attention.out_by_head", 1),
                        ItemIdx(I[:, head_layer_1], except_idx=True),
                    ),
                )
            ),
            op.map(
                ablation_over_ig_log_probs,
                lambda x: cross_entropy_ablation_log_probs if x else get_cross_entropy_integrated_gradients_log_probs(),
            ),
            extra_tree,
        )

        if avg_loss:
            targets = [TargetConf(KeyIdxs("loss.mean_loss"), "loss")]
        else:
            targets = [TargetConf(KeyIdxs("loss.losses"), "loss")]

        if log_probs:
            targets.append(TargetConf(KeyIdxs("cross_entropy.log_probs"), display_name="log_probs"))

        return run_query(runner, Query(targets, tree, use_fwd=fwd))

    return f


# %%

run_probs_just_v = effect_of_layer_0_0_v(ablation_over_ig_log_probs=None)

# %%

out_except = stack_tree(
    run_on_tokens(run_probs_just_v, example_seq, 1, disable_progress=True, prepend_begin=False), stack=False
)
loss_diff = out_except["loss"].squeeze((1, 3))  # get 1 beyond to remove begin tok
summed_over = loss_diff.sum(-1)
_, idxs = jax.lax.top_k(jnp.abs(summed_over), k=10)
loss_diff[idxs]
np.array(example_seq_names)[np.array(idxs)], summed_over[idxs], summed_over

# %%

print(example_seq_names[97])
print_max_min_by_tok_k(loss_diff[97], k=20, get_tok=lambda idxs: np.array(example_seq_names)[np.array(idxs)])

# %%

print(example_seq_names[85])
print_max_min_by_tok_k(loss_diff[85], k=20, get_tok=lambda idxs: np.array(example_seq_names)[np.array(idxs)])

# %%

b_model = model.bind(params)

tok_embeds = b_model.embedding.token_embedding.embedding

all_ov_combined = b_model.get_ov_combined_mats_all_layers()
ov_0_0_to_1_4 = jnp.einsum(" o m, m i -> o i", all_ov_combined[1, 4], all_ov_combined[0, 0])
ov_0_0_to_1_4_to_out = jnp.einsum("v o, o i -> v i", tok_embeds, ov_0_0_to_1_4)
ov_0_0_to_out = jnp.einsum("v m, m i -> v i", tok_embeds, all_ov_combined[0, 0])
ov_0_0_to_1_4_to_out.shape, ov_0_0_to_out.shape

# %%


def print_for_tok(t: Union[str, int], k=50):
    if isinstance(t, str):
        t = single_tokenize(t)

    print(f'"{tok.decode(t)}"')

    preds_full = jnp.einsum("v i, i -> v", ov_0_0_to_1_4_to_out, tok_embeds[t])
    preds_0_0 = jnp.einsum("v i, i -> v", ov_0_0_to_out, tok_embeds[t])

    print("0_0_to_1_4")
    print("logit variance", preds_full.var())
    print_max_min_by_tok_k(preds_full, k=k)
    print()

    # print("0_0")
    # print("logit variance", preds_0_0.var())
    # print_max_min_by_tok_k(preds_0_0, k=k)


# %%

# print more toks as desired
print_for_tok(" dog", k=10)

# %%

# I think this isn't very meaningful due to general log softmax issues when averaging loss?
# Maybe not though.
run_probs_just_v_mean_loss = effect_of_layer_0_0_v(
    ablation_over_ig_log_probs=None,
    avg_loss=True,
    log_probs=False,
    fwd=False,
)

# %%

out_v_path = stack_tree(run_on_tokens(run_probs_just_v_mean_loss, data_subset, 2, disable_progress=False), stack=False)
losses_by_from_tok = out_v_path["loss"].squeeze(-1)
losses_by_from_tok.shape

# %%

n_tokens = tok_embeds.shape[0]

# %%

# offset by begin!
data_subset[:-1].shape, losses_by_from_tok[1:].shape

# %%

loss_deriv_by_tok = jnp.zeros(n_tokens).at[data_subset[:-1]].add(losses_by_from_tok[1:], unique_indices=False)

# %%

print_max_min_by_tok_k(loss_deriv_by_tok, k=30, normalize=False)

# %%

loss_deriv_by_tok.mean()

# %%

run_probs_from_current_loss = effect_of_layer_0_0_v(
    ablation_over_ig_log_probs=True, shape_on_toks=False, avg_loss=False, log_probs=False, fwd=True
)

# %%

batch_size = 2

# %%

out_v_path_q = stack_tree(
    run_on_tokens(run_probs_from_current_loss, data_subset, batch_size, disable_progress=False),
    stack=True,
)
losses_by_q_tok = einops.rearrange(out_v_path_q["loss"], "n (b s) -> (n b) s", b=batch_size)
losses_by_q_tok.shape

# %%

net_loss_by_q_tok = jnp.zeros(n_tokens).at[data_subset[:-1]].add(losses_by_q_tok[1:], unique_indices=False)
net_loss_by_q_tok.mean()

# %%

print_max_min_by_tok_k(net_loss_by_q_tok, k=30, normalize=False)

# %%

most_helpful = losses_by_q_tok.mean(axis=1).argmin()
most_helpful

# %%

print_max_min_by_tok_k(
    losses_by_q_tok[most_helpful][1:],
    k=20,
    get_tok=lambda idxs: toks_to_string_list(data_subset[most_helpful][(idxs,)]),
)

# %%

g_idxs = jnp.array((np.array(toks_to_string_list(data_subset[most_helpful])) == "g").nonzero()[0])
g_idxs

# %%

print(tok.decode(data_subset[most_helpful]))

# %%

run_probs_ablate_a_few_toks = effect_of_layer_0_0_v(
    ablation_over_ig_log_probs=True,
    shape_on_toks=True,
    avg_loss=False,
    log_probs=True,
    fwd=True,
    toks_subset=g_idxs + 2,  # + 1 for [BEGIN] and + 1 again for next tok
)


# %%

out_v_path_from_selected_toks = stack_tree(
    run_on_tokens(
        run_probs_ablate_a_few_toks,
        jnp.expand_dims(data_subset[most_helpful], 0),
        1,
        disable_progress=False,
    ),
    stack=True,
)
losses = out_v_path_from_selected_toks["loss"].squeeze((0, 2, -1))

# %%

losses[419]

# %%

all_virtual_head_outputs = jnp.einsum("o i, n i -> n o", ov_0_0_to_1_4, tok_embeds)
all_virtual_head_outputs.shape


# %%

import torch

x = torch.tensor(np.array(all_virtual_head_outputs))
u, s, v = torch.pca_lowrank(x, q=32)
u, s, v = u.numpy(), s.numpy(), v.numpy()
u.shape, s.shape, v.shape

# %%

# all_virtual_head_outputs[0]
idx = 21
jnp.abs(all_virtual_head_outputs[idx] - (v @ jnp.transpose(v)) @ all_virtual_head_outputs[idx]).max()

# %%

# inp_to_dims
inp_to_dims = jnp.einsum("o i, o d -> d i", ov_0_0_to_1_4, v)
tok_input_to_dims_values = jnp.einsum("d i, t i -> d t", inp_to_dims, tok_embeds)
inp_to_dims.shape, tok_input_to_dims_values.shape

# %%

plt.plot(s)

# %%

s

# %%

direc_to_toks = jnp.einsum("v o, o d -> d v", tok_embeds, v)

# %%

for dim in range(5):
    print("dim", dim)
    print_max_min_by_tok_k(direc_to_toks[dim], k=20)

# %%

print(print_max_min_by_tok_k(tok_input_to_dims_values[0], k=20))

# %%


def get_run_probs_ablate_some_dims(up_to_n):
    return effect_of_layer_0_0_v(
        ablation_over_ig_log_probs=True,
        shape_on_toks=False,
        avg_loss=True,
        log_probs=False,
        fwd=True,
        extra_tree=ModifierCollectionTreeNode(
            ModifyInBasisConf(
                ItemConf(KeyIdxs.single("blocks.attention.out_by_head", 1), ItemIdx(I[:, 4])),
                proj_to=jnp.transpose(v),
                proj_back=v,
                sub_conf=StopGradConf(ItemConf(KeyIdxs(""), ItemIdx(I[..., :up_to_n]))),
            )
        ),
    )


# %%

run_ablate_after_1 = get_run_probs_ablate_some_dims(1)

# %%

out_some_dims = stack_tree(run_on_tokens(run_ablate_after_1, data_subset, 2, disable_progress=True), stack=True)
out_some_dims["loss"], out_some_dims["loss"].shape, out_some_dims["loss"].mean()


# %%

losses = [
    stack_tree(run_on_tokens(get_run_probs_ablate_some_dims(i), data_subset, 2, disable_progress=False), stack=True)[
        "loss"
    ].mean()
    for i in range(0, 10)
]

# %%

plt.plot(-np.array(losses))
