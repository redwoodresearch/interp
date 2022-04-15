# %%

from copy import deepcopy
from functools import partial
import os
from time import time
from typing import Any, Dict, List, Tuple, Union, Optional, Iterable
import math

from tabulate import tabulate
import msgpack
import jax
import jax.numpy as jnp
from attrs import define

from interp.model.gpt_model import (
    Gpt,
    input_dist_zero_cov,
    gpt_apply_to_normal,
    gpt_call,
    gpt_gen_new_dist_to_input,
    gpt_condition_on_input_embeds,
    module_config_dict,
)
from interp.model.apply_to_normal_utils import graft_input_embed, map_dict_keys_covariance
from interp.model.inductive_distribution_generation import (
    get_default_update_and_get_next,
    inductive_distribution_generation,
)
from interp.model.gpt_modules import NEG_INF, AttnApplyToNormalConfig
from interp.model.monte_carlo import (
    get_unigrams_reduction,
    monte_carlo_generative,
    sample_and_print_toks,
    sample_default,
    weighted_sample_except,
    weighted_sample_single,
    log_cache_clean_up_stacked,
    log_cache_combine_to_batch,
)
from interp.tools.compare_multivariate_normal_to_monte import CompareActivation
from interp.tools.multivariate_normal import FlatValueConfig, MultivariateNormal, zero_approximation
from interp.tools.immutable_dict import assign, operate_f, operate
from interp.tools.interpretability_tools import (
    print_max_min_by_tok_k,
    single_tokenize,
    begin_token,
    strings_to_tok_list,
    toks_to_string_list,
    StdoutAndFile,
    compare_estimate_to_monte,
)
from interp.model.model_loading import load_model
from interp.tools.indexer import I
from interp.tools.log import KeyIdxs, LogInfo, LoggerCache, Logger, NoopLogger, LogCache, Idxs
import interp.tools.optional as op
from interp.tools.rrfs import RRFS_DIR
from interp.ui.very_named_tensor import VeryNamedTensor

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

out_dir = "normal_propagation_out/"
os.makedirs(out_dir, exist_ok=True)

# %%

jax.config.update("jax_enable_x64", True)

# %%

models_dir_local = os.path.expanduser("~/interp_models_jax")
model, params, tokenizer = load_model("jan5_attn_only_two_layers/", models_dir=models_dir_local)
model: Gpt = model
model_f64 = Gpt(**{**module_config_dict(model), "dtype": jnp.float64})
model_b = model.bind(params)
params

# %%

start = time()
begin_embeds = model_b.embedding(jnp.array([[begin_token()]]))["tok"]
assert begin_embeds.dtype == jnp.float32  # while we enable f64, we'll be using f32 almost everywhere
out_begin = gpt_apply_to_normal(
    model,
    params,
    jax.random.PRNGKey(8),
    input_dist_zero_cov(begin_embeds),
    config=Gpt.ApplyToNormalConfig(attn_config=AttnApplyToNormalConfig(1)),
)[0]
print(time() - start)
out_begin.mean_as().keys()

# %%

logger = LoggerCache(to_cache={"final_out.inp"})
out_actual, cache_out = gpt_call(model, params, jnp.array([[begin_token()]]), log_info=LogInfo(logger))
cache = LogCache.unwrap(cache_out)
assert jnp.allclose(out_begin.mean_as()["final_embeds"], cache.get(KeyIdxs("final_out.inp")), atol=1e-6)
begin_logits = out_actual

# %%

actual_cov = jnp.cov(
    model_b.embedding.token_embedding.embedding,
    rowvar=False,
    bias=True,
    aweights=jax.nn.softmax(model_b.embedding.unembed(cache.get(KeyIdxs("final_out.inp")))).squeeze((0, 1)),
)

# %%

assert jnp.allclose(
    model_b.new_dist_from_final_embed(jax.random.PRNGKey(1823), out_begin, 1, -1)[0].covariance_as()[
        "new_input_embed_dist"
    ]["new_input_embed_dist"],
    actual_cov,
    atol=5e-5,
)

# %%

unigram = jnp.array(msgpack.unpack(open(f"{RRFS_DIR}/ryan/token_stats/unigram_0_50259.bin", "rb")))

# %%

print_max_min_by_tok_k(unigram / unigram.sum(), k=50, print_min=False)

# %%

default_monte_logger = LoggerCache(to_cache={"final_out.inp"})

# %%

monte_gen = partial(
    monte_carlo_generative,
    key=jax.random.PRNGKey(821),
    model=model,
    params=params,
    batch_size=128,
    reduce=get_unigrams_reduction(model, params),
    logger=default_monte_logger,
    return_prob_ratios=True,
    clean_up_cache_post_scan=log_cache_clean_up_stacked,
    stack_caches=log_cache_combine_to_batch,
)

# %%

default_update_and_get_next = get_default_update_and_get_next(model, params, iters=1_000, jit=True)
estimate_token_dists = jax.jit(model_b.estimate_token_dists, static_argnames=["iters"])

# %%

dist_gen = partial(
    inductive_distribution_generation,
    model=model,
    params=params,
    key=jax.random.PRNGKey(10),
    dist=out_begin,
    update_and_get_next=default_update_and_get_next,
    sampling_eps=4e-3,
    pad_to=11,
)

# %%

to_cache: List[KeyIdxs] = [
    KeyIdxs("blocks.attention.attn_scores", Idxs.all()),
    KeyIdxs("blocks.attention.attn_probs", Idxs.all()),
]
monte_logger_cache = deepcopy(default_monte_logger)
monte_logger_cache.add_all(to_cache)
logger_cache = LoggerCache.from_key_idxs(to_cache)

# %%

monte_unigrams, monte_toks, monte_cache, prob_ratios = monte_gen(
    n_toks=11,
    n_samples=300_000,
    # return_cache=True,
    # logger=monte_logger_cache,
).as_tup()
assert monte_unigrams is not None
assert monte_toks is not None
assert prob_ratios is not None

# %%

sample_and_print_toks(jax.random.PRNGKey(283283832), prob_ratios, monte_toks)

# %%

dist, tok_probs, _, _ = dist_gen(
    pad_to=11,
    additional_toks=10,
    # logger=logger_cache,
    # maybe_invalid_shape=True,
)
tok_probs.shape, tok_probs.dtype

# %%

_ = compare_estimate_to_monte(tok_probs, monte_unigrams)

# %%

# zero_dist, zero_tok_dists, _, _ = dist_gen(approximation=zero_approximation)

# %%

# _ = compare_estimate_to_monte(zero_tok_dists, monte_unigrams)

# %%

# independence_between_types_approximation = partial(
#     map_dict_keys_covariance, f=lambda k_l, k_r, s_l, s_r, v: v if k_l == k_r else jnp.zeros_like(v)
# )
# independence_between_types, independence_between_types_tok_dists, _, _ = dist_gen(approximation=independence_between_types_approximation)

# %%

# _ = compare_estimate_to_monte(independence_between_types_tok_dists, monte_unigrams)

# %%

name_to_seq_axes: Dict[str, List[int]] = {
    "input_embeds": [1],
    "residual": [1],
    "attn_in": [1],
    "k": [2],
    "q": [2],
    "v": [2],
    "attn_scores": [2, 3],
    "attn_probs": [2, 3],
    "combined_v": [2],
    "attn_out": [1],
    "final_embeds": [1],
    "final_embed_for_new_dist": [],
    "new_input_embed_dist": [],
}


def independent_by_type_seq(k_l, k_r, s_l: Tuple[int, ...], s_r: Tuple[int, ...], v, approx_per=lambda k, x: x):
    if k_l != k_r:
        return jnp.zeros_like(v)

    assert s_l == s_r

    seq_axes = name_to_seq_axes[k_l]

    def get_idxs(items):
        slice_idxs = [I[:] for _ in s_l + s_r]
        assert len(seq_axes) == len(items)
        for axis, item in zip(seq_axes, items):
            slice_idxs[axis] = item
            slice_idxs[axis + len(s_l)] = item

        return tuple(slice_idxs)

    shape_removed = tuple(s for i, s in enumerate(s_l) if i not in seq_axes)
    size_removed = math.prod(shape_removed)

    idxs_tup = tuple(jnp.arange(s_l[axis]) for axis in seq_axes)
    idxs = jnp.meshgrid(*idxs_tup)

    def vmap_prod(inner_func, to_map):
        if len(to_map) == 0:
            return inner_func()

        return jax.vmap(lambda i: vmap_prod(partial(inner_func, i), to_map[1:]))(to_map[0])

    print(k_l, jnp.trace(v))

    cov = v.reshape(*s_l, *s_r)
    out = (
        jnp.zeros_like(cov)
        .at[get_idxs(idxs)]
        .set(
            vmap_prod(
                lambda *idxs: approx_per(k_l, cov[get_idxs(idxs)].reshape(size_removed, size_removed)).reshape(
                    *shape_removed, *shape_removed
                ),
                idxs_tup,
            )
        )
    )

    return out.reshape(v.shape[0], v.shape[1])


# %%


# independent_by_type_seq_approximation = partial(map_dict_keys_covariance, f=partial(independent_by_type_seq))
# independent_by_type_seq_dist, independent_by_type_seq_tok_dists, _, _  = dist_gen(approximation=independent_by_type_seq_approximation)

# %%

# _ = compare_estimate_to_monte(independent_by_type_seq_tok_dists, monte_unigrams)

# %%


def coordinate_independent_approx(x: jnp.ndarray, _: Any):
    return jnp.zeros_like(x).at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(jnp.diag(x))


# coordinate_independent_dist, coordinate_independent_tok_dists, _, _  = dist_gen(approximation=coordinate_independent_approx)

# %%

# _ = compare_estimate_to_monte(coordinate_independent_tok_dists, monte_unigrams)

# %%

unit_embeds = model_b.embedding.token_embedding.embedding
unit_embeds = unit_embeds / jnp.linalg.norm(unit_embeds, axis=-1, keepdims=True)

# %%

open_paren = " ("
close_paren = ")"
open_paren_tok = single_tokenize(open_paren)
close_paren_tok = single_tokenize(close_paren)

dist_to_open = jnp.einsum("k, v k -> v", unit_embeds[open_paren_tok], unit_embeds)
dist_to_close = jnp.einsum("k, v k -> v", unit_embeds[close_paren_tok], unit_embeds)

print_max_min_by_tok_k(dist_to_open, k=5, print_min=False)
print_max_min_by_tok_k(dist_to_close, k=5, print_min=False)

k = 40

open_like = jax.lax.top_k(dist_to_open, k=k)[1]
close_like = jax.lax.top_k(dist_to_close, k=k)[1]

print(list(toks_to_string_list(open_like)))
print(list(toks_to_string_list(close_like)))

paren_like = jnp.concatenate([open_like, close_like])

paren_loc = 4


def other_toks(toks):
    # silly but whatever
    return (jnp.arange(model.vocab_size)[None] != toks[:, None]).all(axis=0).nonzero()[0]


non_paren_like_toks = other_toks(paren_like)
non_paren_like_toks.shape


@jax.jit
def sample_paren_and_non_paren_like(key, unnormalized_logits, toks, next_tok_idx):
    return jax.lax.cond(
        next_tok_idx == paren_loc,
        lambda: weighted_sample_single(unnormalized_logits, toks.shape[0], open_paren_tok),
        lambda: weighted_sample_except(key, unnormalized_logits, paren_like),
    )


@partial(jax.jit, static_argnames=["weight"])
def sample_paren(key, unnormalized_logits, toks, next_tok_idx, weight=True):
    return jax.lax.cond(
        next_tok_idx == paren_loc,
        lambda: weighted_sample_single(unnormalized_logits, toks.shape[0], open_paren_tok, weight=weight),
        lambda: sample_default(key, unnormalized_logits, toks, next_tok_idx),
    )


# %%

additional_toks_paren = 10

# %%

paren_monte_unigrams, paren_toks, _, paren_prob_ratios = monte_gen(
    n_toks=additional_toks_paren + 1,
    n_samples=500_000,
    sample=sample_paren_and_non_paren_like,
).as_tup()
assert paren_monte_unigrams is not None

# %%

sample_and_print_toks(jax.random.PRNGKey(2838), paren_prob_ratios, paren_toks, n=30)
paren_toks = None

# %%

paren_monte_unigrams[:, close_paren_tok]

# %%

paren_allow_close_monte_unigrams, paren_allow_close_toks, _, paren_allow_close_prob_ratios = monte_gen(
    n_toks=additional_toks_paren + 1,
    n_samples=500_000,
    sample=sample_paren,
).as_tup()
assert paren_allow_close_monte_unigrams is not None


# %%

sample_and_print_toks(jax.random.PRNGKey(2839), paren_allow_close_prob_ratios, paren_allow_close_toks, n=30)

# %%

paren_allow_close_monte_unigrams[:, close_paren_tok]

# %%


def paren_run_at_seq_idx(
    key: jax.random.KeyArray,
    dist: MultivariateNormal,
    seq_idx: Union[int, jnp.ndarray],
    non_paren_like_toks: jnp.ndarray,
    condition_not_paren: Union[bool, jnp.ndarray],
    set_instead_of_condition: bool,
    iters: int = 1_000,
):
    dist, mean_probs, mean_include_prob = gpt_gen_new_dist_to_input(
        model, params, key, dist, iters=iters, final_embed_seq_idx=seq_idx, is_set=True
    )

    next_tok_idx = seq_idx + 1

    def run_cond(toks):
        return gpt_condition_on_input_embeds(
            model,
            params,
            dist,
            next_tok_idx,
            toks,
            mean_probs,
            set_instead_of_condition=set_instead_of_condition,
            is_dict=True,
        )

    dist = jax.lax.cond(
        next_tok_idx == paren_loc,
        lambda: run_cond(jnp.asarray(open_paren_tok)),
        lambda: jax.lax.cond(condition_not_paren, lambda: run_cond(jnp.asarray(non_paren_like_toks)), lambda: dist),
    )

    return dist, mean_probs, mean_include_prob


jitted_paren_run_at_seq_idx = jax.jit(paren_run_at_seq_idx, static_argnames=["set_instead_of_condition", "iters"])

# %%


def paren_run(
    key: jax.random.KeyArray,
    dist: MultivariateNormal,
    seq_idx: int,
    non_paren_like_toks: jnp.ndarray,
    condition_not_paren: Union[bool, jnp.ndarray],
    set_instead_of_condition: bool,
    fixed_point_iter: bool,
    fixed_point_iter_only_paren: bool,
    reverse: bool,
):
    it = reversed(range(seq_idx + 1)) if reverse else range(seq_idx + 1)

    out_probs: Optional[jnp.ndarray] = None
    out_mean_include: Optional[jnp.ndarray] = None
    for seq_idx_other in it:
        is_end = seq_idx_other == seq_idx
        if not is_end and (not fixed_point_iter or (fixed_point_iter_only_paren and seq_idx_other + 1 != paren_loc)):
            continue

        key, subkey = jax.random.split(key)
        dist, mean_probs, mean_include_prob = jitted_paren_run_at_seq_idx(
            subkey, dist, seq_idx_other, non_paren_like_toks, condition_not_paren, set_instead_of_condition
        )
        if is_end:
            out_probs = mean_probs
            out_mean_include = mean_include_prob

    return dist, op.unwrap(out_probs), op.unwrap(out_mean_include)


# %%

do_fixed_point_iter = False  # slow

for base_name, condition_not_paren, set_instead_of_condition in [
    ("just_set", False, True),
    ("only_cond_paren", False, False),
    ("cond_paren_and_not_paren", True, False),
]:

    for fixed_point_iter, reverse, only_paren in [(False, False, False)] + (
        [] if set_instead_of_condition else [(True, False, False), (True, False, True), (True, True, False)]
    ):
        if fixed_point_iter:
            if not do_fixed_point_iter:
                continue
            name = f"fixed_point{'_reverse' if reverse else ''}{'_fix_only_paren' if only_paren else ''}_{base_name}"
        else:
            name = base_name

        conditioned_on_paren_dist, _, _, _ = dist_gen(
            additional_toks=additional_toks_paren,
            pad_to=11,  # odd
            update_and_get_next=partial(
                paren_run,
                non_paren_like_toks=non_paren_like_toks,
                condition_not_paren=condition_not_paren,
                set_instead_of_condition=set_instead_of_condition,
                fixed_point_iter=fixed_point_iter,
                fixed_point_iter_only_paren=only_paren,
                reverse=reverse,
            ),
        )
        paren_tok_dists = estimate_token_dists(jax.random.PRNGKey(283), conditioned_on_paren_dist)

        with open(f"{out_dir}/paren_dump_{name}.txt", "w") as f:
            compare_estimate_to_monte(
                paren_tok_dists,
                paren_monte_unigrams,
                k=10,
                target_toks=[close_paren_tok, open_paren_tok],
                file=StdoutAndFile(f),
            )
        if not condition_not_paren:
            with open(f"{out_dir}/paren_allow_close_dump_{name}.txt", "w") as f:
                compare_estimate_to_monte(
                    paren_tok_dists,
                    paren_allow_close_monte_unigrams,
                    k=10,
                    target_toks=[close_paren_tok, open_paren_tok],
                    file=StdoutAndFile(f),
                )


# %%

causal_paren_sample = partial(sample_paren, weight=False)

# %%

paren_causal_monte_unigrams, paren_causal_toks, paren_causal_monte_cache_out, paren_causal_prob_ratios = monte_gen(
    n_toks=additional_toks_paren + 1,
    n_samples=200_000,
    sample=causal_paren_sample,
    return_cache=True,
    logger=monte_logger_cache,
).as_tup()
assert paren_causal_monte_unigrams is not None
assert paren_causal_prob_ratios is not None
paren_causal_monte_cache = LogCache.unwrap(paren_causal_monte_cache_out)

# %%

sample_and_print_toks(jax.random.PRNGKey(2839), paren_causal_prob_ratios, paren_causal_toks, n=30)

# %%

paren_causal_monte_unigrams[:, close_paren_tok]

# %%


def paren_causal(
    key: jax.random.KeyArray,
    dist: MultivariateNormal,
    seq_idx: Union[int, jnp.ndarray],
    iters: int = 1_000,
):
    dist, mean_probs, mean_include_prob = gpt_gen_new_dist_to_input(
        model, params, key, dist, iters=iters, final_embed_seq_idx=seq_idx, is_set=True
    )

    next_tok_idx = seq_idx + 1

    dist = jax.lax.cond(
        next_tok_idx == paren_loc,
        lambda: graft_input_embed(dist, paren_loc, model_b.embedding.token_embedding.embedding[open_paren_tok], 0.0),
        lambda: dist,
    )

    return dist, mean_probs, mean_include_prob


jitted_paren_causal = jax.jit(paren_causal, static_argnames=["iters"])

# %%

paren_causal_dist, paren_causal_tok_dists, _, paren_causal_dist_cache_out = dist_gen(
    additional_toks=additional_toks_paren,
    pad_to=11,  # odd
    update_and_get_next=jitted_paren_causal,
    logger=logger_cache,
    maybe_invalid_shape=True,
)
paren_causal_dist_cache = LogCache.unwrap(paren_causal_dist_cache_out)

# %%

compare_estimate_to_monte(
    paren_causal_tok_dists,
    paren_causal_monte_unigrams,
    k=10,
    target_toks=[close_paren_tok, open_paren_tok],
)

# %%

paren_causal_monte_cache.get(KeyIdxs("blocks.attention.attn_scores", Idxs.all())).shape

# %%

causal_paren_compare_attn_scores = CompareActivation(
    model,
    paren_causal_monte_cache,
    paren_causal_dist_cache,
    "attn_scores",
    "attn_scores",
    "attn_scores",
    seq_axis=2,
    remove_upper_triangle=True,
)

# %%

causal_paren_compare_attn_scores.plot_layer_idx(1, (4, 7, 2))

# %%

causal_paren_compare_attn_scores.plot_percentile(0.97)

# %%


def purge_neg_inf(v):
    return jnp.where(v == NEG_INF, float("nan"), v)


# %%

causal_paren_compare_attn_scores.monte_mean.shape

# %%

causal_paren_scores_mean_vnt = VeryNamedTensor(
    jnp.stack(
        [
            purge_neg_inf(causal_paren_compare_attn_scores.monte_mean),
            purge_neg_inf(causal_paren_compare_attn_scores.dist_mean),
            causal_paren_compare_attn_scores.diff_mean,
        ]
    ),
    dim_names="which layer head Q K".split(),
    dim_types="which layer head seq seq".split(),
    dim_idx_names=[
        ["monte", "dist", "diff"],
        [str(i) for i in range(causal_paren_compare_attn_scores.diff_mean.shape[0])],
        [str(i) for i in range(causal_paren_compare_attn_scores.diff_mean.shape[1])],
        [str(i) for i in range(1, causal_paren_compare_attn_scores.diff_mean.shape[2] + 1)],
        [str(i) for i in range(causal_paren_compare_attn_scores.diff_mean.shape[3])],
    ],
    units="mean scores",
    title="mean scores",
)

# %%

import interp.cui as cui

await cui.init(port=6789)  # type: ignore

# %%

await cui.show_tensors(causal_paren_scores_mean_vnt)  # type: ignore

# %%

causal_paren_compare_attn_probs = CompareActivation(
    model,
    paren_causal_monte_cache,
    paren_causal_dist_cache,
    "attn_probs",
    "attn_probs",
    "attn_probs",
    seq_axis=2,
    remove_upper_triangle=True,
)

# %%

causal_paren_compare_attn_probs.plot_layer_idx(0, (0, 7, 6), bin_size=0.01)

# %%

causal_paren_probs_mean_vnt = VeryNamedTensor(
    jnp.stack(
        [
            causal_paren_compare_attn_probs.monte_mean,
            causal_paren_compare_attn_probs.dist_mean,
            causal_paren_compare_attn_probs.diff_mean,
        ]
    ),
    dim_names="which layer head Q K".split(),
    dim_types="which layer head seq seq".split(),
    dim_idx_names=[
        ["monte", "dist", "diff"],
        [str(i) for i in range(causal_paren_compare_attn_probs.diff_mean.shape[0])],
        [str(i) for i in range(causal_paren_compare_attn_probs.diff_mean.shape[1])],
        [str(i) for i in range(causal_paren_compare_attn_probs.diff_mean.shape[2])],
        [str(i) for i in range(causal_paren_compare_attn_probs.diff_mean.shape[3])],
    ],
    units="mean probs",
    title="mean probs",
)

# %%

await cui.show_tensors(causal_paren_probs_mean_vnt)  # type: ignore

# %%

additional_toks_induct_redwood = 7
red_loc = 3
wood_loc = 4
snd_red_loc = 7
red_tok = single_tokenize(" Red")
wood_tok = single_tokenize("wood")


@jax.jit
def sample_induct_redwood(key, unnormalized_logits, toks, next_tok_idx):
    return jax.lax.cond(
        (next_tok_idx == red_loc) | (next_tok_idx == snd_red_loc),
        lambda: weighted_sample_single(unnormalized_logits, toks.shape[0], red_tok),
        lambda: jax.lax.cond(
            next_tok_idx == wood_loc,
            lambda: weighted_sample_single(unnormalized_logits, toks.shape[0], wood_tok),
            lambda: sample_default(key, unnormalized_logits, toks, next_tok_idx),
        ),
    )


# %%

redwood_monte_unigrams, redwood_toks, _, redwood_prob_ratios = monte_gen(
    n_toks=additional_toks_induct_redwood + 1,
    n_samples=500_000,
    sample=sample_induct_redwood,
).as_tup()
assert redwood_monte_unigrams is not None

# %%

sample_and_print_toks(jax.random.PRNGKey(283882), redwood_prob_ratios, redwood_toks, n=30)

# %%

redwood_monte_unigrams[:, wood_tok]

# %%


def redwood_run_at_seq_idx(
    key: jax.random.KeyArray,
    dist: MultivariateNormal,
    seq_idx: Union[int, jnp.ndarray],
    iters: int = 1_000,
):
    dist, mean_probs, mean_include_prob = gpt_gen_new_dist_to_input(
        model, params, key, dist, iters=iters, final_embed_seq_idx=seq_idx, is_set=True
    )

    next_tok_idx = jnp.asarray(seq_idx + 1)

    def run_cond(toks) -> MultivariateNormal:
        return gpt_condition_on_input_embeds(
            model,
            params,
            dist,
            next_tok_idx,
            toks,
            mean_probs,
            set_instead_of_condition=False,
            is_dict=True,
        )

    dist = jax.lax.cond(
        (next_tok_idx == red_loc) | (next_tok_idx == wood_loc) | (next_tok_idx == snd_red_loc),
        lambda: run_cond(jnp.where(next_tok_idx == wood_loc, jnp.asarray(wood_tok), jnp.asarray(red_tok))),
        lambda: dist,
    )

    return dist, mean_probs, mean_include_prob


jitted_redwood_run_at_seq_idx = jax.jit(redwood_run_at_seq_idx, static_argnames=["iters"])

# %%


redwood_dist, _, _, _ = dist_gen(
    additional_toks=additional_toks_induct_redwood,
    pad_to=11,  # odd
    update_and_get_next=jitted_redwood_run_at_seq_idx,
)
redwood_tok_dists = estimate_token_dists(jax.random.PRNGKey(285), redwood_dist)

compare_estimate_to_monte(
    redwood_tok_dists,
    redwood_monte_unigrams,
    k=10,
    target_toks=[wood_tok],
)

# %%

additional_toks_induct = 10

# %%

eq_loc = 4
snd_eq_loc = 10
b_tok = single_tokenize("ley")

# %%


@jax.jit
def sample_induct_b(key, unnormalized_logits, toks, next_tok_idx):
    b_loc = eq_loc + 1
    return jax.lax.cond(
        next_tok_idx == snd_eq_loc,
        lambda: weighted_sample_single(unnormalized_logits, toks.shape[0], toks[:, eq_loc]),
        lambda: jax.lax.cond(
            next_tok_idx == b_loc,
            lambda: weighted_sample_single(unnormalized_logits, toks.shape[0], b_tok),
            lambda: sample_default(key, unnormalized_logits, toks, next_tok_idx),
        ),
    )


# %%

b_monte_unigrams, b_toks, _, b_prob_ratios = monte_gen(
    n_toks=additional_toks_induct + 1,
    n_samples=300_000,
    sample=sample_induct_b,
).as_tup()
assert b_monte_unigrams is not None
assert b_toks is not None
assert b_prob_ratios is not None

# %%

sample_and_print_toks(jax.random.PRNGKey(283882), b_prob_ratios, b_toks, n=30)

# %%

b_toks.shape

# %%

b_monte_unigrams[:, b_tok]

# %%

observed_unigrams = (
    jnp.zeros(
        (
            b_toks.shape[1],
            model.vocab_size,
        ),
        dtype=b_prob_ratios.dtype,
    )
    .at[jnp.arange(b_toks.shape[1])[:, None], jnp.swapaxes(b_toks, 0, 1)]
    .add(b_prob_ratios)
    / b_prob_ratios.sum()
)
observed_unigrams.shape

# %%

empirical_means = jnp.einsum(
    "k v, v h -> k h", observed_unigrams[1 : eq_loc + 1], model_b.embedding.token_embedding.embedding
)
first_means = jnp.concatenate(
    [jnp.expand_dims(model_b.embedding.token_embedding.embedding[begin_token()], 0), empirical_means]
)
empirical_covariances = jax.vmap(
    lambda uni: jnp.cov(model_b.embedding.token_embedding.embedding, aweights=uni, rowvar=False, bias=True)
)(observed_unigrams[1 : eq_loc + 1])
first_covariances = (
    jnp.zeros((*first_means.shape, *first_means.shape), dtype=first_means.dtype)
    .at[jnp.arange(1, first_means.shape[0]), :, jnp.arange(1, first_means.shape[0]), :]
    .set(empirical_covariances)
)


flat_config, flat_means = FlatValueConfig.from_tree(jnp.expand_dims(first_means, 0))

b_initial_dist = MultivariateNormal(
    flat_means, first_covariances.reshape(flat_means.size, flat_means.size), flat_config
).lin_op(lambda x: {"input_embeds": x, "final_embeds": jnp.zeros_like(x)})
b_initial_dist.mean_as()["input_embeds"].shape, empirical_covariances.shape

# %%


def b_run_at_seq_idx(
    key: jax.random.KeyArray,
    dist: MultivariateNormal,
    seq_idx: Union[int, jnp.ndarray],
    iters: int = 1_000,
):
    dist, mean_probs, mean_include_prob = gpt_gen_new_dist_to_input(
        model, params, key, dist, iters=iters, final_embed_seq_idx=seq_idx, is_set=True
    )

    next_tok_idx = jnp.asarray(seq_idx + 1)

    def run_cond(toks) -> MultivariateNormal:
        return gpt_condition_on_input_embeds(
            model,
            params,
            dist,
            next_tok_idx,
            toks,
            mean_probs,
            set_instead_of_condition=False,
            is_dict=True,
        )

    b_loc = eq_loc + 1
    dist = jax.lax.cond(
        next_tok_idx == b_loc,
        lambda: run_cond(jnp.asarray(b_tok)),
        lambda: jax.lax.cond(
            next_tok_idx == snd_eq_loc,
            # conditioning on diff == 0 conditions on the values being equal
            lambda: dist.condition(lambda x: x["input_embeds"][0, eq_loc] - x["input_embeds"][0, snd_eq_loc], 0.0),
            lambda: dist,
        ),
    )

    return dist, mean_probs, mean_include_prob


jitted_b_run_at_seq_idx = jax.jit(b_run_at_seq_idx, static_argnames=["iters"])

# %%

# this conditioning does quite poorly without f64 for whatever reason
b_dist, _, _, _ = dist_gen(
    model=model_f64,
    dist=jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), out_begin),
    additional_toks=additional_toks_induct,
    pad_to=11,  # odd
    sampling_eps=1e-12,
    update_and_get_next=jitted_b_run_at_seq_idx,
)
b_tok_dists = estimate_token_dists(jax.random.PRNGKey(285), b_dist)

# %%

compare_estimate_to_monte(
    b_tok_dists,
    b_monte_unigrams,
    k=10,
    target_toks=[b_tok],
)

# %%


def b_run_at_seq_idx_skip_first(
    key: jax.random.KeyArray,
    dist: MultivariateNormal,
    seq_idx: int,
    iters: int = 1_000,
):
    next_tok_idx = jnp.asarray(seq_idx + 1)
    assert next_tok_idx >= eq_loc - 1
    if next_tok_idx == eq_loc - 1:
        return dist, jnp.zeros((model.vocab_size,), dtype=dist.dtype), jnp.asarray(1.0, dtype=dist.dtype)
    elif next_tok_idx == snd_eq_loc:
        return (
            dist.lin_op(operate_f("input_embeds", "input_embeds", lambda x: x.at[0, snd_eq_loc].set(x[0, eq_loc]))),
            jnp.zeros((model.vocab_size,), dtype=dist.dtype),
            jnp.asarray(1.0, dtype=dist.dtype),
        )
    else:
        return jitted_b_run_at_seq_idx(key, dist, seq_idx, iters=iters)


# %%

# we'll use input embed dist from observed unigrams up through A
b_initial_dist, _, _, _ = dist_gen(
    dist=b_initial_dist,
    additional_toks=additional_toks_induct - (eq_loc - 1),
    pad_to=11,  # odd
    update_and_get_next=b_run_at_seq_idx_skip_first,
    start_toks=eq_loc,
)
b_initial_tok_dists = estimate_token_dists(jax.random.PRNGKey(285), b_initial_dist)

compare_estimate_to_monte(
    b_initial_tok_dists,
    b_monte_unigrams,
    k=10,
    target_toks=[b_tok],
)

# %%


def b_graft_eq(
    key: jax.random.KeyArray,
    dist: MultivariateNormal,
    seq_idx: int,
    just_copy: bool = False,
    iters: int = 1_000,
):
    next_tok_idx = jnp.asarray(seq_idx + 1)
    if next_tok_idx == eq_loc and not just_copy:
        return (
            graft_input_embed(dist, eq_loc, empirical_means[eq_loc - 1], empirical_covariances[eq_loc - 1]),
            jnp.zeros((model.vocab_size,), dtype=dist.dtype),
            jnp.asarray(1.0, dtype=dist.dtype),
        )
    elif next_tok_idx == snd_eq_loc:
        return (
            dist.lin_op(operate_f("input_embeds", "input_embeds", lambda x: x.at[0, snd_eq_loc].set(x[0, eq_loc]))),
            jnp.zeros((model.vocab_size,), dtype=dist.dtype),
            jnp.asarray(1.0, dtype=dist.dtype),
        )
    else:
        return jitted_b_run_at_seq_idx(key, dist, seq_idx, iters=iters)


# %%

b_graft_dist, _, _, _ = dist_gen(
    additional_toks=additional_toks_induct,
    pad_to=11,  # odd
    update_and_get_next=b_graft_eq,
)
b_graft_tok_dists = estimate_token_dists(jax.random.PRNGKey(285), b_graft_dist)

compare_estimate_to_monte(
    b_graft_tok_dists,
    b_monte_unigrams,
    k=10,
    target_toks=[b_tok],
)

# %%

b_just_copy_dist, _, _, _ = dist_gen(
    additional_toks=additional_toks_induct,
    pad_to=11,  # odd
    update_and_get_next=partial(b_graft_eq, just_copy=True),
)
b_just_copy_tok_dists = estimate_token_dists(jax.random.PRNGKey(285), b_just_copy_dist)

compare_estimate_to_monte(
    b_just_copy_tok_dists,
    b_monte_unigrams,
    k=10,
    target_toks=[b_tok],
)

# %%


@jax.jit
def causal_sample_induct_b(key, unnormalized_logits, toks, next_tok_idx):
    b_loc = eq_loc + 1
    return jax.lax.cond(
        next_tok_idx == eq_loc,
        lambda: sample_default(
            key,
            jnp.broadcast_to(jnp.log(observed_unigrams[eq_loc])[None], unnormalized_logits.shape),
            toks,
            next_tok_idx,
        ),
        lambda: jax.lax.cond(
            next_tok_idx == snd_eq_loc,
            lambda: weighted_sample_single(unnormalized_logits, toks.shape[0], toks[:, eq_loc], weight=False),
            lambda: jax.lax.cond(
                next_tok_idx == b_loc,
                lambda: weighted_sample_single(unnormalized_logits, toks.shape[0], b_tok, weight=False),
                lambda: sample_default(key, unnormalized_logits, toks, next_tok_idx),
            ),
        ),
    )


# %%

b_causal_monte_unigrams, b_causal_toks, _, b_causal_prob_ratios = monte_gen(
    n_toks=additional_toks_induct + 1,
    n_samples=300_000,
    sample=causal_sample_induct_b,
).as_tup()
assert b_causal_monte_unigrams is not None
assert b_causal_toks is not None
assert b_causal_prob_ratios is not None

# %%

b_causal_prob_ratios

# %%


sample_and_print_toks(jax.random.PRNGKey(283882), b_causal_prob_ratios, b_causal_toks, n=30)

# %%


def b_causal_run_at_seq_idx(
    key: jax.random.KeyArray,
    dist: MultivariateNormal,
    seq_idx: Union[int, jnp.ndarray],
    b_tok,
    eq_loc,
    snd_eq_loc,
    graft_mean=jnp.zeros((model.hidden_size,)),
    graft_cov=jnp.zeros((model.hidden_size, model.hidden_size)),
    graft_eq_loc: Union[bool, jnp.ndarray] = False,
    b_tok_is_dist: bool = False,
    set_b_tok: Union[bool, jnp.ndarray] = True,
    iters: int = 1_000,
):
    dist, mean_probs, mean_include_prob = gpt_gen_new_dist_to_input(
        model, params, key, dist, iters=iters, final_embed_seq_idx=seq_idx, is_set=True
    )

    graft_mean = jnp.asarray(graft_mean, dtype=dist.dtype)
    graft_cov = jnp.asarray(graft_cov, dtype=dist.dtype)

    next_tok_idx = jnp.asarray(seq_idx + 1)

    if set_b_tok:
        if b_tok_is_dist:
            probs = jax.nn.softmax(b_tok)
            b_tok_mean = jnp.einsum("v, v h -> h", probs, model_b.embedding.token_embedding.embedding)
            b_tok_cov = jnp.cov(model_b.embedding.token_embedding.embedding, rowvar=False, bias=True, aweights=probs)
        else:
            b_tok_mean = model_b.embedding.token_embedding.embedding[b_tok]
            b_tok_cov = 0.0

    b_loc = eq_loc + 1
    dist = jax.lax.cond(
        (next_tok_idx == eq_loc) & graft_eq_loc,
        lambda: graft_input_embed(dist, eq_loc, graft_mean, graft_cov),
        lambda: jax.lax.cond(
            next_tok_idx == snd_eq_loc,
            lambda: dist.lin_op(
                operate_f("input_embeds", "input_embeds", lambda x: x.at[0, snd_eq_loc].set(x[0, eq_loc]))
            ),
            lambda: jax.lax.cond(
                (next_tok_idx == b_loc) & set_b_tok,
                lambda: graft_input_embed(dist, b_loc, b_tok_mean, b_tok_cov),
                lambda: dist,
            ),
        ),
    )

    return dist, mean_probs, mean_include_prob


jitted_b_causal_run_at_seq_idx = jax.jit(b_causal_run_at_seq_idx, static_argnames=["iters", "b_tok_is_dist"])

# %%

b_causal_dist, _, _, _ = dist_gen(
    additional_toks=additional_toks_induct,
    pad_to=11,  # odd
    update_and_get_next=partial(
        jitted_b_causal_run_at_seq_idx,
        b_tok=b_tok,
        eq_loc=eq_loc,
        snd_eq_loc=snd_eq_loc,
        graft_mean=empirical_means[eq_loc - 1],
        graft_cov=empirical_covariances[eq_loc - 1],
        graft_eq_loc=True,
    ),
)
b_causal_tok_dists = estimate_token_dists(jax.random.PRNGKey(285), b_causal_dist)

compare_estimate_to_monte(
    b_causal_tok_dists,
    b_causal_monte_unigrams,
    k=10,
    target_toks=[b_tok],
)

# %%

# normal = use typical distribution for A
@partial(jax.jit, static_argnames=["eq_loc", "snd_eq_loc", "b_tok_is_dist"])
def causal_sample_normal_dist_induct(
    key, unnormalized_logits, toks, next_tok_idx, b_tok, eq_loc, snd_eq_loc, b_tok_is_dist
):
    b_loc = eq_loc + 1
    return jax.lax.cond(
        next_tok_idx == snd_eq_loc,
        lambda: weighted_sample_single(unnormalized_logits, toks.shape[0], toks[:, eq_loc], weight=False),
        lambda: jax.lax.cond(
            next_tok_idx == b_loc,
            lambda: (
                sample_default(key, jnp.broadcast_to(b_tok[None], unnormalized_logits.shape), toks, next_tok_idx)
                if b_tok_is_dist
                else weighted_sample_single(unnormalized_logits, toks.shape[0], b_tok, weight=False)
            ),
            lambda: sample_default(key, unnormalized_logits, toks, next_tok_idx),
        ),
    )


# %%


def monte_causal_induction(
    b_tok,
    eq_loc,
    snd_eq_loc,
    b_tok_is_dist=False,
    return_cache=False,
    n_samples=300_000,
    extra_log: Iterable[KeyIdxs] = [],
):
    new_monte_logger = deepcopy(default_monte_logger)
    new_monte_logger.add_all(extra_log)
    unigrams, toks, cache, prob_ratios = monte_gen(
        n_toks=additional_toks_induct + 1,
        n_samples=n_samples,
        return_cache=return_cache,
        sample=partial(
            causal_sample_normal_dist_induct,
            b_tok=b_tok,
            eq_loc=eq_loc,
            snd_eq_loc=snd_eq_loc,
            b_tok_is_dist=b_tok_is_dist,
        ),
        logger=new_monte_logger,
    ).as_tup()
    assert unigrams is not None
    assert toks is not None
    assert prob_ratios is not None

    return unigrams, toks, cache, prob_ratios


def dist_gen_causal_induction(b_tok, eq_loc, snd_eq_loc, b_tok_is_dist=False, **kwargs):
    return dist_gen(
        additional_toks=additional_toks_induct,
        pad_to=11,  # odd
        update_and_get_next=partial(
            jitted_b_causal_run_at_seq_idx,
            b_tok=b_tok,
            eq_loc=eq_loc,
            snd_eq_loc=snd_eq_loc,
            graft_eq_loc=False,
            b_tok_is_dist=b_tok_is_dist,
        ),
        **kwargs,
    )


def run_and_print_causal_induction(
    b_tok,
    eq_loc,
    snd_eq_loc,
    compute_implied_probs=False,
    config: Gpt.ApplyToNormalConfig = Gpt.ApplyToNormalConfig(),
    extra_monte_log: List[KeyIdxs] = [],
    dist_prop_logger: Logger = NoopLogger(),
    n_samples=300_000,
    file=None,
    b_tok_is_dist=False,
):
    unigrams, toks, monte_log, prob_ratios = monte_causal_induction(
        b_tok,
        eq_loc,
        snd_eq_loc,
        b_tok_is_dist=b_tok_is_dist,
        return_cache=len(extra_monte_log) > 0,
        extra_log=extra_monte_log,
        n_samples=n_samples,
    )

    b_loc = eq_loc + 1

    sample_and_print_toks(jax.random.PRNGKey(283882), prob_ratios, toks, n=30, file=file)

    dist, tok_dists, _, dist_cache = dist_gen_causal_induction(
        b_tok,
        eq_loc,
        snd_eq_loc,
        b_tok_is_dist=b_tok_is_dist,
        config=config,
        logger=dist_prop_logger,
        maybe_invalid_shape=not isinstance(dist_prop_logger, NoopLogger),
    )

    compare_estimate_to_monte(
        tok_dists,
        unigrams,
        k=10,
        target_toks=[] if b_tok_is_dist else [b_tok],
        file=file,
    )

    if not compute_implied_probs:
        return dist, monte_log, dist_cache

    def get_prob_eq_tok(tok, key, iters=16):
        cond_tok_dists = estimate_token_dists(
            key,
            gpt_condition_on_input_embeds(model, params, dist, new_tok_idx=b_loc, toks=tok, is_dict=True),
            iters=iters,
        )

        return cond_tok_dists[:, tok]

    tok_samples = (
        jax.random.categorical(jax.random.PRNGKey(283840), b_tok, shape=(100,)) if b_tok_is_dist else jnp.array([b_tok])
    )

    eq_target = toks[:, b_loc, None, None] == tok_samples[None, None]

    monte_eq_by_tok = ((toks[:, b_loc, None, None] == toks[:, 1:, None]) & eq_target).sum(axis=0) / eq_target.sum(
        axis=0
    )

    total_dist_eq_probs = 0.0
    for i, tok in enumerate(tok_samples):
        dist_eq_probs_per_this = get_prob_eq_tok(
            jnp.asarray(tok), jax.random.PRNGKey(28224 + i), iters=100 if b_tok_is_dist else 500
        )
        total_dist_eq_probs += dist_eq_probs_per_this
        print("for tok", tokenizer.decode(tok))
        print(
            tabulate(
                list(
                    zip(
                        range(1, tok_dists.shape[0] + 1),
                        dist_eq_probs_per_this,
                        monte_eq_by_tok[:, i],
                        dist_eq_probs_per_this - monte_eq_by_tok[:, i],
                    )
                ),
                headers=["token", "dists eq prob", "monte eq prob", "error"],
            ),
        )

    assert not isinstance(total_dist_eq_probs, float)
    dist_eq_probs = total_dist_eq_probs / tok_samples.shape[0]
    monte_eq = (toks[:, b_loc, None] == toks[:, 1:]).mean(axis=0)

    print("overall")
    print(
        tabulate(
            list(
                zip(
                    range(1, tok_dists.shape[0] + 1),
                    dist_eq_probs,
                    monte_eq,
                    dist_eq_probs - monte_eq,
                )
            ),
            headers=["token", "dists eq prob", "monte eq prob", "error"],
        ),
    )

    return dist, monte_log, dist_cache


# %%

_ = run_and_print_causal_induction(b_tok, eq_loc, snd_eq_loc)

# %%

_ = run_and_print_causal_induction(single_tokenize(" James"), eq_loc=1, snd_eq_loc=10)

# %%

_ = run_and_print_causal_induction(single_tokenize("ing"), eq_loc=1, snd_eq_loc=10)

# %%

_ = run_and_print_causal_induction(single_tokenize("s"), eq_loc=4, snd_eq_loc=10)

# %%

_ = run_and_print_causal_induction(single_tokenize("'s"), eq_loc=4, snd_eq_loc=10)

# %%

_ = run_and_print_causal_induction(single_tokenize(":"), eq_loc=4, snd_eq_loc=10)

# %%

_ = run_and_print_causal_induction(single_tokenize(" in"), eq_loc=4, snd_eq_loc=10)

# %%

_ = run_and_print_causal_induction(single_tokenize(" Trump"), eq_loc=4, snd_eq_loc=10)

# %%

doom_dist, doom_monte_cache_out, doom_dist_cache_out = run_and_print_causal_induction(
    single_tokenize(" doom"),
    eq_loc=4,
    snd_eq_loc=10,
    extra_monte_log=to_cache,
    dist_prop_logger=logger_cache,
    n_samples=200_000,
)
doom_monte_cache, doom_dist_cache = LogCache.unwrap(doom_monte_cache_out), LogCache.unwrap(doom_dist_cache_out)

# %%

doom_compare_attn_probs = CompareActivation(
    model,
    doom_monte_cache,
    doom_dist_cache,
    "attn_probs",
    "attn_probs",
    "attn_probs",
    seq_axis=2,
    remove_upper_triangle=True,
)

# %%

layer = 1
# h q k
base_idx = (4, 10, 5)

doom_compare_attn_probs.plot_layer_idx(layer, base_idx, bin_size=0.01)

# %%

doom_probs_mean_vnt = VeryNamedTensor(
    jnp.stack(
        [
            doom_compare_attn_probs.monte_mean,
            doom_compare_attn_probs.dist_mean,
            doom_compare_attn_probs.diff_mean,
        ]
    ),
    dim_names="which layer head Q K".split(),
    dim_types="which layer head seq seq".split(),
    dim_idx_names=[
        ["monte", "dist", "diff"],
        [str(i) for i in range(doom_compare_attn_probs.diff_mean.shape[0])],
        [str(i) for i in range(doom_compare_attn_probs.diff_mean.shape[1])],
        [str(i) for i in range(doom_compare_attn_probs.diff_mean.shape[2])],
        [str(i) for i in range(doom_compare_attn_probs.diff_mean.shape[3])],
    ],
    units="mean probs",
    title="mean probs",
)

# %%

await cui.show_tensors(doom_probs_mean_vnt)  # type: ignore

# %%

# TODO: consider restoring some grafting experiments and/or running more
# grafting experiments to see what matters

# %%

# works for fine for independent B
_ = run_and_print_causal_induction(
    jnp.log(jnp.asarray(unigram / unigram.sum(), dtype=jnp.float32)),
    4,
    10,
    compute_implied_probs=True,
    n_samples=400_000,
    b_tok_is_dist=True,
)

# %%

traditionally_male_names = [" Bob", " James", " Bill", " Jim", " Ryan", " Jacob"]
traditionally_female_names = [" Sally", " Jenny", " Sarah"]

base_m_name_toks = strings_to_tok_list(traditionally_male_names)
base_f_name_toks = strings_to_tok_list(traditionally_female_names)

# %%

mean_sim_male = jnp.einsum("n k, v k -> n v", unit_embeds[jnp.asarray(base_m_name_toks)], unit_embeds).mean(axis=0)
m_name_toks = jax.lax.top_k(mean_sim_male, k=40)[1]

print_max_min_by_tok_k(
    mean_sim_male,
    print_min=False,
    k=40,
)

mean_sim_female = jnp.einsum("n k, v k -> n v", unit_embeds[jnp.asarray(base_f_name_toks)], unit_embeds).mean(axis=0)
f_name_toks = jax.lax.top_k(mean_sim_female, k=40)[1]

print_max_min_by_tok_k(
    mean_sim_female,
    print_min=False,
    k=40,
)

non_m_name_toks = other_toks(m_name_toks)
non_f_name_toks = other_toks(f_name_toks)

non_m_name_toks.shape, non_f_name_toks.shape

# %%

name_loc = 4
additional_toks_name = 10

# %%


@partial(jax.jit, static_argnames=["loc", "weight"])
def sample_except_at_loc(key, unnormalized_logits, toks, next_tok_idx, except_toks, loc, weight=True):
    return jax.lax.cond(
        next_tok_idx == loc,
        lambda: weighted_sample_except(key, unnormalized_logits, except_toks, weight=weight),
        lambda: sample_default(key, unnormalized_logits, toks, next_tok_idx),
    )


# %%


def except_at_loc(
    key: jax.random.KeyArray,
    dist: MultivariateNormal,
    seq_idx: Union[int, jnp.ndarray],
    except_toks,
    include_toks,
    loc,
    condition=False,
    iters: int = 1_000,
):
    gen_new = lambda mask: gpt_gen_new_dist_to_input(
        model,
        params,
        key,
        dist,
        iters=iters,
        final_embed_seq_idx=seq_idx,
        exclude_toks_mask=mask,
        is_set=True,
    )

    if condition:
        dist, mean_probs, mean_include_prob = gen_new(None)

        next_tok_idx = seq_idx + 1

        dist = jax.lax.cond(
            seq_idx == loc,
            lambda: gpt_condition_on_input_embeds(
                model, params, dist, next_tok_idx, include_toks, mean_probs, is_dict=True
            ),
            lambda: dist,
        )

        return dist, mean_probs, mean_include_prob

    else:
        return jax.lax.cond(seq_idx == loc, lambda: gen_new(except_toks), lambda: gen_new(None))


jitted_except_at_loc = jax.jit(except_at_loc, static_argnames=["condition", "iters"])


# %%


def monte_except_at_loc(except_toks, condition, loc=name_loc, n_samples=200_000):

    out_unigrams, out_toks, _, out_prob_ratios = monte_gen(
        n_toks=additional_toks_name + 1,
        n_samples=n_samples,
        sample=partial(sample_except_at_loc, except_toks=except_toks, loc=loc, weight=condition),
    ).as_tup()
    assert out_unigrams is not None
    assert out_prob_ratios is not None
    assert out_toks is not None

    return out_unigrams, out_toks, out_prob_ratios


def dist_gen_except_at_loc(except_toks, include_toks, condition, loc=name_loc):
    dist, mean_probs, _, _ = dist_gen(
        additional_toks=additional_toks_name,
        pad_to=11,  # odd
        update_and_get_next=partial(
            jitted_except_at_loc,
            except_toks=except_toks,
            include_toks=include_toks,
            loc=loc,
            condition=condition,
        ),
    )
    if condition:
        return dist, estimate_token_dists(jax.random.PRNGKey(285), dist)
    else:
        return dist, mean_probs


# %%

m_name_causal_monte_unigrams, m_name_causal_toks, m_name_causal_prob_ratios = monte_except_at_loc(
    non_m_name_toks, condition=False
)

# %%

sample_and_print_toks(jax.random.PRNGKey(2839), m_name_causal_prob_ratios, m_name_causal_toks, n=30)

# %%

m_name_causal_dist, m_name_causal_tok_dists = dist_gen_except_at_loc(non_m_name_toks, m_name_toks, condition=False)

# %%


compare_estimate_to_monte(
    m_name_causal_tok_dists,
    m_name_causal_monte_unigrams,
    k=10,
    target_toks=strings_to_tok_list([" he", " she", " his", " her"]),
)

# %%


m_name_cond_monte_unigrams, m_name_cond_toks, m_name_cond_prob_ratios = monte_except_at_loc(
    non_m_name_toks, condition=True, n_samples=500_000
)


# %%

sample_and_print_toks(jax.random.PRNGKey(2839), m_name_cond_prob_ratios, m_name_cond_toks, n=30)

# %%

m_name_cond_dist, m_name_cond_tok_dists = dist_gen_except_at_loc(non_m_name_toks, m_name_toks, condition=True)

# %%


compare_estimate_to_monte(
    m_name_cond_tok_dists,
    m_name_cond_monte_unigrams,
    k=10,
    target_toks=strings_to_tok_list([" he", " she", " his", " her"]),
)

# %%

f_name_causal_monte_unigrams, f_name_causal_toks, f_name_causal_prob_ratios = monte_except_at_loc(
    non_f_name_toks, condition=False
)

# %%

sample_and_print_toks(jax.random.PRNGKey(2839), f_name_causal_prob_ratios, f_name_causal_toks, n=30)

# %%

f_name_causal_dist, f_name_causal_tok_dists = dist_gen_except_at_loc(non_f_name_toks, f_name_toks, condition=False)

# %%


compare_estimate_to_monte(
    f_name_causal_tok_dists,
    f_name_causal_monte_unigrams,
    k=10,
    target_toks=strings_to_tok_list([" he", " she", " his", " her"]),
)

# %%


f_name_cond_monte_unigrams, f_name_cond_toks, f_name_cond_prob_ratios = monte_except_at_loc(
    non_f_name_toks, condition=True, n_samples=500_000
)


# %%

sample_and_print_toks(jax.random.PRNGKey(2839), f_name_cond_prob_ratios, f_name_cond_toks, n=30)

# %%

f_name_cond_dist, f_name_cond_tok_dists = dist_gen_except_at_loc(non_f_name_toks, f_name_toks, condition=True)

# %%


compare_estimate_to_monte(
    f_name_cond_tok_dists,
    f_name_cond_monte_unigrams,
    k=10,
    target_toks=strings_to_tok_list([" he", " she", " his", " her"]),
)
