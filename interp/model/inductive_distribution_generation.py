from typing import Callable, Optional, Tuple, Any, Union


import jax
import jax.numpy as jnp
from flax.core.scope import FrozenVariableDict
from attrs import evolve
from tqdm import trange

from interp.model.gpt_model import (
    Gpt,
    gpt_apply_to_normal,
    gpt_apply_to_normal_unjit,
    gpt_gen_new_dist_to_input,
    gpt_gen_new_dist_to_input_unjit,
)
from interp.model.apply_to_normal_utils import pad_dist, un_pad_dist
from interp.model.gpt_modules import AttnApplyToNormalConfig
from interp.tools.log import LogInfo, Logger, LogCache
from interp.tools.multivariate_normal import MultivariateNormal
import interp.tools.optional as op

UpdateAndGetNext = Callable[
    [jax.random.KeyArray, MultivariateNormal, int],
    Tuple[MultivariateNormal, jnp.ndarray, jnp.ndarray],
]


def next_dist(
    model: Gpt,
    params: FrozenVariableDict,
    key: jax.random.KeyArray,
    dist: MultivariateNormal,
    update_and_get_next: UpdateAndGetNext,
    seq_idx: int,
    config: Gpt.ApplyToNormalConfig,
    logger: Optional[Logger],
    jit_apply_to: bool = True,
) -> Tuple[MultivariateNormal, jnp.ndarray, jnp.ndarray, Any]:
    key, subkey = jax.random.split(key)
    dist, next_probs, mean_include_prob = update_and_get_next(subkey, dist, seq_idx)

    key, subkey = jax.random.split(key)
    dist, cache = (gpt_apply_to_normal if jit_apply_to else gpt_apply_to_normal_unjit)(
        model,
        params,
        subkey,
        dist.lin_op(lambda x: x["input_embeds"]),
        config=config,
        log_info=op.map(logger, LogInfo),
    )

    return dist, next_probs, mean_include_prob, cache


def get_default_update_and_get_next(model: Gpt, params: FrozenVariableDict, iters=1_000, jit=True):
    return lambda key, x, seq_idx: (gpt_gen_new_dist_to_input if jit else gpt_gen_new_dist_to_input_unjit)(
        model, params, key, x, iters=iters, final_embed_seq_idx=seq_idx, is_set=True
    )


def inductive_distribution_generation(
    model: Gpt,
    params: FrozenVariableDict,
    key,
    dist: MultivariateNormal,
    update_and_get_next: UpdateAndGetNext,
    config: Gpt.ApplyToNormalConfig = Gpt.ApplyToNormalConfig(),
    additional_toks: int = 5,
    pad_to: Optional[int] = None,
    check_valid_per: bool = True,
    sampling_eps: float = 4e-3,
    disable_progress: bool = False,
    start_toks: Optional[int] = None,
    logger: Optional[Logger] = None,
    **kwargs,
):
    if check_valid_per:
        dist.check_valid()
    dist = evolve(dist, sampling_eps=sampling_eps, **kwargs)

    if start_toks is None:
        start_toks = dist.mean_as()["input_embeds"].shape[1]
    assert start_toks is not None

    if pad_to is None:
        pad_to = additional_toks + start_toks

    assert pad_to is not None
    assert pad_to >= additional_toks + start_toks

    dist = pad_dist(dist, pad_to)

    probs = []

    out_cache: Optional[Any] = None

    overall_include_prob = 1.0

    for i in trange(additional_toks, disable=disable_progress):
        dist, next_probs, mean_include_prob, next_cache = next_dist(
            model,
            params,
            key,
            dist,
            update_and_get_next,
            seq_idx=i + start_toks - 1,
            config=config,
            logger=logger,
        )
        if i == additional_toks - 1:
            out_cache = next_cache

        overall_include_prob *= mean_include_prob
        probs.append(next_probs)

        if check_valid_per:
            dist.check_valid()

    key, subkey = jax.random.split(key)
    dist, next_probs, mean_include_prob = update_and_get_next(subkey, dist, start_toks + additional_toks - 1)
    overall_include_prob *= mean_include_prob
    probs.append(next_probs)

    dist = un_pad_dist(dist, start_toks + additional_toks)

    return dist, jnp.stack(probs), overall_include_prob, out_cache
