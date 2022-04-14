from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp

from interp.tools.interpretability_tools import check_close_weak
import interp.tools.optional as op
from interp.model.model_fixtures import tiny_random_model, example_seq, example_seq_no_begin, ModelParams
from interp.model.monte_carlo import (
    GenerativeReturn,
    monte_carlo_generative,
    log_cache_clean_up_stacked,
    log_cache_combine_to_batch,
)
from interp.tools.indexer import I
from interp.tools.log import Idxs, KeyIdxs, LoggerCache, LogCache

_ = (tiny_random_model, example_seq, example_seq_no_begin)


def test_activation_cache(tiny_random_model: ModelParams, example_seq: jnp.ndarray):
    model, params = tiny_random_model
    key_idxs = [KeyIdxs("final_out.inp"), KeyIdxs("blocks.attention.out_by_head", Idxs.all())]
    logger = LoggerCache.from_key_idxs(key_idxs)

    base_seq_len = 17

    n_toks = 4

    def get(pad: bool, act_cache: bool, pad_to: Optional[int]) -> GenerativeReturn:
        return monte_carlo_generative(
            jax.random.PRNGKey(0),
            model,
            params,
            n_toks=n_toks,
            n_samples=16,
            batch_size=4,
            prompt=example_seq.squeeze(0)[:base_seq_len],
            pad=pad,
            pad_to=pad_to,
            activation_cache=act_cache,
            return_toks=True,
            return_cache=True,
            disable_progress=True,
            logger=logger,
            clean_up_cache_post_scan=log_cache_clean_up_stacked,
            stack_caches=partial(log_cache_combine_to_batch, activation_cache=act_cache),
        )

    extra = [0, 1]
    base_seq_axes = [1, 2]
    seq_axes = [b + e for b, e in zip(base_seq_axes, extra)]
    slices = [(I[:],) * s + (I[base_seq_len : base_seq_len + n_toks - 1],) for s in seq_axes]

    base = get(False, False, None)
    cache_b = LogCache.unwrap(base.caches)

    comp_vals = [cache_b.get(ki, check=True)[sl] for ki, sl in zip(key_idxs, slices)]

    for pad, pad_to, act_cache in [
        (False, None, True),
        (True, None, False),
        (True, None, True),
        (True, 30, False),
        (True, 30, True),
    ]:
        compare = get(pad, act_cache, pad_to)
        assert op.unwrap(base.toks).shape == op.unwrap(compare.toks).shape
        op.unwrap(base.toks) == op.unwrap(compare.toks)
        cache_c = LogCache.unwrap(compare.caches)

        for ki, s, sl, other, ex in zip(key_idxs, seq_axes, slices, comp_vals, extra):
            raw = cache_c.get(ki, check=True)
            if act_cache:
                comp_v = raw.swapaxes(ex + 1, s + 1).squeeze(ex + 1)
            else:
                comp_v = raw[sl]
            assert comp_v.shape == other.shape

            check_close_weak(comp_v, other)
