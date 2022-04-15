from typing import Any, Callable, Protocol, Tuple, List, Optional, Union
from functools import partial

from flax.core.scope import FrozenVariableDict
import jax
import jax.numpy as jnp
import einops
from tqdm import trange
from attrs import evolve, frozen

from interp.model.gpt_model import Gpt, gpt_call, to_any
from interp.model.gpt_modules import AttnActivationCache
from interp.tools.interpretability_tools import begin_token, get_interp_tokenizer
from interp.tools.jax_util import stack_tree
from interp.tools.log import KeyIdxs, LogInfo, Logger, LoggerCache, NoopLogger, LogCache
import interp.tools.optional as op


@jax.jit
def sample_default(
    key: jax.random.KeyArray,
    unnormalized_logits: jnp.ndarray,
    toks: jnp.ndarray,
    _,
    temp: Union[float, jnp.ndarray] = 1.0,
):
    return jax.random.categorical(key, unnormalized_logits / temp), jnp.full(
        toks.shape[0], 1.0, dtype=unnormalized_logits.dtype
    )


class RunInner(Protocol):
    def __call__(
        self,
        key: jax.random.KeyArray,
        model: Gpt,
        params: FrozenVariableDict,
        prompt_n_toks: int,
        n_toks: int,
        batch_size: int,
        initial_toks,
        initial_logits,
        cache: LogCache,
        logger: LoggerCache,
        sample,
        pad: bool,
        pad_to: int,
        activation_cache: bool,
        clean_up_cache_post_scan: Callable[[Any], Any],
    ) -> Tuple[jnp.ndarray, List[Logger], jnp.ndarray]:
        def get_next(key, out, toks, next_tok_idx):
            samples, prob_ratios = sample(key, out, toks, next_tok_idx)

            if pad:
                out_toks = toks.at[:, next_tok_idx].set(samples)
            else:
                out_toks = jnp.concatenate([toks, jnp.expand_dims(samples, 1)], axis=-1)

            return out_toks, prob_ratios

        key, subkey = jax.random.split(key)
        toks, prob_ratios = get_next(
            subkey,
            einops.repeat(initial_logits, "1 n -> b n", b=batch_size),
            einops.repeat(initial_toks, "n -> b n", b=batch_size),
            prompt_n_toks,
        )

        keys = jax.random.split(key, n_toks - 1)

        init_v = (toks, prob_ratios, cache)

        def body_fun(last: Tuple[jnp.ndarray, jnp.ndarray, LogCache], i_key):
            i, key = i_key
            idx = i + prompt_n_toks
            toks, prob_ratios, cache = last
            out, new_cache = gpt_call(
                model,
                params,
                jnp.expand_dims(toks[:, idx], -1) if activation_cache else (toks[:, :-1] if pad else toks),
                log_info=LogInfo(logger),
                config=Gpt.CallConfig(
                    activation_cache=AttnActivationCache.from_cache(idx, cache, is_set=pad)
                    if activation_cache
                    else None,
                ),
            )
            assert isinstance(new_cache, LogCache)
            if not activation_cache:
                out = out[:, idx]
            else:
                out = out.squeeze(1)

            toks, new_prob_ratios = get_next(key, out, toks, idx + 1)
            prob_ratios = prob_ratios * new_prob_ratios

            return (toks, prob_ratios, evolve(new_cache, sub_log_cache=None)), new_cache.sub_log_cache

        if pad:
            (last, sub_cache_out) = jax.lax.scan(body_fun, init_v, (jnp.arange(n_toks - 1), keys))

            if activation_cache:
                sub_cache_out = clean_up_cache_post_scan(sub_cache_out)
            else:
                sub_cache_out = jax.tree_util.tree_map(lambda x: x[-1], sub_cache_out)
        else:
            last = init_v
            sub_cache_out = []
            for i_key in enumerate(keys):
                last, sub_log = body_fun(last, i_key)
                sub_cache_out.append(sub_log)

            if activation_cache:
                sub_cache_out = clean_up_cache_post_scan(stack_tree(sub_cache_out))
            else:
                sub_cache_out = sub_cache_out[-1]

        (toks, prob_ratios, _) = last

        return toks[:, : prompt_n_toks + n_toks], sub_cache_out, prob_ratios


class RunInnerUnjit(RunInner):
    ...


run_inner_unjit = RunInnerUnjit()
run_inner: RunInner = to_any(
    jax.jit(
        run_inner_unjit,
        static_argnames=[
            "model",
            "prompt_n_toks",
            "n_toks",
            "batch_size",
            "sample",
            "pad",
            "pad_to",
            "activation_cache",
            "clean_up_cache_post_scan",
        ],
    )
)


@frozen
class GenerativeReturn:
    reduced: Optional[Any]
    toks: Optional[jnp.ndarray]
    caches: Optional[Any]
    prob_ratios: Optional[jnp.ndarray]

    def as_tup(self):
        return (self.reduced, self.toks, self.caches, self.prob_ratios)


def monte_carlo_generative(
    key: jax.random.KeyArray,
    model: Gpt,
    params: FrozenVariableDict,
    n_toks: int,
    n_samples: int,
    prompt=[begin_token()],
    batch_size: int = 16,
    sample=sample_default,
    reduce=None,
    return_toks=True,
    return_cache=False,
    return_prob_ratios=False,
    disable_progress=False,
    logger: Logger = NoopLogger(),
    pad: bool = True,
    pad_to: Optional[int] = None,
    activation_cache: bool = True,
    jit_inner=True,
    clean_up_cache_post_scan: Callable[[Any], Any] = lambda x: x,
    stack_caches: Callable[[List[Any]], Any] = lambda x: x,
) -> GenerativeReturn:
    """
    This is a (hopefully) quite general function for sampling from the model
    and doing computations.

    pad=True improves compile time at theoretical cost of runtime perf (in
    practice, seems free).

    Note that the way log is returned depends on the value of pad: see
    interp/model/test_activation_cache.py for example.
    """
    assert n_toks > 0

    summed_ratios = None
    summed_reduced = None
    toks_all = []
    caches_all = []
    prob_ratios_all = []

    if isinstance(prompt, list) and begin_token() in prompt:
        # a bit silly, but maybe good to check
        assert model.vocab_size > 50257

    prompt = jnp.array(prompt)

    # support batching over prompts when needed
    assert prompt.ndim == 1

    final_size = int(prompt.shape[0]) + n_toks
    if pad_to is None:
        pad_to = final_size
    assert pad_to >= final_size

    padded_prompt = prompt
    if pad:
        padded_prompt = jnp.pad(prompt, (0, pad_to - prompt.shape[0] - 1))

    # maybe we should support passing in initial log + logits?
    # not most efficient with pad, but fine
    activation_logger = LoggerCache.from_key_idxs(AttnActivationCache.to_log() if activation_cache else [])
    initial_logits, initial_cache_op = gpt_call(
        model,
        params,
        jnp.expand_dims(padded_prompt, 0),
        log_info=LogInfo(activation_logger),
    )
    initial_cache: LogCache = op.unwrap(initial_cache_op)
    initial_cache = evolve(
        initial_cache,
        idxed_cache={
            k: evolve(v, values=jnp.broadcast_to(v.values, (v.values.shape[0], batch_size, *v.values.shape[2:])))
            for k, v in initial_cache.idxed_cache.items()
        },
    )
    initial_logits = initial_logits[:, prompt.shape[0] - 1]

    if pad:
        # pad one more for inserted last token
        padded_prompt = jnp.pad(padded_prompt, (0, 1))

    inner = run_inner if jit_inner else run_inner_unjit

    for _ in trange(n_samples // batch_size, disable=disable_progress):
        key, subkey = jax.random.split(key)
        toks, iter_cache, prob_ratios = inner(
            subkey,
            model,
            params,
            prompt.shape[0],
            n_toks,
            batch_size,
            padded_prompt,
            initial_logits,
            initial_cache,
            evolve(activation_logger, sub_log=logger),
            sample,
            pad,
            pad_to,
            activation_cache,
            clean_up_cache_post_scan,
        )

        if return_toks:
            toks_all.append(toks)

        if return_cache:
            caches_all.append(iter_cache)

        if return_prob_ratios:
            prob_ratios_all.append(prob_ratios)

        sum_ratios = prob_ratios.sum()
        summed_ratios = op.unwrap_or(op.map(summed_ratios, lambda c: c + sum_ratios), sum_ratios)  # type: ignore[misc]

        if reduce is not None:
            reduced_vals = reduce(iter_cache, prob_ratios, toks)
            summed_reduced = op.unwrap_or(op.map(summed_reduced, lambda c: c + reduced_vals), reduced_vals)  # type: ignore[misc]

    assert summed_ratios is not None

    reduced_out = jax.tree_util.tree_map(lambda x: x / summed_ratios, summed_reduced)  # type: ignore[misc]

    if return_toks:
        toks_out = jnp.concatenate(toks_all)
    else:
        toks_out = None

    if return_cache:
        # maybe should always be handled outside...
        cache_out = stack_caches(caches_all)
    else:
        cache_out = None

    if return_prob_ratios:
        prob_ratios_out = jnp.concatenate(prob_ratios_all)
    else:
        prob_ratios_out = None

    return GenerativeReturn(reduced_out, toks_out, cache_out, prob_ratios_out)


def log_cache_clean_up_stacked(cache: Optional[LogCache]):
    return op.map(cache, lambda x: x.cleanup_stack_dim())


def log_cache_combine_to_batch(caches: List[LogCache], activation_cache: bool = True):
    stacked: LogCache = stack_tree(caches).cleanup_stack_dim()

    def combine_to_batch(x):
        extra = "seq" if activation_cache else ""
        return einops.rearrange(x, f"batches {extra} batch ... -> (batches batch) {extra} ...")

    return stacked.map(combine_to_batch)


def sample_monte_vals(key: jax.random.KeyArray, prob_ratios, vals, shape: Tuple[int, ...]):
    logprobs = jnp.log(jnp.clip(prob_ratios / prob_ratios.sum(), a_min=1e-10, a_max=None))
    select = jax.random.categorical(key, logprobs, shape=shape)
    return vals[select], prob_ratios[select], select


def sample_and_print_toks(
    key: jax.random.KeyArray, prob_ratios, toks, n: int = 10, print_prob: bool = False, file=None
):
    prob_sum = prob_ratios.sum()
    seqs, probs, _ = sample_monte_vals(key, prob_ratios, toks, (n,))
    for seq, prob in zip(seqs, probs):
        if print_prob:
            print("prob", prob / prob_sum, file=file)
        print(get_interp_tokenizer().decode(seq), file=file)


@partial(jax.jit, static_argnames=["model"])
def unigrams_reduction(cache: LogCache, prob_ratios, _, model: Gpt, params: FrozenVariableDict):
    """
    Assumes padded version of logs
    """
    return jnp.einsum(
        "s b l t, b -> s t",
        jax.nn.softmax(model.bind(params).embedding.unembed(cache.get(KeyIdxs("final_out.inp")))),
        prob_ratios,
    )


def get_unigrams_reduction(model: Gpt, params: FrozenVariableDict):
    """
    Assumes padded version of logs
    """
    return partial(unigrams_reduction, model=model, params=params)


@partial(jax.jit, static_argnames=["batch_size", "weight"])
def weighted_sample_single(unnormalized_logits, batch_size, tok, weight: bool = True):
    out_toks = jnp.broadcast_to(jnp.array(tok), (batch_size,))
    return out_toks, jax.nn.softmax(unnormalized_logits)[jnp.arange(batch_size), out_toks] if weight else jnp.full(
        (batch_size,), 1.0, dtype=unnormalized_logits.dtype
    )


@partial(jax.jit, static_argnames=["weight"])
def weighted_sample_except(key, unnormalized_logits, toks, weight: bool = True):
    assert unnormalized_logits.ndim == 2
    batch_size = unnormalized_logits.shape[0]
    if toks.ndim == 1:
        toks = toks[None]
    assert toks.ndim == 2
    return jax.random.categorical(key, unnormalized_logits.at[jnp.arange(batch_size)[:, None], toks].set(-1000.0)), (
        1.0 - jax.nn.softmax(unnormalized_logits)[jnp.arange(batch_size)[:, None], toks].sum(axis=-1)
    ) if weight else jnp.full((batch_size,), 1.0, dtype=unnormalized_logits.dtype)
