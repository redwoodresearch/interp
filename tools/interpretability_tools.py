import sys
import math
from typing import Any, Optional, Protocol, Tuple, Callable, List, TypeVar, Union
from functools import partial, lru_cache
from copy import copy

from tabulate import tabulate
from einops.einops import rearrange
import transformers
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import entr
import tqdm
from attrs import frozen

from interp.tools.grad_modify_query import (
    ModifierCollectionTreeNode,
    ModifierCollectionTreeNodeStack,
    Query,
    TargetConf,
    run_queries,
)
from interp.tools.grad_modify_query_items import ReplaceFuncConf, ItemConf
from interp.tools.log import KeyIdxs, LogInfo, Logger, MutLogCache, construct_mut_log_cache
from interp.tools.custom_jvp import ablation_custom_jvp, integrated_gradients_custom_jvp
import interp.tools.optional as op


@lru_cache(maxsize=None)
def get_interp_tokenizer():
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer._add_tokens(["[BEGIN]", "[END]"])
    tokenizer.pad_token = "[END]"
    tokenizer.eos_token = "[END]"
    return tokenizer


def toks_to_string_list(toks: Union[jnp.ndarray, List[int], List[List[int]]]):
    return [get_interp_tokenizer().decode(tok) for tok in toks]


def single_tokenize(s: str) -> int:
    out = get_interp_tokenizer()(s, padding=False)["input_ids"]
    assert len(out) == 1
    return out[0]


def sequence_tokenize(s: str, add_begin_token=False) -> jnp.ndarray:
    return batch_tokenize([s], add_begin_token).squeeze(0)


def add_begin_token(s: str):
    if s[:8] != "[BEGIN] ":
        s = "[BEGIN] " + s
    return s


def batch_tokenize(s: Union[str, List[str]], add_begin_token=False) -> jnp.ndarray:
    if add_begin_token:
        s = add_begin_token(s)
    return get_interp_tokenizer()(s, padding=False, return_tensors="jax")["input_ids"]


def strings_to_tok_list(strs):
    return [single_tokenize(s) for s in strs]


@lru_cache(maxsize=None)
def begin_token():
    return single_tokenize("[BEGIN]")


# BE VERY AFRAID! jax doesn't bounds check so if your labels are out of bounds
# this will silently do the wrong thing (UB-like, but nasal demons still
# imprisoned)
def cross_entropy_loss_no_reduce(logits, labels, log: Optional[MutLogCache] = None):
    log_v = op.unwrap_or(log, MutLogCache.noop())

    assert labels.ndim == 1
    assert logits.ndim >= 2
    assert logits.shape[0] == labels.shape[0], f"logits.shape[0]={logits.shape[0]}, labels.shape[0]={labels.shape[0]}"
    logits = log_v.log_and_modify(logits, "inp")
    log_probs = log_v.log_and_modify(jax.nn.log_softmax(logits, axis=1), "log_probs")

    return -log_probs[jnp.arange(labels.shape[0]), labels]


def cross_entropy_replace_log_probs(replacement):
    return ModifierCollectionTreeNode(
        ReplaceFuncConf(
            ItemConf(KeyIdxs("cross_entropy.log_probs")),
            from_key_idxs=KeyIdxs("cross_entropy.inp"),
            replacement=replacement,
        )
    )


def cross_entropy_wrap_log_probs(wrapper):
    return cross_entropy_replace_log_probs(wrapper(partial(jax.nn.log_softmax, axis=1)))


# mask + reshape
def cross_entropy_loss_helper(logits, token_ids, targets, label_dim: int = 2, log: Optional[MutLogCache] = None):
    # dims after label_dim are kept for batching over computation with same token_ids
    logits = jnp.reshape(logits, (math.prod(logits.shape[:label_dim]),) + logits.shape[label_dim:])
    losses = cross_entropy_loss_no_reduce(logits=logits, labels=targets.flatten(), log=log)
    return jnp.where(
        jnp.expand_dims((token_ids != 50258).flatten(), tuple(range(-(logits.ndim - label_dim), 0))), losses, 0.0
    )


T = TypeVar("T")


def run_on_tokens(
    # often worthwhile to jit the callable!
    run: Callable[[jnp.ndarray, Optional[jnp.ndarray]], T],
    tokenized_data: jnp.ndarray,
    batch_size: int,
    seqlen: int = 512,
    disable_progress: bool = False,
    prepend_begin: bool = True,
    use_targets: bool = True,
) -> List[T]:
    items = []
    seqlen = min(seqlen, tokenized_data.shape[1] - (0 if prepend_begin else 1) + (0 if use_targets else 1))

    for i in tqdm.trange(0, len(tokenized_data), batch_size, disable=disable_progress):
        batch = tokenized_data[i : i + batch_size]
        seqlen_from_batch = seqlen - (1 if prepend_begin else 0)
        token_ids = batch[:, :seqlen_from_batch]
        if prepend_begin:
            token_ids = jnp.concatenate([jnp.full((batch.shape[0], 1), begin_token()), token_ids], axis=1)

        if use_targets:
            if prepend_begin:
                targets = batch[:, :seqlen]
            else:
                targets = batch[:, 1 : seqlen + 1]
        else:
            targets = None

        items.append(run(token_ids, targets))

    return items


class CallModel(Protocol):
    def __call__(
        self,
        token_ids: jnp.ndarray,
        *,
        log_info: Optional[LogInfo] = None,
        log_cache: Optional[Any] = None,
    ) -> Tuple[jnp.ndarray, Optional[Any]]:
        ...


def losses_runner(call_model: CallModel, just_avg_loss=True):
    def run(
        token_ids,
        targets,
        log_info: Optional[LogInfo] = None,
        log_cache: Optional[Any] = None,
    ):
        log = op.unwrap_or(construct_mut_log_cache(log_info, log_cache), MutLogCache.noop())
        logits, log.cache = call_model(token_ids, log_info=log.log_info, log_cache=log.cache)
        losses = cross_entropy_loss_helper(logits, token_ids, targets, log=log.sub("cross_entropy"))

        loss_log = log.sub("loss")

        losses = loss_log.log_and_modify(losses, "losses")
        loss_log.log(losses.mean(), "mean_loss")

        log.check_finish()

        if just_avg_loss:
            return losses.mean(), log.cache
        else:
            return (losses, token_ids.flatten()), log.cache

    return run


def losses_runner_no_log(call_model: CallModel, just_avg_loss=True):
    def run(token_ids, targets):
        return losses_runner(call_model, just_avg_loss=just_avg_loss)(token_ids, targets)[0]

    return run


def losses_runner_log(call_model: CallModel, token_ids, targets):
    def run(logger: Logger):
        return losses_runner(call_model)(token_ids, targets, log_info=LogInfo(logger))[1]

    return run


def get_avg_loss(x: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> float:
    totals = np.array([v[0].mean() for v in x])
    weights = np.array([v[0].size for v in x])
    return np.dot(totals, weights) / weights.sum()


@frozen
class LossesRunnerTreeConfig:
    avg_loss: bool = True
    get_loss_no_deriv: bool = False

    # NOTE: getting log probs takes a considerable amount of memory
    get_log_probs: bool = False

    use_fwd: bool = False


class GetTree(Protocol):
    def __call__(self, runner, seq_len: int) -> ModifierCollectionTreeNodeStack:
        ...


def losses_runner_tree(
    call_model: CallModel,
    get_tree: GetTree,
    config: LossesRunnerTreeConfig = LossesRunnerTreeConfig(),
):
    def f(toks, targets):
        runner = losses_runner_log(call_model, toks, targets)

        loss_target = [TargetConf(KeyIdxs(f"loss.{'mean_loss' if config.avg_loss else 'losses'}"), display_name="loss")]
        if config.get_log_probs:
            loss_target.append(TargetConf(KeyIdxs("cross_entropy.log_probs"), display_name="log_probs"))
        queries = {
            "deriv": Query(
                targets=loss_target, modifier_collection_tree=get_tree(runner, toks.shape[1]), use_fwd=config.use_fwd
            )
        }

        if config.get_loss_no_deriv:
            queries["no_deriv"] = Query(targets=loss_target)

        vals = run_queries(runner, queries)

        out = {"deriv": copy(vals["deriv"])}
        if config.get_loss_no_deriv:
            out["no_deriv"] = copy(vals["no_deriv"])

        if config.avg_loss:
            reshape = lambda x: x
        else:
            reshape = partial(rearrange, pattern="(b n) ... -> b n ...", b=toks.shape[0])

        out["deriv"]["loss"] = reshape(out["deriv"]["loss"])
        if config.get_loss_no_deriv:
            out["no_deriv"]["loss"] = reshape(out["no_deriv"]["loss"])

        if not config.avg_loss:
            out["toks"] = toks

        return out

    return f


# top k which operates overall dimensions and returns the overall idx
def top_k_overall(x: jnp.ndarray, k: int):
    vals, idxs = jax.lax.top_k(x.flatten(), k=k)

    return vals, jnp.unravel_index(idxs, x.shape)


cross_entropy_ablation_log_probs = cross_entropy_wrap_log_probs(ablation_custom_jvp)


def get_cross_entropy_integrated_gradients_log_probs(min_mul=0.0, max_mul=1.0, n=30):
    return cross_entropy_wrap_log_probs(partial(integrated_gradients_custom_jvp, min_mul=min_mul, max_mul=max_mul, n=n))


def print_max_min_by_tok_k(
    vals, k=50, get_tok=toks_to_string_list, normalize=False, print_max: bool = True, print_min: bool = True, file=None
):
    assert vals.ndim == 1
    if normalize:
        vals = vals - vals.mean()
    max_vals, max_idxs = jax.lax.top_k(vals, k=k)
    if print_max:
        print("max", file=file)
        print(tabulate(list(zip(max_vals, max_idxs, [f'"{s}"' for s in get_tok(max_idxs)]))), file=file)
    min_vals, min_idxs = jax.lax.top_k(-vals, k=k)
    min_vals = -min_vals
    if print_min:
        print("min", file=file)
        print(tabulate(list(zip(min_vals, min_idxs, [f'"{s}"' for s in get_tok(min_idxs)]))), file=file)


# Maybe should be renamed. Maybe should be generalized somewhat.
def compare_estimate_to_monte(
    dists,
    monte_dists,
    k=10,
    target_toks=[],
    get_scatter=False,
    scatter_trendline=False,
    file=None,
):
    """
    Currently highly unstable function for comparing dist gen unigrams to monte
    carlo'd unigrams.

    Assumes that dists start 1 before monte (due to how activation caching works).
    """
    dists = dists[1:]
    min_toks = min(monte_dists.shape[0], dists.shape[0])
    dists = dists[:min_toks]
    monte = monte_dists[:min_toks]
    for tok, (tok_dist, unigram_dist) in enumerate(zip(dists, monte)):
        print(f"{2 + tok} token dist", file=file)
        print("normal prop", file=file)
        print_max_min_by_tok_k(tok_dist, k=k, print_min=False, file=file)
        print(file=file)
        print("actual", file=file)
        print_max_min_by_tok_k(unigram_dist, k=k, print_min=False, file=file)
        print(file=file)
        print("error", file=file)
        print_max_min_by_tok_k(unigram_dist - tok_dist, k=k, file=file)

    print("actual entropy", entr(monte).sum(axis=-1), file=file)
    print("estimation entropy", entr(dists).sum(axis=-1), file=file)

    print("kl", jnp.where(monte == 0.0, 0.0, monte * jnp.log(monte / dists)).sum(axis=-1), file=file)

    print(file=file)
    for tok in target_toks:
        dists_at_tok = dists[:, tok]
        monte_at_tok = monte[:, tok]
        print(f'for token "{get_interp_tokenizer().decode([tok])}"', file=file)
        print(
            tabulate(
                list(
                    zip(
                        range(1, 1 + len(dists_at_tok)),
                        dists_at_tok,
                        monte_at_tok,
                        dists_at_tok - monte_at_tok,
                    )
                ),
                headers=["token", "dists", "monte", "error"],
            ),
            file=file,
        )

    if get_scatter:
        import plotly.express as px

        fig = px.scatter(
            dict(
                monte_carlo=monte.flatten(),
                gaussian_prop=dists.flatten(),
                tok=toks_to_string_list(jnp.arange(dists.shape[1])) * dists.shape[0],
                tok_n=sum([[t for _ in range(dists.shape[1])] for t in range(2, dists.shape[0] + 2)], []),
            ),
            facet_row="tok_n",
            x="monte_carlo",
            y="gaussian_prop",
            hover_name="tok",
            log_x=True,
            log_y=True,
            trendline="ols" if scatter_trendline else None,
        )
        fig.update_layout(autosize=False, width=800, height=1400)

        return fig


class StdoutAndFile(object):
    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)


# a sad function indeed
# probably there is a better way...
def check_close_weak(actual: jnp.ndarray, expected: jnp.ndarray, atol=1e-4, norm_div_tol=1e-3):
    assert actual.shape == expected.shape
    assert (
        jnp.allclose(actual, expected, atol=atol)
        or (jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)) < norm_div_tol
    )
