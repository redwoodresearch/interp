from __future__ import annotations

import dataclasses
from functools import lru_cache, partial
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple, Union, Any, Protocol, Set, List

from flax.core.scope import CollectionFilter, FrozenVariableDict, RNGSequences, VariableDict
import flax.linen as nn
import jax
import jax.numpy as jnp
from attrs import frozen

from interp.model.gpt_modules import (
    AttnActivationCache,
    AttnApplyToNormalConfig,
    Embedding,
    GptBlock,
    PosEncType,
    id_norm,
    MlpActType,
)
from interp.model.blocks import Y, Carry, LayerNorm, ScanRunOnLogConfig, scan_run_on_log
from interp.tools.jax_util import stack_tree
from interp.tools.assert_never import assert_never
from interp.tools.multivariate_normal import MultivariateNormal, make_sym
from interp.tools.immutable_dict import assign, operate, operate_f, remove_f
from interp.tools.jax_tree_util import AttrsPartiallyStatic
from interp.tools.variable_dict import variable_dict_replace, variable_dict_replace_params
from interp.tools.log import (
    Idxs,
    KeyIdxs,
    LogInfo,
    Logger,
    LoggerCache,
    MutLogCache,
    NoopLogger,
    construct_mut_log_cache,
)
import interp.tools.optional as op

NormType = Literal["none", "layer_norm", "batch_norm"]


class Gpt(nn.Module):
    num_layers: int = 2
    num_heads: int = 8
    vocab_size: int = 50259
    hidden_size: int = 256
    max_sequence_len: int = 512
    dropout_rate: float = 0.1
    embed_dropout_rate: float = 0.1
    attn_probs_dropout_rate: float = 0.1
    norm_type: NormType = "none"
    attn_bias: bool = False
    layer_norm_epsilon: float = 1e-5
    pos_enc_type: PosEncType = "gpt"
    use_mlp: bool = False
    mlp_act_type: MlpActType = "gelu"
    use_norm_output: bool = True
    tied_embed_unembed: bool = True
    dtype: Any = jnp.float32

    def setup(self):
        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_size=self.hidden_size,
            max_position_embeddings=self.max_sequence_len,
            dropout_rate=self.embed_dropout_rate,
            pos_enc_type=self.pos_enc_type,
            tied_embed_unembed=self.tied_embed_unembed,
            dtype=self.dtype,
        )

        def get_norm(norm_type=self.norm_type):
            if norm_type == "none":
                return id_norm
            elif norm_type == "layer_norm":
                return LayerNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype)
            elif norm_type == "batch_norm":
                raise NotImplementedError()
            else:
                assert_never(norm_type)

        self.blocks = [
            GptBlock(
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
                attn_probs_dropout_rate=self.attn_probs_dropout_rate,
                attn_bias=self.attn_bias,
                get_norm=get_norm,
                max_sequence_len=self.max_sequence_len,
                use_mlp=self.use_mlp,
                mlp_act_type=self.mlp_act_type,
                dtype=self.dtype,
            )
            for _ in range(self.num_layers)
        ]

        self.norm_output = get_norm() if self.use_norm_output else get_norm("none")

    @frozen
    class BaseCallConfig(AttrsPartiallyStatic):
        log_finish: bool = True
        scan_run_on_log_config: ScanRunOnLogConfig = ScanRunOnLogConfig()

    @jax.tree_util.register_pytree_node_class
    @frozen
    class CallConfig(BaseCallConfig):
        allow_position_extrapolation: bool = False
        training: bool = False
        activation_cache: Optional[AttnActivationCache] = None

        def non_static_names(self) -> Set[str]:
            return {"activation_cache"}

    def scan_on_blocks(
        self,
        run_on_log: Callable[[MutLogCache, Carry, GptBlock], Tuple[Carry, Y]],
        carry: Carry,
        log: MutLogCache,
        config: ScanRunOnLogConfig,
    ) -> Tuple[Carry, Y]:
        if config.use_for_loop:
            get_block = lambda i: self.blocks[i]
        else:
            all_params = stack_tree([self.block_params(i) for i in range(self.num_layers)])
            block = self.blocks[0].clone(parent=None)
            get_block = lambda i: block.bind(jax.tree_util.tree_map(lambda x: x[i], all_params))

        return scan_run_on_log(
            run_on_log,
            carry,
            get_block,
            n=self.num_layers,
            log=log,
            init_idxed=not isinstance(log.log_info.logger, NoopLogger),
            config=config,
        )

    def __call__(  # type: ignore[override]
        self,
        token_ids: jnp.ndarray,
        log: Optional[MutLogCache] = None,
        config: CallConfig = CallConfig(),
    ) -> jnp.ndarray:
        log_v = op.unwrap_or(log, MutLogCache.noop())

        embeds = self.embedding(
            token_ids,
            log_v.sub("embedding"),
            training=config.training,
            pos_idxs=op.map(config.activation_cache, lambda x: x.as_idxs()),
        )
        x = embeds["tok"]

        def run_on_log(log: MutLogCache, x, block: GptBlock):
            return (
                block(
                    x,
                    log=log,
                    allow_position_extrapolation=config.allow_position_extrapolation,
                    pos_embeds=embeds.get("pos"),
                    training=config.training,
                    activation_cache=config.activation_cache,
                ),
                None,
            )

        x, _ = self.scan_on_blocks(
            run_on_log,
            x,
            log=log_v.sub("blocks"),
            config=config.scan_run_on_log_config,
        )

        logits = self.out(x, log_v.sub("final_out"))

        if config.log_finish:
            log_v.check_finish()

        return logits

    def out(
        self,
        x: jnp.ndarray,
        log: Optional[MutLogCache] = None,
    ) -> jnp.ndarray:
        log_v = op.unwrap_or(log, MutLogCache.noop())

        x = log_v.log_and_modify(x, "inp")
        x = log_v.log_and_modify(self.norm_output(x, log_v.sub("norm")), "norm")
        logits = log_v.log_and_modify(self.embedding.unembed(x), "logits")
        return logits

    def block_params(self, i):
        # Note: for this to work, params must have already been inited!
        return FrozenVariableDict({"params": self.variables["params"][f"blocks_{i}"]})

    @jax.tree_util.register_pytree_node_class
    @frozen
    class ApplyToNormalConfig(BaseCallConfig):
        """
        Doesn't do much atm.
        """

        attn_config: AttnApplyToNormalConfig = AttnApplyToNormalConfig()

    # TODO: maybe should support arbitrary additional inputs (not very hard)
    def apply_to_normal(
        self,
        key: jax.random.KeyArray,
        x: MultivariateNormal,
        log: Optional[MutLogCache] = None,
        config: ApplyToNormalConfig = ApplyToNormalConfig(),
    ):
        log_v = op.unwrap_or(log, MutLogCache.noop())

        batch_size, seq_len, _ = x.mean_as().shape
        assert batch_size == 1, "we keep same shape by convention, but batch != 1 is currently insane"

        assert self.embedding.pos_enc_type == "shortformer", "we currently assume shortformer for lazyness"
        pos_embeds = jnp.expand_dims(self.embedding.position_embedding(jnp.arange(seq_len)), 0)

        # dup out orig embed distribution
        x = x.lin_op(lambda x: {"input_embeds": x, "residual": x})

        def run_on_log(log: MutLogCache, x_key, block: GptBlock):
            x, key = x_key
            key, subkey = jax.random.split(key)
            return (
                (block.apply_to_normal(subkey, x, log=log, pos_embeds=pos_embeds, config=config.attn_config), key),
                None,
            )

        (x, key), _ = self.scan_on_blocks(
            run_on_log, (x, key), log=log_v.sub("blocks"), config=config.scan_run_on_log_config
        )

        x = x.lin_op(operate_f("residual", "final_embeds"))

        assert self.norm_output is id_norm

        if config.log_finish:
            log_v.check_finish()

        return x

    def gen_new_dist_to_input(
        self,
        key: jax.random.KeyArray,
        x: MultivariateNormal,
        iters: int,
        final_embed_seq_idx: Union[int, jnp.ndarray] = -1,
        # excluding toks like this doesn't do any fancy conditioning
        exclude_toks_mask: Optional[jnp.ndarray] = None,
        is_set: bool = False,
    ):
        x, mean_probs, mean_include_prob = self.new_dist_from_final_embed(
            key,
            x,
            iters,
            final_embed_seq_idx=final_embed_seq_idx,
            exclude_toks_mask=exclude_toks_mask,
        )
        x = x.lin_op(
            lambda x: (
                remove_f(["new_input_embed_dist"])
                @ operate_f(
                    "input_embeds",
                    "input_embeds",
                    lambda inp: inp.at[0, final_embed_seq_idx + 1].set(x["new_input_embed_dist"])
                    if is_set
                    else jnp.concatenate([inp, jnp.expand_dims(x["new_input_embed_dist"], (0, 1))], axis=1),
                )
            )(x)
        )

        return x, mean_probs, mean_include_prob

    def new_dist_from_final_embed(
        self,
        key: jax.random.KeyArray,
        x: MultivariateNormal,
        iters: int,
        final_embed_seq_idx: Union[int, jnp.ndarray],
        # excluding toks like this doesn't do any fancy conditioning
        exclude_toks_mask: Optional[jnp.ndarray] = None,
    ):
        sample_selector = lambda x: x["final_embeds"][0, final_embed_seq_idx]

        key, subkey = jax.random.split(key)
        samples = x.lin_op(sample_selector).sample(subkey, (iters,))

        logits = self.embedding.unembed(samples)

        probs = jax.nn.softmax(logits)

        mean_include_prob = jnp.ones_like(probs, shape=(), dtype=probs.dtype)
        if exclude_toks_mask is not None:
            exclude_probs = probs[..., exclude_toks_mask].sum(axis=-1, keepdims=True)
            probs = probs.at[..., exclude_toks_mask].set(0.0) / (1.0 - exclude_probs)

            mean_include_prob = 1.0 - exclude_probs.mean()

        means = jnp.einsum("n v, v h -> n h", probs, self.embedding.token_embedding.embedding)
        mean_probs = probs.mean(axis=0)

        cov = make_sym(jnp.cov(self.embedding.token_embedding.embedding, rowvar=False, bias=True, aweights=mean_probs))

        _, orig_new_cov = x.get_beta_orig_new(samples, means, sample_selector)

        combine = lambda new, orig: assign(orig, "new_input_embed_dist", new)

        return (
            x.combine_dist(combine, cov, orig_new_cov, means.mean(axis=0)),
            mean_probs,
            mean_include_prob,
        )

    def estimate_token_dist_raw(self, key: jax.random.KeyArray, dist: MultivariateNormal, iters=2_000):
        return jax.nn.softmax(self.embedding.unembed(dist.sample(key, (iters,)))).mean(axis=0)

    def estimate_token_dist(
        self, key: jax.random.KeyArray, dist: MultivariateNormal, tok: Union[jnp.ndarray, int], iters=2_000
    ):
        return jax.nn.softmax(
            self.embedding.unembed(dist.lin_op(lambda x: x["final_embeds"][:, tok]).sample(key, (iters,)))
        ).mean(axis=0)

    def estimate_token_dists(self, key: jax.random.KeyArray, dist: MultivariateNormal, iters=2_000):
        return jax.vmap(lambda tok: self.estimate_token_dist(key, dist, tok, iters=iters))(
            jnp.arange(dist.mean_as()["final_embeds"].shape[1])
        )

    def condition_on_input_embeds(
        self,
        dist: MultivariateNormal,
        new_tok_idx: Union[int, jnp.ndarray],
        toks: jnp.ndarray,
        mean_probs: Optional[jnp.ndarray] = None,
        set_instead_of_condition: bool = False,
        is_dict: bool = False,
    ):
        if is_dict:
            selector = lambda x: x["input_embeds"][:, new_tok_idx]
            setter = lambda x, val: operate(
                x, "input_embeds", "input_embeds", lambda inp: inp.at[:, new_tok_idx].set(val)
            )
        else:
            selector = lambda x: x[:, new_tok_idx]
            setter = lambda x, val: x.at[:, new_tok_idx].set(val)

        assert toks.ndim in {0, 1}
        if set_instead_of_condition:
            if toks.ndim == 0:
                return dist.set(jnp.expand_dims(self.embedding.token_embedding.embedding[toks], 0), setter)
            else:
                assert mean_probs is not None
                return dist.set(
                    jnp.expand_dims(
                        (self.embedding.token_embedding.embedding[toks] * mean_probs[toks, None]).sum(axis=0), 0
                    ),
                    setter,
                )

        if toks.ndim == 0:
            return dist.condition(
                selector,
                jnp.expand_dims(self.embedding.token_embedding.embedding[toks], 0),
                setter=setter,
            )
        else:
            assert mean_probs is not None
            return dist.condition_mixture(
                selector,
                jnp.expand_dims(self.embedding.token_embedding.embedding[toks], 1),
                mean_probs[toks],
                setter=setter,
            )

    def get_qkv_mats_all_layers(self) -> jnp.ndarray:
        return jnp.stack([b.attention.get_qkv_mats() for b in self.blocks], axis=1)

    # a bit of swizzling never hurt anyone
    def get_q_mats_all_layers(self) -> jnp.ndarray:
        return self.get_qkv_mats_all_layers()[0]

    def get_k_mats_all_layers(self) -> jnp.ndarray:
        return self.get_qkv_mats_all_layers()[1]

    def get_v_mats_all_layers(self) -> jnp.ndarray:
        return self.get_qkv_mats_all_layers()[2]

    def get_qk_mats_all_layers(self) -> jnp.ndarray:
        return self.get_qkv_mats_all_layers()[:2]

    def get_o_mats_all_layers(self) -> jnp.ndarray:
        return jnp.stack([b.attention.get_o_mats() for b in self.blocks], axis=0)

    def get_qk_combined_mats_all_layers(self) -> jnp.ndarray:
        return jnp.stack([b.attention.get_qk_combined_mats() for b in self.blocks], axis=0)

    def get_ov_combined_mats_all_layers(self) -> jnp.ndarray:
        return jnp.stack([b.attention.get_ov_combined_mats() for b in self.blocks], axis=0)

    # allows for doing computation with different q_enc and k_enc
    # ignores bias
    def qk_all_layers(self, q_enc: jnp.ndarray, k_enc: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        qs, ks = zip(*(b.attention.qk(q_enc, k_enc) for b in self.blocks))
        q = jnp.stack(qs, axis=0)
        k = jnp.stack(ks, axis=0)

        return q, k

    # allows for doing attn score computation with different q_enc and k_enc
    # ignores bias terms
    # requires shape [term, batch, seq, d]
    def compute_attn_scores_all_layers(self, q_enc: jnp.ndarray, k_enc: jnp.ndarray, mask=True) -> jnp.ndarray:
        return jnp.stack([b.attention.compute_attn_scores(q_enc, k_enc, mask=mask) for b in self.blocks], axis=0)

    @nn.nowrap
    @staticmethod
    def get_block_param_args(block: int) -> Tuple[str]:
        return (f"blocks_{block}",)

    def set_emb_weights(self, mat: str, new_weights: jnp.ndarray) -> FrozenVariableDict:
        assert mat in {"t_emb", "p_emb"}
        return variable_dict_replace(
            self.variables,
            new_weights,
            "params",
            "embedding",
            ("token" if mat == "t_emb" else "position") + "_embedding",
            "embedding",
        )

    def set_linear_weights(self, mat: str, new_weights: jnp.ndarray, l: int) -> FrozenVariableDict:
        return variable_dict_replace_params(
            self.variables,
            self.blocks[l].set_linear_weights(mat, new_weights),
            *self.get_block_param_args(l),
        )

    # override for typing
    def bind(
        self, variables: VariableDict, *args, rngs: Optional[RNGSequences] = None, mutable: CollectionFilter = False
    ) -> Gpt:
        return super().bind(variables, *args, rngs=rngs, mutable=mutable)

    def set_attn_weights(
        self,
        mat: str,
        new_weights: jnp.ndarray,
        l: int,
        h_low: int,
        h_high: Optional[int] = None,
    ) -> FrozenVariableDict:
        return variable_dict_replace_params(
            self.variables,
            self.blocks[l].set_attn_weights(mat, new_weights, h_low, h_high),
            *self.get_block_param_args(l),
        )

    @nn.nowrap
    def activation_cache_to_log(self) -> List[KeyIdxs]:
        return [KeyIdxs("blocks.attention.final_k", Idxs.all()), KeyIdxs("blocks.attention.final_v", Idxs.all())]


# using jank cache because I want to return the same block by reference every time
# so jitted functions rejit, even though model and params aren't hashable
block_fn_cache: List[Tuple[Gpt, GptBlock]] = []


def get_gpt_block_fn(model: Gpt, params: FrozenVariableDict):
    matching = [x for x in block_fn_cache if (x[0] is model)]
    if len(matching) > 0:
        return matching[0][1]
    block = model.bind(params).blocks[0].clone(parent=None)
    block_fn_cache.append((model, block))
    assert isinstance(block, GptBlock)
    return block


def module_config_dict(module: nn.Module):
    out = dataclasses.asdict(module)
    del out["parent"]
    del out["name"]

    return out


def to_any(x) -> Any:
    """
    Type forget
    """
    return x


@partial(jax.jit, static_argnames=["model"])
def gpt_init(model: Gpt, key: jax.random.KeyArray):
    return model.init(
        key, jnp.array([[0]]), config=Gpt.CallConfig(scan_run_on_log_config=ScanRunOnLogConfig(use_for_loop=True))
    )


# ideally would be static in Gpt, but some issues with that...
class GptCall(Protocol):
    def __call__(
        self,
        model: Gpt,
        params: FrozenVariableDict,
        token_ids: jnp.ndarray,
        log_info: Optional[LogInfo] = None,
        log_cache: Optional[Any] = None,
        config: Gpt.CallConfig = Gpt.CallConfig(),
    ) -> Tuple[jnp.ndarray, Optional[Any]]:
        log = construct_mut_log_cache(log_info, log_cache)
        out = model.bind(params)(token_ids=token_ids, log=log, config=config)

        return (out, op.map(log, lambda log: log.cache))


class GptCallUnjit(GptCall):
    ...


gpt_call_unjit = GptCallUnjit()
gpt_call: GptCall = to_any(jax.jit(gpt_call_unjit, static_argnames=["model"]))


def gpt_call_no_log(
    model: Gpt,
    params: FrozenVariableDict,
    token_ids: jnp.ndarray,
    config: Gpt.CallConfig = Gpt.CallConfig(),
    jit: bool = True,
) -> jnp.ndarray:
    call = gpt_call if jit else gpt_call_unjit

    return call(model=model, params=params, token_ids=token_ids, config=config)[0]


# typing on this is kinda sad...
def partial_gpt_call(
    model: Gpt, params: FrozenVariableDict, config: Gpt.CallConfig = Gpt.CallConfig(), jit: bool = True
):
    return partial(gpt_call if jit else gpt_call_unjit, model, params, config=config)


# typing on this is kinda sad...
def partial_gpt_call_no_log(
    model: Gpt, params: FrozenVariableDict, config: Gpt.CallConfig = Gpt.CallConfig(), jit: bool = True
):
    return partial(gpt_call_no_log, model, params, config=config, jit=jit)


# typing on this is kinda sad...
def partial_gpt_call_just_logger(model: Gpt, params: FrozenVariableDict, token_ids: jnp.ndarray, jit: bool = True):
    return lambda logger, *args, **kwargs: (gpt_call if jit else gpt_call_unjit)(
        model, params, token_ids, LogInfo(logger), *args, **kwargs
    )[1]


# TODO on below
def inference(
    model: Gpt,
    params: FrozenVariableDict,
    token_ids: jnp.ndarray,
    to_log: Iterable[str] = [],
    to_log_idxed: Iterable[KeyIdxs] = [],
    jit: bool = True,
) -> Dict[str, Any]:
    logger = LoggerCache(to_cache=set([*to_log, "final_out.logits"]))
    logger.add_all(to_log_idxed)
    call = gpt_call if jit else gpt_call_unjit
    _, cache = call(model, params, token_ids, log_info=LogInfo(logger))
    cache = to_any(cache)  # idk, for some reason jitting and such makes mypy wrong about the type here
    return {**cache.cache, **{k: v.values for k, v in cache.idxed_cache.items()}}


class GptApplyToNormal(Protocol):
    def __call__(
        self,
        model: Gpt,
        params: FrozenVariableDict,
        key: jax.random.KeyArray,
        x: MultivariateNormal,
        log_info: Optional[LogInfo] = None,
        log_cache: Optional[Any] = None,
        config: Gpt.ApplyToNormalConfig = Gpt.ApplyToNormalConfig(),
    ) -> Tuple[MultivariateNormal, Optional[Any]]:
        log = construct_mut_log_cache(log_info, log_cache)
        out = model.bind(params).apply_to_normal(
            key=key,
            x=x,
            log=log,
            config=config,
        )

        return (out, op.map(log, lambda log: log.cache))


class GptApplyToNormalUnjit(GptApplyToNormal):
    ...


gpt_apply_to_normal_unjit = GptApplyToNormalUnjit()
gpt_apply_to_normal: GptApplyToNormal = to_any(jax.jit(gpt_apply_to_normal_unjit, static_argnames=["model"]))


def input_dist_zero_cov(embeds: jnp.ndarray):
    flat_embeds = embeds.flatten()
    return MultivariateNormal(
        flat_embeds, jnp.zeros((flat_embeds.size, flat_embeds.size), dtype=flat_embeds.dtype)
    ).lin_op(lambda x: x.reshape(embeds.shape))


def input_dist_zero_cov_toks(model: Gpt, params: FrozenVariableDict, toks: jnp.ndarray):
    return input_dist_zero_cov(model.bind(params).embedding(toks)["tok"])


class GptNewDistFromFinalEmbed(Protocol):
    def __call__(
        self,
        model: Gpt,
        params: FrozenVariableDict,
        key: jax.random.KeyArray,
        x: MultivariateNormal,
        iters: int,
        final_embed_seq_idx: Union[int, jnp.ndarray],
        # excluding toks like this doesn't do any fancy conditioning
        exclude_toks_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[MultivariateNormal, jnp.ndarray, jnp.ndarray]:
        return model.bind(params).new_dist_from_final_embed(
            key=key,
            x=x,
            iters=iters,
            final_embed_seq_idx=final_embed_seq_idx,
            exclude_toks_mask=exclude_toks_mask,
        )


class GptNewDistFromFinalEmbedUnjit(GptNewDistFromFinalEmbed):
    ...


gpt_new_dist_from_final_embed_unjit = GptNewDistFromFinalEmbedUnjit()
gpt_new_dist_from_final_embed: GptNewDistFromFinalEmbed = to_any(
    jax.jit(gpt_new_dist_from_final_embed_unjit, static_argnames=["model", "iters"])
)


class GptGenNewDistToInput(Protocol):
    def __call__(
        self,
        model: Gpt,
        params: FrozenVariableDict,
        key: jax.random.KeyArray,
        x: MultivariateNormal,
        iters: int,
        final_embed_seq_idx: Union[int, jnp.ndarray] = -1,
        # excluding toks like this doesn't do any fancy conditioning
        exclude_toks_mask: Optional[jnp.ndarray] = None,
        is_set: bool = False,
    ) -> Tuple[MultivariateNormal, jnp.ndarray, jnp.ndarray]:
        return model.bind(params).gen_new_dist_to_input(
            key=key,
            x=x,
            iters=iters,
            final_embed_seq_idx=final_embed_seq_idx,
            exclude_toks_mask=exclude_toks_mask,
            is_set=is_set,
        )


class GptGenNewDistToInputUnjit(GptGenNewDistToInput):
    ...


gpt_gen_new_dist_to_input_unjit = GptGenNewDistToInputUnjit()
gpt_gen_new_dist_to_input: GptGenNewDistToInput = to_any(
    jax.jit(gpt_gen_new_dist_to_input_unjit, static_argnames=["model", "iters", "is_set"])
)


class GptConditionOnInputEmbeds(Protocol):
    def __call__(
        self,
        model: Gpt,
        params: FrozenVariableDict,
        dist: MultivariateNormal,
        new_tok_idx: Union[int, jnp.ndarray],
        toks: jnp.ndarray,
        mean_probs: Optional[jnp.ndarray] = None,
        set_instead_of_condition: bool = False,
        is_dict: bool = False,
    ) -> MultivariateNormal:
        return model.bind(params).condition_on_input_embeds(
            dist=dist,
            new_tok_idx=new_tok_idx,
            toks=toks,
            mean_probs=mean_probs,
            set_instead_of_condition=set_instead_of_condition,
            is_dict=is_dict,
        )


class GptConditionOnInputEmbedsUnjit(GptConditionOnInputEmbeds):
    ...


gpt_condition_on_input_embeds_unjit = GptConditionOnInputEmbedsUnjit()
gpt_condition_on_input_embeds: GptConditionOnInputEmbeds = to_any(
    jax.jit(gpt_condition_on_input_embeds_unjit, static_argnames=["model", "set_instead_of_condition", "is_dict"])
)


def loss_by_token_loggable(model, params, input_ids):
    def loggable(logger: Logger):
        log_info = LogInfo(logger)
        logits, cache = gpt_call(model, params, input_ids, log_info=log_info, config=Gpt.CallConfig(log_finish=False))
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        loss_predicting_next = jnp.take_along_axis(
            logprobs,
            jnp.expand_dims(jnp.concatenate([input_ids[:, 1:], jnp.full((input_ids.shape[0], 1), 0)], 1), axis=-1),
            2,
        )[:, :, 0]
        cache = log_info.sub("loss").log(loss_predicting_next, "loss_by_token", cache)

        logger.check_finish_cache(cache)
        return cache

    return loggable


def logprobs_on_correct(logits, input_ids):
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    loss_predicting_next = jnp.take_along_axis(
        logprobs,
        jnp.expand_dims(jnp.concatenate([input_ids[:, 1:], jnp.full((input_ids.shape[0], 1), 0)], 1), axis=-1),
        2,
    )[:, :, 0]
    return loss_predicting_next
