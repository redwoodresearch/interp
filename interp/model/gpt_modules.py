from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Union, Tuple, Any, Set
from copy import copy

from einops import rearrange
from flax.core.scope import CollectionFilter, FrozenVariableDict, RNGSequences, VariableDict
import flax.linen as nn
import jax.numpy as jnp
import jax
from attrs import frozen
from transformers.utils.dummy_pt_objects import LogitsProcessorList

from interp.model.blocks import Dense, PointwiseNonlinearity, gelu
from interp.tools import assert_never
from interp.tools.jax_util import maybe_static_cond
from interp.tools.jax_tree_util import AttrsPartiallyStaticDefaultNonStatic
from interp.tools.multivariate_normal import MultivariateNormal
from interp.tools.variable_dict import variable_dict_replace_params
from interp.tools.log import Idxs, KeyIdxs, LogInfo, Logger, LoggerCache, MutLogCache, LogCache
from interp.tools.immutable_dict import (
    assign,
    operate_f,
    remove,
    remove_f,
    get_f,
    assign_f,
)
import interp.tools.optional as op


NEG_INF = -1.0e4
MlpActType = Literal["gelu", "relu"]


@jax.tree_util.register_pytree_node_class
@frozen
class AttnActivationCache(AttrsPartiallyStaticDefaultNonStatic):
    size_cached: Union[int, jnp.ndarray]
    cached_k: jnp.ndarray
    cached_v: jnp.ndarray
    # by default concat instead of set
    is_set: bool = False

    @staticmethod
    def to_log() -> List[KeyIdxs]:
        return [KeyIdxs("blocks.attention.final_k", Idxs.all()), KeyIdxs("blocks.attention.final_v", Idxs.all())]

    @classmethod
    def from_cache(
        cls, size_cached: Union[int, jnp.ndarray], cache: LogCache, is_set: bool = False, check: bool = False
    ):
        k, v = cls.to_log()
        return cls(size_cached, cache.get(k, check=check), cache.get(v, check=check), is_set=is_set)

    def static_names(self) -> Set[str]:
        return {"is_set"}

    def get_full(self, value: jnp.ndarray, which: Literal["k", "v"], log_idx: Optional[jnp.ndarray]) -> jnp.ndarray:
        orig_value = self.cached_k if which == "k" else self.cached_v
        if log_idx is not None:
            orig_value = orig_value[log_idx]

        # broadcast over batch dim
        broadcasted_orig = jnp.broadcast_to(orig_value, (value.shape[0], *orig_value.shape[1:]))

        if self.is_set:
            return broadcasted_orig.at[:, :, self.size_cached].set(value.squeeze(2))
        else:
            out = jnp.concatenate([broadcasted_orig, value], axis=2)
            return out

    def as_idxs(self):
        return jnp.full((1,), self.size_cached)


@frozen
class AttnApplyToNormalConfig:
    softmax_iters: int = 1000


class UnidirectionalAttn(nn.Module):
    num_heads: int
    hidden_size: int
    bias: bool
    attn_probs_dropout_rate: float
    max_sequence_len: int
    dtype: Any = jnp.float32

    def setup(self):
        # no input dim because flax is weird
        self.attn_weights = Dense(3 * self.hidden_size, use_bias=self.bias, dtype=self.dtype)
        self.project_output = Dense(self.hidden_size, use_bias=self.bias, dtype=self.dtype)
        self.attn_probs_dropout = nn.Dropout(self.attn_probs_dropout_rate)

        self.head_size = self.hidden_size // self.num_heads

    def __call__(  # type: ignore[override]
        self,
        x: jnp.ndarray,
        log: Optional[MutLogCache] = None,
        allow_position_extrapolation=False,
        pos_embeds: Optional[jnp.ndarray] = None,
        training: bool = False,
        activation_cache: Optional[AttnActivationCache] = None,
    ) -> jnp.ndarray:
        log_v = op.unwrap_or(log, MutLogCache.noop())

        assert x.ndim >= 2
        seq_len = x.shape[-2]

        q, k, v = [log_v.log_and_modify(x, s) for x, s in zip(self.compute_qkv(x, log_v.sub("qkv")), ["q", "k", "v"])]

        if pos_embeds is not None:
            pos_q, pos_k = self.get_pos_q_k(
                pos_embeds,
                seq_len,
                allow_position_extrapolation=allow_position_extrapolation,
                log=log_v,
            )
            q = q + pos_q
            k = k + pos_k

        q = log_v.log_and_modify(q, "final_q")

        if activation_cache is not None:
            k = activation_cache.get_full(k, "k", log_v.log_info.log_idx)
        k = log_v.log_and_modify(k, "final_k")

        if activation_cache is not None:
            v = activation_cache.get_full(v, "v", log_v.log_info.log_idx)
        v = log_v.log_and_modify(v, "final_v")

        scores = log_v.log_and_modify(
            self.attn_scores(q, k, q_idxs=op.map(activation_cache, lambda x: x.as_idxs())),
            "attn_scores",
        )
        probs = log_v.log_and_modify(jax.nn.softmax(scores), "attn_probs")
        probs = log_v.log_and_modify(self.attn_probs_dropout(probs, deterministic=not training), "attn_probs_dropout")

        combined_v = log_v.log_and_modify(self.mul_probs_v(probs, v), "combined_v")

        project_output_name = "project_output"

        def run_by_head(combined_v, log_info: LogInfo, cache: Any):
            log = MutLogCache.new_info(log_info, cache)
            by_head = log.log_and_modify(
                jnp.einsum("n o h, ... n q h -> ... n q o", self.get_o_mats(), combined_v),
                "out_by_head",
            )

            out = by_head.sum(axis=1)

            proj_log = log.sub(project_output_name)

            # names + shape + type should match Dense
            out = proj_log.log_and_modify(out, "mul_kernel")
            if self.project_output.use_bias:
                bias = proj_log.log_and_modify(self.project_output.get_bias(), "bias")
                out = proj_log.log_and_modify(out + jnp.expand_dims(bias, (0, 1)), "added_bias")

            return out, log.cache

        def run_default(combined_v, log_info: LogInfo, cache: Any):
            log = MutLogCache.new_info(log_info, cache)
            out = self.project_output(rearrange(combined_v, "... n q h -> ... q (n h)"), log.sub(project_output_name))

            return (out, log.cache)

        out, log_v.cache = maybe_static_cond(
            log_v.log_info.would_log_or_modify("out_by_head"),
            run_by_head,
            run_default,
            combined_v,
            log_v.log_info,
            log_v.cache,
        )

        return log_v.log_and_modify(out, "project_output")

    def apply_to_normal(
        self,
        key: jax.random.KeyArray,
        x: MultivariateNormal,
        log: Optional[MutLogCache] = None,
        pos_embeds: Optional[jnp.ndarray] = None,
        config: AttnApplyToNormalConfig = AttnApplyToNormalConfig(),
    ) -> MultivariateNormal:
        log_v = op.unwrap_or(log, MutLogCache.noop())

        assert not self.bias, "unsupported atm"

        _, seq_len, _ = x.mean_as()["attn_in"].shape

        def qkv(x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
            q, k, v = self.compute_qkv(x["attn_in"])
            return {**remove(x, ["attn_in"]), **dict(q=q, k=k, v=v)}

        x = x.lin_op(qkv)
        x = log_v.log_and_modify(x, "qkv")

        if pos_embeds is not None:

            def add_pos_q_k(x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
                x = copy(x)
                assert pos_embeds is not None
                pos_q, pos_k = self.get_pos_q_k(pos_embeds, seq_len)
                x["q"] += pos_q
                x["k"] += pos_k

                return x

            x = x.add(add_pos_q_k)

        x = log_v.log_and_modify(x, "final_qkv")

        x = self.attn_scores_apply_to_normal(x)
        x = log_v.log_and_modify(x, "attn_scores")

        key, subkey = jax.random.split(key)
        x = x.monte_carlo_non_linearity(
            subkey,
            lambda x, _: jax.nn.softmax(x),
            iters=config.softmax_iters,
            sample_selector=get_f("attn_scores"),
            combine=lambda probs, x: (remove_f(["attn_scores"]) @ assign_f("attn_probs", probs))(x),
        )
        x = log_v.log_and_modify(x, "attn_probs")

        x = self.mul_probs_v_apply_to_normal(x)

        x = log_v.log_and_modify(x, "mul_probs_v")

        x = x.lin_op(
            operate_f("combined_v", "attn_out", lambda x: self.project_output(rearrange(x, "b n q h -> b q (n h)")))
        )
        x = log_v.log_and_modify(x, "combined_v")

        return x

    def compute_qkv(self, x: jnp.ndarray, log: Optional[MutLogCache] = None) -> jnp.ndarray:
        return rearrange(self.attn_weights(x, log), "... s (k n h) -> k ... n s h", k=3, n=self.num_heads)

    def get_pos_q_k(
        self, pos_embeds: jnp.ndarray, seq_len, allow_position_extrapolation=False, log: Optional[MutLogCache] = None
    ):
        log_v = op.unwrap_or(log, MutLogCache.noop())

        if not allow_position_extrapolation:
            assert seq_len <= self.max_sequence_len
        seq_len_min = min(self.max_sequence_len, seq_len)
        pos = pos_embeds[:seq_len_min]
        if seq_len_min < seq_len:
            # repeat last
            pos = jnp.concatenate([pos] + [pos[-1].unsqueeze(0)] * (seq_len - seq_len_min), axis=0)

        pos = log_v.log_and_modify(pos, "pos")

        return jnp.stack(
            [
                log_v.log_and_modify(x, s)
                for x, s in zip(self.compute_qkv(pos, log_v.sub("qkv_pos")), ["pos_q", "pos_k"])
            ]
        )

    @nn.nowrap
    def attn_scores(
        self, q: jnp.ndarray, k: jnp.ndarray, mask=True, q_idxs: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        return self.attn_scores_static(q, k, self.head_size, mask=mask, q_idxs=q_idxs)

    @nn.nowrap
    @classmethod
    def attn_scores_static(
        cls, q: jnp.ndarray, k: jnp.ndarray, head_size: int, mask=True, q_idxs: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        out = jnp.einsum("... q h, ... k h -> ... q k", q, k) / (head_size ** 0.5)
        if mask:
            return cls.mask_out_upper_triangle(out, l_idxs=q_idxs)
        return out

    @nn.nowrap
    @classmethod
    def attn_scores_apply_to_normal(cls, dist: MultivariateNormal, mask=True) -> MultivariateNormal:
        _, _, _, head_size = dist.mean_as()["q"].shape
        out = dist.mul_select(
            get_f("q"),
            ["batch", "head", "seq_q", "head_size"],
            get_f("k"),
            ["batch", "head", "seq_k", "head_size"],
            ["batch", "head", "seq_q", "seq_k"],
            lambda scores, x: (remove_f(["q", "k"]) @ assign_f("attn_scores", scores))(x),
        ).lin_op(operate_f("attn_scores", "attn_scores", lambda x: x / head_size ** 0.5))
        if mask:
            return out.set(
                NEG_INF,
                lambda x, to: operate_f("attn_scores", "attn_scores", lambda x: cls.mask_out_upper_triangle(x, to))(x),
            )
        else:
            return out

    @nn.nowrap
    @staticmethod
    def mul_probs_v(probs: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("... q k, ... k h -> ... q h", probs, v)

    @nn.nowrap
    @staticmethod
    def mul_probs_v_apply_to_normal(dist: MultivariateNormal) -> MultivariateNormal:
        return dist.mul_select(
            get_f("attn_probs"),
            ["batch", "head", "seq_q", "seq_k"],
            get_f("v"),
            ["batch", "head", "seq_k", "head_size"],
            ["batch", "head", "seq_q", "head_size"],
            lambda combined, x: (remove_f(["attn_probs", "v"]) @ assign_f("combined_v", combined))(x),
        )

    @nn.nowrap
    @staticmethod
    def strict_upper_triangle_mask_idxs(idxs_l, idxs_r):
        return jnp.expand_dims(idxs_l, 1) < jnp.expand_dims(idxs_r, 0)

    @nn.nowrap
    @classmethod
    def strict_upper_triangle_mask(cls, seq_length):
        return cls.strict_upper_triangle_mask_idxs(jnp.arange(seq_length), jnp.arange(seq_length))

    @nn.nowrap
    @classmethod
    def mask_out_upper_triangle(
        cls,
        a: jnp.ndarray,
        d: Union[float, bool, int, jnp.ndarray] = NEG_INF,
        l_idxs: Optional[jnp.ndarray] = None,
        r_idxs: Optional[jnp.ndarray] = None,
    ):
        if l_idxs is None and r_idxs is None:
            assert a.shape[-2] == a.shape[-1]

        mask = cls.strict_upper_triangle_mask_idxs(
            op.unwrap_or(l_idxs, jnp.arange(a.shape[-2])),
            op.unwrap_or(r_idxs, jnp.arange(a.shape[-1])),
        )

        return jnp.where(mask, d, a)

    def get_qkv_mats(self) -> jnp.ndarray:
        return rearrange(self.attn_weights.variables["params"]["kernel"], "c (k h d) -> k h d c", k=3, h=self.num_heads)

    # a bit of swizzling never hurt anyone
    def get_q_mats(self) -> jnp.ndarray:
        return self.get_qkv_mats()[0]

    def get_k_mats(self) -> jnp.ndarray:
        return self.get_qkv_mats()[1]

    def get_v_mats(self) -> jnp.ndarray:
        return self.get_qkv_mats()[2]

    def get_qk_mats(self) -> jnp.ndarray:
        return self.get_qkv_mats()[:2]

    def get_o_mats(self) -> jnp.ndarray:
        return rearrange(self.project_output.variables["params"]["kernel"], "(n h) o -> n o h", n=self.num_heads)

    def get_qk_combined_mats(self) -> jnp.ndarray:
        q, k = self.get_qk_mats()
        return jnp.einsum("h v q, h v k -> h q k", q, k)

    def get_ov_combined_mats(self) -> jnp.ndarray:
        return jnp.einsum("h o d, h d v -> h o v", self.get_o_mats(), self.get_v_mats())

    # allows for doing computation with different q_enc and k_enc
    # ignores bias
    def qk(self, q_enc: jnp.ndarray, k_enc: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q_mat, k_mat = self.get_qk_mats()
        q = jnp.einsum("n h d, ... s d -> ... n s h", q_mat, q_enc)
        k = jnp.einsum("n h d, ... s d -> ... n s h", k_mat, k_enc)
        return q, k

    # TODO: maybe should support seq len batching to save memory when needed
    # allows for doing attn score computation with different q_enc and k_enc
    # ignores bias terms
    # requires shape [term, batch, seq, d]
    def compute_attn_scores(self, q_enc: jnp.ndarray, k_enc: jnp.ndarray, mask=True):
        q, k = self.qk(q_enc, k_enc)
        return self.attn_scores(q, k, mask=mask)

    def set_weights(
        self, mat: str, new_weights: jnp.ndarray, h_low: int, h_high: Optional[int] = None
    ) -> FrozenVariableDict:
        assert mat in {"q", "k", "o", "v"}, mat
        if h_high is None:
            h_high = h_low + 1
        if mat in {"q", "k", "v"}:
            if mat == "k":
                h_low += self.num_heads
                h_high += self.num_heads
            elif mat == "v":
                h_low += 2 * self.num_heads
                h_high += 2 * self.num_heads

            return variable_dict_replace_params(
                self.variables,
                self.attn_weights.replace_weights(
                    lambda x: x.at[h_low * self.head_size : h_high * self.head_size].set(new_weights)
                ),
                "attn_weights",
            )
        elif mat == "o":
            return variable_dict_replace_params(
                self.variables,
                self.project_output.replace_weights(
                    lambda x: x.at[:, h_low * self.head_size : h_high * self.head_size].set(new_weights)
                ),
                "project_output",
            )
        else:
            assert False


id_norm = lambda x, _log: x


class GptBlock(nn.Module):
    num_heads: int
    hidden_size: int
    dropout_rate: float
    attn_probs_dropout_rate: float
    attn_bias: bool
    get_norm: Callable[[], Callable[[jnp.ndarray, MutLogCache], jnp.ndarray]]
    max_sequence_len: int
    use_mlp: bool
    mlp_act_type: MlpActType
    dtype: Any = jnp.float32

    def setup(self):
        self.attention = UnidirectionalAttn(
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            bias=self.attn_bias,
            attn_probs_dropout_rate=self.attn_probs_dropout_rate,
            max_sequence_len=self.max_sequence_len,
            dtype=self.dtype,
        )

        self.norm1 = self.get_norm()

        if self.use_mlp:
            self.norm2 = self.get_norm()
            self.linear1 = Dense(self.hidden_size * 4, dtype=self.dtype)
            if self.mlp_act_type == "gelu":
                self.activation = PointwiseNonlinearity(gelu)
            elif self.mlp_act_type == "relu":
                self.activation = PointwiseNonlinearity(jax.nn.relu)
            else:
                assert_never(self.mlp_act_type)
            self.linear2 = Dense(self.hidden_size, dtype=self.dtype)

        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(  # type: ignore[override]
        self,
        encodings: jnp.ndarray,
        log: Optional[MutLogCache] = None,
        allow_position_extrapolation=False,
        pos_embeds: Optional[jnp.ndarray] = None,
        training: bool = False,
        activation_cache: Optional[AttnActivationCache] = None,
    ) -> jnp.ndarray:
        log_v = op.unwrap_or(log, MutLogCache.noop())

        attn = self.attn(
            encodings,
            log_v.sub("attention"),
            allow_position_extrapolation=allow_position_extrapolation,
            pos_embeds=pos_embeds,
            training=training,
            activation_cache=activation_cache,
        )
        attn_res = log_v.log_and_modify(attn, "attn_res")
        inp_res = log_v.log_and_modify(encodings, "inp_res")

        if self.use_mlp:
            inp_res_for_mlp = log_v.log_and_modify(encodings, "inp_res_for_mlp")
            res1 = attn + inp_res_for_mlp
            mlp = self.mlp(res1, log_v.sub("mlp"), training=training)
            return log_v.log_and_modify(mlp + attn_res + inp_res, "out")
        else:
            return log_v.log_and_modify(attn_res + inp_res, "out")

    def mlp(self, x: jnp.ndarray, log: Optional[MutLogCache] = None, training: bool = False) -> jnp.ndarray:
        log_v = op.unwrap_or(log, MutLogCache.noop())

        def apply_layer(x, layer, name):
            return log_v.log_and_modify(layer(x, log_v.sub(name)), name)

        x = log_v.log_and_modify(x, "inp")
        x = apply_layer(x, self.norm2, "norm2")
        x = apply_layer(x, self.linear1, "linear1")
        x = apply_layer(x, self.activation, "gelu")

        def run_by_neuron(x, log: MutLogCache):
            log = log.clone()
            orig_type = x.dtype
            x = log.log_and_modify(
                jnp.einsum("b s n, o n-> b n s o", x, self.linear2.get_weights()),
                "out_by_neuron",
            )

            # names + shape + type should match Dense
            linear2_log = log.sub("linear2")
            x = linear2_log.log_and_modify(jnp.asarray(jnp.sum(x, axis=1), dtype=orig_type), "mul_kernel")
            if self.linear2.use_bias:
                bias = linear2_log.log_and_modify(self.linear2.get_bias(), "bias")
                x = linear2_log.log_and_modify(x + jnp.expand_dims(bias, (0, 1)), "added_bias")

            return x, log.cache

        def run_default(x, log: MutLogCache):
            log = log.clone()
            x = self.linear2(x, log.sub("linear2"))

            return x, log.cache

        x, log_v.cache = maybe_static_cond(
            log_v.log_info.would_log_or_modify("out_by_neuron"), run_by_neuron, run_default, x, log_v
        )
        x = log_v.log_and_modify(x, "linear2")
        x = log_v.log_and_modify(self.dropout(x, deterministic=not training), "dropout")
        x = log_v.log_and_modify(x, "out")

        return x

    def attn(
        self,
        x: jnp.ndarray,
        log: Optional[MutLogCache] = None,
        allow_position_extrapolation: bool = False,
        pos_embeds: Optional[jnp.ndarray] = None,
        training: bool = False,
        activation_cache: Optional[AttnActivationCache] = None,
    ) -> jnp.ndarray:
        log_v = op.unwrap_or(log, MutLogCache.noop())

        x = log_v.log_and_modify(x, "inp")
        x = log_v.log_and_modify(self.norm1(x, log_v.sub("norm1")), "norm1")
        x = self.attention(
            x,
            log_v,
            allow_position_extrapolation=allow_position_extrapolation,
            pos_embeds=pos_embeds,
            training=training,
            activation_cache=activation_cache,
        )
        x = log_v.log_and_modify(x, "out")
        return x

    def apply_to_normal(
        self,
        key: jax.random.KeyArray,
        x: MultivariateNormal,
        log: Optional[MutLogCache] = None,
        pos_embeds: Optional[jnp.ndarray] = None,
        config: AttnApplyToNormalConfig = AttnApplyToNormalConfig(),
    ):
        log_v = op.unwrap_or(log, MutLogCache.noop())

        assert not self.use_mlp, "currently unsupported"
        assert self.norm1 is id_norm, "norm currently unsupported"

        key, subkey = jax.random.split(key)
        x = self.attention.apply_to_normal(
            subkey,
            x.lin_op(lambda x: assign(x, "attn_in", x["residual"])),
            log=log_v.sub("attention"),
            pos_embeds=pos_embeds,
            config=config,
        )
        x = x.lin_op(lambda x: (remove_f(["attn_out"]) @ assign_f("residual", x["attn_out"] + x["residual"]))(x))
        x = log_v.log_and_modify(x, "out")

        return x

    # override for typing
    def bind(
        self, variables: VariableDict, *args, rngs: Optional[RNGSequences] = None, mutable: CollectionFilter = False
    ) -> GptBlock:
        return super().bind(variables, *args, rngs=rngs, mutable=mutable)

    def set_linear_weights(self, mat: str, new_weights: jnp.ndarray) -> FrozenVariableDict:
        assert self.use_mlp
        assert mat in {"l1", "l2"}
        return variable_dict_replace_params(
            self.variables,
            (self.linear1 if mat == "l1" else self.linear2).set_weights(new_weights),
            "linear1" if mat == "l1" else "linear2",
        )

    def set_attn_weights(
        self, mat: str, new_weights: jnp.ndarray, h_low: int, h_high: Optional[int] = None
    ) -> FrozenVariableDict:
        return variable_dict_replace_params(
            self.variables, self.attention.set_weights(mat, new_weights, h_low, h_high), "attention"
        )


PosEncType = Literal["gpt", "shortformer"]


class Embedding(nn.Module):
    vocab_size: int
    embedding_size: int
    max_position_embeddings: int
    dropout_rate: float
    pos_enc_type: PosEncType
    tied_embed_unembed: bool
    dtype: Any = jnp.float32

    def setup(self):
        self.token_embedding = nn.Embed(self.vocab_size, self.embedding_size, dtype=self.dtype)
        self.position_embedding = nn.Embed(self.max_position_embeddings, self.embedding_size, dtype=self.dtype)
        self.dropout = nn.Dropout(self.dropout_rate)
        if not self.tied_embed_unembed:
            self._token_unembedding = nn.Embed(self.vocab_size, self.embedding_size, dtype=self.dtype)

    def __call__(  # type: ignore[override]
        self,
        input_ids: jnp.ndarray,
        log: Optional[MutLogCache] = None,
        pos_idxs: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        log_v = op.unwrap_or(log, MutLogCache.noop())

        seq_len = input_ids.shape[1]
        if pos_idxs is None:
            pos_idxs = jnp.arange(seq_len)

        tok_embed = log_v.log_and_modify(self.token_embedding(input_ids), "tok")
        pos_embed = log_v.log_and_modify(jnp.expand_dims(self.position_embedding(pos_idxs), 0), "pos")

        overall_embed = tok_embed
        if self.pos_enc_type == "gpt":
            overall_embed = overall_embed + pos_embed

        overall_embed = log_v.log_and_modify(overall_embed, "overall")
        overall_embed = log_v.log_and_modify(self.dropout(overall_embed, deterministic=not training), "dropout")
        out_dict = {"tok": overall_embed}

        if self.pos_enc_type == "shortformer":
            out_dict["pos"] = pos_embed

        return out_dict

    @property
    def token_unembedding(self):
        return self.token_embedding if self.tied_embed_unembed else self._token_unembedding

    def unembed(self, encodings: jnp.ndarray) -> jnp.ndarray:
        unembedding = self.token_unembedding.embedding
        return jnp.einsum("... x, v x -> ... v", encodings, unembedding)

    def bind(
        self, variables: VariableDict, *args, rngs: Optional[RNGSequences] = None, mutable: CollectionFilter = False
    ) -> Embedding:
        return super().bind(variables, *args, rngs=rngs, mutable=mutable)
