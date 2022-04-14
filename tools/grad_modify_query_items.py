import operator
from typing import Any, Callable, List, Set, Tuple, Optional, Dict, Union
from functools import partial

import jax
import jax.numpy as jnp
from jax.interpreters.ad import JVPTracer
from attrs import define, frozen, Factory
from jax.tree_util import register_pytree_node_class

from interp.tools.grad_modify_query import GradModifierConf
from interp.tools.custom_jvp import different_function_custom_jvp
from interp.tools.log import EnableMod, EnableModSetup, KeyIdxs, Mod, ModInfo, LogCache
from interp.tools.jax_tree_util import AttrsPartiallyStatic, AttrsPartiallyStaticDefaultNonStatic
import interp.tools.optional as op


@register_pytree_node_class
@frozen
class ItemIdx(AttrsPartiallyStatic):
    idx_or_mask: Any = ...
    is_mask: bool = False
    is_static: bool = True
    get_idx_or_mask: Callable[[Any, Optional[jnp.ndarray]], Any] = lambda x, _: x
    except_idx: bool = False

    def non_static_names(self) -> Set[str]:
        return set() if self.is_static else {"idx_or_mask"}

    def apply(self, x: jnp.ndarray, idx: Optional[jnp.ndarray], op, op_on_idxed: bool = True):
        idx_or_mask = self.get_idx_or_mask(self.idx_or_mask, idx)
        if self.is_mask:
            # except_idx has no effect in this case!
            mask, not_mask = idx_or_mask, ~idx_or_mask
            return x * not_mask + op(x) * mask
        else:
            idx = idx_or_mask
            if self.except_idx:
                return op(x).at[idx].set(x[idx], unique_indices=True)
            else:
                return x.at[idx].set(op(x[idx]) if op_on_idxed else op(x)[idx], unique_indices=True)


# note: hash by id
@frozen(eq=False)
class ItemConf:
    log_key_idxs: KeyIdxs
    item_idx: ItemIdx = ItemIdx()
    positive: bool = True
    enable_setup: EnableModSetup = EnableModSetup()

    def mod_info(self, mod: Mod, to_log: List[KeyIdxs] = []) -> ModInfo:
        return ModInfo(self.log_key_idxs, EnableMod(mod, self.enable_setup), to_log)


@frozen(eq=False)
class BaseConf:
    conf: ItemConf

    def is_positive(self) -> bool:
        return self.conf.positive


@register_pytree_node_class
@frozen
class ItemIdxMod(AttrsPartiallyStaticDefaultNonStatic):
    item_idx: ItemIdx
    op: Callable[[Any, jnp.ndarray], jnp.ndarray]
    args: Any = None
    op_static: bool = True
    args_static: bool = False
    op_on_idxed: bool = True

    def static_names(self) -> Set[str]:
        out = {"op_static", "args_static", "op_on_idxed"}
        if self.op_static:
            out.add("op")
        if self.args_static:
            out.add("args")
        return out

    def __call__(self, x: jnp.ndarray, _, idx: Optional[jnp.ndarray]) -> jnp.ndarray:
        return self.item_idx.apply(x, idx, partial(self.op, self.args), op_on_idxed=self.op_on_idxed)


@frozen(eq=False)
class MulConf(BaseConf, GradModifierConf):
    shape: Tuple[int, ...] = ()

    def get_mod(self, alpha: JVPTracer, _) -> ModInfo:
        return self.conf.mod_info(ItemIdxMod(self.conf.item_idx, op=operator.mul, args=alpha))

    def get_val(self) -> jnp.ndarray:
        # TODO: handle shape issues more generally!
        # probably should get broadcasted somehow or something?
        return jnp.ones(self.shape)


@frozen(eq=False)
class AddConf(BaseConf, GradModifierConf):
    shape: Tuple[int, ...] = ()
    multiplier: Union[float, jnp.ndarray] = 1.0

    def get_mod(self, beta: JVPTracer, _) -> ModInfo:
        return self.conf.mod_info(ItemIdxMod(self.conf.item_idx, op=operator.add, args=beta * self.multiplier))

    def get_val(self) -> jnp.ndarray:
        return jnp.zeros(self.shape)


def stop_grad_op(_, x):
    return jax.lax.stop_gradient(x)


@frozen(eq=False)
class StopGradConf(BaseConf, GradModifierConf):
    def get_mod(self, *_) -> ModInfo:
        return self.conf.mod_info(ItemIdxMod(self.conf.item_idx, op=stop_grad_op))

    def get_val(self) -> Optional[jnp.ndarray]:
        return None


@register_pytree_node_class
@define
class ReplaceOp(AttrsPartiallyStatic):
    item_idx: ItemIdx
    replacement: Callable[[Any], Any]
    from_key_idxs: KeyIdxs
    get_from_key_idxs_for_cache: Callable[[KeyIdxs, Optional[jnp.ndarray]], KeyIdxs]

    def non_static_names(self) -> Set[str]:
        return {"item_idx", "from_key_idxs"}

    def __call__(self, x: jnp.ndarray, cache: LogCache, i: Optional[jnp.ndarray]) -> jnp.ndarray:
        # we can't apply the function on just the indexed part as the
        # function could be non-local in general (we must trust in jax)
        new_x = self.replacement(cache.get(self.get_from_key_idxs_for_cache(self.from_key_idxs, i)))
        return self.item_idx.apply(x, i, lambda _: new_x, op_on_idxed=False)


# defined outside for constant hash
def identity_get_key_idxs(key_idxs: KeyIdxs, _):
    return key_idxs


def single_same_get_key_idxs(key_idxs: KeyIdxs, idx: Optional[jnp.ndarray]):
    return KeyIdxs.single(key_idxs.key, op.unwrap(idx))


@frozen(eq=False)
class ReplaceFuncConf(BaseConf, GradModifierConf):
    from_key_idxs: KeyIdxs
    # if you want this function to have identical primal behavior, you should
    # check that yourself
    replacement: Callable[[Any], Any]

    # by default uses same index as was passed if idxed
    get_from_key_idxs_for_cache: Callable[[KeyIdxs, Optional[jnp.ndarray]], KeyIdxs] = Factory(
        lambda x: single_same_get_key_idxs if x.from_key_idxs.idxs.is_idxed else identity_get_key_idxs, takes_self=True
    )

    def get_mod(self, *_) -> ModInfo:
        return self.conf.mod_info(
            ReplaceOp(
                self.conf.item_idx,
                replacement=self.replacement,
                from_key_idxs=self.from_key_idxs,
                get_from_key_idxs_for_cache=self.get_from_key_idxs_for_cache,
            ),
            to_log=[self.from_key_idxs],
        )

    def get_val(self) -> None:
        return None


@frozen(eq=False)
class NoneConf(GradModifierConf):
    positive: bool = True

    def get_mod(self, *_) -> None:
        return None

    def get_val(self) -> None:
        return None

    def is_positive(self) -> bool:
        return self.positive


@register_pytree_node_class
@define
class ModifyInBasisOp(AttrsPartiallyStaticDefaultNonStatic):
    item_idx: ItemIdx
    proj_to: jnp.ndarray
    proj_back: jnp.ndarray
    sub_mod: Optional[Mod]
    just_jvp: bool

    def static_names(self) -> Set[str]:
        return {"just_jvp"}

    def __call__(self, x: jnp.ndarray, cache: LogCache, i: Optional[jnp.ndarray]) -> jnp.ndarray:
        def op(to_proj):
            projected = jnp.einsum("p i, ... i -> ... p", self.proj_to, to_proj)
            if self.sub_mod is not None:
                projected = self.sub_mod(projected, cache, i)
            return jnp.einsum("o p, ... p -> ... o", self.proj_back, projected)

        def run(x):
            return self.item_idx.apply(x, i, op)

        if self.just_jvp:
            return different_function_custom_jvp(lambda x: x, run)(x)
        else:
            return run(x)


@frozen(eq=False)
class ModifyInBasisConf(BaseConf, GradModifierConf):
    proj_to: jnp.ndarray
    proj_back: jnp.ndarray

    # name doesn't matter for this
    sub_conf: GradModifierConf

    just_jvp: bool = True

    def get_mod(self, my_val: JVPTracer, all_vals: Dict[GradModifierConf, JVPTracer]) -> ModInfo:
        sub_mod_info = self.sub_conf.get_mod(my_val, all_vals)

        return self.conf.mod_info(
            ModifyInBasisOp(
                self.conf.item_idx,
                proj_to=self.proj_to,
                proj_back=self.proj_back,
                sub_mod=op.map(sub_mod_info, lambda info: info.mod),
                just_jvp=self.just_jvp,
            ),
            op.unwrap_or(op.map(sub_mod_info, lambda info: info.to_log), []),
        )

    def get_val(self) -> Optional[Union[jnp.ndarray, float]]:
        return self.sub_conf.get_val()
