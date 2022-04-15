from __future__ import annotations

from collections import defaultdict
from typing import Dict, Generic, Iterable, List, Protocol, Set, Callable, Any, TypeVar, Union, Optional, Tuple
from copy import copy

from attrs import define, Factory, Factory, asdict, frozen, evolve
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class, tree_map
import numpy as np

from interp.tools.jax_util import maybe_static_cond, stack_tree
from interp.tools.immutable_dict import assign, operate
from interp.tools.jax_tree_util import AttrsPartiallyStaticDefaultNonStatic, AttrsPartiallyStatic
import interp.tools.optional as op

# also implementation of NoopLog
class Logger(Protocol):
    def would_log(self, key: str, idx: Optional[jnp.ndarray] = None) -> Union[jnp.ndarray, bool]:
        return False

    def would_log_or_modify(self, key: str, idx: Optional[jnp.ndarray] = None) -> Union[jnp.ndarray, bool]:
        return False

    def init_cache(self) -> Any:
        return None

    def log(self, value: Any, key: str, cache: Any, idx: Optional[jnp.ndarray] = None) -> Any:
        return cache

    def log_and_modify(self, value: Any, key: str, cache: Any, idx: Optional[jnp.ndarray] = None) -> Tuple[Any, Any]:
        return value, cache

    def init_idxed_for_shapes(self, cache: Any, shapes: Dict[str, Any], count: int) -> Any:
        return cache

    def check_finish_cache(self, cache: Any) -> None:
        return


@register_pytree_node_class
class NoopLogger(Logger, AttrsPartiallyStatic):
    ...


@register_pytree_node_class
class ShapeLogger(Logger, AttrsPartiallyStatic):
    """
    Tracks shapes through a dummy pass so that other caches can be created with
    the right shape (helpful for jax compilation)
    """

    def would_log(self, key: str, idx: Optional[jnp.ndarray] = None) -> Union[jnp.ndarray, bool]:
        return True

    def would_log_or_modify(self, key: str, idx: Optional[jnp.ndarray] = None) -> Union[jnp.ndarray, bool]:
        return True

    def init_cache(self) -> Any:
        return {}

    def log(self, value: Any, key: str, cache: Any, idx: Optional[jnp.ndarray] = None) -> Any:
        return assign(cache, key, value, check_absent=True)

    def log_and_modify(self, value: Any, key: str, cache: Any, idx: Optional[jnp.ndarray] = None) -> Tuple[Any, Any]:
        return value, self.log(value, key, cache, idx)


@register_pytree_node_class
@frozen
class LogInfo(AttrsPartiallyStatic):
    logger: Logger
    log_prefix: Optional[str] = None
    log_idx: Optional[jnp.ndarray] = None

    def key(self, s: str):
        return s if self.log_prefix is None else f"{self.log_prefix}.{s}"

    def would_log(self, s: str) -> Union[jnp.ndarray, bool]:
        return self.logger.would_log(self.key(s), self.log_idx)

    def would_log_or_modify(self, s: str) -> Union[jnp.ndarray, bool]:
        return self.logger.would_log_or_modify(self.key(s), self.log_idx)

    def log(self, value: Any, s: str, cache: Any) -> Any:
        return self.logger.log(value, self.key(s), cache, self.log_idx)

    def log_and_modify(self, value: Any, s: str, cache: Any) -> Tuple[Any, Any]:
        return self.logger.log_and_modify(value, self.key(s), cache, self.log_idx)

    def sub(self, name: Optional[str] = None, log_idx: Optional[Union[jnp.ndarray, int]] = None) -> LogInfo:
        if self.log_idx is not None:
            # just add, cartesian product/unravel must be handled separately
            log_idx = self.log_idx + op.unwrap_or(log_idx, 0)

        return evolve(
            self, log_prefix=self.log_prefix if name is None else self.key(name), log_idx=op.map(log_idx, jnp.array)
        )

    def non_static_names(self) -> Set[str]:
        return {"logger", "log_idx"}


T = TypeVar("T")


@define
class Ref(Generic[T]):
    v: T


# not sure if this is way...
@define
class MutLogCache:
    """
    Tracks a mutable reference to caches, thus allowing for various convenience functions that can operate on
    the stored cache while allowing the Cache classes to be entirely immuntable.

    Intentionally not pytree node class, split into parts if you want to pass.
    """

    log_info: LogInfo
    # use ref so that cache mutation persists through 'sub'
    cache_ref: Ref[Any]

    @property
    def cache(self):
        return self.cache_ref.v

    @cache.setter
    def cache(self, cache):
        self.cache_ref.v = cache

    @classmethod
    def new_info(cls, log_info: LogInfo, cache: Optional[Any] = None):
        return cls.new(**asdict(log_info, recurse=False), cache=cache)

    @classmethod
    def new(
        cls,
        logger: Logger,
        log_prefix: Optional[str] = None,
        log_idx: Optional[jnp.ndarray] = None,
        cache: Optional[Any] = None,
    ):
        return cls(
            LogInfo(logger, log_prefix=log_prefix, log_idx=log_idx), Ref(op.unwrap_or(cache, logger.init_cache()))
        )

    @classmethod
    def noop(cls):
        return cls.new(NoopLogger())

    def log(self, value: Any, s: str) -> None:
        self.cache = self.log_info.log(value, s, self.cache)

    def log_and_modify(self, value: Any, s: str) -> Any:
        value, out_cache = self.log_info.log_and_modify(value, s, self.cache)
        self.cache = out_cache
        return value

    def sub(self, name: Optional[str] = None, log_idx: Optional[Union[jnp.ndarray, int]] = None) -> MutLogCache:
        return evolve(self, log_info=self.log_info.sub(name, log_idx))

    def clone(self):
        return evolve(self, cache_ref=copy(self.cache_ref))

    def check_finish(self):
        self.log_info.logger.check_finish_cache(self.cache)

    def init_idxed_for_shapes(self, shapes: Dict[str, Any], count: int) -> None:
        self.cache = self.log_info.logger.init_idxed_for_shapes(self.cache, shapes, count)


def tuple_set(tup, i, value):
    return tup[:i] + (value,) + tup[i + 1 :]


@register_pytree_node_class
@frozen
class IdxedValues(AttrsPartiallyStaticDefaultNonStatic):
    idxs: jnp.ndarray
    is_set: jnp.ndarray
    values: Any

    def set(self, idx: jnp.ndarray, value: Any) -> IdxedValues:
        which = jnp.argmax(jnp.concatenate([self.idxs == idx]))

        return jax.lax.cond(
            (self.idxs == idx).any(),
            lambda: evolve(
                self,
                is_set=self.is_set.at[which].set(True),
                values=tree_map(lambda orig, new: orig.at[which].set(new), self.values, value),
            ),
            lambda: self,
        )

    def get(self, idx: Union[int, jnp.ndarray], check=False) -> Any:
        """
        Danger! This is unchecked by default and everything must be known statically if you want to check!
        """
        return tree_map(lambda x: x.squeeze(0), self.get_idxs(jnp.asarray(idx)[None], check=check))

    def get_idxs(self, idxs: Optional[jnp.ndarray], check=False) -> Any:
        """
        Danger! This is unchecked by default and everything must be known statically if you want to check!
        """
        if idxs is None:
            return self.values

        assert idxs.ndim == 1

        eq = self.idxs[None] == idxs[:, None]
        if check:
            assert eq.any(axis=-1).all()
        which = jnp.argmax(eq, axis=-1)
        if check:
            assert self.is_set[which].all()
        return tree_map(lambda x: x[which], self.values)

    def cleanup_stack_dim(self, check=False):
        assert self.idxs.ndim > 1
        assert self.is_set.ndim > 1

        if check:
            assert (self.idxs[None, 0] == self.idxs).all()
            assert (self.is_set[None, 0] == self.is_set).all()

        return IdxedValues(
            idxs=self.idxs[0], is_set=self.is_set[0], values=tree_map(lambda x: jnp.swapaxes(x, 0, 1), self.values)
        )

    def map(self, f) -> IdxedValues:
        return evolve(self, values=jax.vmap(lambda i: f(self.get(i)))(jnp.arange(self.idxs.shape[0])))


@register_pytree_node_class
@frozen
class Idxs(AttrsPartiallyStatic):
    idxs: Optional[Union[jnp.ndarray, np.ndarray]] = None
    is_single: bool = False
    is_idxed: bool = Factory(lambda x: x.idxs is not None, takes_self=True)

    @classmethod
    def single(cls, idx: Union[jnp.ndarray, np.ndarray, int]):
        if isinstance(idx, int):
            idx = np.array(idx)
        assert idx.ndim == 0
        return cls(idxs=idx[None], is_single=True, is_idxed=True)

    @classmethod
    def all(cls):
        return cls(idxs=None, is_idxed=True)

    def __attrs_post_init__(self):
        if not self.is_idxed:
            assert self.idxs is None
        if self.is_single:
            assert self.idxs is None or self.idxs.shape == (1,)

    def non_static_names(self) -> Set[str]:
        return {"idxs"}  # might not do what you want if idxs is numpy array!


@register_pytree_node_class
@frozen
class KeyIdxs(AttrsPartiallyStatic):
    key: str
    idxs: Idxs = Idxs()

    def non_static_names(self) -> Set[str]:
        return {"idxs"}

    @classmethod
    def single(cls, key: str, idx: Union[jnp.ndarray, np.ndarray, int]):
        return cls(key, Idxs.single(idx))


@register_pytree_node_class
@define
class LogCache(AttrsPartiallyStaticDefaultNonStatic):
    cache: Dict[str, Any]
    idxed_cache: Dict[str, IdxedValues]
    sub_log_cache: Any

    @staticmethod
    def unwrap(cache: Optional[Any]) -> LogCache:
        out: Any = op.unwrap(cache)
        assert isinstance(out, LogCache)
        return out

    def get(self, key_idxs: KeyIdxs, check=False) -> Any:
        if key_idxs.idxs.is_idxed:
            idxed_v = self.idxed_cache[key_idxs.key]
            if key_idxs.idxs.is_single:
                return idxed_v.get(jnp.array(op.unwrap(key_idxs.idxs.idxs)).squeeze(), check=check)
            else:
                return idxed_v.get_idxs(op.map(key_idxs.idxs.idxs, jnp.array), check=check)
        else:
            return self.cache[key_idxs.key]

    def cleanup_stack_dim(self, check=False):
        assert self.sub_log_cache is None
        return evolve(self, idxed_cache={k: v.cleanup_stack_dim(check=check) for k, v in self.idxed_cache.items()})

    def map(self, f) -> LogCache:
        """
        Be warned, doesn't map sub_log_cache!
        """

        return evolve(
            self,
            cache={k: f(v) for k, v in self.cache.items()},
            idxed_cache={k: v.map(f) for k, v in self.idxed_cache.items()},
        )


def maybe_static_or(l: Union[jnp.ndarray, bool], r: Union[jnp.ndarray, bool]) -> Union[jnp.ndarray, bool]:
    l_b = isinstance(l, bool)
    r_b = isinstance(r, bool)
    if l_b and r_b:
        return l or r
    if (l_b and l) or (r_b and r):
        return True
    return l | r


@register_pytree_node_class
@define
class LoggerCacheAll(Logger, AttrsPartiallyStatic):
    """Cache that logs everything possible by default"""

    exclude: Set[str] = set()  # allows for excluding some keys, for a bit of convenience & avoiding oom errors
    sub_log: Logger = NoopLogger()

    def would_log_base(self, key: str, idx: Optional[jnp.ndarray]) -> Union[jnp.ndarray, bool]:
        return key not in self.exclude

    def would_log(self, key: str, idx: Optional[jnp.ndarray] = None) -> Union[jnp.ndarray, bool]:
        return maybe_static_or(self.would_log_base(key, idx), self.sub_log.would_log(key, idx))

    def would_log_or_modify(self, key: str, idx: Optional[jnp.ndarray] = None) -> Union[jnp.ndarray, bool]:
        """
        If you override modify, you should also override this method to indicate when your modification applies,
        so that the underlying model can decide not to do the modification plumbing. If you don't do this, then
        sometimes your modifications will work anyway, but sometimes they won't.
        """
        return maybe_static_or(self.would_log(key, idx), self.sub_log.would_log_or_modify(key, idx))

    def init_cache(self) -> LogCache:
        return LogCache(cache={}, idxed_cache={}, sub_log_cache=self.sub_log.init_cache())

    def base_log(self, value: Any, key: str, cache: LogCache, idx: Optional[jnp.ndarray]) -> LogCache:
        if idx is None:
            return maybe_static_cond(
                self.would_log_base(key, idx),
                lambda: evolve(cache, cache=assign(cache.cache, key, value, check_absent=True)),
                lambda: cache,
            )
        else:
            return maybe_static_cond(
                self.would_log_base(key, idx),
                lambda: evolve(
                    cache,
                    idxed_cache=operate(cache.idxed_cache, key, key, lambda idxed_values: idxed_values.set(idx, value)),
                ),
                lambda: cache,
            )

    def log(self, value: Any, key: str, cache: LogCache, idx: Optional[jnp.ndarray] = None) -> LogCache:
        return evolve(
            self.base_log(value, key, cache, idx), sub_log_cache=self.sub_log.log(value, key, cache.sub_log_cache, idx)
        )

    def modify(self, value: Any, key: str, cache: LogCache, idx: Optional[jnp.ndarray]) -> Tuple[Any, LogCache]:
        return value, cache

    def modify_after_log(
        self, value: Any, key: str, cache: LogCache, idx: Optional[jnp.ndarray]
    ) -> Tuple[Any, LogCache]:
        return value, cache

    def log_and_modify(
        self, value: Any, key: str, cache: LogCache, idx: Optional[jnp.ndarray] = None
    ) -> Tuple[Any, LogCache]:
        value, cache = self.modify(value, key, cache, idx)
        cache = self.base_log(value, key, cache, idx)
        value, cache = self.modify_after_log(value, key, cache, idx)
        value, sub_log_cache = self.sub_log.log_and_modify(value, key, cache.sub_log_cache, idx)
        return value, evolve(cache, sub_log_cache=sub_log_cache)

    def get_log_idxs_for_keys(self, keys: Iterable[str], count: int) -> Dict[str, jnp.ndarray]:
        return {k: jnp.arange(count) for k in keys if k not in self.exclude}

    def init_idxed_for_shapes(self, cache: LogCache, shapes: Dict[str, Any], count: int) -> LogCache:
        """
        Creates a new cache of the right shape (given by shapes, usually originating from a dummy pass with ShapeLogger)
        Again, this is for the sake of jax compiling."""
        key_idxs = self.get_log_idxs_for_keys(shapes.keys(), count)
        new_idxed_values = {
            k: IdxedValues(
                idxs,
                jnp.zeros((idxs.shape[0],), dtype=jnp.bool_),
                tree_map(lambda x: jnp.full((idxs.shape[0],) + x.shape, float("nan"), dtype=x.dtype), shapes[k]),
            )
            for k, idxs in key_idxs.items()
        }
        for k in new_idxed_values.keys():
            assert k not in cache.idxed_cache, f"{k} already present in idxed_cache"
        return evolve(
            cache,
            idxed_cache={**cache.idxed_cache, **new_idxed_values},
            sub_log_cache=self.sub_log.init_idxed_for_shapes(cache.sub_log_cache, shapes, count),
        )

    def check_finish_cache(self, cache: LogCache) -> None:
        self.sub_log.check_finish_cache(cache.sub_log_cache)

    def non_static_names(self) -> Set[str]:
        return {"sub_log"}


class UnusedKeyError(Exception):
    pass


@register_pytree_node_class
@define
class LoggerCache(LoggerCacheAll):
    to_cache: Set[str] = Factory(set)
    # if left as None, cache all
    to_cache_idxed: defaultdict[str, Optional[Union[jnp.ndarray, np.ndarray]]] = Factory(
        lambda: defaultdict(lambda: jnp.array([], dtype=jnp.int32))
    )
    idxs_as_numpy: bool = False
    check_all_acquired: bool = True

    @classmethod
    def from_key_idxs(cls, key_idxs: Iterable[KeyIdxs], idxs_as_numpy: bool = False, check_all_acquired: bool = True):
        out = cls(idxs_as_numpy=idxs_as_numpy, check_all_acquired=check_all_acquired)
        out.add_all(key_idxs)

        return out

    @property
    def idx_np(self):
        return np if self.idxs_as_numpy else jnp

    # NOTE: idxs must be static for this to work! (typically use numpy to ensure this)
    def dedup_idxs(self):
        return jax.tree_util.tree_map(lambda x: self.idx_np.unique(x)[0], self.to_cache_idxed)

    def add_all(self, key_idxs: Iterable[KeyIdxs]):
        for x in key_idxs:
            self.add(x)

    def add(self, key_idxs: KeyIdxs):
        if key_idxs.idxs.is_idxed:
            if key_idxs.idxs.idxs is None or self.to_cache_idxed[key_idxs.key] is None:
                self.to_cache_idxed[key_idxs.key] = None
            else:
                # concat here is a bit silly, but can be deduped later.
                self.to_cache_idxed[key_idxs.key] = self.idx_np.concatenate([self.to_cache_idxed[key_idxs.key], key_idxs.idxs.idxs])  # type: ignore
        else:
            self.to_cache.add(key_idxs.key)

    def would_log_base(self, key: str, idx: Optional[jnp.ndarray]) -> Union[jnp.ndarray, bool]:
        if idx is None:
            return key in self.to_cache
        else:
            if key not in self.to_cache_idxed:
                return False
            idxs = self.to_cache_idxed[key]
            if idxs is None:
                return True
            return (idxs == idx).any()

    def get_log_idxs_for_keys(self, keys: Iterable[str], count: int) -> Dict[str, jnp.ndarray]:
        return {
            k: op.unwrap_or(op.map(self.to_cache_idxed[k], jnp.array), jnp.arange(count))
            for k in keys
            if k in self.to_cache_idxed
        }

    def check_finish_cache(self, cache: LogCache) -> None:
        super().check_finish_cache(cache)
        if self.check_all_acquired:
            unused = self.to_cache - cache.cache.keys()
            unused_idxed = set(self.to_cache_idxed.keys()) - cache.idxed_cache.keys()
            if unused or unused_idxed:
                raise UnusedKeyError(
                    f"LoggerCache: check_all_acquired enabled but some keys not acquired.\n"
                    f"non-indexed: {unused}, indexed: {unused_idxed}\n"
                    f"used non-indexed {cache.cache.keys()}, used indexed: {cache.idxed_cache.keys()}\n"
                    f"expected non-indexed {self.to_cache}, expected indexed: {self.to_cache_idxed.keys()}\n"
                )

    def non_static_names(self) -> Set[str]:
        return super().non_static_names().union({"to_cache_idxed"})


Mod = Callable[[Any, LogCache, Optional[jnp.ndarray]], Any]


@register_pytree_node_class
@frozen
class EnableModSetup(AttrsPartiallyStatic):
    enable: Union[bool, jnp.ndarray] = True
    is_enable_static: bool = True
    # by idx allows for separate enable value for each idx
    enable_by_idx: bool = False

    def __attrs_post_init__(self):
        if self.enable_by_idx:
            assert not isinstance(self.enable, bool)
            assert self.enable.ndim == 1

    def get(self, idx: Optional[jnp.ndarray]):
        if self.enable_by_idx:
            assert not isinstance(self.enable, bool)
            return self.enable[op.unwrap(idx)]
        else:
            return self.enable

    def non_static_names(self) -> Set[str]:
        if not self.is_enable_static:
            return {"enable"}
        else:
            return set()


@register_pytree_node_class
@frozen
class EnableMod(AttrsPartiallyStaticDefaultNonStatic):
    mod: Mod
    enable_setup: EnableModSetup

    def __call__(self, value: Any, cache: LogCache, idx: Optional[jnp.ndarray]):
        return maybe_static_cond(
            self.enable_setup.get(idx),
            lambda *args: self.mod(*args),
            (lambda x, *_: x),
            value,
            cache,
            idx,
            force_is_static=self.enable_setup.is_enable_static,
        )


@register_pytree_node_class
@frozen
class ModIdxs(AttrsPartiallyStaticDefaultNonStatic):
    mod: Mod
    # if left as None, mod all
    idxs: Optional[jnp.ndarray]

    def __call__(self, value: Any, cache: LogCache, idx: Optional[jnp.ndarray]) -> Any:
        return jax.lax.cond(
            self.idxs is None or (self.idxs == op.unwrap(idx)).any(),
            lambda *args: self.mod(*args),
            lambda x, *_: x,
            value,
            cache,
            idx,
        )


@register_pytree_node_class
@frozen
class StaticMod(AttrsPartiallyStatic):
    mod: Mod

    def __call__(self, value: Any, cache: LogCache, idx: Optional[jnp.ndarray]) -> Any:
        return self.mod(value, cache, idx)


@frozen
class ModInfo:
    key_idxs: KeyIdxs
    mod: Mod
    to_log: List[KeyIdxs] = Factory(list)


@register_pytree_node_class
@define
class SubLogModifyCache(LogCache):
    used_mods: Set[str] = Factory(set)
    used_mods_idxed: Set[str] = Factory(set)

    def use_mod(self, key: str, idx: Optional[jnp.ndarray]):
        if idx is None:
            return evolve(self, used_mods=self.used_mods.union([key]))
        else:
            return evolve(self, used_mods_idxed=self.used_mods_idxed.union([key]))

    def static_names(self) -> Set[str]:
        return super().static_names().union({"used_mods", "used_mods_idxed"})


@register_pytree_node_class
@define
class LoggerCacheAndModify(LoggerCache):
    """A class that provides a convenient way to modify values and confirm they were modified.

    Overriding the modify method yourself creates the following nuisances:
    - you have to get would_log_or_modify correct too, and it can be non-obvious when you haven't,
    - if the computation doesn't use your modification, it can be hard to notice.

    This class provides a correct would_log_or_modify, and a finish method that checks all the
    modifications were used before it was called.
    """

    mods: defaultdict[str, List[Mod]] = Factory(lambda: defaultdict(list))
    mods_idxed: defaultdict[str, List[ModIdxs]] = Factory(lambda: defaultdict(list))
    check_all_mods_used: bool = True

    def add_mod(self, mod_info: ModInfo):
        for name_idx in mod_info.to_log:
            self.add(name_idx)

        if mod_info.key_idxs.idxs.is_idxed:
            self.mods_idxed[mod_info.key_idxs.key].append(
                ModIdxs(mod_info.mod, op.map(mod_info.key_idxs.idxs.idxs, jnp.array))
            )
        else:
            self.mods[mod_info.key_idxs.key].append(mod_info.mod)

    def would_log_or_modify(self, key: str, idx: Optional[jnp.ndarray] = None) -> Union[jnp.ndarray, bool]:
        val = False
        if idx is None:
            val = key in self.mods and len(self.mods[key]) > 0
        else:
            if key in self.mods_idxed:
                val = any(mi.idxs is None for mi in self.mods_idxed[key]) or jnp.any(
                    jnp.array([(mi.idxs == idx).any() for mi in self.mods_idxed[key]])
                )

        return maybe_static_or(super().would_log_or_modify(key, idx), val)

    def init_cache(self) -> SubLogModifyCache:
        return SubLogModifyCache(**asdict(super().init_cache(), recurse=False))

    def init_idxed_for_shapes(self, cache: LogCache, shapes: Dict[str, Any], count: int) -> SubLogModifyCache:
        assert isinstance(cache, SubLogModifyCache)
        cache_out = super().init_idxed_for_shapes(cache, shapes, count)
        assert isinstance(cache_out, SubLogModifyCache)
        return evolve(
            cache_out,
            used_mods_idxed=cache_out.used_mods_idxed.union({k for k in shapes.keys() if k in self.mods_idxed}),
        )

    def modify(
        self, value: Any, key: str, cache: LogCache, idx: Optional[jnp.ndarray]
    ) -> Tuple[Any, SubLogModifyCache]:
        assert isinstance(cache, SubLogModifyCache)
        cache_v = cache
        mods_d: Union[defaultdict[str, List[Mod]], defaultdict[str, List[ModIdxs]]] = (
            self.mods if idx is None else self.mods_idxed
        )

        if key in mods_d:
            for mod in mods_d[key]:
                value = mod(value, cache_v, idx)
            cache_v = cache_v.use_mod(key, idx)

        return value, cache_v

    def check_finish_cache(self, cache: LogCache) -> None:
        assert isinstance(cache, SubLogModifyCache)
        super().check_finish_cache(cache)
        if self.check_all_mods_used:
            unused = set(self.mods.keys()) - cache.used_mods
            unused_idxed = set(self.mods_idxed.keys()) - cache.used_mods_idxed
            if unused or unused_idxed:
                raise UnusedKeyError(
                    "LoggerCacheAndModify: check_all_mods_used is enabled and there are unused mods.\n"
                    f"non-indexed: {unused}, indexed: {unused_idxed}\n"
                    f"used non-indexed {cache.used_mods}, used indexed: {cache.used_mods_idxed}\n"
                    f"expected non-indexed {self.mods.keys()}, expected indexed: {self.mods_idxed.keys()}\n"
                )

    def non_static_names(self) -> Set[str]:
        return super().non_static_names().union({"mods", "mods_idxed"})


def construct_mut_log_cache(
    log_info: Optional[LogInfo] = None,
    log_cache: Optional[Any] = None,
) -> Optional[MutLogCache]:
    return op.map(log_info, lambda log_info: MutLogCache.new_info(log_info, log_cache))
