from __future__ import annotations

import jax.numpy as jnp
import pytest

from interp.tools.log import KeyIdxs, LoggerCacheAndModify, ModInfo, MutLogCache, UnusedKeyError


def use_log(log: MutLogCache, actually_do_not=False):
    value = jnp.array([1.0])
    if actually_do_not or not log.log_info.would_log_or_modify("result"):
        result = value
    else:
        result = log.log_and_modify(value, "result")

    log.check_finish()

    return result


def test_no_modifications():
    log = MutLogCache.new(logger=LoggerCacheAndModify())
    result = use_log(log)
    assert jnp.allclose(result, jnp.array([1.0]))


def test_modification():
    logger = LoggerCacheAndModify()
    logger.add_mod(ModInfo(KeyIdxs("result"), lambda value, *_: value * 10))
    log = MutLogCache.new(logger=logger)
    result = use_log(log)
    assert jnp.allclose(result, jnp.array([10.0]))


def test_log():
    logger = LoggerCacheAndModify(to_cache={"result"})
    log = MutLogCache.new(logger=logger)
    use_log(log)
    assert jnp.allclose(log.cache.cache["result"], jnp.array([1.0]))


def test_log_and_modify():
    logger = LoggerCacheAndModify(to_cache={"result"})
    logger.add_mod(ModInfo(KeyIdxs("result"), lambda value, *_: value * 10))
    log = MutLogCache.new(logger=logger)
    result = use_log(log)
    assert jnp.allclose(result, jnp.array([10.0]))
    assert jnp.allclose(log.cache.cache["result"], jnp.array([10.0]))


def test_unused_modify():
    logger = LoggerCacheAndModify()
    logger.add_mod(ModInfo(KeyIdxs("result"), lambda value, *_: value * 10))
    log = MutLogCache.new(logger=logger)
    with pytest.raises(UnusedKeyError):
        use_log(log, actually_do_not=True)


def test_bad_key():
    logger = LoggerCacheAndModify()
    logger.add_mod(ModInfo(KeyIdxs("bad_key"), lambda value, *_: value * 10))
    log = MutLogCache.new(logger=logger)
    with pytest.raises(UnusedKeyError):
        use_log(log)


def test_set_of_used_changes_is_per_instance():
    logger = LoggerCacheAndModify()
    logger.add_mod(ModInfo(KeyIdxs("result"), lambda value, *_: value * 10))
    log = MutLogCache.new(logger=logger)
    use_log(log)
    logger = LoggerCacheAndModify()
    logger.add_mod(ModInfo(KeyIdxs("result"), lambda value, *_: value * 10))
    log = MutLogCache.new(logger=logger)
    with pytest.raises(UnusedKeyError):
        use_log(log, actually_do_not=True)
