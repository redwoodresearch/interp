import jax
from jax.tree_util import tree_unflatten, tree_flatten
import jax.numpy as jnp

from interp.tools.log import Idxs, KeyIdxs, LoggerCacheAndModify, ModInfo


# also tests interp/tools/jax_tree_util.py
def test_log_flatten():
    logger = LoggerCacheAndModify(to_cache={"A", "b"}, idxs_as_numpy=True, check_all_acquired=False)
    logger.add(KeyIdxs.single("a", 3))
    logger.add(KeyIdxs("b", Idxs(jnp.arange(3))))
    logger.add_mod(ModInfo(KeyIdxs(""), lambda x, *_: x))
    logger.add_mod(
        ModInfo(
            KeyIdxs("d", Idxs(jnp.arange(2))),
            lambda x, *_: x,
            to_log=[KeyIdxs("slkdfj"), KeyIdxs("kdfj", Idxs(jnp.arange(2)))],
        )
    )
    logger.add_mod(
        ModInfo(
            KeyIdxs("e", Idxs(jnp.arange(3))),
            lambda x, *_: x,
            to_log=[KeyIdxs("slkdfj"), KeyIdxs("__kdfj", Idxs(jnp.arange(2)))],
        )
    )

    leaves, tree_def = tree_flatten(logger)
    new_logger: LoggerCacheAndModify = tree_unflatten(tree_def, leaves)

    _, new_tree_def = tree_flatten(new_logger)
    assert tree_def == new_tree_def

    assert new_logger.to_cache == logger.to_cache
    assert new_logger.idxs_as_numpy == logger.idxs_as_numpy
    assert new_logger.check_all_acquired == logger.check_all_acquired
    assert set(new_logger.to_cache_idxed.keys()) == set(logger.to_cache_idxed.keys())
    assert set(new_logger.mods.keys()) == set(logger.mods.keys())
    assert set(new_logger.mods_idxed.keys()) == set(logger.mods_idxed.keys())
    for k, ms_l in logger.mods.items():
        assert ms_l == new_logger.mods[k]
    for k, ms_l in logger.mods_idxed.items():
        for m_l, m_r in zip(ms_l, new_logger.mods_idxed[k]):
            m_l.mod == m_r.mod
            if m_l.idxs is None:
                assert m_r.idxs is None
            else:
                assert m_r.idxs is not None
                assert (m_l.idxs == m_r.idxs).all()
