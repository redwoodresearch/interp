from functools import partial
from typing import Set, Union
from operator import mul

from attrs import evolve, frozen
import jax.numpy as jnp
import jax
import pytest
from interp.model.blocks import scan_run_on_log

from interp.tools.log import (
    Idxs,
    Logger,
    LoggerCache,
    LoggerCacheAndModify,
    ModInfo,
    MutLogCache,
    KeyIdxs as KI,
    ShapeLogger,
    LogCache,
)
from interp.tools.indexer import I
from interp.tools.grad_modify_query import (
    run_queries,
    Query,
    ModifierCollectionTreeNode as MCTN,
    TargetConf,
)
from interp.tools.grad_modify_query_items import (
    ItemIdx,
    StopGradConf as SGC,
    MulConf as MC,
    ItemConf as IC,
    NoneConf,
)
from interp.tools.grad_modify_query_utils import MulBuilder, as_op, compose_trees as ct
import interp.tools.optional as op


def scalar_func(log: MutLogCache, finish=True):
    x_0 = (
        log.log_and_modify(jnp.array(10.0), "x_0_10")
        + log.log_and_modify(jnp.array(7.0), "x_0_7")
        + log.log_and_modify(jnp.array(5.0), "x_0_5")
    )
    x_0 = log.log_and_modify(x_0, "x_0")
    x_1 = (
        4.0 * log.log_and_modify(x_0, "x_0_x_1_4")
        + 5.0 * log.log_and_modify(x_0, "x_0_x_1_5")
        + log.log_and_modify(jnp.array(121.0), "x_1_121")
    )
    x_1 = log.log_and_modify(x_1, "x_1")
    x_2 = (
        23.0 * log.log_and_modify(x_0, "x_0_x_2_23")
        + 41.0 * log.log_and_modify(x_0, "x_0_x_2_41")
        + 13.0 * log.log_and_modify(x_1, "x_1_x_2_13")
        + 3.0 * log.log_and_modify(x_1, "x_1_x_2_3")
    )
    x_2 = log.log_and_modify(x_2, "x_2")

    x_3 = log.log_and_modify(x_1, "x_1_x_3_mul_x_2") * log.log_and_modify(
        x_2, "x_2_x_3_mul_x_1"
    ) + 4 * log.log_and_modify(x_0, "x_0_x_3_4")
    x_3 = log.log_and_modify(x_3, "x_3")

    if finish:
        log.check_finish()

    return log.cache


def scalar_func_for_logger(logger: Logger):
    return scalar_func(MutLogCache.new(logger))


def scalar_func_for_logger_indexed(logger: Logger, n=3):
    def run_on_log(log: MutLogCache, *_):
        scalar_func(log, finish=False)
        return None, None

    log = MutLogCache.new(logger)
    scan_run_on_log(run_on_log, None, lambda _: None, n=n, log=log)

    return log.cache


def simple_indexed(logger: Logger, n=10):
    def run_on_log(log: MutLogCache, x, _):
        return log.log_and_modify(x + log.log_and_modify(jnp.array(10.0), "add"), "x"), None

    log = MutLogCache.new(logger)
    scan_run_on_log(run_on_log, 0, lambda _: None, n=n, log=log)

    return log.cache


@pytest.mark.parametrize("idx", [None, 0, 2])
def test_simple_func(idx):
    ki = op.unwrap_or(op.map(idx, lambda idx: partial(KI.single, idx=idx)), KI)
    run_this = scalar_func_for_logger if idx is None else scalar_func_for_logger_indexed

    @jax.jit
    def run():
        queries = {
            "x_0_via_10": Query(targets=[TargetConf(ki("x_0"))], modifier_collection_tree=MCTN(MC(IC(ki("x_0_10"))))),
            "x_0_via_10_7": Query(
                targets=[TargetConf(ki("x_0"))],
                modifier_collection_tree=MCTN([MC(IC(ki("x_0_10"))), MC(IC(ki("x_0_7")))]),
            ),
            "x_1_via_10_to_5": Query(
                targets=[TargetConf(ki("x_1")), TargetConf(ki("x_0"))],
                modifier_collection_tree=ct(MCTN(MC(IC(ki("x_0_10")))), MCTN(MC(IC(ki("x_0_x_1_5"))))),
            ),
            "many_nones": Query(
                targets=[TargetConf(ki("x_1")), TargetConf(ki("x_0"))],
                modifier_collection_tree=ct(
                    MCTN(NoneConf()),
                    MCTN(MC(IC(ki("x_0_10")))),
                    MCTN(NoneConf()),
                    MCTN(MC(IC(ki("x_0_x_1_5")))),
                    MCTN(NoneConf()),
                    MCTN(NoneConf()),
                    MCTN(NoneConf()),
                ),
            ),
            "x_2_via_7_to_4_to_13": Query(
                targets=[TargetConf(ki("x_2"))],
                modifier_collection_tree=ct(
                    MCTN(MC(IC(ki("x_0_7")))),
                    MCTN(MC(IC(ki("x_0_x_1_4")))),
                    MCTN(MC(IC(ki("x_1_x_2_13")))),
                ),
            ),
            "x_2_via_7_10_to_4_5_to_13": Query(
                targets=[TargetConf(ki("x_2"))],
                modifier_collection_tree=ct(
                    MCTN([MC(IC(ki("x_0_7"))), MC(IC(ki("x_0_10")))]),
                    MCTN([MC(IC(ki("x_0_x_1_4"))), MC(IC(ki("x_0_x_1_5")))]),
                    MCTN(MC(IC(ki("x_1_x_2_13")))),
                ),
            ),
            "x_2_via_7_10_to_4_5_to_13_stack_4_5": Query(
                targets=[TargetConf(ki("x_2"))],
                modifier_collection_tree=ct(
                    MCTN([MC(IC(ki("x_0_7"))), MC(IC(ki("x_0_10")))]),
                    MCTN([[MC(IC(ki("x_0_x_1_4")))], [MC(IC(ki("x_0_x_1_5")))]]),
                    MCTN(MC(IC(ki("x_1_x_2_13")))),
                ),
            ),
            "x_2_via_7_10_5_to_4_5_to_13_stack_4_5_stack_7_10_5": Query(
                targets=[TargetConf(ki("x_2"))],
                modifier_collection_tree=ct(
                    MCTN([[MC(IC(ki("x_0_7")))], [MC(IC(ki("x_0_10")))], [MC(IC(ki("x_0_5")))], [NoneConf()]]),
                    MCTN([[MC(IC(ki("x_0_x_1_4")))], [MC(IC(ki("x_0_x_1_5")))], [NoneConf()]]),
                    MCTN(MC(IC(ki("x_1_x_2_13")))),
                ),
            ),
            "branching": Query(
                targets=[TargetConf(ki("x_2"))],
                modifier_collection_tree=as_op(
                    [
                        MCTN(
                            [MC(IC(ki("x_0_7"))), MC(IC(ki("x_0_10")))],
                            next_item=as_op(
                                [
                                    ct(MCTN(MC(IC(ki("x_0_x_1_4")))), MCTN(MC(IC(ki("x_1_x_2_3"))))),
                                    ct(MCTN(MC(IC(ki("x_0_x_1_5")))), MCTN(MC(IC(ki("x_1_x_2_13"))))),
                                    MCTN(MC(IC(ki("x_0_x_2_23")))),
                                ]
                            ),
                        ),
                        MCTN(
                            MC(IC(ki("x_0_5"))),
                            next_item=as_op(
                                [
                                    MCTN(MC(IC(ki("x_0_x_2_41")))),
                                    ct(
                                        MCTN(MC(IC(ki("x_0_x_1_5")))),
                                        MCTN(MC(IC(ki("x_1_x_2_13")))),
                                        MCTN(NoneConf()),
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            "branching_both_concat": Query(
                targets=[TargetConf(ki("x_2"))],
                modifier_collection_tree=as_op(
                    [
                        [
                            MCTN(
                                [[MC(IC(ki("x_0_7")))], [MC(IC(ki("x_0_10")))]],
                                next_item=as_op(
                                    [
                                        ct(MCTN(MC(IC(ki("x_0_x_1_4")))), MCTN(MC(IC(ki("x_1_x_2_3"))))),
                                        ct(MCTN(MC(IC(ki("x_0_x_1_5")))), MCTN(MC(IC(ki("x_1_x_2_13"))))),
                                        MCTN(MC(IC(ki("x_0_x_2_23")))),
                                    ]
                                ),
                            )
                        ],
                        [
                            MCTN(
                                [[MC(IC(ki("x_0_5")))], [MC(IC(ki("x_0_10")))]],
                                next_item=as_op(
                                    [
                                        MCTN(MC(IC(ki("x_0_x_2_41")))),
                                        ct(MCTN(MC(IC(ki("x_0_x_1_5")))), MCTN(MC(IC(ki("x_1_x_2_13"))))),
                                    ]
                                ),
                            )
                        ],
                        [
                            MCTN(
                                [[MC(IC(ki("x_0_5")))], [MC(IC(ki("x_0_10")))]],
                                next_item=as_op(
                                    [
                                        ct(MCTN(MC(IC(ki("x_0_x_1_4")))), MCTN(MC(IC(ki("x_1_x_2_3"))))),
                                        MCTN(MC(IC(ki("x_0_x_2_23")))),
                                    ]
                                ),
                            )
                        ],
                    ]
                ),
            ),
            "x_2_via_x_0_10_stop_x_1": Query(
                targets=[TargetConf(ki("x_2"))],
                modifier_collection_tree=ct(MCTN(MC(IC(ki("x_0_10")))), MCTN(SGC(IC(ki("x_1"))))),
            ),
            "mul_no_stop": Query(targets=[TargetConf(ki("x_3"))], modifier_collection_tree=MCTN(MC(IC(ki("x_0_5"))))),
            "mul_no_stop_term_x_1": Query(
                targets=[TargetConf(ki("x_3"))],
                modifier_collection_tree=ct(MCTN(MC(IC(ki("x_0_5")))), MCTN(MC(IC(ki("x_1_x_3_mul_x_2"))))),
            ),
            "mul_no_stop_term_x_2": Query(
                targets=[TargetConf(ki("x_3"))],
                modifier_collection_tree=ct(MCTN(MC(IC(ki("x_0_5")))), MCTN(MC(IC(ki("x_2_x_3_mul_x_1"))))),
            ),
        }

        return run_queries(
            run_this,
            {n: evolve(q, allow_non_first_deriv=True) for n, q in queries.items()},
        )

    results = run()

    assert results["x_0_via_10"]["x_0"].shape == ()

    assert jnp.allclose(results["x_0_via_10"]["x_0"], 10.0)
    assert jnp.allclose(results["x_0_via_10_7"]["x_0"], 10 + 7)
    assert jnp.allclose(results["x_1_via_10_to_5"]["x_0"], 0.0)  # earlier nodes are zero
    assert jnp.allclose(results["x_1_via_10_to_5"]["x_1"], 5 * 10)
    assert jnp.allclose(results["many_nones"]["x_0"], 0)
    assert jnp.allclose(results["many_nones"]["x_1"], 5 * 10)
    assert jnp.allclose(results["x_2_via_7_to_4_to_13"]["x_2"], 7 * 4 * 13)
    assert jnp.allclose(results["x_2_via_7_10_to_4_5_to_13"]["x_2"], (7 + 10) * (4 + 5) * 13)
    assert jnp.allclose(
        results["x_2_via_7_10_to_4_5_to_13_stack_4_5"]["x_2"], jnp.array([(7 + 10) * 4 * 13, (7 + 10) * 5 * 13])
    )
    assert jnp.allclose(
        results["x_2_via_7_10_5_to_4_5_to_13_stack_4_5_stack_7_10_5"]["x_2"],
        jnp.array(
            [
                [7 * 4 * 13, 10 * 4 * 13, 5 * 4 * 13, (7 + 10 + 5) * 4 * 13],
                [7 * 5 * 13, 10 * 5 * 13, 5 * 5 * 13, (7 + 10 + 5) * 5 * 13],
                [7 * (4 + 5) * 13, 10 * (4 + 5) * 13, 5 * (4 + 5) * 13, ((7 + 10 + 5) * (4 + 5) + 121) * 13],
            ]
        ),
    )
    assert jnp.allclose(results["branching"]["x_2"], (7 + 10) * (4 * 3 + 5 * 13 + 23) + 5 * (41 + 5 * 13))
    assert jnp.allclose(
        results["branching_both_concat"]["x_2"],
        jnp.array(
            [
                [7 * (4 * 3 + 5 * 13 + 23), 5 * (41 + 5 * 13), 5 * (4 * 3 + 23)],
                [10 * (4 * 3 + 5 * 13 + 23), 10 * (41 + 5 * 13), 10 * (4 * 3 + 23)],
            ]
        ),
    )

    assert jnp.allclose(results["x_2_via_x_0_10_stop_x_1"]["x_2"], 10 * (23 + 41))

    val_cache: LogCache = scalar_func_for_logger(LoggerCache(to_cache={"x_0", "x_1", "x_2"}))

    def add_ablates(logger: LoggerCacheAndModify, to_ablate: Set[str]):
        for ablate in to_ablate:
            logger.add_mod(ModInfo(KI(ablate), lambda x, *_: jnp.zeros_like(x)))

    # ablate other than alpha
    x_0_5_val_log = LoggerCacheAndModify(to_cache={"x_0", "x_1", "x_2"})
    add_ablates(x_0_5_val_log, {"x_0_10", "x_0_7", "x_1_121"})
    x_0_5_val_cache: LogCache = scalar_func_for_logger(x_0_5_val_log)

    x_0_7_val_log = LoggerCacheAndModify(to_cache={"x_0", "x_1", "x_2"})
    add_ablates(x_0_7_val_log, to_ablate={"x_0_10", "x_0_5", "x_1_121"})
    x_0_7_val_cache: LogCache = scalar_func_for_logger(x_0_7_val_log)

    quad_term = val_cache.get(KI("x_2")) * x_0_5_val_cache.get(KI("x_1")) + val_cache.get(
        KI("x_1")
    ) * x_0_5_val_cache.get(KI("x_2"))
    paths_through_term = quad_term - x_0_5_val_cache.get(KI("x_1")) * x_0_5_val_cache.get(KI("x_2"))

    assert jnp.allclose(results["mul_no_stop"]["x_3"], quad_term + 4 * 5)
    assert jnp.allclose(results["mul_no_stop_term_x_1"]["x_3"], quad_term)
    assert jnp.allclose(results["mul_no_stop_term_x_2"]["x_3"], quad_term)

    @jax.jit
    def run_muls():
        queries = {
            f"mul_stop_x_1": Query(
                targets=[TargetConf(ki("x_3"))],
                modifier_collection_tree=ct(MCTN(MC(IC(ki("x_0_5")))), MCTN(SGC(IC(ki("x_1_x_3_mul_x_2"))))),
            ),
            f"mul_stop_x_2": Query(
                targets=[TargetConf(ki("x_3"))],
                modifier_collection_tree=ct(MCTN(MC(IC(ki("x_0_5")))), MCTN(SGC(IC(ki("x_2_x_3_mul_x_1"))))),
            ),
            f"mul_stop_x_1_selective": Query(
                targets=[TargetConf(ki("x_3"))],
                modifier_collection_tree=ct(
                    MCTN(MC(IC(ki("x_0_5")))),
                    MCTN(MC(IC(ki("x_0_x_1_4")))),
                    MCTN(MC(IC(ki("x_1_x_2_13")))),
                    MCTN(SGC(IC(ki("x_1_x_3_mul_x_2")))),
                ),
            ),
            f"mul_actual_pair": Query(
                targets=[TargetConf(ki("x_3"))],
                modifier_collection_tree=MulBuilder(
                    run_this,
                    MCTN(MC(IC(ki("x_0_x_2_23")))),
                    MCTN(MC(IC(ki("x_0_x_1_4")))),
                    ki("x_2_x_3_mul_x_1"),
                    ki("x_1_x_3_mul_x_2"),
                    mul,
                    ki("x_3"),
                ).conjunctive(),
            ),
            "mul_base_conj": Query(
                targets=[TargetConf(ki("x_3"))],
                modifier_collection_tree=MulBuilder(
                    run_this,
                    MCTN(MC(IC(ki("x_0_5")))),
                    MCTN(MC(IC(ki("x_0_5")))),
                    ki("x_2_x_3_mul_x_1"),
                    ki("x_1_x_3_mul_x_2"),
                    mul,
                    ki("x_3"),
                ).conjunctive(),
            ),
            "mul_non_conj": Query(
                targets=[TargetConf(ki("x_3"))],
                modifier_collection_tree=ct(
                    as_op(
                        MulBuilder(
                            run_this,
                            MCTN(MC(IC(ki("x_0_5")))),
                            MCTN(MC(IC(ki("x_0_5")))),
                            ki("x_2_x_3_mul_x_1"),
                            ki("x_1_x_3_mul_x_2"),
                            mul,
                            ki("x_3"),
                        ).remove_conjunctive_from_sum()
                    ),
                    MCTN(SGC(IC(ki("x_0_x_3_4")))),
                ),
            ),
            "mul_non_conj_diff": Query(
                targets=[TargetConf(ki("x_3"))],
                modifier_collection_tree=ct(
                    as_op(
                        MulBuilder(
                            run_this,
                            MCTN(MC(IC(ki("x_0_5")))),
                            MCTN(MC(IC(ki("x_0_7")))),
                            ki("x_1_x_3_mul_x_2"),
                            ki("x_2_x_3_mul_x_1"),
                            mul,
                            ki("x_3"),
                        ).remove_conjunctive_from_sum()
                    ),
                    MCTN(SGC(IC(ki("x_0_x_3_4")))),
                ),
            ),
        }
        return run_queries(run_this, {n: evolve(q, allow_non_first_deriv=True) for n, q in queries.items()})

    mul_results = run_muls()

    extra_x_0_mul = 4

    assert jnp.allclose(
        mul_results["mul_stop_x_1"]["x_3"],
        val_cache.get(KI("x_1")) * x_0_5_val_cache.get(KI("x_2")) + x_0_5_val_cache.get(KI("x_0")) * extra_x_0_mul,
    )
    assert jnp.allclose(
        mul_results["mul_stop_x_2"]["x_3"],
        val_cache.get(KI("x_2")) * x_0_5_val_cache.get(KI("x_1")) + x_0_5_val_cache.get(KI("x_0")) * extra_x_0_mul,
    )
    assert jnp.allclose(mul_results["mul_stop_x_1_selective"]["x_3"], val_cache.get(KI("x_1")) * 5 * 4 * 13)
    assert jnp.allclose(
        mul_results["mul_actual_pair"]["x_3"], 23 * val_cache.get(KI("x_0")) * 4.0 * val_cache.get(KI("x_0"))
    )
    assert jnp.allclose(
        mul_results["mul_base_conj"]["x_3"], x_0_5_val_cache.get(KI("x_1")) * x_0_5_val_cache.get(KI("x_2"))
    )
    assert jnp.allclose(mul_results["mul_non_conj"]["x_3"], paths_through_term)
    quad_term_diff = val_cache.get(KI("x_2")) * x_0_5_val_cache.get(KI("x_1")) + val_cache.get(
        KI("x_1")
    ) * x_0_7_val_cache.get(KI("x_2"))
    paths_through_term_diff = quad_term_diff - x_0_5_val_cache.get(KI("x_1")) * x_0_7_val_cache.get(KI("x_2"))
    assert jnp.allclose(mul_results["mul_non_conj_diff"]["x_3"], paths_through_term_diff)


def test_simple_indexed():
    targets = [TargetConf(KI("x", Idxs.all()))]

    locs = [0, 2, 7]

    @jax.jit
    def run_alpha_at_different_points():
        queries = {
            str(i): Query(targets=targets, modifier_collection_tree=MCTN(MC(IC(KI.single("add", i))))) for i in locs
        }

        return run_queries(simple_indexed, queries)

    cache = run_alpha_at_different_points()

    for i in locs:
        assert (cache[str(i)]["x"][:i] == 0.0).all()
        assert (cache[str(i)]["x"][i:] == 10.0).all()


def test_stop_grad_idxing():
    def run_for_logger(logger: Logger):
        log = MutLogCache.new(logger)
        log.log_and_modify(jnp.ones((4, 5, 7)), "start")
        return log.cache

    targets = [TargetConf(KI("start"))]

    mask = jax.random.bernoulli(jax.random.PRNGKey(238), shape=(4, 5, 7))

    queries = {
        f"stop_grad_{name}": Query(targets, ct(MCTN(MC(IC(KI("start")))), MCTN(SGC(IC(KI("start"), item_idx)))))
        for name, item_idx in [
            ("all", ItemIdx()),
            ("single", ItemIdx(I[:, 3])),
            ("except", ItemIdx(I[:, 3], except_idx=True)),
            ("mask", ItemIdx(mask, is_mask=True)),
        ]
    }
    out = run_queries(run_for_logger, queries)

    assert out["stop_grad_all"]["start"].shape == (4, 5, 7)
    assert (out["stop_grad_all"]["start"] == 0).all()

    assert (out["stop_grad_single"]["start"][:, 3] == 0).all()
    assert (out["stop_grad_single"]["start"][:, :3] == 1).all()
    assert (out["stop_grad_single"]["start"][:, 4:] == 1).all()

    assert (out["stop_grad_except"]["start"][:, 3] == 1).all()
    assert (out["stop_grad_except"]["start"][:, :3] == 0).all()
    assert (out["stop_grad_except"]["start"][:, 4:] == 0).all()

    assert (out["stop_grad_mask"]["start"][mask] == 0).all()
    assert (out["stop_grad_mask"]["start"][~mask] == 1).all()
