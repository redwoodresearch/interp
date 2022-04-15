from interp.tools.grad_modify_query_items import (
    NoneConf,
    ReplaceFuncConf,
    StopGradConf,
    MulConf,
    ReplaceFuncConf,
    ItemConf,
)
from interp.tools.log import KeyIdxs


def test_conf_equality_is_id():
    def run(get):
        x = get()
        assert x != get()
        assert x == x
        assert x is x

    run(NoneConf)
    run(lambda: ReplaceFuncConf(ItemConf(KeyIdxs("")), from_key_idxs=KeyIdxs(""), replacement=lambda x: x))
    run(lambda: StopGradConf(ItemConf(KeyIdxs(""))))
    run(lambda: MulConf(ItemConf(KeyIdxs(""))))
