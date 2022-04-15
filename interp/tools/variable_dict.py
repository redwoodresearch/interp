from typing import Any, Callable

from flax.core.frozen_dict import FrozenDict
from flax.core.scope import FrozenVariableDict, VariableDict


# we'll do these operations immutably for now to support frozen dicts


def variable_dict_operate_on(
    d: VariableDict, op: Callable[[Any], Any], first_key: str, *rest_keys: str
) -> FrozenVariableDict:
    assert first_key in d
    if len(rest_keys) == 0:
        new_value = op(d[first_key])
    else:
        new_value = variable_dict_operate_on(d[first_key], op, *rest_keys)

    return FrozenDict({**d, first_key: new_value})


def variable_dict_replace(d: VariableDict, value, first_key: str, *rest_keys: str) -> FrozenVariableDict:
    return variable_dict_operate_on(d, lambda _: value, first_key, *rest_keys)


def variable_dict_replace_params(params: VariableDict, value, *keys: str) -> FrozenVariableDict:
    return variable_dict_replace(params, value["params"], "params", *keys)
