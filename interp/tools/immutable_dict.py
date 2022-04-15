from copy import copy
import functools

# this whole file might be a mistake, but does make working with dictionaries
# in functional way better


class Composable:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __matmul__(self, other):
        return Composable(lambda *args, **kw: self.func(other.func(*args, **kw)))

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)

    @classmethod
    def mk(cls, f, *args):
        return cls(lambda x: f(x, *args))


def remove(x, keys):
    x = copy(x)
    for key in keys:
        del x[key]
    return x


def remove_f(keys):
    return Composable.mk(remove, keys)


def operate(x, old_key, new_key, func=lambda x: x, delete_old: bool = True):
    x = copy(x)
    x[new_key] = func(x[old_key])
    if new_key != old_key and delete_old:
        del x[old_key]
    return x


def operate_f(old_key, new_key, func=lambda x: x, delete_old: bool = True):
    return Composable.mk(operate, old_key, new_key, func, delete_old)


def assign(x, key, value, check_absent=False, check_present=False):
    if check_absent:
        assert key not in x
    if check_present:
        assert key in x
    return {**x, key: value}


def assign_f(key, value, check_absent=False, check_present=False):
    return Composable.mk(assign, key, value, check_absent, check_present)


def get_f(key):
    return Composable(lambda x: x[key])


def gets_f(keys):
    return Composable(lambda x: {key: x[key] for key in keys})


def keep_only(x, keys):
    return {k: x[k] for k in keys}


def keep_only_f(keys):
    return Composable.mk(keep_only, keys)
