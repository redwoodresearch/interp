from typing import NoReturn

# https://github.com/python/typing/issues/735
def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError(f"Invalid value: {x!r}")
