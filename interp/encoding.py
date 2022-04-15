import os
import numpy
import numpy as np
import jax.numpy as jnp
import torch
from typing import *
import msgpack

callables_table = {}


def _msgpack_special_encoder(obj):
    if isinstance(obj, numpy.ndarray):
        dtype = obj.dtype.name
        # Unfortunately there is no Int64Array type in Javascript.
        # So we convert int64 arrays to int32 arrays, and encode them over the wire that way.
        # However, we maintain the original type to allow for (somewhat) reversible decoding.
        # This is pretty absurdly confusing, so I might change this later.
        if obj.dtype == numpy.int64:
            obj = obj.astype(numpy.int32)
        if obj.dtype == numpy.uint64:
            obj = obj.astype(numpy.uint32)
        if obj.dtype == numpy.float16:
            obj = obj.astype(
                numpy.float32
            )  # very sad that we're sending float32, but JS doesn't have float16 and I can't be bothered to use some WASM for f16
        return {
            "$$type$$": "ndarray",
            "dtype": obj.dtype.name,
            "shape": obj.shape,
            "data": obj.tobytes(),
            "v": 0,  # I might change the over-the-wire encoding later, so I want a version tag.
        }
    if isinstance(obj, torch.Tensor):
        array_encoding = _msgpack_special_encoder(obj.detach().cpu().numpy())
        return {"$$type$$": "torch", "array": array_encoding, "device": str(obj.device)}
    if isinstance(obj, jnp.ndarray):
        array_encoding = _msgpack_special_encoder(np.array(obj))
        return {"$$type$$": "jax", "array": array_encoding, "device": str(obj.device)}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if callable(obj):
        function_token = os.urandom(12).hex()
        callables_table[function_token] = obj
        return {"$$type$$": "local-callback", "id": function_token}
    raise TypeError("cannot serialize %r object" % type(obj))


def _decode_walk_tree(x):
    if isinstance(x, dict):
        if "$$type$$" in x:
            if x["$$type$$"] == "ndarray":
                array = numpy.frombuffer(
                    x["data"],
                    dtype={
                        "int8": numpy.int8,
                        "uint8": numpy.uint8,
                        "int16": numpy.int16,
                        "uint16": numpy.uint16,
                        "int32": numpy.int32,
                        "uint32": numpy.uint32,
                        "int64": numpy.int32,  # Intentional mismatch! See comment above.
                        "uint64": numpy.uint32,  # Intentional mismatch! See comment above.
                        "float32": numpy.float32,
                        "float64": numpy.float64,
                    }[x["dtype"]],
                ).reshape(x["shape"])
                if x["dtype"] == "int64":
                    array = array.astype(numpy.int64)
                if x["dtype"] == "uint64":
                    array = array.astype(numpy.uint64)
                return array
            elif x["$$type$$"] == "torch":
                array = _decode_walk_tree(x["array"])
                return torch.tensor(array)
                # return torch.tensor(array, device=x["device"])
            else:
                raise ValueError("Bad serialized $$type$$: %r" % x["$$type$$"])
        return {k: _decode_walk_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode_walk_tree(v) for v in x]
    return x


def msgpack_encode(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_msgpack_special_encoder)


def msgpack_decode(raw_data: bytes) -> Any:
    return _decode_walk_tree(msgpack.unpackb(raw_data))


def get_callback_result(msg):
    if msg["id"] in callables_table:
        function = callables_table[msg["id"]]
        data = msg["data"]
        # print("called on", data)
        value = function(*data)
        response = dict(kind="callbackResult", token=msg["token"], buffer=value)
    else:
        response = dict(kind="callbackResult", token=msg["token"], buffer="unknown callback token")
    return response
