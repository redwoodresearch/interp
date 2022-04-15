from functools import partial

from interp.model.model_loading import MODELS_DIR, load_model
from interp.tools.interpretability_tools import add_begin_token, sequence_tokenize, single_tokenize
from interp.ui.attribution_backend import AttributionBackend
from interp.ui.unembedder import unembed_to_topk
from interp.ui.very_named_tensor import LazyVeryNamedTensor, VeryNamedTensor
from typing import Any, Union
import asyncio
import websockets
import json
import argparse
from interp.encoding import get_callback_result, msgpack_encode
import time
import jax.numpy as jnp
from interp.tensor_makers import tensor_makers, attribution_backend_maker
import os

mpts = {}


AVAILABLE_MODELS_NAMES = [
    "attention_only_two_layers",
    "gelu_two_layers",
    "gelu_two_layers_2",
    "attention_only_two_layers_untrained",
    "attention_only_two_layers_2",
    "gelu_twelve_layers",
]
if not os.environ.get("REACT_APP_LOW_MEMORY"):
    AVAILABLE_MODELS_NAMES.append("gelu_twenty_four_layers")


PRELOADED_MODEL_NAMES = ["attention_only_two_layers", "gelu_two_layers"]


def deep_unpad_vnt(x: Any):
    if isinstance(x, VeryNamedTensor) or isinstance(x, LazyVeryNamedTensor):
        return unpad_vnt(x)
    elif isinstance(x, AttributionBackend):
        print("have attribution backend")
        x.token_strs = [t for t in x.token_strs if t != "[END]"]
        return x
    elif isinstance(x, dict):
        return {k: deep_unpad_vnt(v) for k, v in x.items()}
    else:
        return x


def unpad_vnt(vnt: Union[VeryNamedTensor, LazyVeryNamedTensor]):
    return vnt.getitem(
        tuple([slice(0, len([x for x in names if x != "[END]"])) for names in vnt.dim_idx_names]), no_title_change=True
    )


AVAILABLE_MODELS = [
    {"name": n, "info": json.load(open(f"{MODELS_DIR}/{n}/model_info.json"))} for n in AVAILABLE_MODELS_NAMES
]


PAD_LENGTHS = [32, 128, 256]


def add_begin_token_and_pad_from_model_name(x: str, model_name: str):
    info = [x["info"] for x in AVAILABLE_MODELS if x["name"] == model_name]
    assert len(info) == 1
    if info[0]["model_class"] == "GPTBeginEndToks":
        x = add_begin_token(x)
    ntoks = len(sequence_tokenize(x))
    next_pad_length = [x for x in PAD_LENGTHS if x >= ntoks][0]
    x += "[END]" * (next_pad_length - ntoks)
    return x


def get_model(string):
    if string not in mpts:
        mpts[string] = load_model(
            string, dtype=jnp.float16, models_dir=os.path.expanduser("~/interpretability_models_jax")
        )
        print("loaded", string)
    return mpts[string]


def get_lvnt_makers():

    result = []
    for maker_name, fn_wrapper in tensor_makers.items():

        def fn(func, str, model_name):
            return func(*get_model(model_name), add_begin_token_and_pad_from_model_name(str, model_name))

        result.append(
            {
                "name": maker_name,
                "fn": partial(fn, fn_wrapper["fn"]),
                "required_model_info_subtree": fn_wrapper["required_model_info_subtree"],
            }
        )
    return result


async def handler(websocket):
    async for buf in websocket:
        msg = json.loads(buf)
        kind = msg.get("kind")
        stime = time.time()
        if kind == "init":
            pass
        elif kind == "getTensorMakers":
            result = {
                "kind": "availableLVNTMakers",
                # "availableLVNTMakers": [],
                "availableLVNTMakers": get_lvnt_makers(),
                "attributionBackend": lambda str, nm: attribution_backend_maker["value"](
                    *get_model(nm), add_begin_token_and_pad_from_model_name(str, nm), model_name=nm
                ),
                "unembedder": lambda vector, nm: unembed_to_topk(*get_model(nm), vector),
                "availableModels": AVAILABLE_MODELS,
            }
            await websocket.send(msgpack_encode(result))
        elif kind == "callback":
            try:
                enc = msgpack_encode(deep_unpad_vnt(get_callback_result(msg)))
            except KeyError:
                enc = msgpack_encode("unknown callback key")
            await websocket.send(enc)
        else:
            print("Unrecognized message: ", msg)
        print(kind, "took", time.time() - stime)
    print("WEBSOCKET EXITED")


def is_subtree(super, sub):
    if isinstance(sub, dict):
        pass
    elif isinstance(sub, list):
        raise Exception("is_subtree doesn't support ")


def full_warmup():
    test_texts = ["[END]" * x for x in PAD_LENGTHS]
    lvnt_makers = get_lvnt_makers()
    local_model_path = os.path.expanduser("~/interpretability_models_jax")
    if not os.path.exists(local_model_path):
        os.mkdir(local_model_path)

    for available_mname in AVAILABLE_MODELS_NAMES:
        os.system(f"rsync -r {MODELS_DIR}/{available_mname}/ {local_model_path}/{available_mname}")
        for test_text in test_texts[:-1]:
            mtup = get_model(available_mname)
            for maker in lvnt_makers:
                if len(maker["required_model_info_subtree"]) == 0:  # this should actually check req model info
                    lvnt = maker["fn"](test_text, available_mname)
                    view_args = ["axis" for _ in lvnt.dim_idx_names]
                    lvnt.getView(view_args)
            backend = AttributionBackend(*mtup, test_text, available_mname)
            seq_idx = 3
            backend.startTree(
                {
                    "kind": "logprob",
                    "data": {"seqIdx": seq_idx, "tokString": " Hi", "comparisonTokString": None},
                    "direct": "False",
                },
                fuse_neurons=True,
            )
            backend.expandTreeNode(
                [{"layerWithIO": 2, "token": seq_idx, "isMlp": False, "headOrNeuron": 4}], False, fuse_neurons=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", dest="port", type=int, default=6789)
    parser.add_argument("--host", dest="host", type=str, default="localhost")
    parser.add_argument("--full-warmup", dest="warmup", type=bool, default=False)
    args = parser.parse_args()
    start_server = websockets.serve(handler, args.host, args.port)  # type: ignore
    asyncio.get_event_loop().run_until_complete(start_server)
    for name in PRELOADED_MODEL_NAMES:
        get_model(name)
    if args.warmup:
        print("performing full warmup")
        stime = time.time()
        full_warmup()
        print("full warmup took", time.time() - stime)
    asyncio.get_event_loop().run_forever()
