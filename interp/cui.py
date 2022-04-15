from functools import partial
import time
from interp.ui.attribution_backend import AttributionBackend
from interp.ui.very_named_tensor import LazyVeryNamedTensor, VeryNamedTensor
from typing import Any
import asyncio
import websockets
import json
from interp.model.model_loading import load_model
from interp.encoding import get_callback_result, callables_table, msgpack_encode
import threading
from threading import Lock

from IPython.core.display import display, HTML
from typing import *

served_things: Dict[str, Any] = {}
thread = None
lock = Lock()
sockets = []


async def handler(websocket):
    global sockets
    print("handler")
    this_obj = {"websocket": websocket, "name": None}
    sockets.append(this_obj)
    async for buf in websocket:
        msg = json.loads(buf)
        kind = msg.get("kind")
        print(kind)
        if kind == "init":
            pass
        elif kind == "nameStartup":
            name = msg["name"]
            print("name", name)
            this_obj["name"] = name
            with lock:
                thing = served_things.get(name, None)
            await websocket.send(msgpack_encode({"kind": "nameStartup", "data": thing}))
        elif kind == "callback":
            enc = msgpack_encode(get_callback_result(msg))
            await websocket.send(enc)
        else:
            print("Unrecognized message: ", msg)


def loop_in_thread(loop, port):
    print("started")

    asyncio.set_event_loop(loop)
    loop.run_until_complete(websockets.serve(handler, "localhost", port))
    loop.run_forever()


async def init(port=6789):
    global thread
    port = str(port)
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop_in_thread, args=(loop, port))
    t.start()
    thread = t
    print(
        "Composable UI initialized! Make sure you've run `npm install` and `npm run start` in /interp/app before using"
    )


async def show_tensors(*lvnts, name="untitled"):
    lvnts = [lvnt.to_lvnt() if not isinstance(lvnt, LazyVeryNamedTensor) else lvnt for lvnt in lvnts]
    with lock:
        did_exist = name in served_things
        served_things[name] = lvnts
    if did_exist:
        encoded = msgpack_encode({"kind": "nameStartup", "data": lvnts})
        try:
            await [socket for socket in sockets if socket["name"] == name][0]["websocket"].send(encoded)
        except:
            pass
    link = f"http://127.0.0.1:3000/#/tensors/{name}"
    return display(
        HTML(f'<h1><a href="{link}" target="_blank">Link</a><script>window.open({link},"_blank")</script></h1>')
    )


async def show_fns(*lvnt_makers, name="untitled"):
    with lock:
        did_exist = name in served_things
        served_things[name] = lvnt_makers
    link = f"http://127.0.0.1:3000/#/functions/{name}"
    if did_exist:
        await [socket for socket in sockets if socket["name"] == name][0]["websocket"].send(
            msgpack_encode({"kind": "nameStartup", "data": lvnt_makers})
        )

    return display(
        HTML(f'<h1><a href="{link}" target="_blank">Link</a><script>window.open({link},"_blank")</script></h1>')
    )


async def show_attribution(model, params, tokenizer, name="untitled"):
    func = partial(AttributionBackend, model, params, tokenizer)
    with lock:
        did_exist = name in served_things
        served_things[name] = func
    if did_exist:
        await [socket for socket in sockets if socket["name"] == name][0]["websocket"].send(
            msgpack_encode({"kind": "nameStartup", "data": func})
        )
    link = f"http://127.0.0.1:3000/#/attribution/{name}"
    return display(
        HTML(f'<h1><a href="{link}" target="_blank">Link</a><script>window.open({link},"_blank")</script></h1>')
    )
