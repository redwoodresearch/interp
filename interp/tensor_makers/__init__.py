from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import os
import importlib
import inspect
import interp.ui.attribution_backend as ab
from typing import *

tensor_makers: Dict[str, Any] = {}
attribution_backend_maker = {"value": None}

print("cwd", os.getcwd())


def load_tensor_makers():
    global tensor_makers
    tensor_makers.clear()
    for dir in os.listdir(os.path.dirname(__file__)):
        reload(dir)
    reload("attribution_backend.py")


def reload(dir):
    global tensor_makers
    if dir == "attribution_backend.py":
        importlib.reload(ab)
        attribution_backend_maker["value"] = ab.AttributionBackend
    elif dir[-3:] == ".py" and dir[:2] != "__":
        modname = dir[:-3]
        modstring = f"interp.tensor_makers.{modname}"
        try:
            if modname in tensor_makers:
                i = importlib.reload(inspect.getmodule(tensor_makers[modname["fn"]]))
                print("reloaded", modname)
            else:
                i = importlib.import_module(modstring)
            tensor_makers[modname] = {
                "fn": i.get_lvnt,
                "required_model_info_subtree": i.required_model_info_subtree
                if hasattr(i, "required_model_info_subtree")
                else {},
            }
            # for some reason this doesn't think newly created modules have get_lvnt, idk why
        except Exception as e:
            print("reloading failed", modname)
            print(e)


load_tensor_makers()


def handler_fn(event):
    global last_sync_time
    if not event.is_directory:
        reload(event.src_path.split("/")[-1])


def start_watching():
    patterns = ["*"]
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_directories=None, ignore_patterns=None)
    my_event_handler.on_modified = handler_fn
    my_event_handler.on_created = handler_fn
    my_observer = Observer()
    my_observer.schedule(my_event_handler, __file__, recursive=True)
    my_observer.schedule(my_event_handler, ab.__file__, recursive=True)
    my_observer.start()


start_watching()
