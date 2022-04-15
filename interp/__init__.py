import os

"""
Normally, creating a tokenizer for the first time per Python process causes 7 separate HTTP requests to HuggingFace's servers
to check for updates to the tokenizer's data, even if the data is already cached locally.

Now we disable this by default, unless the user explicitly requested being online by specifying the environment variable TRANSFORMERS_OFFLINE=0. 

This saves approximately 2 seconds per process startup.
"""
if "TRANSFORMERS_OFFLINE" not in os.environ:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
