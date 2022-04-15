# Composable UI (CUI)

This package provides a web interface to explore multi-dimensional tensors defined either in a notebook or in a Python script.

## Websocket Server Installation
- To use models and data, you'll need access to those files. You can obtain these by training new models, downloading from RRFS if you have access, or downloading from [here](https://github.com/redwoodresearch/interp-assets). Then save the `model.bin` and `model_info.json` in a `model_name` subdirectory of one of the following locations. The code will try to find the model assets in the following locations in order:
  - environment variables `INTERPRETABILITY_MODELS_DIR` for models and `INTERP_DATA_DIR` for data
  - local folder `~/datasets` for data, whatever local folder you pass to `load_model` for models (e.g. `~/interp_models_jax`)
  - RRFS. This is generally slower so it's preferred to cache models locally.
- `pip install -r requirements.txt`
- If you see an error about attrs, do `pip uninstall attrs; pip install attrs`

## React Client Installation
- Install npm: one way is to do:
    `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash`
    `source ~/.zshrc`
- Go to the interp/app folder
- `npm install`

## Running in a Jupyter notebook (or VS Code notebook)

- See `cui_demo.ipynb` for instructions. 

## Running using tensor_makers

- You can write Python modules in the tensor_makers folder and then run `python local_dev.py` to launch a server that automatically detects these modules and reloads them when they change.
