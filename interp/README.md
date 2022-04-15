## Setup
- To use models and data, you'll need access to those files. You can obtain these by training new models, downloading from RRFS if you have access, or downloading from S3. To download from S3 you need the S3 cli, but not an AWS account:
  ```
  # See what is available
  > aws s3 ls s3://rrserve/interp-assets/ --no-sign-request --recursive
  2022-04-14 17:23:14          0 interp-assets/
  2022-04-15 10:48:04          0 interp-assets/models/
  2022-04-15 10:48:31          0 interp-assets/models/attention_only_two_layers/
  2022-04-15 10:48:30   54087052 interp-assets/models/attention_only_two_layers/model.bin
  2022-04-15 10:48:31        670 interp-assets/models/attention_only_two_layers/model_info.json
  2022-04-15 10:48:32          0 interp-assets/models/attention_only_two_layers_2/
  2022-04-15 10:48:33   54091250 interp-assets/models/attention_only_two_layers_2/model.bin
  2022-04-15 10:48:32        668 interp-assets/models/attention_only_two_layers_2/model_info.json
  2022-04-15 10:48:31          0 interp-assets/models/attention_only_two_layers_untrained/
  2022-04-15 10:48:32   54087025 interp-assets/models/attention_only_two_layers_untrained/model.bin
  2022-04-15 10:48:31        607 interp-assets/models/attention_only_two_layers_untrained/model_info.json
  2022-04-15 10:48:30          0 interp-assets/models/gelu_twelve_layers/
  2022-04-15 10:48:30  497764481 interp-assets/models/gelu_twelve_layers/model.bin
  2022-04-15 10:48:30        597 interp-assets/models/gelu_twelve_layers/model_info.json
  2022-04-15 10:48:29          0 interp-assets/models/gelu_twenty_four_layers/
  2022-04-15 10:48:29 1419302987 interp-assets/models/gelu_twenty_four_layers/model.bin
  2022-04-15 10:48:29        719 interp-assets/models/gelu_twenty_four_layers/model_info.json
  2022-04-15 10:48:27          0 interp-assets/models/gelu_two_layers/
  2022-04-15 10:48:27   58310669 interp-assets/models/gelu_two_layers/model.bin
  2022-04-15 10:48:27        711 interp-assets/models/gelu_two_layers/model_info.json
  2022-04-15 10:48:28          0 interp-assets/models/gelu_two_layers_2/
  2022-04-15 10:48:28   59875241 interp-assets/models/gelu_two_layers_2/model.bin
  2022-04-15 10:48:28        612 interp-assets/models/gelu_two_layers_2/model_info.json

  # Download all available models locally
  > mkdir ~/rr-models/
  > aws s3 cp s3://rrserve/interp-assets/models/ --no-sign-request ~/rr-models/ --recursive
  ```
  The code will try to find assets in the following locations in order:
    - environment variables `INTERPRETABILITY_MODELS_DIR` for models and `INTERP_DATA_DIR` for data. If you followed the instructions above, you can set `INTERPRETABILITY_MODELS_DIR=~/rr-models`.
    - local folder `~/datasets` for data, whatever local folder you pass to `load_model` for models (e.g. `~/rr-models`)
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
