## Setup
- To use models and data, you'll need access to those files. You can obtain these by training new models, downloading from RRFS if you have access, or downloading from S3. To download from S3 you need the S3 cli, but not an AWS account:
  ```
  > aws s3 ls s3://rrserve/interp-assets/ --no-sign-request
                            PRE attention_only_two_layers/
                            PRE attention_only_two_layers_2/
                            PRE attention_only_two_layers_untrained/
                            PRE gelu_twelve_layers/
                            PRE gelu_twenty_four_layers/
                            PRE gelu_two_layers/
                            PRE gelu_two_layers_2/
  2022-04-14 17:23:14          0
  > aws s3 ls s3://rrserve/interp-assets/attention_only_two_layers/ --no-sign-request
  2022-04-14 17:27:01          0
  2022-04-14 17:27:01   54087052 model.bin
  2022-04-14 17:27:01        670 model_info.json
  > aws s3 cp s3://rrserve/interp-assets/attention_only_two_layers/model_info.json --no-sign-request /tmp
  download: s3://rrserve/interp-assets/attention_only_two_layers/model_info.json to ../../tmp/model_info.json
  ```
  Then save the `model.bin` and `model_info.json` in a `model_name` subdirectory of one of the following locations. The code will try to find the model assets in the following locations in order:
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
