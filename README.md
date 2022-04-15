## Setup
- You need access to pretrained models (or train them yourself), and OpenWebText data. You can download our pretrained models (and OpenAI models ported into our codebase) fro s3. To download these from S3 you need the S3 cli, but not an AWS account:
  ```
  # Download all available models locally
  > mkdir ~/rr-models/
  > aws s3 cp s3://rrserve/interp-assets/models/ --no-sign-request ~/rr-models/ --recursive
  # List available models
  > aws s3 ls s3://rrserve/interp-assets/ --no-sign-request --recursive
  ```
  Our code looks for models in the path specified by the environment variable `INTERPRETABILITY_MODELS_DIR` for models and `INTERP_DATA_DIR` for data, so set `INTERPRETABILITY_MODELS_DIR` to `~/rr-models` or something.
- `pip install -r requirements.txt`
- If you see an error about attrs, do `pip uninstall attrs; pip install attrs`

## React Client Installation
- Install npm: one way is to do:
    `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash`
    `source ~/.zshrc`
- Go to the interp/app folder, and install our npm dependencies
- `npm install`

## Running our interpretability GUI

- Go to /interp
- `python local_dev.py`
- In a seperate terminal, go to interp/app
- `npm run start`
- Your browser should open to localhost:3000, and the GUI should start working after a few seconds.
- You can write Python modules in the tensor_makers folder and they will immediately become available in the website.

## Running in a Jupyter notebook (or VS Code notebook)

- See `cui_demo.ipynb` for instructions. 
