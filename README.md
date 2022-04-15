# Redwood Interpretability Tools

This is an open source mirror of Redwood Research's transformer interpretability tools. This is all the code behind our UI at [our demo website](http://interp-tools.redwoodresearch.org), as well as our interpretability-specific transformer implementation and gradient query framework.


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

### Running our interpretability GUI

- Go to /interp
- `python local_dev.py`
- In a seperate terminal, go to interp/app
- `npm run start`
- Your browser should open to localhost:3000, and the GUI should start working after a few seconds.
- You can write Python modules in the tensor_makers folder and they will immediately become available in the website.

## Using our Composable UI (CUI) from a Jupyter notebook:

- Start our JS server, as described above
- In a Jupyter notebook, write code like the following:
```
import interp.cui as cui
import numpy as np
from interp.ui.very_named_tensor import VeryNamedTensor

await cui.init(port=6789)

hidden_size = 20
token_strings = ["Hello","There"]
# using fake data in the shape of a transformer hidden state
tensor = np.random.normal(0,1,(len(token_strings),hidden_size))

my_vnt = VeryNamedTensor(
    tensor,
    dim_names=["Token","Embedding"],
    dim_types=["seq","hidden"],
    dim_idx_names=[
        token_strings,
        [str(i) for i in hidden_size]
    ],
    units="activation",
    title="Random Data",
)

await cui.show_tensors(my_vnt)
```
- See `cui_demo.ipynb` for another example using the CUI. 
