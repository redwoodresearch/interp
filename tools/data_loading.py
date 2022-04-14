from functools import lru_cache
import itertools
import torch
import os
import numpy as np
from einops import rearrange
from tqdm import tqdm
import random
from interp.tools.interpretability_tools import begin_token
from interp.tools.rrfs import RRFS_DIR

DATA_DIR = os.environ.get(
    "INTERP_DATASETS_DIR",
    os.path.expanduser("~/datasets/")
    if os.path.exists(os.path.expanduser("~/datasets/"))
    else f"{RRFS_DIR}/pretraining_datasets/",
)


@lru_cache()
def get_val_seqs(train=False, n_files=2):
    dirname = DATA_DIR + ("owt_tokens_int16" if train else "owt_tokens_int16_val")

    fnames = os.listdir(dirname)[:n_files]
    print("loading data")
    all_tokens = [torch.load(f"{dirname}/{f}") for f in tqdm(fnames)]
    data_pt = list(itertools.chain(*[torch.split(x["tokens"], x["lens"].tolist()) for x in all_tokens]))
    max_size = 511

    data = torch.stack(
        [data_pt_val[:max_size].to(torch.int64) + 32768 for data_pt_val in data_pt if data_pt_val.size(0) >= max_size],
        dim=0,
    ).numpy()
    data = np.concatenate([np.full((data.shape[0], 1), begin_token()), data], axis=1)
    np.random.shuffle(data)
    print("data shape", data.shape)
    return data


@lru_cache()
def get_train_full_length_seqs(n_files=2):
    dirname = DATA_DIR + "owt_tokens_int16"

    fnames = os.listdir(dirname)[:n_files]
    print("loading data")
    all_tokens = [torch.load(f"{dirname}/{f}") for f in tqdm(fnames)]
    data_pt = list(itertools.chain(*[torch.split(x["tokens"], x["lens"].tolist()) for x in all_tokens]))
    data = [datum.to(torch.int32) + 32768 for datum in data_pt]
    print("total num toks", sum([x.shape[0] for x in data]))
    return data


def to_binned_batches(tensors, tokens_per_batch, max_length):
    tensors = [x[:max_length] for x in tensors]
    lengths_tensor = torch.tensor([x.shape[0] for x in tensors])
    idxs_by_length = torch.argsort(lengths_tensor)
    tensors_by_length = [tensors[x] for x in idxs_by_length]
    num_buckets = 10
    cumsum = torch.cumsum(lengths_tensor, dim=0)
    num_tokens = cumsum[-1]
    batches = []
    for bucket in range(num_buckets):
        start_ntoks = (num_tokens * bucket) // num_buckets
        end_ntoks = (num_tokens * (bucket + 1)) // num_buckets
        start_idx = torch.sum((cumsum <= start_ntoks))
        end_idx = torch.sum((cumsum <= end_ntoks))
        bucket_list = tensors_by_length[start_idx:end_idx]
        bucket = torch.stack([x[: bucket_list[0].shape[0]] for x in bucket_list])
        bucket = torch.cat([torch.full((bucket.shape[0], 1), begin_token()), bucket], axis=1)
        batch_size = tokens_per_batch // bucket_list[0].shape[0]
        bucket_batches = np.array(
            rearrange(bucket[: bucket.shape[0] // batch_size * batch_size], "(a b) c -> a b c", b=batch_size)
        )
        batches.extend(bucket_batches)
    print("ntoks", sum([x.size for x in batches]))
    random.shuffle(batches)
    return batches


def to_batches(tensor, batch_size):
    n_seqs = tensor.shape[0]
    dataset_here = tensor[: n_seqs // batch_size * batch_size]
    dataset_batches = rearrange(dataset_here, "(n b) ... -> n b ...", b=batch_size)
    return dataset_batches


def np_log_softmax(x):
    mx = np.max(x, axis=-1).reshape(-1, 1)
    e_x = np.exp(x - mx)
    result = x - mx - np.log(e_x.sum(axis=-1).reshape(-1, 1))
    return result
