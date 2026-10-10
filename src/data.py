"""Data loading.

The whole dataset is kept on the compute device as tensors (MNIST is tiny), which avoids
DataLoader overhead and makes each run take seconds instead of minutes.

Split (fixed for EVERY run, so comparisons are valid):
    60,000 train images -> 54,000 train / 6,000 validation   (split seed = 0)
    10,000 official test images -> test (touched once per run, at the best-validation epoch)
Preprocessing: pad 28x28 -> 32x32 with background (as in the LeNet-5 paper), then standardise.
"""
import torch
import torch.nn.functional as F

STATS = {"mnist": (0.1307, 0.3081), "fashion": (0.2860, 0.3530)}
N_VAL = 6000
SPLIT_SEED = 0


def _prep(images_uint8, mean, std):
    x = images_uint8.float().div(255.0)
    x = F.pad(x, (2, 2, 2, 2), value=0.0)  # 28x28 -> 32x32
    x = (x - mean) / std
    return x.unsqueeze(1)  # (N, 1, 32, 32)


def build_splits(train_x, train_y, test_x, test_y, dataset, device, train_subset=None):
    mean, std = STATS[dataset]
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(train_x), generator=g)
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
    if train_subset:  # only used by --smoke
        tr_idx = tr_idx[:train_subset]
    x_all, y_all = _prep(train_x, mean, std), train_y.long()
    xt, yt = _prep(test_x, mean, std), test_y.long()
    out = {
        "train": (x_all[tr_idx], y_all[tr_idx]),
        "val": (x_all[val_idx], y_all[val_idx]),
        "test": (xt, yt),
    }
    return {k: (x.to(device), y.to(device)) for k, (x, y) in out.items()}


def load_data(dataset, root, device, train_subset=None):
    from torchvision import datasets  # lazy import

    cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[dataset]
    tr = cls(root, train=True, download=True)
    te = cls(root, train=False, download=True)
    return build_splits(tr.data, tr.targets, te.data, te.targets, dataset, device, train_subset)
