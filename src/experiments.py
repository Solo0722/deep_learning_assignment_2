"""All experiment definitions live here. One factor changes at a time versus `baseline`.

Baseline = LeNet-5 (LeCun et al., 1998) as commonly reimplemented:
    32x32 input -> conv(6,5x5) -> tanh -> avgpool -> conv(16,5x5) -> tanh -> avgpool
    -> FC120 -> tanh -> FC84 -> tanh -> FC10
Known deviations from the paper (state them in your report): plain average pooling instead of
trainable subsampling, full S2->C3 connectivity instead of the sparse table, softmax/cross-entropy
instead of RBF output units, Adam instead of the paper's SGD schedule.

Extra conv layers (depth > 2) use 'same' padding, no pooling, and the channel count of the last
baseline conv, so the spatial size stays valid.
"""


def _conv(out, k, pad, pool):
    return {"out": out, "k": k, "pad": pad, "pool": pool}


def make(name, *, channels=(6, 16), k=5, n_conv=2, fc=(120, 84), activation="tanh",
         pool="avg", batchnorm=False, dropout=0.0):
    convs = []
    for i in range(n_conv):
        if i < 2:
            convs.append(_conv(channels[i], k, 0, True))
        else:
            convs.append(_conv(channels[-1], k, k // 2, False))
    return {"name": name, "convs": convs, "fc": list(fc), "activation": activation,
            "pool": pool, "batchnorm": batchnorm, "dropout": dropout}


CONFIGS = {}


def add(name, **kw):
    CONFIGS[name] = make(name, **kw)


add("baseline")

# 1. Width (out channels of conv1, conv2)
add("width_3_8", channels=(3, 8))
add("width_12_32", channels=(12, 32))
add("width_24_64", channels=(24, 64))
add("width_48_128", channels=(48, 128))

# 2a. Conv depth
add("conv_1", n_conv=1)
add("conv_3", n_conv=3)
add("conv_4", n_conv=4)

# 2b. FC depth (hidden layers before the 10-way output)
add("fc_0", fc=())
add("fc_1", fc=(120,))
add("fc_3", fc=(120, 84, 64))

# 3. Activation
add("act_relu", activation="relu")
add("act_leaky_relu", activation="leaky_relu")
add("act_gelu", activation="gelu")

# 4. Pooling
add("pool_max", pool="max")

# 5. Kernel size (applies to both conv layers)
add("kernel_3", k=3)
add("kernel_7", k=7)

# 6. Regularisation
add("batchnorm", batchnorm=True)
add("dropout_20", dropout=0.2)
add("dropout_50", dropout=0.5)

# 7. Best combined model: fill this in AFTER looking at validation results, then run
#    `python -m src.run_all --factors combined`. It is its own configuration, tested with all seeds.
#    Example:  add("combined_1", channels=(12, 32), activation="relu", pool="max", dropout=0.2)
#    then:     COMBINED_NAMES = ["combined_1"]
COMBINED_NAMES = []

FACTORS = {
    "width": ["width_3_8", "baseline", "width_12_32", "width_24_64", "width_48_128"],
    "conv_depth": ["conv_1", "baseline", "conv_3", "conv_4"],
    "fc_depth": ["fc_0", "fc_1", "baseline", "fc_3"],
    "activation": ["baseline", "act_relu", "act_leaky_relu", "act_gelu"],
    "pooling": ["baseline", "pool_max"],
    "kernel": ["kernel_3", "baseline", "kernel_7"],
    "regularization": ["baseline", "batchnorm", "dropout_20", "dropout_50"],
    "combined": ["baseline"] + COMBINED_NAMES,
}
