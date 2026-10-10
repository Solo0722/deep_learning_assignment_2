"""Config-driven LeNet. Every architectural change in the study is expressed through the config dict."""
import torch
import torch.nn as nn

ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "gelu": nn.GELU,
}


class LeNetVariant(nn.Module):
    """
    cfg keys:
        convs      list of {"out", "k", "pad", "pool"} dicts, one per conv layer
        fc         list of hidden FC sizes (e.g. [120, 84]); final 10-way layer is always added
        activation key of ACTIVATIONS
        pool       "avg" | "max"
        batchnorm  bool (after each conv, before activation)
        dropout    float (after each hidden FC activation)
    The flattened size is inferred with a dummy forward pass, so any conv change just works.
    """

    def __init__(self, cfg, in_ch=1, in_size=32, n_classes=10):
        super().__init__()
        act = ACTIVATIONS[cfg["activation"]]
        Pool = nn.AvgPool2d if cfg["pool"] == "avg" else nn.MaxPool2d

        layers, c = [], in_ch
        for L in cfg["convs"]:
            layers.append(nn.Conv2d(c, L["out"], L["k"], padding=L["pad"]))
            if cfg["batchnorm"]:
                layers.append(nn.BatchNorm2d(L["out"]))
            layers.append(act())
            if L["pool"]:
                layers.append(Pool(2, 2))
            c = L["out"]
        self.features = nn.Sequential(*layers)

        self.features.eval()
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, in_ch, in_size, in_size)).numel()
        self.features.train()

        head, d = [nn.Flatten()], n_flat
        for h in cfg["fc"]:
            head += [nn.Linear(d, h), act()]
            if cfg["dropout"] > 0:
                head.append(nn.Dropout(cfg["dropout"]))
            d = h
        head.append(nn.Linear(d, n_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
