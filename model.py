import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    """
    it's just a Linear module which have mask over the weight matrix
    in order to turn off some weights during the forward propagation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight, dtype=torch.bool))

        self.optimized = False

    def forward(self, input):
        if not self.optimized:
            return F.linear(input, self.weight.masked_fill(self.mask, 0), self.bias)
        else:
            result = torch.sparse.mm(self.weight, input.t()).t()
            if self.bias is not None:
                result = result + self.bias

            return result

    def prune(self, k, type):

        self.mask.zero_()

        if type == "weight":
            # find lowest k% of weights.
            # .numel() is the total number of elements in the tensor
            _, prune_indices = torch.topk(
                self.weight.abs().flatten(),
                int(self.weight.numel() * k / 100),
                largest=False,
            )

            mask = self.mask.flatten()
            mask[prune_indices] = True
            self.mask = mask.view(*self.weight.size())
        elif type == "unit":
            norms = self.weight.norm(dim=1)

            _, prune_indices = torch.topk(
                norms, int(norms.numel() * k / 100), largest=False
            )

            mask = torch.zeros_like(norms, dtype=torch.bool)
            mask[prune_indices] = True
            self.mask = mask.unsqueeze(-1).repeat(1, self.weight.size(-1))

    def optimize_pruned(self):
        # optimize computations for sparse weights
        self.weight = nn.Parameter(self.weight.masked_fill(self.mask, 0).to_sparse())

        self.optimized = True


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.net = nn.Sequential(
            MaskedLinear(input_size, 1000, bias=False),
            nn.ReLU(),
            MaskedLinear(1000, 1000, bias=False),
            nn.ReLU(),
            MaskedLinear(1000, 500, bias=False),
            nn.ReLU(),
            MaskedLinear(500, 200, bias=False),
            nn.ReLU(),
            nn.Linear(200, output_size),
        )

    def forward(self, input):
        return self.net(input.flatten(1))

    def prune(self, k, type):

        assert type in ["weight", "unit"]

        if k > 0:
            for layer in self.modules():
                if isinstance(layer, MaskedLinear):
                    layer.prune(k, type)

    def optimize_pruned(self):
        for layer in self.modules():
            if isinstance(layer, MaskedLinear):
                layer.optimize_pruned()
