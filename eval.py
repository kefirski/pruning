import argparse
import random
from time import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Model
from utils import eval_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="forai")
    parser.add_argument("--test-batch", default=10000, type=int)
    parser.add_argument("--model", default="checkpoint/model_99.pt", type=str)
    parser.add_argument("--sparcity", default="weight", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    writer = SummaryWriter(str(args.sparcity))

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # BORROWED THIS PART FROM PYTORCH EXAMPLES
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch,
        shuffle=False,
        **kwargs,
    )
    # ------------------------------------------

    model = Model(28 ** 2, 10).to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # accuracy test
    for k in [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]:
        model.prune(k, args.sparcity)

        test_accuracy = eval_model(model, test_loader, device)

        print(f"k = {k}, test accuracy {test_accuracy}")
        writer.add_scalar("test accuracy-pruning", test_accuracy, k)

    # speed test
    model.prune(99, args.sparcity)
    test_input = torch.randn(100, 1, 28, 28)

    opt_time = []

    for _ in range(1000):
        start = time()
        ort_inputs = model(test_input)
        end = time()
        opt_time.append(end - start)

    print(f"no optimization time {sum(opt_time) / len(opt_time)}")

    model.optimize_pruned()

    opt_time = []

    for _ in range(1000):
        start = time()
        ort_inputs = model(test_input)
        end = time()
        opt_time.append(end - start)

    print(f"optimization time {sum(opt_time) / len(opt_time)}")
