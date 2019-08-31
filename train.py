import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import eval_model

from model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="forai")
    parser.add_argument("--batch", default=1000, type=int)
    parser.add_argument("--test-batch", default=10000, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    name = "training"

    writer = SummaryWriter(name)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # BORROWED THIS PART FROM PYTORCH EXAMPLES
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch,
        shuffle=True,
        **kwargs,
    )
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    nll = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):

        for input, target in train_loader:
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(input)
            loss = nll(output, target)

            loss.backward()
            optimizer.step()

        model.eval()
        test_accuracy = eval_model(model, test_loader, device)

        writer.add_scalar("test accuracy", test_accuracy, epoch)
        print(f"epoch {epoch}, test accuracy {test_accuracy}")

        model.train()

        torch.save(model.state_dict(), f"checkpoint/model_{str(epoch)}.pt")
