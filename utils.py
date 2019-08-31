import torch


def eval_model(model, loader, device):
    with torch.no_grad():
        test_accuracy = []
        for input, target in loader:
            input, target = input.to(device), target.to(device)

            output = model(input)
            batch_accuracy = (output.argmax(dim=1) == target).float()

            test_accuracy.append(batch_accuracy)

        return torch.cat(test_accuracy, 0).mean().item()
