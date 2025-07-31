import torch
import torch.nn.functional as F
import time
from loguru import logger


def cross_entropy(logits, label):
    target = F.one_hot(torch.tensor([label]), num_classes=10)
    logits_softmax = F.softmax(logits, dim=-1)
    loss = -torch.sum(torch.log(logits_softmax) * target)
    return loss


def main():
    a = torch.randn(1, 10, requires_grad=True)  # .to('mps:0')
    # optimizer = torch.optim.SGD([a], lr=0.001)
    optimizer = torch.optim.Adam([a], lr=0.001)

    label = 0
    for i in range(1000000000):
        optimizer.zero_grad()
        b = a / 10.0
        loss = cross_entropy(b, label)
        loss.backward()
        optimizer.step()
        if i % 1000000 == 0:
            logger.info(f"{a[0][label].item()}")
            # print(loss.item())


main()
