import torch
import torch.nn.functional as F
import time
from loguru import logger


def cross_entropy(logits, label):
    target = F.one_hot(torch.tensor([label]), num_classes=10)

    # logits_softmax = F.softmax(logits, dim=-1)
    # logits_log_softmax = torch.log(logits_softmax)

    logits_log_softmax = torch.log_softmax(logits, dim=-1)
    loss = -torch.sum(logits_log_softmax * target)
    return loss


def main():
    a = torch.randn(1, 10, requires_grad=True)  # .to('mps:0')
    optimizer = torch.optim.SGD([a], lr=0.001)
    # optimizer = torch.optim.Adam([a], lr=0.001)

    label = 0
    for i in range(1000000000):
        if i % 10000 == 0:
            logger.info(f"{a[0][label].item()}")
        # optimizer.zero_grad()
        if a.grad is not None:
            a.grad.zero_()

        # b = torch.exp(a)
        b = (1 - a) * 100
        # b = a
        loss = cross_entropy(b, label)
        loss.backward()
        with torch.no_grad():
            a.sub_(a.grad, alpha=0.001)
            # a -= a.grad * 0.001
        # a = a + a.grad.detach() * 0.001
        # optimizer.step()
            # print(loss.item())


main()
