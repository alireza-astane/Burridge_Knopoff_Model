import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats

dtype = torch.float32

device = "cuda"


eps = 0.01
F0 = 1
m = 1
h = 0.1


def f(x, v, t, alpha, u, k1, k2, eps):
    xIntoLeft = torch.roll(x, -1, dims=0)
    xIntoRight = torch.roll(x, 1, dims=0)

    xIntoLeft[-1, :] = 0
    xIntoRight[0, :] = 0

    return torch.where(
        v == 0,
        torch.where(
            torch.abs(-x * (2 * k1 + k2) + (xIntoRight + xIntoLeft) * k1 + k2 * u * t)
            < F0,
            -x * (2 * k1 + k2) + (xIntoRight + xIntoLeft) * k1 + k2 * u * t,
            F0 * (1 - eps),
        ),
        F0 * (1 - eps) / (1 + alpha * torch.abs(v) / (1 - eps)),
    )


def acc(x, v, t, alpha, u, k1, k2, eps):
    xIntoLeft = torch.roll(x, -1, dims=0)
    xIntoRight = torch.roll(x, 1, dims=0)

    xIntoLeft[-1, :] = 0
    xIntoRight[0, :] = 0

    return (
        -x * (2 * k1 + k2)
        + (xIntoRight + xIntoLeft) * k1
        + k2 * u * t
        - f(x, v, t, alpha, u, k1, k2, eps)
    ) / m


def tenstion(x, v, t, alpha, u, k1, k2, eps):
    return m * acc(x, v, t, alpha, u, k1, k2, eps) + f(x, v, t, alpha, u, k1, k2, eps)


def step(x, v, t, alpha, u, k1, k2, eps):
    t = torch.where(
        (v == 0).all(0),
        t + (F0 - tenstion(x, v, t, alpha, u, k1, k2, eps).max(0)[0]) / (k2 * u),
        t,
    )

    k_1 = v
    l_1 = acc(x, v, t, alpha, u, k1, k2, eps)

    k_2 = v + l_1 * h / 2
    l_2 = acc(x + k_1 * h / 2, v + l_1 * h / 2, t + h / 2, alpha, u, k1, k2, eps)

    k_3 = v + l_2 * h / 2
    l_3 = acc(x + k_2 * h / 2, v + l_2 * h / 2, t + h / 2, alpha, u, k1, k2, eps)

    k_4 = v + l_3 / 2
    l_4 = acc(x + k_3 * h, v + l_3 * h, t + h, alpha, u, k1, k2, eps)

    deltax = (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h / 6

    # x = x +  deltax

    x = x + torch.where(deltax > 0, deltax, 0)
    v = v + (l_1 + 2 * l_2 + 2 * l_3 + l_4) * h / 6

    v = torch.where(v < 0, 0, v)
    return x, v, t + h


def run(x, v, t, steps, alpha, u, width, L, k1, k2, eps):
    xdata = torch.zeros((steps, L, width), dtype=dtype, device=device)
    vdata = torch.zeros((steps, L, width), dtype=dtype, device=device)
    tdata = torch.zeros((steps, width), dtype=dtype, device=device)
    for i in range(steps):
        x, v, t = step(x, v, t, alpha, u, k1, k2, eps)
        xdata[i] = x
        vdata[i] = v

    slipping = (vdata > 0).any(dim=1)

    slippingR = slipping.roll(+1, dims=0)
    slippingL = slipping.roll(-1, dims=0)
    slippingR[0] = False
    slippingL[-1] = False

    ends = torch.logical_xor(slipping, (slipping * slippingL))
    starts = torch.logical_xor(slipping, (slipping * slippingR))

    totalX = xdata.sum(1)

    eventsSize = totalX.T[ends.T] - totalX.T[starts.T]

    eventsStep = (torch.argwhere(ends.T) - torch.argwhere(starts.T))[:, 1]

    eventsTime = tdata.T[ends.T] - tdata.T[starts.T]

    evetnsBlock = (
        xdata.permute(2, 0, 1)[ends.T] != xdata.permute(2, 0, 1)[starts.T]
    ).sum(dim=1)

    return x, v, t, eventsSize, eventsStep, eventsTime, evetnsBlock


def getData(iteration, steps, width, alpha, u, L, k1, k2, eps):
    means = torch.zeros(iteration)
    x = 0.1 * (torch.rand((L, width), dtype=dtype, device=device) - 1)
    v = torch.zeros((L, width), dtype=dtype, device=device)
    t = torch.zeros(width, dtype=dtype, device=device)
    eventsSizes = torch.empty((0), dtype=dtype, device=device)
    eventsSteps = torch.empty((0), dtype=dtype, device=device)
    eventsTimes = torch.empty((0), dtype=dtype, device=device)
    evetnsBlocks = torch.empty((0), dtype=dtype, device=device)

    for i in tqdm(range(iteration)):
        x, v, t, eventsSize, eventsStep, eventsTime, evetnsBlock = run(
            x, v, t, steps, alpha, u, width, L, k1, k2, eps
        )
        eventsSizes = torch.concatenate((eventsSizes, eventsSize))
        eventsSteps = torch.concatenate((eventsSteps, eventsStep))
        eventsTimes = torch.concatenate((eventsTimes, eventsTime))
        evetnsBlocks = torch.concatenate((evetnsBlocks, evetnsBlock))
        means[i] = eventsSize.mean()

    return eventsSizes, eventsSteps, eventsTimes, means, evetnsBlocks


def getDataWithRest(iteration, steps, width, alpha, u, L, k1, k2, eps):
    means = torch.zeros(iteration)
    eventsSizes = torch.empty((0), dtype=dtype, device=device)
    eventsSteps = torch.empty((0), dtype=dtype, device=device)
    eventsTimes = torch.empty((0), dtype=dtype, device=device)
    evetnsBlocks = torch.empty((0), dtype=dtype, device=device)

    for i in tqdm(range(iteration)):
        x = 0.1 * (torch.rand((L, width), dtype=dtype, device=device) - 1)
        v = torch.zeros((L, width), dtype=dtype, device=device)
        t = torch.zeros(width, dtype=dtype, device=device)
        x, v, t, eventsSize, eventsStep, eventsTime, evetnsBlock = run(
            x, v, t, steps, alpha, u, width, L, k1, k2, eps
        )
        eventsSizes = torch.concatenate((eventsSizes, eventsSize))
        eventsSteps = torch.concatenate((eventsSteps, eventsStep))
        eventsTimes = torch.concatenate((eventsTimes, eventsTime))
        evetnsBlocks = torch.concatenate((evetnsBlocks, evetnsBlock))

        means[i] = eventsSize.mean()

    return eventsSizes, eventsSteps, eventsTimes, means, evetnsBlocks
