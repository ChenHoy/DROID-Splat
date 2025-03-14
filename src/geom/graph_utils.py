from collections import OrderedDict

import lietorch
import numpy as np
import torch


def graph_to_edge_list(graph):
    ii, jj, kk = [], [], []
    for s, u in enumerate(graph):
        for v in graph[u]:
            ii.append(u)
            jj.append(v)
            kk.append(s)

    ii = torch.as_tensor(ii)
    jj = torch.as_tensor(jj)
    kk = torch.as_tensor(kk)
    return ii, jj, kk


def keyframe_indicies(graph):
    return torch.as_tensor([u for u in graph])


def meshgrid(m, n, device="cuda"):
    ii, jj = torch.meshgrid(torch.arange(m), torch.arange(n))
    return ii.reshape(-1).to(device), jj.reshape(-1).to(device)


def neighbourhood_graph(n, r):
    ii, jj = meshgrid(n, n)
    d = (ii - jj).abs()
    keep = (d >= 1) & (d <= r)
    return ii[keep], jj[keep]
