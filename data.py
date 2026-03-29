import torch

def normalize(x, lower, upper):
    freq = x.abs() / x.std() * (upper-lower) / 4 + lower
    return freq

def pca_reduce(x, q):
    x = x - x.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(x, q=q)
    return U @ torch.diag(S)

def to_uniform(x, lower, upper):
    T, V = x.shape
    rankings = x.flatten().argsort().argsort().view(T, V)
    normalized = rankings / torch.max(rankings)
    return normalized * (upper - lower) + lower