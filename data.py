import torch

def normalize(x, lower, upper):
    z = (x - x.mean()) / x.std()
    freq = (z + 2) * (upper-lower) / 4 + lower
    return freq
    # return torch.clamp(freq, lower, upper)

def pca_reduce(x, q):
    x = x - x.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(x, q=q)
    return U @ torch.diag(S)

def to_uniform(x, lower, upper):
    T, V = x.shape
    rankings = x.flatten().argsort().argsort().view(T, V)
    normalized = rankings / torch.max(rankings)
    return normalized * (upper - lower) + lower