import torch

def normalize(x, lower=50, upper=2050):
    z = (x - x.mean()) / x.std()
    return (z + 2) * (upper-lower) / 4 + lower

def pca_reduce(x, q):
    x = x - x.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(x, q=q)
    return U @ torch.diag(S)