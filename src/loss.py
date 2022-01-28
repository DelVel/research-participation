import torch
from einops import reduce
from torch.nn.functional import normalize


def cross_loss(img, text):
    assert img.shape[0] == text.shape[0]
    batch = img.shape[0]
    mat = get_mat(img, text)
    max_elem = torch.diagonal(mat).mean()
    min_elem = torch.triu(mat, diagonal=1).sum() / (batch * (batch - 1) / 2)
    return min_elem - max_elem


def get_mat(img, text):
    img = normalize(img, dim=2)
    text = normalize(text, dim=2)
    covar = torch.inner(img, text)
    covar = reduce(covar, 'b1 i b2 t -> b1 b2 t', 'max')
    covar = reduce(covar, 'b1 b2 t -> b1 b2', 'mean')
    return covar
