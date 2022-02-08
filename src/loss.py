import torch
from einops import reduce, rearrange
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


class SomeLoss(torch.nn.Module):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Loss Config")
        parser.add_argument('--loss_temperature', type=float, default=1.0)
        return parent_parser

    def __init__(self, temperature):
        super(SomeLoss, self).__init__()
        self.temperature = temperature
        self.covar = None

    def forward(self, img, text):
        self._get_inner(img, text)
        covar_temp = reduce(self.covar, 'bi i bt t -> bi bt t', 'max')
        covar_temp *= self.temperature

        loss = 0
        loss += self._i2t(covar_temp)
        loss += self._t2i(covar_temp)
        return loss

    def _i2t(self, covar_temp):
        loss = 0
        loss += self._i2t_first_term(covar_temp)
        loss += self._i2t_second_term(covar_temp)
        return loss

    @staticmethod
    def _i2t_first_term(covar_temp):
        mean = reduce(covar_temp, 'bi bt t -> bi bt', 'mean')
        sigma = torch.diagonal(mean).sum()
        return -sigma

    @staticmethod
    def _i2t_second_term(covar_temp):
        covar_temp_exp = covar_temp.exp()
        covar_second = rearrange(covar_temp_exp, 'bi bt t -> bi (bt t)')
        first_sigma = reduce(covar_second, 'bi t -> bi', 'sum')
        return first_sigma.log().sum()

    def _t2i(self, covar_temp):
        loss = 0
        loss += self._t2i_first_term(covar_temp)
        loss += self._t2i_second_term(covar_temp)
        return loss

    @staticmethod
    def _t2i_first_term(covar_temp):
        first_term_sigma = reduce(covar_temp, 'bi bt t -> bi bt', 'sum')
        first_term_sigma = torch.diagonal(first_term_sigma).sum()
        return -first_term_sigma

    @staticmethod
    def _t2i_second_term(covar_temp):
        covar_temp_exp = covar_temp.exp()
        covar_second = rearrange(covar_temp_exp, 'bi bt t -> bi (bt t)')
        first_sigma = reduce(covar_second, 'bi t -> t', 'sum')
        return first_sigma.log().sum()

    def _get_inner(self, img, text):
        img = normalize(img, dim=2)
        text = normalize(text, dim=2)
        self.covar = torch.inner(img, text)
