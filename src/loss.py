from abc import ABCMeta, abstractmethod

import torch
from einops import reduce
from torch.nn import Module
from torch.nn.functional import normalize


class InnerLoss(Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.covar = None
        self.batch = None

    def forward(self, img, text):
        """
        Loss function.

        :param img: Shape of (batch, z_per_i, embedding_dim)
        :param text: Shape of (batch, captions, embedding_dim)
        :return: Loss value.
        """
        assert img.shape[0] == text.shape[0]
        assert img.shape[2] == text.shape[2]
        self._get_inner(img, text)
        self.batch = img.shape[0]
        return self.post_forward(img, text)

    @abstractmethod
    def post_forward(self, img, text):
        """
        Forward pass after getting inner product.

        :return: self.covar for inner product tensor, self.batch for batch size
        of the input.
        """
        pass

    def _get_inner(self, img, text):
        img = normalize(img, dim=2)
        text = normalize(text, dim=2)
        self.covar = torch.inner(img, text)


class CrossLoss(InnerLoss):
    def post_forward(self, img, text):
        covar = self.covar
        covar = reduce(covar, 'b1 i b2 t -> b1 b2 t', 'max')
        covar = reduce(covar, 'b1 b2 t -> b1 b2', 'mean')
        max_elem = covar.trace() / self.batch
        min_elem = 0
        if self.batch > 1:
            min_elem = torch.triu(covar, diagonal=1).sum() / (
                    self.batch * (self.batch - 1) / 2)
        return min_elem - max_elem


class ContrastiveLoss(InnerLoss):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Loss Config")
        parser.add_argument('--loss_temperature', type=float, default=1.0)
        return parent_parser

    def __init__(self, temperature, epsilon=10):
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def post_forward(self, img, text):
        self.covar = reduce(self.covar, 'bi i bt t -> bi bt t', 'max')
        self.covar *= self.temperature
        return self.sum_up_loss()

    def sum_up_loss(self):
        loss = 0
        loss += self._i2t(self.covar)
        loss += self._t2i(self.covar)
        return loss

    def _i2t(self, covar):
        loss = 0
        loss += self._i2t_term_1(covar)
        loss += self._i2t_term_2(covar)
        return loss

    @staticmethod
    def _i2t_term_1(covar):
        mean = reduce(covar, 'bi bt t -> bi bt', 'mean')
        sigma = mean.trace()
        return -sigma

    def _i2t_term_2(self, covar):
        covar_exp = covar.exp()
        sigma = reduce(covar_exp, 'bi bt t -> bi', 'sum')
        sigma += self.epsilon
        return sigma.log().sum()

    def _t2i(self, covar):
        loss = 0
        loss += self._t2i_term_1(covar)
        loss += self._t2i_term_2(covar)
        return loss

    @staticmethod
    def _t2i_term_1(covar_temp):
        mean = reduce(covar_temp, 'bi bt t -> bi bt', 'sum')
        sigma = mean.trace()
        return -sigma

    def _t2i_term_2(self, covar_temp):
        covar_exp = covar_temp.exp()
        sigma = reduce(covar_exp, 'bi bt t -> bt t', 'sum')
        sigma += self.epsilon
        return sigma.log().sum()
