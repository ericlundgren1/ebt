"""Loss functions for ``NeuralNetModel``.

A loss is a callable ``loss(prediction, labels) -> scalar tensor``. ``LossFunction``
adds the ``create`` / ``create_partial`` factory hooks that ``NeuralNetModel`` uses to
build the loss from its hyperparameters.
"""
from functools import partial
import torch


class LossFunction():
    def __call__(self, *args, **kwargs):
        self.result = self.calculate_loss(*args, **kwargs)
        if type(self.result) == dict:
            return self.result['loss']
        return self.result

    def calculate_loss(self, prediction: torch.Tensor, labels: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    def create(klass, *args, **kwargs):
        return klass(**kwargs)

    @classmethod
    def create_partial(klass, *args, **kwargs):
        return partial(klass.create, *args, **kwargs)


class MSELoss(LossFunction):
    """Mean squared error over the label entries that are not NaN."""
    def __init__(self, *args, **kwargs):
        pass

    def calculate_loss(self, prediction, labels, *args, **kwargs):
        mask = ~labels.isnan()
        diff = torch.where(mask, prediction - labels, torch.zeros_like(prediction))
        return (diff ** 2).sum() / mask.sum().clamp(min=1)
