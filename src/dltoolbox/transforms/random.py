import numpy as np
import torch
from typing import Dict, Optional, Union

from .core import Transformer, TransformerWithMode


class RandomChoices(TransformerWithMode):
    """Randomly chose one or multiple transforms from a collection of transforms based on its probability."""

    def __init__(
            self,
            choices: Dict[Transformer, float],
            samples: int = 1,
            replace: bool = True,
    ):
        super().__init__()
        self._transforms: Optional[Dict[int, Transformer]] = None
        self._samples = samples
        self._replace = replace
        if len(choices) < samples and not replace:
            raise ValueError("The number of available choices must be greater than or equal to "
                             "the number of samples to be drawn when sampling without replacement.")
        if len(choices) > 0:
            self._transforms = {id(k): k for k in choices.keys()}
            # a-array for np.random.choice
            self._a = np.array(list(self._transforms.keys()))
            # normalize the sum of probabilities to 1.0
            alpha = 1.0 / sum(choices.values())
            # p-array for np.random.choice
            self._p = np.array(list(alpha * v for v in choices.values()))

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.is_eval_mode() or self._transforms is None:
            return x

        chosen_indices = np.random.choice(self._a, size=self._samples, replace=self._replace, p=self._p)
        for chosen_index in chosen_indices:
            x = self._transforms[chosen_index](x)

        return x
