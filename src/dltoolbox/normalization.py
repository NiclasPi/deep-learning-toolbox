import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass
class Normalization:
    mean: Union[float, np.ndarray, torch.Tensor]
    std: Union[float, np.ndarray, torch.Tensor]
    perm: Optional[Tuple[int, ...]] = None


class WelfordEstimator:
    """Compute the mean and standard deviation using Welford's iterative algorithm. Data is processed in batches
    which enables the computation on very large datasets which do not fit into main memory.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance"""

    def __init__(self, dim: Optional[Tuple[int, ...]] = None):
        self._initialized = False
        self._dim = dim
        self._permute: Optional[Tuple[int, ...]] = None

        self.count: torch.Tensor
        self.mean: torch.Tensor
        self.std: torch.Tensor
        self.m2: torch.Tensor

    def _try_initialize(self, shape: torch.Size, device: torch.device) -> None:
        if not self._initialized:
            self.count = torch.zeros(1, dtype=torch.float64, device=device)
            if self._dim is None:
                self.mean = torch.zeros(1, dtype=torch.float64, device=device)
                self.std = torch.zeros(1, dtype=torch.float64, device=device)
                self.m2 = torch.zeros(1, dtype=torch.float64, device=device)
            else:
                result_shape = tuple([shape[i] for i in range(len(shape)) if i not in self._dim])
                self.mean = torch.zeros(result_shape, dtype=torch.float64, device=device)
                self.std = torch.zeros(result_shape, dtype=torch.float64, device=device)
                self.m2 = torch.zeros(result_shape, dtype=torch.float64, device=device)

                if any(v > i for i, v in enumerate(self._dim)):
                    # input needs permutation such that the requested dims are the leading dimensions
                    # otherwise computations cannot be applied to the intermediate tensors
                    self._permute = tuple([*self._dim] + [i for i in range(len(shape)) if i not in self._dim])

            self._initialized = True

    def _count_features(self, input_shape: torch.Size) -> torch.Tensor:
        if self._dim is None:
            return torch.prod(torch.tensor(input_shape))
        else:
            return torch.prod(torch.tensor([input_shape[i] for i in range(len(input_shape)) if i in self._dim]))

    def update(self,
               data: torch.Tensor,
               ) -> None:
        """
        Update the internal state by incorporating the input data.

        Args:
            data (torch.Tensor): Input data.
        """

        self._try_initialize(data.shape, data.device)

        data = data.to(dtype=torch.float64)  # convert input to float64 to match the precision of the internal tensors

        self.count += self._count_features(data.shape)
        if self._permute is not None:
            data = torch.permute(data, self._permute)

        delta1 = torch.sub(data, self.mean)
        self.mean += torch.sum(delta1 / self.count, dim=tuple(range(len(self._dim))) if self._dim else None)
        delta2 = torch.sub(data, self.mean)
        self.m2 += torch.sum(delta1 * delta2, dim=tuple(range(len(self._dim))) if self._dim else None)

    def finalize(self,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = None,
                 ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[int, ...]]]:
        """
        Return the mean and standard deviation tensors.

        Additionally, if an input sample requires reordering of dimensions (permutation) before  applying the mean and
        standard deviation, the function returns the dimension ordering needed for this permutation.

        Args:
            dtype (torch.dtype): Data type of the output tensors Defaults to torch.float64.
            device (torch.device, optional): Location of output tensors. Defaults to the device of the input data.

        Returns:
            mean (torch.Tensor): The computed mean values.
            std (torch.Tensor): The computed standard deviation values.
            permutation (tuple, optional): The ordering of dimensions for permutation, if applicable.
        """

        return (self.mean.to(dtype=dtype, device=device),
                torch.sqrt(self.m2 / self.count).to(dtype=dtype, device=device),
                tuple(v - 1 for i, v in enumerate(self._permute) if i > 0) if self._permute is not None else None)
