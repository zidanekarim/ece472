from dataclasses import InitVar, dataclass, field

import numpy as np

from .model import LinearModel


@dataclass
class Data:
    """Handles generation of synthetic data for linear regression."""

    model: LinearModel
    rng: InitVar[np.random.Generator]
    num_features: int
    num_samples: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """Generate synthetic data based on the model."""
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(0, 1.0, size=(self.num_samples, self.num_features)) # manually changed from (0.1, 0.9) to (0, 1)
        #clean_y = self.x @ self.model.weights[:, np.newaxis] + self.model.bias
        # I changed clean_y from the weights + bias here, to the sin wave output. 
        # This is because in training:loss_fn, the y^ is already taken from the model, so I didn't really understand why I should basically replicate that in this step. emailed Prof Curro about this
        #clean_y = np.sin(2*np.pi*self.x)         
        self.y = rng.normal(loc=clean_y, scale=self.sigma) # this produces the + epsilon_i from my understanding

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()
