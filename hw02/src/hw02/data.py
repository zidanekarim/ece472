from dataclasses import InitVar, dataclass, field

import numpy as np

from .model import LinearModel


@dataclass
class Data:
    """Handles generation of synthetic data for linear regression."""
    noise: float
    def two_spiral_generator(
        self, n_turns: int = 3, points: int=100, *, rng: np.random.Generator
    ):
        """Generates two spirals in order to avoid over-parameterizing this function"""
        temp_radius = np.linspace(1, 10, points)
        radius = rng.normal(loc=temp_radius, scale=self.noise)
        theta = np.linspace(0.0, turns * 2.0 * np.pi, points)

        t0 = theta - np.pi / 2
        x1_0 = radius * np.cos(t0)
        x2_0 = radius * np.sin(t0)
        set0 = np.column_stack([x1_0, x2_0]) # combines coordinates, sort of opposite of flatten



        set1 = -set0

        X = np.vstack([set0, set1])
        Y = np.concatenate([np.zeros(points), np.ones(points)]).astype(int)

        idx = rng.permutation(len(y))
        return X[idx], Y[idx]
        

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()
