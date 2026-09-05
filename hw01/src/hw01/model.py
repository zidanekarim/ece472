from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
# Credit to Josh Miao, EE '27 for helping me understand the problem, and Ahikara Sandrasagra EE '28, for providing me with the material to understand

@dataclass
class LinearModel:
    """Represents a simple linear model."""

    weights: np.ndarray
    bias: float

@dataclass 
class Parameters: # avoided calling this simple BasisExpansion since it's not really the same
    mu: np.ndarray # n-dimensional array
    sigma: np.ndarray
    weights: np.ndarray
    bias: float


class NNXLinearModel(nnx.Module):
    """A Flax NNX module for a linear regression model."""

    def __init__(self, *, rngs: nnx.Rngs, num_features: int):
        self.num_features = num_features
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, (self.num_features, 1)))
        self.b = nnx.Param(jnp.zeros((1, 1)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        return jnp.squeeze(x @ self.w.value + self.b.value, axis=-1)

    @property
    def model(self) -> LinearModel:
        """Returns the underlying simple linear model."""
        return LinearModel(
            weights=np.array(self.w.value).reshape([self.num_features]),
            bias=float(np.array(self.b.value).squeeze()),
        )

class NNXBasisExpansion(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs, num_features: int):
        self.num_features = num_features
        key1, key2 = jax.random.split(rngs.params())
        # src: https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/variables.html
        # param required for learnable & automatic diff
        self.mu = nnx.Param(jax.random.normal(key1, (self.num_features, 1)))
        self.sig = nnx.Param(jnp.ones((self.num_features)) * 0.16) # why 0.16? let M=6, 1/M



    def __call__(self, x: jax.Array):
        """Predicts the output for a given input.""" # copied from NNXLinearModel
        return jnp.exp(- (x-self.mu.value)**2 / self.sig.value**2)

        



class NNXRegressionModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs, num_features: int):
        self.basis = NNXBasisExpansion(num_features=num_features)
        self.linear = NNXLinearModel(rngs=rngs, num_features=num_features)

    def __call__(self, x: jax.Array) -> jax.Array:
        phi_x = self.basis(x)
        return self.linear(phi_x) # we can pass phi_x in place of x as according to given functional form
        # and we need to read from this regression rather than the straight linear
    @property
    def model(self) -> Parameters:
        return Parameters(
            mu=np.array(self.basis.mu.value),
            sigma=np.array(self.basis.sig.value),
            weights=np.array(self.linear.w.value).reshape([self.linear.num_features]),
            bias=float(np.array(self.linear.b.value).squeeze()),
        )