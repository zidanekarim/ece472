from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class LinearModel:
    """Represents a simple linear model."""

    weights: np.ndarray
    bias: float


class Conv2d(nnx.Module):

    def __init__(self, in_features, out_features, kernel_size, strides, padding, use_bias, *, rngs):
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            rngs=rngs,
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x)

class Classifier(nnx.Module):

    def __init__(self, input_channels: int,
                layer_channels: list[int],
                kernel_sizes: list[tuple[int, int]],
                strides: list[int],
                num_classes: int,
                ):
        self.conv_layers = []
        self.input_channels = input_channels
        

         



class NNXLinearModel(nnx.Module):
    """A Flax NNX module for a linear regression model, with multi-dimension support added to solve MLP"""

    def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int):
        key = rngs.params()
        self.in_features = in_features  # bc of MLP
        self.out_features = out_features
        std = jnp.sqrt(1.0 / in_features)  # prevents vanishing gradients
        self.w = nnx.Param(jax.random.normal(key, (in_features, out_features)) * std)
        self.b = nnx.Param(jnp.zeros((1, out_features)))  # broadcasts for output

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        return x @ self.w.value + self.b.value  # squeeze gone to maintain hidden layers

    # @property
    # def model(self) -> LinearModel:
    #     """Returns the underlying simple linear model."""
    #     return LinearModel(
    #         weights=np.array(self.w.value).reshape([self.num_features]),
    #         bias=float(np.array(self.b.value).squeeze()),
    #     )