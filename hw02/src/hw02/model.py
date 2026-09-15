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

class MLP(nnx.Module):
    

    def __init__(self, num_inputs: int, num_outputs: int, num_hidden_layers: int, hidden_layer_width: int, hidden_activation=nnx.gelu, output_activation=nnx.identity, *, rngs: nnx.Rngs):
        layers = []
        self.hidden_activation = hidden_activation
        self.output_activation=output_activation
        
        layers.append(NNXLinearModel(rngs, num_inputs, hidden_layer_width)) # first layer hidden
        for i in range(num_hidden_layers-1):
            layers.append(Linear(hidden_layer_width, hidden_layer_width, rngs=rngs))
        
        self.hidden_layers = nnx.List(layers)
        self.out_layer = NNXLinearModel(hidden_layer_width, num_outputs, rngs=rngs) 

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.hidden_layers:
            x = self.hidden_activation(layer(x)) # at each layer, compute layer(x) = xW + b. then the hidden actrivation is applying the sigmoid or gelu. this then feeds as the next input (sort of recursive calls?)
        return self.output_activation(self.out_layer(x)) # take final-1 input, feed it to out_layer, and feed output_activation function to that

class NNXLinearModel(nnx.Module):
    """A Flax NNX module for a linear regression model, with multi-dimension support added to solve MLP"""

    def __init__(self, *, rngs: nnx.Rngs, in_features: int, out_features: int):
        key = rngs.params()
        self.in_features = in_features # bc of MLP
        self.out_features = out_features
        std = jnp.sqrt(1.0 / in_features) # prevents vanishing gradients 
        self.w = nnx.Param(jax.random.normal(key, (in_features, out_features)) * std)
        self.b = nnx.Param(jnp.zeros((1, out_features))) # broadcasts for output

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        return x @ self.w.value + self.b.value # squeeze gone to maintain hidden layers 

    # @property
    # def model(self) -> LinearModel:
    #     """Returns the underlying simple linear model."""
    #     return LinearModel(
    #         weights=np.array(self.w.value).reshape([self.num_features]),
    #         bias=float(np.array(self.b.value).squeeze()),
    #     )
