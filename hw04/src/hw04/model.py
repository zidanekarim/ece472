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


class GroupNorm(nnx.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        epsilon: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.scale = nnx.Param(jnp.ones((1, 1, 1, num_channels), dtype=jnp.float32)) # (1, 1, 1, C) like paper
        self.bias = nnx.Param(jnp.zeros((1, 1, 1, num_channels), dtype=jnp.float32)) # biases init to zero

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        x_grouped = x.reshape((B, self.num_groups, -1))

        mean = jnp.mean(x_grouped, axis=-1, keepdims=True)
        variance = jnp.var(x_grouped, axis=-1, keepdims=True)

        x_norm = (x_grouped - mean) / jnp.sqrt(variance + self.epsilon) # formula
        x_norm = x_norm.reshape((B, H, W, C)) # standard group norm 

        return x_norm * self.scale.value + self.bias.value

class ResidualBlock(nnx.Module): # consulting previous resnet assignment
    def __init__(self,
            input_channels: int,
            output_channels: list[int],
            strides: int,
            num_groups: int,
            *,
            rngs):
        self.strides = (strides, strides)
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv_layer1 = Conv2d(in_features=input_channels, out_features=output_channels,  
                                  kernel_size=(3,3), strides=self.strides, padding="SAME", use_bias=False, rngs=rngs)
        self.conv_layer2 = Conv2d(in_features=output_channels, out_features=output_channels, 
                                          kernel_size=(3,3), strides=(1, 1), padding="SAME", use_bias=False, rngs=rngs)

        self.group_norm1 = GroupNorm(num_groups=num_groups, num_channels=output_channels, epsilon=1e-5, rngs=rngs)
        self.group_norm2 = GroupNorm(num_groups=num_groups, num_channels=output_channels, epsilon=1e-5, rngs=rngs)

        if self.strides[0] != 1 or input_channels != output_channels:
            self.shortcut = Conv2d(
                in_features=input_channels,
                out_features=output_channels,
                kernel_size=(1, 1),
                strides=self.strides,
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            )
            self.shortcut_norm = GroupNorm(
                num_groups=num_groups, num_channels=output_channels, rngs=rngs
            )
        else:
            self.shortcut = None # bc we only do this if dimensions change
            self.shortcut_norm = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        identity = x # we are going to using an identity skip connection #awesomesauce
        out = self.conv_layer1(x)
        out = self.group_norm1(out)
        out = nnx.relu(out)
        out = self.conv_layer2(out)
        out = self.group_norm2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)
            identity = self.shortcut_norm(identity)
        return nnx.relu(out + identity)





class Conv2d(nnx.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        strides,
        padding,
        use_bias, 
        *,
        rngs,
    ):
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
    def __init__(
        self,
        input_channels: int,
        layer_channels: list[int],
        strides: list[int],
        num_classes: int,
        num_groups: int,
        *,
        rngs,
    ):
        self.blocks = nnx.List([])
        self.input_channels = input_channels

        for out_channel, stride in zip(layer_channels, strides, strict=True): # whichever smaller
            self.blocks.append(
                ResidualBlock(
                    input_channels=self.input_channels ,
                    output_channels=out_channel,
                    strides=stride,
                    num_groups=num_groups,
                    rngs=rngs,
                )
            )

            self.blocks.append(
                ResidualBlock(
                    input_channels=out_channel,
                    output_channels=out_channel,
                    strides=1,
                    num_groups=num_groups,
                    rngs=rngs,
                )
            )
            self.input_channels = out_channel

        # Linear classification head mapping the final channels to num_classes
        self.linear_layer = NNXLinearModel(
            rngs, in_features=layer_channels[-1], out_features=num_classes
        )
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x)
        x = jnp.mean(
            x, axis=(1, 2)
        )  # global average pooling so model is invariant to input resolution
        return self.linear_layer(x)



class NNXLinearModel(nnx.Module):
    """A Flax NNX module for a linear regression model, with multi-dimension support added to solve MLP"""

    def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int):
        key = rngs.params()
        self.in_features = in_features  # bc of MLP
        self.out_features = out_features
        std = jnp.sqrt(
            2.0 / in_features
        )  # prevents vanishing gradients, switched to He initialization
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
     