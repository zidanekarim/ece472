from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass
class LinearModel:
    """Represents a simple linear model."""

    weights: np.ndarray
    bias: float


# straight pasted from assignment 2
class SK_NNXClassifier(BaseEstimator, ClassifierMixin):
    """Allows a Flax NNX model to be used with scikit-learn. DecisionBoundaryDisplay
    builds a 2d grid of points and calls the model to get predictions for each point. This wrapper is necessary because the model is not a scikit-learn model and does not have a predict method.
    """

    def __init__(self, model):
        self.model = model
        # 0, 1 because its either red or blue. this is used by DecisionBoundaryDisplay to determine the classes of the model.

    def predict_proba(self, X):
        """Returns the predicted class probabilities for the input data."""
        X = jnp.asarray(X)
        logits = self.model(X)
        probs = jax.nn.softmax(logits, axis=-1)
        return np.array(probs)  # convert back to numpy array for sklearn

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=-1)  # returns the index of the max probability

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self  # had to do this because sklearn fails otherwise


class MLP(nnx.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden_layers: int,
        hidden_layer_width: int,
        hidden_activation=nnx.gelu,
        output_activation=nnx.identity,
        *,
        rngs: nnx.Rngs,
    ):
        layers = []
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        layers.append(
            NNXLinearModel(rngs, num_inputs, hidden_layer_width)
        )  # first layer hidden
        for i in range(num_hidden_layers - 1):
            layers.append(NNXLinearModel(rngs, hidden_layer_width, hidden_layer_width))

        self.hidden_layers = nnx.List(layers)
        self.out_layer = NNXLinearModel(rngs, hidden_layer_width, num_outputs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.hidden_layers:
            x = self.hidden_activation(
                layer(x)
            )  # at each layer, compute layer(x) = xW + b. then the hidden actrivation is applying the sigmoid or gelu. this then feeds as the next input (sort of recursive calls?)
        return self.output_activation(
            self.out_layer(x)
        )  # take final-1 input, feed it to out_layer, and feed output_activation function to that


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


class SparseAutoencoder(nnx.Module):

  def __init__(
      self, d_in: int, m_latents: int, k: int | None = None, *, rngs: nnx.Rngs
  ):
    self.d_in = d_in
    self.m_latents = m_latents
    self.k = k  # Set k (e.g., 16 or 32) for Top-k SAE; None for ReLU + L1

    # Initialize encoder and decoder weights
    # Tip: Initializing W_enc as W_dec.T (unit-norm columns) stabilizes early training
    w_dec_init = jax.nn.initializers.lecun_normal()(
        rngs.params(), (m_latents, d_in)
    )
    w_dec_init = w_dec_init / (
        jnp.linalg.norm(w_dec_init, axis=-1, keepdims=True) + 1e-8
    )

    self.W_dec = nnx.Param(w_dec_init)  # Shape: (m, d)
    self.W_enc = nnx.Param(w_dec_init.T.copy())  # Shape: (d, m)
    self.b_enc = nnx.Param(jnp.zeros((m_latents,)))
    self.b_dec = nnx.Param(jnp.zeros((d_in,)))

  def encode(self, x: jax.Array) -> jax.Array:
    # Center input using learned decoder bias
    x_cent = x - self.b_dec.value
    pre_acts = x_cent @ self.W_enc.value + self.b_enc.value

    if self.k is not None:
      # Top-k activation sparsity
      top_vals, top_idx = jax.lax.top_k(pre_acts, self.k)
      top_vals = jax.nn.relu(top_vals)
      z = jnp.zeros_like(pre_acts)
      z = z.at[
          jnp.arange(pre_acts.shape[0])[:, None], top_idx
      ].set(top_vals)
      return z
    else:
      # Standard ReLU for L1 sparsity
      return jax.nn.relu(pre_acts)

  def decode(self, z: jax.Array) -> jax.Array:
    return z @ self.W_dec.value + self.b_dec.value

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    z = self.encode(x)
    x_hat = self.decode(z)
    return x_hat, z