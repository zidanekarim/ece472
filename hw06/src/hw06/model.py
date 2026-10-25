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


class Attention(nnx.Module):
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array | None, causal: bool=False):
        d_k = q.shape[-1] # dimension of the key
        t_q = q.shape[-2] # sequence length of the query
        t_k = k.shape[-2] # sequence length of the key
        mask = mask if mask is not None else jnp.ones((t_q, t_k))

        scale = 1.0 / jnp.sqrt(d_k)
        scores = jnp.einsum('...qd,...kd->...qk', q, k) * scale

        if causal:
            causal_mask = jnp.tril(jnp.ones((t_q, t_k))) # lower triangular matrix
            mask = mask * causal_mask
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9) # apply mask

        weights = nnx.softmax(scores, axis=-1)
        return jnp.einsum('...qk,...vd->...qd', weights, v)


class MultiHeadAttention(nnx.Module):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O"""
    """and head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)"""
    def __init__(self, d_model, num_heads, *, rngs, use_bias):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_bias = use_bias

        scale = 1.0 / jnp.sqrt(self.d_model)
        self.w_q = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, d_model)) * scale
        )
        self.w_k = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, d_model)) * scale
        )
        self.w_v = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, d_model)) * scale
        )
        self.w_o = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, d_model)) * scale
        )

        if self.use_bias:
            self.b_q = nnx.Param(jnp.zeros((d_model,))) # query bias
            self.b_k = nnx.Param(jnp.zeros((d_model,))) # key bias
            self.b_v = nnx.Param(jnp.zeros((d_model,))) # value bias
            self.b_o = nnx.Param(jnp.zeros((d_model,))) # output bias
        else:
            self.b_q = None # define for python type checking
            self.b_k = None
            self.b_v = None
            self.b_o = None
        self.attention = Attention() # initialize the attention module

    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array | None = None, causal: bool = False):
        # Linear projections
        q_proj = jnp.einsum('...qd,df->...qf', q, self.w_q.value)
        k_proj = jnp.einsum('...kd,df->...kf', k, self.w_k.value)
        v_proj = jnp.einsum('...vd,df->...vf', v, self.w_v.value)

        if self.use_bias:
            q_proj += self.b_q.value
            k_proj += self.b_k.value
            v_proj += self.b_v.value

        q_proj = q_proj.reshape(q.shape[:-1] + (self.num_heads, self.head_dim))
        k_proj = k_proj.reshape(k.shape[:-1] + (self.num_heads, self.head_dim))
        v_proj = v_proj.reshape(v.shape[:-1] + (self.num_heads, self.head_dim))

        # Attention(Q W_i^Q, K W_i^K, V W_i^V)
        attn_output = self.attention(q_proj, k_proj, v_proj, mask=mask, causal=causal)

        # MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        attn_output = attn_output.reshape(attn_output.shape[:-2] + (self.d_model,))
        output = jnp.einsum('...qd,df->...qf', attn_output, self.w_o.value)

        if self.use_bias:
            output += self.b_o.value

        return output


class TransformerBlock(nnx.Module):
    """
    y = x + Attention(LayerNorm(x))
    Output = y + FFN(LayerNorm(y))
    """

    def __init__(self, d_model, num_heads, d_ff, *, rngs, use_bias):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_bias = use_bias

        self.mha = MultiHeadAttention(d_model, num_heads, rngs=rngs, use_bias=use_bias)
        self.ffn_w1 = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, d_ff)) * (1.0 / jnp.sqrt(d_model))
        )
        self.ffn_w2 = nnx.Param(
            jax.random.normal(rngs.params(), (d_ff, d_model)) * (1.0 / jnp.sqrt(d_ff))
        )

        if self.use_bias:
            self.ffn_b1 = nnx.Param(jnp.zeros((d_ff,)))
            self.ffn_b2 = nnx.Param(jnp.zeros((d_model,)))
        else:
            self.ffn_b1 = None
            self.ffn_b2 = None